"""Federated Learning Security Auditing, Operations Lifecycle, and Candidate Promotion."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, Field

from agrifed_khaos.crypto import compute_manifest_hmac, compute_sha256

HEAD_PREFIXES = ("fc.", "classifier.", "head.", "model.head.", "linear.")


class ClientUpdateAuditResult(BaseModel):
    """Structured security and lineage audit result for a candidate client update."""

    file_name: str
    file_path: str
    client_id: str = "unknown"
    size_bytes: int = 0
    sha256: str = ""
    expected_sha256: str | None = None
    base_model_sha256: str | None = None
    target_base_sha256: str | None = None
    lineage_matched: bool = False
    num_samples: int = 0
    training_scope: str = "unknown"
    has_hmac: bool = False
    hmac_valid: bool | None = None
    schema_valid: bool = False
    all_finite: bool = False
    is_eligible: bool = False
    rejection_reasons: list[str] = Field(default_factory=list)
    created_at: str | None = None


class FederatedAggregationRequest(BaseModel):
    """Payload schema for requesting a preflight-checked federated aggregation."""

    target_model_slug: str
    selected_update_paths: list[str]
    server_momentum: float = 0.9
    server_lr: float = 1.0
    allow_new_classes: bool = True
    explicit_val_dir: str | None = None
    max_acc_drop: float = 0.03


class FederatedOperationRecord(BaseModel):
    """Traceable record of an asynchronous federated aggregation and promotion cycle."""

    operation_id: str
    created_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    target_model_slug: str
    base_checkpoint: str
    base_checkpoint_sha256: str
    candidate_checkpoint: str | None = None
    selected_updates: list[str] = Field(default_factory=list)
    aggregation_params: dict[str, Any] = Field(default_factory=dict)
    guardrail_status: str = "PENDING_EVALUATION"  # PASSED | REJECTED | PENDING_EVALUATION
    pre_aggregation_acc: float | None = None
    pre_aggregation_loss: float | None = None
    post_aggregation_acc: float | None = None
    post_aggregation_loss: float | None = None
    accuracy_delta: float | None = None
    loss_delta: float | None = None
    val_dir_used: str | None = None
    # PENDING | RUNNING | COMPLETED | REJECTED_REGRESSION | FAILED | PROMOTED
    status: str = "PENDING"
    failure_reasons: list[str] = Field(default_factory=list)
    promoted_at: str | None = None


def resolve_meta_path(client_file: Path, explicit_meta: str | None = None) -> Path:
    """Resolve corresponding metadata JSON path for a client update checkpoint."""
    if explicit_meta:
        return Path(explicit_meta)
    cand_dot = client_file.with_name(f"{client_file.name}.meta.json")
    if cand_dot.is_file():
        return cand_dot
    cand_suffix = client_file.with_suffix(".meta.json")
    if cand_suffix.is_file():
        return cand_suffix
    meta_in_dir = client_file.parent / "metadata.json"
    if meta_in_dir.is_file():
        return meta_in_dir
    return cand_suffix


_AUDIT_CACHE: dict[tuple[str, int, int, int, str | None, str | None], ClientUpdateAuditResult] = {}
_AUDIT_CACHE_MAX = 256


def audit_client_update(
    client_path: Path,
    target_base_checkpoint: Path | None = None,
    hmac_key: str | None = None,
    target_arch: str = "resnet18",
) -> ClientUpdateAuditResult:
    """Rigorous preflight security audit of a client parameters artifact."""
    res = ClientUpdateAuditResult(
        file_name=client_path.name,
        file_path=str(client_path),
        rejection_reasons=[],
    )

    # 1. Existence and basic size sanity check
    if not client_path.is_file():
        res.rejection_reasons.append(f"El archivo '{client_path}' no existe en disco.")
        return res

    st = client_path.stat()
    res.size_bytes = st.st_size
    if res.size_bytes < 500:
        # Stub or corrupt tiny file (e.g. mock test stubs < 100 bytes)
        res.rejection_reasons.append(
            f"Artefacto corrupto o stub no reconocible ({res.size_bytes}B). Mínimo requerido: 500B."
        )
        return res

    meta_path = resolve_meta_path(client_path)
    meta_mtime_ns = meta_path.stat().st_mtime_ns if meta_path.is_file() else 0
    target_sha_for_cache: str | None = None
    if target_base_checkpoint and target_base_checkpoint.is_file():
        try:
            target_sha_for_cache = compute_sha256(target_base_checkpoint)
        except Exception:
            target_sha_for_cache = None

    cache_key = (
        str(client_path.resolve()),
        st.st_mtime_ns,
        st.st_size,
        meta_mtime_ns,
        target_sha_for_cache,
        hmac_key,
    )
    cached_res = _AUDIT_CACHE.get(cache_key)
    if cached_res is not None:
        copy_res = cached_res.model_copy(deep=True)
        copy_res.file_path = str(client_path)
        copy_res.file_name = client_path.name
        return copy_res

    # 2. Checksum verification
    try:
        res.sha256 = compute_sha256(client_path)
    except Exception as exc:
        res.rejection_reasons.append(f"No se pudo calcular el hash SHA-256 del archivo: {exc}")
        return res

    # 3. Load metadata manifest if present
    meta_path = resolve_meta_path(client_path)
    client_meta: dict[str, Any] = {}
    if meta_path.is_file():
        try:
            client_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            res.expected_sha256 = client_meta.get("sha256")
            res.client_id = client_meta.get("client_id", "unknown")
            res.base_model_sha256 = client_meta.get("base_model_sha256")
            res.training_scope = client_meta.get("training_scope", "unknown")
            res.num_samples = int(client_meta.get("num_samples", 0))
            res.created_at = client_meta.get("created_at")
        except Exception as exc:
            res.rejection_reasons.append(
                f"Manifiesto de metadatos corrupto ({meta_path.name}): {exc}"
            )

    # 4. PyTorch state dict loading and schema check
    state_dict: dict[str, Any] = {}
    bundle_meta: dict[str, Any] = {}
    try:
        loaded = torch.load(client_path, map_location="cpu")
        if isinstance(loaded, dict) and "state_dict" in loaded:
            state_dict = loaded["state_dict"]
            bundle_meta = loaded.get("metadata", {})
        elif isinstance(loaded, dict):
            state_dict = loaded
        else:
            res.rejection_reasons.append(
                f"Estructura no válida: se esperaba dict/bundle, se obtuvo {type(loaded).__name__}"
            )
            return res
    except Exception as exc:
        res.rejection_reasons.append(f"Error al decodificar checkpoint de PyTorch: {exc}")
        return res

    # Merge embedded metadata if manifest didn't provide fields
    if not res.client_id or res.client_id == "unknown":
        res.client_id = bundle_meta.get("client_id", "unknown")
    if not res.base_model_sha256:
        res.base_model_sha256 = bundle_meta.get("base_model_sha256")
    if res.training_scope == "unknown":
        res.training_scope = bundle_meta.get("training_scope", "head_only")
    if res.num_samples <= 0:
        res.num_samples = int(bundle_meta.get("num_samples", 0))
    if not res.created_at:
        res.created_at = bundle_meta.get("created_at")

    # 5. Checksum verification against manifest
    if res.expected_sha256 and res.expected_sha256 != res.sha256:
        res.rejection_reasons.append(
            f"Suma de verificación alterada: esperado {res.expected_sha256[:12]}..., "
            f"obtenido {res.sha256[:12]}..."
        )

    # 6. Finite values check across all tensors
    has_non_finite = False
    for _k, t in state_dict.items():
        if isinstance(t, torch.Tensor):
            if not torch.isfinite(t).all().item():
                has_non_finite = True
                break
    res.all_finite = not has_non_finite
    if has_non_finite:
        res.rejection_reasons.append(
            "El checkpoint contiene valores numéricos no finitos (NaN o Inf)."
        )

    # 7. Sample count integrity
    if res.num_samples <= 0:
        res.rejection_reasons.append(
            f"Número de muestras inválido ({res.num_samples}); debe ser entero positivo."
        )

    # 8. Scope validation
    if res.training_scope == "head_only":
        modified_non_head = [
            k for k in state_dict.keys()
            if not any(k.startswith(pfx) for pfx in HEAD_PREFIXES)
            and "num_batches_tracked" not in k
        ]
        if modified_non_head:
            res.rejection_reasons.append(
                f"Ámbito declarado 'head_only' pero modifica tensores de backbone: "
                f"{modified_non_head[:3]}"
            )

    # 9. Lineage verification against target base model
    if target_base_checkpoint and target_base_checkpoint.is_file():
        try:
            target_sha = compute_sha256(target_base_checkpoint)
            res.target_base_sha256 = target_sha
            if res.base_model_sha256 == target_sha:
                res.lineage_matched = True
            else:
                res.lineage_matched = False
                res.rejection_reasons.append(
                    f"Incompatibilidad de linaje: base_model_sha256 "
                    f"({str(res.base_model_sha256)[:10]}...) no coincide con base "
                    f"({target_sha[:10]}...)."
                )
        except Exception as exc:
            res.rejection_reasons.append(
                f"No se pudo auditar linaje frente a '{target_base_checkpoint.name}': {exc}"
            )

    # 10. HMAC authentication check
    sig = client_meta.get("hmac_signature") or bundle_meta.get("hmac_signature")
    res.has_hmac = sig is not None
    if hmac_key:
        if not sig:
            res.rejection_reasons.append(
                "El servidor requiere autenticación HMAC pero el artefacto no está firmado."
            )
            res.hmac_valid = False
        else:
            import hmac
            expected_sig = compute_manifest_hmac(
                key=hmac_key,
                client_sha256=res.sha256,
                base_model_sha256=res.base_model_sha256 or "",
                training_scope=res.training_scope,
                client_id=res.client_id,
                num_samples=res.num_samples,
            )
            is_valid = hmac.compare_digest(str(sig), expected_sig)
            res.hmac_valid = is_valid
            if not is_valid:
                res.rejection_reasons.append("Firma HMAC del artefacto inválida o manipulada.")

    # Schema valid if we have at least one valid tensor
    res.schema_valid = len(state_dict) > 0 and res.all_finite
    res.is_eligible = len(res.rejection_reasons) == 0

    if len(_AUDIT_CACHE) >= _AUDIT_CACHE_MAX:
        _AUDIT_CACHE.clear()
    _AUDIT_CACHE[cache_key] = res.model_copy(deep=True)

    return res


def resolve_validation_directory_for_model(
    classes: list[str],
    explicit_val_dir: str | Path | None = None,
) -> tuple[Path | None, str]:
    """Find a validation dataset directory strictly compatible with the model's classes."""
    candidates: list[Path] = []
    if explicit_val_dir:
        candidates.append(Path(explicit_val_dir))

    candidates.extend(
        [
            Path("/data/EnfermedadesHoja_v1.0/dataset/val"),
            Path("data/EnfermedadesHoja_v1.0/dataset/val"),
            Path("/data/EnfermedadesHoja_v1.0/dataset/test"),
            Path("data/EnfermedadesHoja_v1.0/dataset/test"),
            Path("data/val"),
            Path("data/test"),
            Path("models/hitl_queue/verified"),
        ]
    )
    # Search common uploaded dataset test/val splits
    uploads_dir = Path("data/uploads")
    if uploads_dir.is_dir():
        for split_cand in ("test", "val"):
            candidates.extend(sorted(uploads_dir.glob(f"*/dataset/{split_cand}")))
            candidates.extend(sorted(uploads_dir.glob(f"*/{split_cand}")))

    norm_classes = {c.strip().lower() for c in classes}

    for cand in candidates:
        if not cand.is_dir():
            continue
        subdirs = {
            d.name.strip().lower()
            for d in cand.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        }
        # Check if subdirs match the model classes exactly or contain all required classes
        if norm_classes and subdirs == norm_classes:
            return cand, "MATCH_EXACT"
        if norm_classes and norm_classes.issubset(subdirs):
            return cand, "MATCH_SUPERSET"

    return None, "INCOMPATIBLE_OR_MISSING"


class FederatedOperationsManager:
    """Manages the persistence of federated operations, staged candidates, and promotions."""

    def __init__(self, storage_path: Path | str = "models/federated_operations.json") -> None:
        self.storage_path = Path(storage_path)

    def load_operations(self) -> list[FederatedOperationRecord]:
        if not self.storage_path.is_file():
            return []
        try:
            raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
            return [FederatedOperationRecord(**op) for op in raw.get("operations", [])]
        except Exception:
            return []

    def save_operations(self, operations: list[FederatedOperationRecord]) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"operations": [op.model_dump() for op in operations]}
        content = json.dumps(payload, indent=2, ensure_ascii=False)
        temp_p = self.storage_path.with_suffix(".tmp")
        try:
            temp_p.write_text(content, encoding="utf-8")
            os.replace(temp_p, self.storage_path)
        finally:
            temp_p.unlink(missing_ok=True)

    def record_operation(self, op: FederatedOperationRecord) -> None:
        ops = self.load_operations()
        for idx, existing in enumerate(ops):
            if existing.operation_id == op.operation_id:
                ops[idx] = op
                self.save_operations(ops)
                return
        ops.append(op)
        self.save_operations(ops)

    def get_operation(self, operation_id: str) -> FederatedOperationRecord | None:
        for op in self.load_operations():
            if op.operation_id == operation_id:
                return op
        return None

    def get_latest_candidate(self, slug: str) -> FederatedOperationRecord | None:
        matching = [
            op
            for op in self.load_operations()
            if op.target_model_slug == slug and op.candidate_checkpoint
        ]
        if matching:
            return matching[-1]
        return None
