#!/usr/bin/env python3
"""Master Checkpoint Aggregator for AgriFed-KHAOS.

Inspects incoming client parameter checkpoints using ParameterValidator (numerical
and structural sanity checks, L2 perturbation limits) and incorporates them into
the global model via SanitizedFedProx / Multi-Client Weighted FedOpt aggregation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from agrifed_khaos.crypto import compute_manifest_hmac, compute_sha256
from agrifed_khaos.quality import audit_regression_guardrail
from agrifed_khaos.wrappers import EvaluationStats, ParameterValidator

HEAD_PREFIXES = ("fc.", "classifier.", "head.", "model.head.")
BN_BUFFER_SUFFIXES = (".running_mean", ".running_var", ".num_batches_tracked")


def resolve_meta_path(client_file: Path, explicit_meta: str | None = None) -> Path:
    """Resolve the .meta.json manifest path corresponding to a client parameter file."""
    if explicit_meta:
        return Path(explicit_meta)
    candidate1 = client_file.with_suffix(".meta.json")
    if candidate1.is_file():
        return candidate1
    candidate2 = client_file.parent / f"{client_file.name}.meta.json"
    if candidate2.is_file():
        return candidate2
    return candidate1


class ClientUpdate:
    """Encapsulates a fully verified and audited client parameter update."""

    def __init__(
        self,
        path: Path,
        meta_path: Path | None,
        client_id: str,
        sha256: str,
        num_samples: int,
        delta_norm: float,
        relative_norm: float,
        scope: str,
        state_dict: dict[str, torch.Tensor] | None,
        target_keys: list[str],
        classes: list[str] | None = None,
        num_classes: int | None = None,
    ) -> None:
        self.path = path
        self.meta_path = meta_path
        self.client_id = client_id
        self.sha256 = sha256
        self.num_samples = num_samples
        self.delta_norm = delta_norm
        self.relative_norm = relative_norm
        self.scope = scope
        self.state_dict = state_dict
        self.target_keys = target_keys
        self.classes = classes
        self.num_classes = num_classes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Master Parameter Gate & Multi-Client FedOpt Aggregator"
    )
    parser.add_argument(
        "--base-model", type=str, default="models/base_model.pt", help="Path to base weights"
    )
    parser.add_argument(
        "--client-weights",
        type=str,
        nargs="+",
        default=["models/client_parameters.pt"],
        help="One or more client parameter files to aggregate",
    )
    parser.add_argument(
        "--output", type=str, default="models/global_model.pt", help="Output global checkpoint"
    )
    parser.add_argument(
        "--metadata", type=str, default="models/metadata.json", help="Path to model metadata"
    )
    parser.add_argument(
        "--eta", type=float, default=0.6, help="Aggregation learning rate / weight (default: 0.6)"
    )
    parser.add_argument(
        "--server-momentum",
        type=float,
        default=0.9,
        help="Server momentum factor beta in [0, 1) for FedAvgM (default: 0.9)",
    )
    parser.add_argument(
        "--momentum-buffer",
        type=str,
        default=None,
        help="Path to save/load server momentum buffer (default: <output_dir>/server_momentum.pt)",
    )
    parser.add_argument(
        "--max-norm", type=float, default=150.0, help="Max perturbation norm (default: 150.0)"
    )
    parser.add_argument(
        "--max-relative-norm",
        type=float,
        default=None,
        help="Max relative perturbation norm ||ΔW||_2 / ||W_base||_2 (optional)",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default=None,
        help="Validation dataset directory to execute the Regression Guardrail",
    )
    parser.add_argument(
        "--max-acc-drop",
        type=float,
        default=0.03,
        help="Maximum allowed validation accuracy drop before triggering rollback (default: 0.03)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Model architecture (default: resolved from metadata.json or 'resnet18')",
    )
    parser.add_argument(
        "--client-meta",
        type=str,
        default=None,
        help=(
            "Path to client metadata JSON manifest (only used when aggregating "
            "a single client; default: <client_weights>.meta.json)"
        ),
    )
    parser.add_argument(
        "--skip-manifest-check",
        "--insecure-skip-provenance",
        dest="skip_manifest_check",
        action="store_true",
        help="Skip lineage and checksum verification (NOT recommended for production)",
    )
    parser.add_argument(
        "--hmac-key",
        type=str,
        default=None,
        help="Pre-shared secret key for HMAC-SHA256 client authentication (optional)",
    )
    parser.add_argument(
        "--skip-invalid-clients",
        action="store_true",
        help=(
            "Skip invalid clients failing security/structural/quality gates "
            "instead of aborting the round"
        ),
    )
    parser.add_argument(
        "--max-samples-per-client",
        type=int,
        default=None,
        help=(
            "Maximum allowed sample weight cap per client to defend against "
            "Byzantine inflation (optional)"
        ),
    )
    parser.add_argument(
        "--allow-new-classes",
        action="store_true",
        help="Allow dynamic classifier expansion and class-incremental federated aggregation",
    )
    return parser.parse_args()


def evaluate_weights(
    state_dict: dict[str, Any],
    val_dir: Path,
    num_classes: int,
    device: torch.device,
    architecture: str = "resnet18",
) -> EvaluationStats:
    """Evaluate model state on a validation directory to assess accuracy and loss."""
    from agrifed_khaos.models import build_model

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.ImageFolder(str(val_dir), transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    net = build_model(architecture=architecture, num_classes=num_classes).to(device)
    net.load_state_dict(state_dict)
    net.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = net(x)
            total_loss += criterion(out, y).item() * x.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += x.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return EvaluationStats(examples=total, loss=avg_loss, metrics={"accuracy": acc})


def align_client_head(
    client_state: dict[str, torch.Tensor],
    client_classes: list[str] | None,
    base_state: dict[str, torch.Tensor],
    union_classes: list[str],
) -> dict[str, torch.Tensor]:
    """Align and pad client classification head parameters to match base_state/union_classes shape.

    If a client has fewer classes than union_classes, missing classes are initialized with
    the exact weights from base_state (implying Delta W = 0 for unseen classes).
    """
    if not union_classes:
        return client_state

    for k, b_t in base_state.items():
        if any(k.startswith(pfx) for pfx in HEAD_PREFIXES) and k in client_state:
            c_t = client_state[k]
            if (
                b_t.dim() in (1, 2)
                and b_t.dim() == c_t.dim()
                and b_t.shape[0] == len(union_classes)
                and (
                    c_t.shape != b_t.shape
                    or (client_classes is not None and client_classes != union_classes)
                )
            ):
                padded = b_t.clone()
                if client_classes:
                    for local_idx, c_name in enumerate(client_classes):
                        if c_name in union_classes and local_idx < c_t.shape[0]:
                            u_idx = union_classes.index(c_name)
                            padded[u_idx] = c_t[local_idx]
                else:
                    min_c = min(c_t.shape[0], b_t.shape[0])
                    padded[:min_c] = c_t[:min_c]
                client_state[k] = padded

    return client_state


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")

    base_path = Path(args.base_model)
    out_path = Path(args.output)
    meta_file = Path(args.metadata)

    if not base_path.is_file():
        print(f"[!] Error: Base model file not found: {base_path}", file=sys.stderr)
        sys.exit(1)

    raw_client_weights = args.client_weights
    if isinstance(raw_client_weights, str):
        candidate_paths = [Path(raw_client_weights)]
    else:
        candidate_paths = [Path(p) for p in raw_client_weights]

    client_paths: list[Path] = []
    for cp in candidate_paths:
        if cp.is_dir():
            cand_param = cp / "parameters.pt"
            if cand_param.is_file():
                client_paths.append(cand_param)
                continue
            cand_named = cp / f"{cp.name}_parameters.pt"
            if cand_named.is_file():
                client_paths.append(cand_named)
                continue
            # Direct client parameter files in the directory
            nested_pts = sorted([p for p in cp.glob("*.pt") if not p.name.startswith(".")])
            if nested_pts:
                client_paths.extend(nested_pts)
                continue
            # Updates folder with client stamp subdirectories
            updates_folder = (
                cp / "updates"
                if (cp / "updates").is_dir()
                else (cp if cp.name == "updates" else None)
            )
            if updates_folder and updates_folder.is_dir():
                discovered: list[Path] = []
                for client_dir in sorted(updates_folder.iterdir()):
                    if client_dir.is_dir() and not client_dir.name.startswith("."):
                        stamps = sorted(
                            [
                                d
                                for d in client_dir.iterdir()
                                if d.is_dir() and not d.name.startswith(".")
                            ],
                            key=lambda d: d.name,
                        )
                        if stamps:
                            latest_param = stamps[-1] / "parameters.pt"
                            if latest_param.is_file():
                                discovered.append(latest_param)
                if discovered:
                    client_paths.extend(discovered)
                    continue
        client_paths.append(cp)

    if not client_paths:
        print("[!] Error: No client parameter files provided.", file=sys.stderr)
        sys.exit(1)

    from agrifed_khaos.models import load_checkpoint_bundle, save_checkpoint_bundle

    base_state, base_bundle_meta = load_checkpoint_bundle(base_path, device=device)
    actual_base_sha = compute_sha256(base_path)

    # Resolve architecture from CLI, metadata, or bundle
    arch = args.arch
    if not arch and meta_file.is_file():
        try:
            m_data = json.loads(meta_file.read_text(encoding="utf-8"))
            arch = m_data.get("architecture")
        except Exception:
            pass
    if not arch:
        arch = base_bundle_meta.get("architecture")
    arch = arch or "resnet18"

    print("=" * 65)
    print("AGRIFED-KHAOS: PARAMETER GATE & FEDPROX AGGREGATOR")
    print("=" * 65)
    print(f"[*] Architecture:     {arch}")
    print(f"[*] Base Model:       {base_path}")
    print(f"[*] Client Updates:   {len(client_paths)} candidate checkpoint(s)")
    for cp in client_paths:
        print(f"  • {cp}")

    base_classes = list(base_bundle_meta.get("classes", []))
    if not base_classes and meta_file.is_file():
        try:
            m_data = json.loads(meta_file.read_text(encoding="utf-8"))
            base_classes = list(m_data.get("classes", []))
        except Exception:
            pass

    union_classes = list(base_classes)
    if args.allow_new_classes:
        from agrifed_khaos.models import expand_state_dict_classifier

        for cp in client_paths:
            if not cp.is_file():
                continue
            try:
                _, c_bmeta = load_checkpoint_bundle(cp, device=device)
                c_cls = c_bmeta.get("classes")
                if not c_cls:
                    mp = resolve_meta_path(cp)
                    if mp.is_file():
                        c_meta_json = json.loads(mp.read_text(encoding="utf-8"))
                        c_cls = c_meta_json.get("classes")
                if c_cls:
                    for c_name in c_cls:
                        if c_name not in union_classes:
                            union_classes.append(c_name)
            except Exception:
                pass

        if len(union_classes) > len(base_classes):
            print(
                f"[*] Dynamic Classifier Expansion (--allow-new-classes): "
                f"Expanding global model from {len(base_classes)} to {len(union_classes)} classes."
            )
            print(f"    Base classes:  {base_classes}")
            print(f"    Union classes: {union_classes}")
            expand_state_dict_classifier(
                base_state, len(union_classes), class_names=union_classes
            )
            base_bundle_meta["classes"] = union_classes
            base_bundle_meta["num_classes"] = len(union_classes)

    base_keys = set(base_state.keys())
    base_clean_keys = {k for k in base_keys if "num_batches_tracked" not in k}
    base_head_keys = {k for k in base_clean_keys if k.startswith(HEAD_PREFIXES)}

    validated_clients: list[ClientUpdate] = []

    for client_path in client_paths:
        print("-" * 50)
        print(f"[*] Inspecting Client Update: {client_path}")

        if not client_path.is_file():
            if args.skip_invalid_clients:
                print(
                    f"[!] WARNING: Skipping client '{client_path}': File not found.",
                    file=sys.stderr,
                )
                continue
            print(f"[!] Error: Client parameter file not found: {client_path}", file=sys.stderr)
            sys.exit(1)

        # 0. Lineage & Checksum Verification (+ Optional HMAC Authentication)
        client_meta: dict[str, Any] = {}
        actual_client_sha = compute_sha256(client_path)

        if not args.skip_manifest_check:
            meta_cand = (
                Path(args.client_meta)
                if (len(client_paths) == 1 and args.client_meta)
                else resolve_meta_path(client_path)
            )
            if not meta_cand.is_file():
                if args.skip_invalid_clients:
                    print(
                        f"[!] WARNING: Skipping client '{client_path}': "
                        f"Missing metadata manifest {meta_cand}.",
                        file=sys.stderr,
                    )
                    continue
                print(
                    f"[!] SECURITY REJECTION: Missing client metadata manifest: {meta_cand}.\n"
                    "    Lineage and integrity verification cannot be established.\n"
                    "    Client parameters must be accompanied by a .meta.json manifest.\n"
                    "    (Pass --skip-manifest-check to bypass only during test/debugging).",
                    file=sys.stderr,
                )
                sys.exit(1)

            try:
                client_meta = json.loads(meta_cand.read_text(encoding="utf-8"))
            except Exception as exc:
                if args.skip_invalid_clients:
                    print(
                        f"[!] WARNING: Skipping client '{client_path}': "
                        f"Corrupt metadata {meta_cand}: {exc}",
                        file=sys.stderr,
                    )
                    continue
                print(
                    f"[!] SECURITY REJECTION: Corrupt or unreadable client metadata "
                    f"{meta_cand}: {exc}",
                    file=sys.stderr,
                )
                sys.exit(1)

            # 0.a Verify client parameter file checksum
            expected_client_sha = client_meta.get("sha256")
            if expected_client_sha != actual_client_sha:
                if args.skip_invalid_clients:
                    print(
                        f"[!] WARNING: Skipping client '{client_path}': Checksum mismatch.",
                        file=sys.stderr,
                    )
                    continue
                print(
                    f"[!] SECURITY REJECTION: Client weights checksum mismatch!\n"
                    f"    Manifest SHA-256: {expected_client_sha}\n"
                    f"    Actual SHA-256:   {actual_client_sha}",
                    file=sys.stderr,
                )
                sys.exit(1)

            # 0.b Verify base model lineage
            expected_base_sha = client_meta.get("base_model_sha256")
            if expected_base_sha != actual_base_sha:
                global_cand = Path("models/global_model.pt")
                if global_cand.is_file() and compute_sha256(global_cand) == expected_base_sha:
                    extra_hint = (
                        f"\n    [HINT] Client update was trained on global model '{global_cand}' "
                        f"({expected_base_sha[:16]}...).\n"
                        "    For subsequent rounds (Round 2+), specify: "
                        "'--base-model models/global_model.pt' to aggregate."
                    )
                else:
                    extra_hint = ""

                if args.skip_invalid_clients:
                    print(
                        f"[!] WARNING: Skipping client '{client_path}': "
                        f"Base lineage mismatch.{extra_hint}",
                        file=sys.stderr,
                    )
                    continue
                print(
                    f"[!] SECURITY REJECTION: Base model lineage mismatch!\n"
                    f"    Client update was trained from model: {expected_base_sha}\n"
                    f"    Target aggregation model hash is:     {actual_base_sha}{extra_hint}",
                    file=sys.stderr,
                )
                sys.exit(1)

            # 0.c Validate num_samples
            raw_samples = client_meta.get("num_samples")
            if raw_samples is None:
                num_samples = 1
                num_samples_for_hmac = None
            else:
                if (
                    not isinstance(raw_samples, int)
                    or isinstance(raw_samples, bool)
                    or raw_samples <= 0
                ):
                    err_msg = (
                        f"Invalid or non-positive 'num_samples' in client manifest: {raw_samples}. "
                        "Expected positive integer >= 1."
                    )
                    if args.skip_invalid_clients:
                        print(
                            f"[!] WARNING: Skipping client '{client_path}': {err_msg}",
                            file=sys.stderr,
                        )
                        continue
                    print(f"[!] SECURITY REJECTION: {err_msg}", file=sys.stderr)
                    sys.exit(1)
                num_samples = raw_samples
                num_samples_for_hmac = raw_samples

            # Byzantine Sample Inflation Guardrail
            if (
                args.max_samples_per_client is not None
                and num_samples > args.max_samples_per_client
            ):
                print(
                    f"[!] SECURITY AUDIT: Client '{client_path}' claimed {num_samples} samples, "
                    f"capped to --max-samples-per-client ({args.max_samples_per_client}).",
                    file=sys.stderr,
                )
                num_samples = args.max_samples_per_client

            # 0.d Optional HMAC Authentication over Canonical Manifest Metadata
            if args.hmac_key:
                import hmac

                manifest_sig = client_meta.get("hmac_signature")
                if not manifest_sig:
                    if args.skip_invalid_clients:
                        print(
                            f"[!] WARNING: Skipping client '{client_path}': "
                            "Missing HMAC signature.",
                            file=sys.stderr,
                        )
                        continue
                    print(
                        "[!] SECURITY REJECTION: Server requires HMAC authentication (--hmac-key), "
                        "but client manifest does not contain 'hmac_signature'.",
                        file=sys.stderr,
                    )
                    sys.exit(1)

                client_id = client_meta.get("client_id", "client")
                claimed_scope = client_meta.get("training_scope", "")
                expected_sig = compute_manifest_hmac(
                    key=args.hmac_key,
                    client_sha256=actual_client_sha,
                    base_model_sha256=actual_base_sha,
                    training_scope=claimed_scope,
                    client_id=client_id,
                    num_samples=num_samples_for_hmac,
                )
                if not hmac.compare_digest(manifest_sig, expected_sig):
                    if args.skip_invalid_clients:
                        print(
                            f"[!] WARNING: Skipping client '{client_path}': "
                            "Invalid HMAC signature.",
                            file=sys.stderr,
                        )
                        continue
                    print(
                        "[!] SECURITY REJECTION: Invalid HMAC signature in client manifest! "
                        "Canonical payload metadata was tampered with or client lacks the "
                        "shared key.",
                        file=sys.stderr,
                    )
                    sys.exit(1)

            print("[*] Lineage & Integrity Verification:")
            print(f"  • Client parameter hash:   verified ({actual_client_sha[:16]}...)")
            print(f"  • Base model lineage:      verified ({actual_base_sha[:16]}...)")
            print(f"  • Dataset sample count:    verified ({num_samples} samples)")
            if args.hmac_key:
                print(
                    "  • HMAC authenticity:       verified "
                    "(Client authenticated with shared key)"
                )
        else:
            print(
                "[!] WARNING: Running with --skip-manifest-check. "
                "Lineage and checksum checks bypassed."
            )
            num_samples = 1
            client_id = f"client_{len(validated_clients) + 1}"

        # 1. Update Scope Auto-Detection and Strict Schema Validation
        try:
            client_state, client_bundle_meta = load_checkpoint_bundle(client_path, device=device)
            if not client_meta and client_bundle_meta:
                client_meta = client_bundle_meta
        except Exception as exc:
            err_msg = f"Failed to load PyTorch tensor file {client_path}: {exc}"
            if args.skip_invalid_clients:
                print(f"[!] WARNING: Skipping client '{client_path}': {err_msg}", file=sys.stderr)
                continue
            print(f"[!] Error: {err_msg}", file=sys.stderr)
            sys.exit(1)

        c_classes = client_meta.get("classes") or client_bundle_meta.get("classes")
        if c_classes is None:
            c_classes = list(base_classes)
        if args.allow_new_classes and union_classes:
            client_state = align_client_head(client_state, c_classes, base_state, union_classes)

        client_keys = set(client_state.keys())
        client_clean_keys = {k for k in client_keys if "num_batches_tracked" not in k}

        # Schema A: Head-Only update
        is_head_only = (
            len(client_clean_keys) > 0
            and all(k.startswith(HEAD_PREFIXES) for k in client_clean_keys)
            and client_clean_keys == base_head_keys
        )
        # Schema B: Full-Model update
        is_full_model = client_clean_keys == base_clean_keys

        if is_head_only:
            update_scope = "head_only"
            target_keys = sorted(client_clean_keys)
            print(
                f"[*] Scope: Verified HEAD-ONLY Update ({len(target_keys)} tensors: "
                f"{', '.join(target_keys)})"
            )
            print("  • Feature backbone: FROZEN & INVARIANT")
            print("  • Selective gate: Validating only classification head parameters")
        elif is_full_model:
            update_scope = "full_model"
            target_keys = sorted(base_clean_keys)
            print(f"[*] Scope: Verified FULL-MODEL Update ({len(target_keys)} tensors)")
        else:
            extra_keys = client_clean_keys - base_clean_keys
            missing_keys = base_clean_keys - client_clean_keys
            if args.skip_invalid_clients:
                print(
                    f"[!] WARNING: Skipping client '{client_path}': Schema mismatch "
                    f"(extra={len(extra_keys)}, missing={len(missing_keys)}).",
                    file=sys.stderr,
                )
                continue
            print(
                "[!] SECURITY REJECTION: Checkpoint failed structural schema validation.",
                file=sys.stderr,
            )
            if extra_keys:
                print(
                    f"  • Unauthorized extraneous keys ({len(extra_keys)}): "
                    f"{sorted(extra_keys)[:5]}",
                    file=sys.stderr,
                )
            if missing_keys:
                print(
                    f"  • Missing model backbone/head keys ({len(missing_keys)}): "
                    f"{sorted(missing_keys)[:5]}",
                    file=sys.stderr,
                )
            print(
                "  • Allowed schemas: exact full-model match OR "
                "isolated classification head match.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Validate training scope consistency against closed set {"head_only", "full", "full_model"}
        if not args.skip_manifest_check:
            claimed_scope = client_meta.get("training_scope")
            if claimed_scope not in {"head_only", "full", "full_model"}:
                if args.skip_invalid_clients:
                    print(
                        f"[!] WARNING: Skipping client '{client_path}': "
                        f"Invalid training_scope '{claimed_scope}'.",
                        file=sys.stderr,
                    )
                    continue
                print(
                    f"[!] SECURITY REJECTION: Invalid or missing training_scope in client "
                    f"manifest: '{claimed_scope}'. "
                    f"Expected closed set {{'head_only', 'full', 'full_model'}}.",
                    file=sys.stderr,
                )
                sys.exit(1)

            expected_scope = "head_only" if claimed_scope == "head_only" else "full_model"
            if update_scope != expected_scope:
                if args.skip_invalid_clients:
                    print(
                        f"[!] WARNING: Skipping client '{client_path}': Scope mismatch "
                        f"(claimed '{claimed_scope}', detected '{update_scope}').",
                        file=sys.stderr,
                    )
                    continue
                print(
                    f"[!] SECURITY REJECTION: Manifest claimed scope '{claimed_scope}', "
                    f"but detected update scope is '{update_scope}'.",
                    file=sys.stderr,
                )
                sys.exit(1)

        trainable_target_keys = [
            k for k in target_keys if not any(k.endswith(sfx) for sfx in BN_BUFFER_SUFFIXES)
        ]
        if update_scope == "full_model":
            print(
                f"  • Trainable parameters for FL aggregation: {len(trainable_target_keys)} tensors"
            )
            print("  • BatchNorm statistics: FROZEN & INVARIANT (preserving base model statistics)")

        # 2. ParameterValidator Audit on Trainable Target Parameters
        c_params = [client_state[k].detach().cpu().numpy() for k in trainable_target_keys]
        b_params = [base_state[k].detach().cpu().numpy() for k in trainable_target_keys]

        validator = ParameterValidator(
            max_update_norm=args.max_norm,
            max_relative_norm=args.max_relative_norm,
            enforce_finite=True,
            enforce_structural_match=True,
        )
        validation = validator.validate(
            c_params, reference_parameters=b_params, check_delta_norm=True
        )

        delta_norm = validation.delta_norm if validation.delta_norm is not None else 0.0
        rel_norm = validation.relative_norm if validation.relative_norm is not None else 0.0
        print(f"[*] Quality Gate: Structural Match & Finiteness = {validation.valid}")
        rel_str = f" | Relative: {rel_norm:.4f}"
        if args.max_relative_norm is not None:
            rel_str += f" (Limit: {args.max_relative_norm:.2f})"
        print(
            f"[*] Divergence Norm ||ΔW||_2: {delta_norm:.4f} "
            f"(Limit: {args.max_norm:.1f}){rel_str}"
        )

        if not validation.valid:
            if args.skip_invalid_clients:
                print(
                    f"[!] WARNING: Skipping client '{client_path}': "
                    f"Rejected by parameter gate: {validation.issues}",
                    file=sys.stderr,
                )
                continue
            print(
                f"[!] REJECTED: Candidate update rejected by parameter gate: {validation.issues}",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"[*] Client '{client_path}' APPROVED by security gate.")
        client_id_val = client_meta.get("client_id", f"client_{len(validated_clients) + 1}")
        validated_clients.append(
            ClientUpdate(
                path=client_path,
                meta_path=meta_cand if not args.skip_manifest_check else None,
                client_id=client_id_val,
                sha256=actual_client_sha,
                num_samples=num_samples,
                delta_norm=delta_norm,
                relative_norm=rel_norm,
                scope=update_scope,
                state_dict=client_state,
                target_keys=trainable_target_keys,
                classes=c_classes,
                num_classes=len(c_classes),
            )
        )

    # Validate that at least one client succeeded
    if not validated_clients:
        print(
            "[!] Error: No valid clients remaining after validation. Aggregation aborted.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 3. Multi-Client Sanitized FedProx / FedOpt Weighted Incorporation
    total_samples = sum(c.num_samples for c in validated_clients)
    print("=" * 65)
    print(
        f"[*] AGGREGATING {len(validated_clients)} VALIDATED CLIENT(S) "
        f"(Total Samples: {total_samples}):"
    )
    for idx, c in enumerate(validated_clients, start=1):
        rel_weight = c.num_samples / total_samples
        print(
            f"  [{idx}] {c.client_id} ({c.path.name}): {c.num_samples} samples "
            f"({rel_weight * 100:.2f}%) | ||ΔW||_2: {c.delta_norm:.4f} | Scope: {c.scope}"
        )

    all_head_only = all(c.scope == "head_only" for c in validated_clients)
    all_full_model = all(c.scope == "full_model" for c in validated_clients)

    if all_head_only:
        round_scope = "head_only"
        round_target_keys = sorted(base_head_keys)
    elif all_full_model:
        round_scope = "full_model"
        round_target_keys = sorted(base_clean_keys)
    else:
        round_scope = "mixed"
        round_target_keys = sorted(base_clean_keys)

    round_trainable_keys = [
        k for k in round_target_keys if not any(k.endswith(sfx) for sfx in BN_BUFFER_SUFFIXES)
    ]

    new_state: dict[str, Any] = {k: v.clone() for k, v in base_state.items()}
    delta_fed_accum: dict[str, torch.Tensor] = {
        k: torch.zeros_like(base_state[k], dtype=torch.float32)
        for k in round_trainable_keys
        if base_state[k].dtype.is_floating_point
    }

    num_union_classes = len(union_classes)
    total_samples_per_class: list[int] = [0] * num_union_classes
    for idx_cls, cls_name in enumerate(union_classes):
        total_samples_per_class[idx_cls] = sum(
            c.num_samples
            for c in validated_clients
            if c.classes and cls_name in c.classes
        )

    for c in validated_clients:
        weight_k = c.num_samples / total_samples
        if c.state_dict is not None:
            c_state = c.state_dict
        else:
            c_state, _ = load_checkpoint_bundle(str(c.path), device=device)
            if args.allow_new_classes and union_classes:
                c_state = align_client_head(c_state, c.classes, base_state, union_classes)

        for k in round_trainable_keys:
            if k not in c_state or k not in delta_fed_accum:
                continue

            is_head_key = any(k.startswith(pfx) for pfx in HEAD_PREFIXES)
            delta_k = c_state[k].float() - base_state[k].float()

            if (
                is_head_key
                and num_union_classes > 0
                and delta_k.dim() in (1, 2)
                and delta_k.shape[0] == num_union_classes
            ):
                for idx_cls, cls_name in enumerate(union_classes):
                    if (
                        c.classes
                        and cls_name in c.classes
                        and total_samples_per_class[idx_cls] > 0
                    ):
                        weight_kc = c.num_samples / total_samples_per_class[idx_cls]
                        delta_fed_accum[k][idx_cls] += weight_kc * delta_k[idx_cls]
            else:
                delta_fed_accum[k] += weight_k * delta_k

        # Free memory immediately to prevent O(K) checkpoint RAM buildup
        c.state_dict = None

    # Server momentum (FedAvgM) handling
    momentum_buffer_path: Path | None = None
    prev_momentum: dict[str, torch.Tensor] = {}
    new_momentum_buffer: dict[str, torch.Tensor] = {}

    if args.server_momentum > 0.0:
        if not 0.0 <= args.server_momentum < 1.0:
            print(
                f"[!] Error: --server-momentum must be in [0.0, 1.0), got {args.server_momentum}",
                file=sys.stderr,
            )
            sys.exit(1)
        momentum_buffer_path = (
            Path(args.momentum_buffer)
            if args.momentum_buffer
            else Path(args.output).parent / "server_momentum.pt"
        )
        if momentum_buffer_path.is_file():
            try:
                raw_loaded = torch.load(
                    str(momentum_buffer_path), map_location=device, weights_only=True
                )
                if isinstance(raw_loaded, dict) and "velocity" in raw_loaded:
                    stored_parent = raw_loaded.get("parent_base_sha256")
                    stored_global = raw_loaded.get("produced_global_sha256")
                    stored_arch = raw_loaded.get("architecture")

                    lineage_match = (
                        actual_base_sha in (stored_parent, stored_global)
                        if (stored_parent or stored_global)
                        else True
                    )
                    arch_match = (stored_arch == arch) if stored_arch else True

                    if not lineage_match:
                        print(
                            f"[!] Lineage Notice: Momentum buffer at {momentum_buffer_path} "
                            f"was generated for model {stored_global or stored_parent} "
                            f"(current base is {actual_base_sha}).\n"
                            "    Resetting velocity buffer to zero to prevent stale momentum drift."
                        )
                        prev_momentum = {}
                    elif not arch_match:
                        print(
                            f"[!] Warning: Server momentum buffer architecture '{stored_arch}' "
                            f"does not match current '{arch}'. Discarding buffer."
                        )
                        prev_momentum = {}
                    else:
                        prev_momentum = raw_loaded.get("velocity", {})
                        print(
                            f"[*] Loaded server momentum buffer from {momentum_buffer_path} "
                            f"(FedAvgM beta={args.server_momentum:.2f}, lineage verified)"
                        )
                elif isinstance(raw_loaded, dict):
                    print(
                        f"[*] Loaded legacy server momentum buffer from {momentum_buffer_path} "
                        f"(FedAvgM beta={args.server_momentum:.2f})"
                    )
                    prev_momentum = raw_loaded
                else:
                    prev_momentum = {}
            except Exception as e:
                print(
                    f"[!] Warning: Failed to load server momentum buffer ({e}), initializing fresh."
                )
                prev_momentum = {}
        else:
            print(
                f"[*] Initialized fresh server momentum velocity buffer "
                f"(FedAvgM beta={args.server_momentum:.2f})"
            )

    for k in round_trainable_keys:
        if base_state[k].dtype.is_floating_point:
            delta_k = delta_fed_accum[k]
            if args.server_momentum > 0.0:
                v_prev = prev_momentum.get(k, torch.zeros_like(delta_k))
                if v_prev.shape != delta_k.shape:
                    v_prev = torch.zeros_like(delta_k)
                v_t = args.server_momentum * v_prev.to(device) + delta_k
                new_momentum_buffer[k] = v_t.cpu()
                new_state[k] = base_state[k] + args.eta * v_t
            else:
                new_state[k] = base_state[k] + args.eta * delta_k
        else:
            new_state[k] = base_state[k]

    # Compute aggregated delta norm ||ΔW_fed||_2 and relative norm
    agg_delta_sq = 0.0
    base_l2_sq = 0.0
    for k in round_trainable_keys:
        if base_state[k].dtype.is_floating_point:
            diff = (new_state[k].float() - base_state[k].float()) / (
                args.eta if args.eta != 0 else 1.0
            )
            agg_delta_sq += torch.sum(diff**2).item()
            base_l2_sq += torch.sum(base_state[k].float() ** 2).item()
    aggregated_delta_norm = agg_delta_sq**0.5
    base_l2_norm = base_l2_sq**0.5
    aggregated_relative_norm = aggregated_delta_norm / max(base_l2_norm, 1e-8)
    print(
        f"[*] Aggregated Delta Norm ||ΔW_fed||_2: {aggregated_delta_norm:.4f} | "
        f"Relative Norm: {aggregated_relative_norm:.4f}"
    )

    # 4. Regression Guardrail Verification (Optional Holdout Benchmark)
    guardrail_status = "NOT_EVALUATED"
    if args.val_dir:
        val_path = Path(args.val_dir)
        if val_path.is_dir():
            print(f"[*] Running Regression Guardrail on benchmark: {val_path}...")
            num_classes = None
            if meta_file.is_file():
                try:
                    meta_content = json.loads(meta_file.read_text(encoding="utf-8"))
                    num_classes = meta_content.get("num_classes")
                except Exception:
                    pass
            if num_classes is None:
                for head_key in (
                    "fc.weight",
                    "classifier.weight",
                    "classifier.3.weight",
                    "head.fc.weight",
                    "model.head.weight",
                ):
                    if head_key in base_state and hasattr(base_state[head_key], "shape"):
                        num_classes = int(base_state[head_key].shape[0])
                        break
            if num_classes is None:
                num_classes = 9

            base_stats = evaluate_weights(
                base_state, val_path, num_classes, device, architecture=arch
            )
            cand_stats = evaluate_weights(
                new_state, val_path, num_classes, device, architecture=arch
            )
            guardrail = audit_regression_guardrail(
                base_stats, cand_stats, max_allowed_acc_drop=args.max_acc_drop
            )
            print(
                f"  • Pre-Aggregation Accuracy:  {base_stats.metrics['accuracy'] * 100:.2f}% | "
                f"Loss: {base_stats.loss:.4f}"
            )
            print(
                f"  • Post-Aggregation Accuracy: {cand_stats.metrics['accuracy'] * 100:.2f}% | "
                f"Loss: {cand_stats.loss:.4f}"
            )
            print(
                f"  • Shift: ΔAcc = {guardrail.accuracy_delta * 100:+.2f}%, "
                f"ΔLoss = {guardrail.loss_delta:+.4f}"
            )
            if not guardrail.approved:
                print(
                    f"[!] REJECTED BY REGRESSION GUARDRAIL: Rollback triggered! "
                    f"Issues: {'; '.join(guardrail.reasons)}",
                    file=sys.stderr,
                )
                sys.exit(1)
            print("[*] Regression Guardrail PASSED: Model performance preserved/improved.")
            guardrail_status = "PASSED"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 5. Provenance & Metadata Governance
    meta_file = Path(args.metadata)
    meta: dict[str, Any] = {}
    if meta_file.is_file():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    elif base_bundle_meta:
        meta = dict(base_bundle_meta)

    # Propagate base metadata fields if missing
    meta["architecture"] = arch
    if union_classes:
        meta["classes"] = union_classes
        meta["num_classes"] = len(union_classes)
    else:
        if "num_classes" not in meta and "num_classes" in base_bundle_meta:
            meta["num_classes"] = base_bundle_meta["num_classes"]
        if "classes" not in meta and "classes" in base_bundle_meta:
            meta["classes"] = base_bundle_meta["classes"]
    if "img_size" not in meta and "img_size" in base_bundle_meta:
        meta["img_size"] = base_bundle_meta["img_size"]
    if "calibrated_thresholds" not in meta and "calibrated_thresholds" in base_bundle_meta:
        meta["calibrated_thresholds"] = base_bundle_meta["calibrated_thresholds"]

    if (
        meta.get("calibrated_thresholds")
        and isinstance(meta["calibrated_thresholds"], dict)
        and union_classes
    ):
        if not all(c in meta["calibrated_thresholds"] for c in union_classes):
            print(
                "[*] Calibrated thresholds invalidated due to class expansion. "
                "Will require re-calibration on new class set."
            )
            meta.pop("calibrated_thresholds", None)

    meta["latest_aggregated_checkpoint"] = str(out_path)
    history = meta.get("aggregation_history", [])

    clients_record = [
        {
            "client_path": str(c.path),
            "client_id": c.client_id,
            "client_sha256": c.sha256,
            "num_samples": c.num_samples,
            "sample_weight": round(c.num_samples / total_samples, 6),
            "delta_norm": round(c.delta_norm, 4),
            "relative_norm": round(c.relative_norm, 4),
            "scope": c.scope,
        }
        for c in validated_clients
    ]

    history_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_sha256": actual_base_sha,
        "client_sha256": (
            validated_clients[0].sha256
            if len(validated_clients) == 1
            else [c.sha256 for c in validated_clients]
        ),
        "delta_norm": round(aggregated_delta_norm, 4),
        "relative_norm": round(aggregated_relative_norm, 4),
        "eta": args.eta,
        "server_momentum": args.server_momentum,
        "scope": round_scope,
        "num_clients": len(validated_clients),
        "total_samples": total_samples,
        "clients": clients_record,
        "updated_tensors": round_trainable_keys,
        "status": "APPROVED",
        "guardrail_status": guardrail_status,
    }
    history.append(history_entry)
    meta["aggregation_history"] = history

    save_checkpoint_bundle(out_path, new_state, meta)
    out_sha256 = compute_sha256(out_path)
    meta["latest_aggregated_sha256"] = out_sha256
    history_entry["global_sha256"] = out_sha256

    try:
        meta_file.parent.mkdir(parents=True, exist_ok=True)
        meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[*] Metadata updated at:   {meta_file}")
    except (OSError, PermissionError) as exc:
        print(f"[!] Warning: Unable to update metadata file at {meta_file}: {exc}", file=sys.stderr)

    # Do not advance optimizer state until the candidate checkpoint has been safely committed.
    if args.server_momentum > 0.0 and momentum_buffer_path is not None:
        momentum_bundle = {
            "format_version": 2,
            "parent_base_sha256": actual_base_sha,
            "produced_global_sha256": out_sha256,
            "architecture": arch,
            "classes": union_classes,
            "beta": args.server_momentum,
            "velocity": new_momentum_buffer,
            "updated_at": time.time(),
        }
        momentum_buffer_path.parent.mkdir(parents=True, exist_ok=True)
        temp_momentum = momentum_buffer_path.with_name(
            f".{momentum_buffer_path.name}.{uuid.uuid4().hex}.tmp"
        )
        torch.save(momentum_bundle, str(temp_momentum))
        os.replace(temp_momentum, momentum_buffer_path)
        print(
            f"[*] Persisted verified server momentum buffer (bound to global {out_sha256[:16]}...) "
            f"to {momentum_buffer_path}"
        )

    print(f"[*] Aggregation complete! Saved updated global model: {out_path}")
    print(f"[*] Global SHA-256 Digest: {out_sha256}")
    print("=" * 65)
    print("PROXIMOS PASOS OPERATIVOS:")
    print("  1. Re-calibrar umbrales de decisión sobre partición de validación:")
    print(
        f"     python scripts/evaluate.py --model {out_path} "
        f"--data-dir dataset/val --calibrate-threshold"
    )
    print("     [Nota: Usar siempre datos de validación separados para no sobreajustar umbrales]")
    print("")
    print("  2. Inferencia y Servicio REST:")
    print("     El microservicio (scripts/serve.py) detecta y recarga este modelo automáticamente.")
    print("     Los clientes pueden enviar muestras para diagnosticar en T1:")
    print("     curl -X POST http://<IP_SERVIDOR>:8000/predict -F 'file=@$MUESTRA'")
    print("=" * 65)


if __name__ == "__main__":
    main()
