#!/usr/bin/env python3
"""Master Checkpoint Aggregator for AgriFed-KHAOS.

Inspects incoming client parameter checkpoints using ParameterValidator (numerical
and structural sanity checks, L2 perturbation limits) and incorporates them into
the global model via SanitizedFedProx / Multi-Client Weighted FedOpt aggregation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from agrifed_khaos.crypto import compute_manifest_hmac, compute_sha256
from agrifed_khaos.quality import audit_regression_guardrail
from agrifed_khaos.wrappers import EvaluationStats, ParameterValidator


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
        client_paths = [Path(raw_client_weights)]
    else:
        client_paths = [Path(p) for p in raw_client_weights]

    if not client_paths:
        print("[!] Error: No client parameter files provided.", file=sys.stderr)
        sys.exit(1)

    # Resolve architecture from metadata or CLI
    arch = args.arch
    if not arch and meta_file.is_file():
        try:
            m_data = json.loads(meta_file.read_text(encoding="utf-8"))
            arch = m_data.get("architecture")
        except Exception:
            pass
    arch = arch or "resnet18"

    print("=" * 65)
    print("AGRIFED-KHAOS: PARAMETER GATE & FEDPROX AGGREGATOR")
    print("=" * 65)
    print(f"[*] Architecture:     {arch}")
    print(f"[*] Base Model:       {base_path}")
    print(f"[*] Client Updates:   {len(client_paths)} candidate checkpoint(s)")
    for cp in client_paths:
        print(f"  • {cp}")

    base_state = torch.load(str(base_path), map_location=device, weights_only=True)
    actual_base_sha = compute_sha256(base_path)

    HEAD_PREFIXES = ("fc.", "classifier.", "head.", "model.head.")
    BN_BUFFER_SUFFIXES = (".running_mean", ".running_var", ".num_batches_tracked")
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
                if args.skip_invalid_clients:
                    print(
                        f"[!] WARNING: Skipping client '{client_path}': Base lineage mismatch.",
                        file=sys.stderr,
                    )
                    continue
                print(
                    f"[!] SECURITY REJECTION: Base model lineage mismatch!\n"
                    f"    Client update was trained from base model: {expected_base_sha}\n"
                    f"    Current server base model hash is:        {actual_base_sha}",
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
            client_state = torch.load(str(client_path), map_location=device, weights_only=True)
        except Exception as exc:
            err_msg = f"Failed to load PyTorch tensor file {client_path}: {exc}"
            if args.skip_invalid_clients:
                print(f"[!] WARNING: Skipping client '{client_path}': {err_msg}", file=sys.stderr)
                continue
            print(f"[!] Error: {err_msg}", file=sys.stderr)
            sys.exit(1)

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

    for c in validated_clients:
        weight_k = c.num_samples / total_samples
        c_state = (
            c.state_dict
            if c.state_dict is not None
            else torch.load(str(c.path), map_location=device, weights_only=True)
        )
        for k in round_trainable_keys:
            if k in c_state and k in delta_fed_accum:
                delta_fed_accum[k] += weight_k * (c_state[k].float() - base_state[k].float())
        # Free memory immediately to prevent O(K) checkpoint RAM buildup
        c.state_dict = None

    for k in round_trainable_keys:
        if base_state[k].dtype.is_floating_point:
            new_state[k] = base_state[k] + args.eta * delta_fed_accum[k]
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
    torch.save(new_state, str(out_path))

    # 5. Provenance & Metadata Governance
    out_sha256 = compute_sha256(out_path)

    meta_file = Path(args.metadata)
    meta: dict[str, Any] = {}
    if meta_file.is_file():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    meta["latest_aggregated_checkpoint"] = str(out_path)
    meta["latest_aggregated_sha256"] = out_sha256
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
        "scope": round_scope,
        "num_clients": len(validated_clients),
        "total_samples": total_samples,
        "clients": clients_record,
        "updated_tensors": round_trainable_keys,
        "global_sha256": out_sha256,
        "status": "APPROVED",
        "guardrail_status": guardrail_status,
    }
    history.append(history_entry)
    meta["aggregation_history"] = history
    try:
        meta_file.parent.mkdir(parents=True, exist_ok=True)
        meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[*] Metadata updated at:   {meta_file}")
    except (OSError, PermissionError) as exc:
        print(f"[!] Warning: Unable to update metadata file at {meta_file}: {exc}", file=sys.stderr)

    print(f"[*] Aggregation complete! Saved updated global model: {out_path}")
    print(f"[*] Global SHA-256 Digest: {out_sha256}")
    print("=" * 65)


if __name__ == "__main__":
    main()
