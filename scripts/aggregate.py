#!/usr/bin/env python3
"""Master Checkpoint Aggregator for AgriFed-KHAOS.

Inspects incoming client parameter checkpoints using ParameterValidator (numerical
and structural sanity checks, L2 perturbation limits) and incorporates them into
the global model via SanitizedFedProx aggregation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from torchvision import models

from agrifed_khaos.wrappers import ParameterValidator


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file for cryptographic provenance verification."""
    sha = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Master Parameter Gate & Aggregator")
    parser.add_argument(
        "--base-model", type=str, default="models/base_model.pt", help="Path to base weights"
    )
    parser.add_argument(
        "--client-weights",
        type=str,
        default="models/client_parameters.pt",
        help="Client parameters",
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
    return parser.parse_args()


def get_model(num_classes: int = 9) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return cast(nn.Module, model)


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")

    base_path = Path(args.base_model)
    client_path = Path(args.client_weights)
    out_path = Path(args.output)

    if not base_path.is_file():
        print(f"[!] Error: Base model file not found: {base_path}", file=sys.stderr)
        sys.exit(1)
    if not client_path.is_file():
        print(f"[!] Error: Client parameter file not found: {client_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 65)
    print("AGRIFED-KHAOS: PARAMETER GATE & FEDPROX AGGREGATOR")
    print("=" * 65)
    print(f"[*] Base Model:       {base_path}")
    print(f"[*] Client Update:     {client_path}")

    base_state = torch.load(str(base_path), map_location=device, weights_only=True)
    client_state = torch.load(str(client_path), map_location=device, weights_only=True)

    # 1. ParameterValidator Audit
    c_params = [
        client_state[k].detach().cpu().numpy()
        for k in client_state
        if "num_batches_tracked" not in k
    ]
    b_params = [
        base_state[k].detach().cpu().numpy() for k in base_state if "num_batches_tracked" not in k
    ]

    validator = ParameterValidator(
        max_update_norm=args.max_norm, enforce_finite=True, enforce_structural_match=True
    )
    validation = validator.validate(c_params, reference_parameters=b_params, check_delta_norm=True)

    delta_norm = validation.delta_norm if validation.delta_norm is not None else 0.0
    print(f"[*] Quality Gate: Structural Match & Finiteness = {validation.valid}")
    print(f"[*] Divergence Norm ||ΔW||_2: {delta_norm:.4f} (Tolerance limit: {args.max_norm:.1f})")

    if not validation.valid:
        print(
            f"[!] REJECTED: Candidate update rejected by parameter gate: {validation.issues}",
            file=sys.stderr,
        )
        sys.exit(1)

    print("[*] Update APPROVED by security gate.")

    # 2. Sanitized FedProx Weighted Incorporation: W_global = W_base + eta * (W_client - W_base)
    new_state: dict[str, Any] = {}
    for k in base_state:
        if base_state[k].dtype.is_floating_point:
            delta = client_state[k].float() - base_state[k].float()
            new_state[k] = base_state[k] + args.eta * delta
        else:
            new_state[k] = client_state[k]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_state, str(out_path))

    # 3. Provenance & Metadata Governance
    base_sha256 = compute_sha256(base_path)
    client_sha256 = compute_sha256(client_path)
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
    history.append(
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "base_sha256": base_sha256,
            "client_sha256": client_sha256,
            "delta_norm": round(delta_norm, 4),
            "eta": args.eta,
            "global_sha256": out_sha256,
            "status": "APPROVED",
        }
    )
    meta["aggregation_history"] = history
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[*] Aggregation complete! Saved updated global model: {out_path}")
    print(f"[*] Global SHA-256 Digest: {out_sha256}")
    print(f"[*] Metadata updated at:   {meta_file}")
    print("=" * 65)


if __name__ == "__main__":
    main()
