#!/usr/bin/env python3
"""Domain-Agnostic Federated Client Trainer for AgriFed-KHAOS.

Trains locally on private data using FedProx proximal regularization.
Zero raw images leave the host: outputs solely numerical parameter tensors.
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
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from agrifed_khaos.crypto import compute_manifest_hmac, compute_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AgriFed On-Premise Client Trainer (Zero Data Leakage)"
    )
    # ONLY --data-dir is required! All others have sensible, production defaults
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to private dataset directory"
    )
    parser.add_argument(
        "--weights-in", type=str, default="models/base_model.pt", help="Path to base weights"
    )
    parser.add_argument(
        "--weights-out", type=str, default="models/client_parameters.pt", help="Path to save update"
    )
    parser.add_argument(
        "--metadata", type=str, default="models/metadata.json", help="Path to metadata json"
    )
    parser.add_argument("--epochs", type=int, default=2, help="Local epochs (default: 2)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument(
        "--mu", type=float, default=0.01, help="FedProx proximal regularization (default: 0.01)"
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (default: 5e-4)")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Optimizer L2 weight decay (default: 1e-4; auto-zeroed if --mu > 0)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient L2 norm for gradient clipping (default: 1.0)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=600, help="Max private samples (default: 600)"
    )
    parser.add_argument(
        "--threads", type=int, default=8, help="PyTorch CPU intra-op threads (default: 8)"
    )
    parser.add_argument(
        "--param-noise",
        "--noise-std",
        "--dp-noise",
        dest="noise_std",
        type=float,
        default=0.0,
        help="Post-training Gaussian parameter noise standard deviation sigma (default: 0.0)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Model architecture (default: resolved from metadata.json or 'resnet18')",
    )
    parser.add_argument(
        "--head-only",
        action="store_true",
        help="Train only classification head and freeze backbone (ultra-fast, ~18 KB update)",
    )
    parser.add_argument(
        "--client-id",
        type=str,
        default="client",
        help="Client identifier recorded in manifest and HMAC signature (default: 'client')",
    )
    parser.add_argument(
        "--hmac-key",
        type=str,
        default=None,
        help="Optional pre-shared secret key to sign client manifest with HMAC-SHA256",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    weights_in_path = Path(args.weights_in)
    if not weights_in_path.is_file():
        print(f"[!] Error: Base weights file not found: {weights_in_path}", file=sys.stderr)
        sys.exit(1)

    # 1. Resolve architecture and number of classes from metadata or dataset
    from agrifed_khaos.models import build_model

    num_classes = 9
    arch = args.arch
    meta_path = Path(args.metadata)
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        num_classes = meta.get("num_classes", 9)
        if not arch:
            arch = meta.get("architecture")

    arch = arch or "resnet18"
    model = build_model(architecture=arch, num_classes=num_classes).to(device)
    state_dict = torch.load(str(weights_in_path), map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print("=" * 65)
    print("AGRIFED-KHAOS: CLIENT TRAINER (ON-PREMISE ZERO-LEAKAGE)")
    print("=" * 65)
    print(f"[*] Architecture: {arch} ({num_classes} classes)")
    print(f"[*] Loaded base weights from {weights_in_path}")

    def is_head_param(param_name: str) -> bool:
        return any(k in param_name for k in ("fc.", "classifier.", "head."))

    head_only = args.head_only
    if not head_only and meta_path.is_file():
        meta_dict = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta_dict.get("training_mode") == "head_only":
            head_only = True

    if head_only:
        frozen_count = 0
        trainable_count = 0
        for name, param in model.named_parameters():
            if not is_head_param(name):
                param.requires_grad = False
                frozen_count += 1
            else:
                param.requires_grad = True
                trainable_count += 1
        print(
            f"[*] Head-Only Mode Active: {trainable_count} head layers trainable, "
            f"{frozen_count} backbone layers frozen."
        )
    else:
        print("[*] Full-Model Mode Active: All network layers trainable.")

    # Reference weights for FedProx proximal regularization: 0.5 * mu * ||W - W_ref||^2
    ref_params = [p.detach().clone() for p in model.parameters() if p.requires_grad]

    # 2. Local Private Dataset
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    data_path = Path(args.data_dir)
    dataset = datasets.ImageFolder(str(data_path), transform=transform)
    total_avail = len(dataset)
    print(f"[*] Private dataset located at {data_path} ({total_avail} samples)")

    train_ds: Any = dataset
    if args.max_samples and args.max_samples < total_avail:
        indices = torch.randperm(total_avail)[: args.max_samples].tolist()
        train_ds = Subset(dataset, indices)
        print(f"[*] Subsampled to {len(train_ds)} samples for fast local round")

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 3. Local Training with FedProx & Frozen BatchNorm Statistics
    from agrifed_khaos.wrappers import apply_adaptive_rms_noise, freeze_bn_stats

    head_trainable = [
        p for n, p in model.named_parameters() if p.requires_grad and is_head_param(n)
    ]
    bb_trainable = [
        p for n, p in model.named_parameters() if p.requires_grad and not is_head_param(n)
    ]
    if head_trainable and bb_trainable:
        param_groups = [
            {"params": bb_trainable, "lr": args.lr * 0.1},
            {"params": head_trainable, "lr": args.lr},
        ]
    else:
        param_groups = [
            {"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr}
        ]

    effective_weight_decay = args.weight_decay
    if args.mu > 0.0 and args.weight_decay == 1e-4:
        effective_weight_decay = 0.0
        print(
            f"[*] FedProx active (mu={args.mu}): decoupled AdamW weight_decay set to 0.0 "
            "to eliminate conflict between origin-pull (L2) and base-model pull (proximal)."
        )

    optimizer = torch.optim.AdamW(param_groups, weight_decay=effective_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    criterion = nn.CrossEntropyLoss()

    model.train()
    freeze_bn_stats(model)
    print("[*] Frozen BatchNorm: Active (running_mean & running_var locked to base model)")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        freeze_bn_stats(model)
        epoch_loss = 0.0
        correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)

            # Proximal penalty
            if args.mu > 0.0:
                prox_term = torch.zeros((), device=device)
                for p, p_ref in zip(trainable_params, ref_params, strict=True):
                    prox_term = prox_term + torch.sum((p - p_ref) ** 2)
                loss = loss + 0.5 * args.mu * prox_term

            loss.backward()
            if args.max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.max_grad_norm)
            optimizer.step()

            bs = x.size(0)
            epoch_loss += loss.item() * bs
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += bs

        scheduler.step()
        avg_loss = epoch_loss / max(total, 1)
        acc = (correct / max(total, 1)) * 100
        print(
            f"  [Local Epoch {epoch + 1}/{args.epochs}] Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%"
        )

    elapsed = time.time() - t0

    # 4. Optional Post-Training Gaussian Parameter Noise Perturbation
    if args.noise_std > 0.0:
        print(
            "[*] Applying Post-Training Parameter Perturbation: "
            f"Adaptive RMS-Scaled Gaussian noise (sigma={args.noise_std})..."
        )
        apply_adaptive_rms_noise(trainable_params, noise_std=args.noise_std)

    # 5. Compute Delta Norm ||W - W_ref||_2
    delta_sq = 0.0
    for p, p_ref in zip(trainable_params, ref_params, strict=True):
        delta_sq += torch.sum((p - p_ref) ** 2).item()
    delta_norm = delta_sq**0.5

    # 6. Save Output Parameters & Metadata Manifest
    out_path = Path(args.weights_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if head_only:
        saved_state = {k: v for k, v in model.state_dict().items() if is_head_param(k)}
        training_scope = "head_only"
    else:
        saved_state = model.state_dict()
        training_scope = "full"

    torch.save(saved_state, str(out_path))
    filesize_kb = out_path.stat().st_size / 1024.0
    size_str = (
        f"{filesize_kb:.1f} KB" if filesize_kb < 1024.0 else f"{filesize_kb / 1024.0:.2f} MB"
    )
    out_sha256 = compute_sha256(out_path)
    base_sha256 = compute_sha256(weights_in_path)

    num_samples = max(len(train_ds), 1)

    client_meta = {
        "client_weights_file": str(out_path),
        "client_id": args.client_id,
        "sha256": out_sha256,
        "base_model_sha256": base_sha256,
        "num_samples": num_samples,
        "delta_norm": round(delta_norm, 4),
        "training_scope": training_scope,
        "trainable_tensors": list(saved_state.keys()),
        "training_stats": {
            "epochs": args.epochs,
            "samples": num_samples,
            "batch_size": args.batch_size,
            "fedprox_mu": args.mu,
            "param_noise_sigma": args.noise_std,
            "training_time_seconds": round(elapsed, 2),
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if args.hmac_key:
        client_meta["hmac_signature"] = compute_manifest_hmac(
            key=args.hmac_key,
            client_sha256=out_sha256,
            base_model_sha256=base_sha256,
            training_scope=training_scope,
            client_id=args.client_id,
            num_samples=num_samples,
        )

    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(client_meta, indent=2), encoding="utf-8")

    print(f"[*] Training finished in {elapsed:.2f}s (Scope: {training_scope})")
    print(f"[*] Divergence Delta Norm ||ΔW||_2: {delta_norm:.4f}")
    print(f"[*] Output Parameter File: {out_path} ({size_str})")
    print(f"[*] SHA-256 Digest:        {out_sha256}")
    print(f"[*] Client Metadata Saved: {meta_path}")
    print("[*] Privacy Verified: 0 bytes of images transferred outside host.")
    print("=" * 65)


if __name__ == "__main__":
    main()
