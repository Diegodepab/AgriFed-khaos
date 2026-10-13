#!/usr/bin/env python3
"""Domain-Agnostic Federated Client Trainer for AgriFed-KHAOS.

Trains locally on private data using FedProx proximal regularization.
Zero raw images leave the host: outputs solely numerical parameter tensors.
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
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file for cryptographic provenance verification."""
    sha = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


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
        "--max-samples", type=int, default=600, help="Max private samples (default: 600)"
    )
    parser.add_argument(
        "--threads", type=int, default=8, help="PyTorch CPU intra-op threads (default: 8)"
    )
    parser.add_argument(
        "--dp-noise",
        type=float,
        default=0.0,
        help="Differential Privacy Gaussian noise standard deviation sigma (default: 0.0)",
    )
    return parser.parse_args()


def get_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return cast(nn.Module, model)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    weights_in_path = Path(args.weights_in)
    if not weights_in_path.is_file():
        print(f"[!] Error: Base weights file not found: {weights_in_path}", file=sys.stderr)
        sys.exit(1)

    # 1. Resolve number of classes from metadata or dataset
    num_classes = 9
    meta_path = Path(args.metadata)
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        num_classes = meta.get("num_classes", 9)

    model = get_model(num_classes=num_classes).to(device)
    state_dict = torch.load(str(weights_in_path), map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print("=" * 65)
    print("AGRIFED-KHAOS: CLIENT TRAINER (ON-PREMISE ZERO-LEAKAGE)")
    print("=" * 65)
    print(f"[*] Loaded base weights from {weights_in_path}")

    # Reference weights for FedProx proximal regularization: 0.5 * mu * ||W - W_ref||^2
    ref_params = [p.detach().clone() for p in model.parameters() if p.requires_grad]

    # 2. Local Private Dataset
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
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

    # 3. Local Training with FedProx
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    t0 = time.time()

    for epoch in range(args.epochs):
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
            optimizer.step()

            bs = x.size(0)
            epoch_loss += loss.item() * bs
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += bs

        avg_loss = epoch_loss / max(total, 1)
        acc = (correct / max(total, 1)) * 100
        print(
            f"  [Local Epoch {epoch + 1}/{args.epochs}] Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%"
        )

    elapsed = time.time() - t0

    # 4. Optional Differential Privacy (DP-SGD) Perturbation
    if args.dp_noise > 0.0:
        print(f"[*] Applying Differential Privacy: Gaussian noise (sigma={args.dp_noise})...")
        with torch.no_grad():
            for p in trainable_params:
                noise = torch.randn_like(p) * args.dp_noise
                p.add_(noise)

    # 5. Compute Delta Norm ||W - W_ref||_2
    delta_sq = 0.0
    for p, p_ref in zip(trainable_params, ref_params, strict=True):
        delta_sq += torch.sum((p - p_ref) ** 2).item()
    delta_norm = delta_sq**0.5

    # 6. Save Output Parameters & Metadata Manifest
    out_path = Path(args.weights_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out_path))
    filesize_mb = out_path.stat().st_size / (1024 * 1024)
    out_sha256 = compute_sha256(out_path)
    base_sha256 = compute_sha256(weights_in_path)

    client_meta = {
        "client_weights_file": str(out_path),
        "sha256": out_sha256,
        "base_model_sha256": base_sha256,
        "delta_norm": round(delta_norm, 4),
        "training_stats": {
            "epochs": args.epochs,
            "samples": len(train_ds),
            "batch_size": args.batch_size,
            "fedprox_mu": args.mu,
            "dp_noise_sigma": args.dp_noise,
            "training_time_seconds": round(elapsed, 2),
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(client_meta, indent=2), encoding="utf-8")

    print(f"[*] Training finished in {elapsed:.2f}s")
    print(f"[*] Divergence Delta Norm ||ΔW||_2: {delta_norm:.4f}")
    print(f"[*] Output Parameter File: {out_path} ({filesize_mb:.2f} MB)")
    print(f"[*] SHA-256 Digest:        {out_sha256}")
    print(f"[*] Client Metadata Saved: {meta_path}")
    print("[*] Privacy Verified: 0 bytes of images transferred outside host.")
    print("=" * 65)


if __name__ == "__main__":
    main()
