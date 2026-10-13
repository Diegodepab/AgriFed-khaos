#!/usr/bin/env python3
"""Domain-Agnostic Base Model Trainer for AgriFed-KHAOS.

Trains a base vision classification backbone on any dataset with class subdirectories.
Outputs model weights and architecture metadata for federated orchestration.
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
    parser = argparse.ArgumentParser(description="AgriFed Domain-Agnostic Base Model Trainer")
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to training directory (class folders)"
    )
    parser.add_argument(
        "--output", type=str, default="models/base_model.pt", help="Path to save base model weights"
    )
    parser.add_argument(
        "--metadata", type=str, default="models/metadata.json", help="Path to save model metadata"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument(
        "--max-samples", type=int, default=1800, help="Max balanced samples to train on CPU"
    )
    parser.add_argument(
        "--threads", type=int, default=8, help="PyTorch CPU intra-op threads (default: 8)"
    )
    return parser.parse_args()


def get_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return cast(nn.Module, model)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    data_path = Path(args.data_dir)
    if not data_path.is_dir():
        print(f"[!] Error: data directory not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    # 1. Inspect classes dynamically from subfolders (Domain-Agnostic)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = datasets.ImageFolder(str(data_path), transform=transform)
    classes = dataset.classes
    num_classes = len(classes)
    total_samples = len(dataset)

    print("=" * 65)
    print("AGRIFED-KHAOS: BASE MODEL TRAINER (DOMAIN-AGNOSTIC)")
    print("=" * 65)
    print(f"[*] Dataset: {data_path}")
    print(f"[*] Discovered classes ({num_classes}): {classes}")
    print(f"[*] Total samples in dataset: {total_samples}")

    # Subsample if max_samples is set for manageable CPU runtime
    train_ds: Any = dataset
    if args.max_samples and args.max_samples < total_samples:
        indices = torch.randperm(total_samples)[: args.max_samples].tolist()
        train_ds = Subset(dataset, indices)
        print(f"[*] Training on {len(train_ds)} samples across {num_classes} classes")

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 2. Initialize Model
    model = get_model(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 3. Training Loop
    model.train()
    t0 = time.time()
    for ep in range(args.epochs):
        ep_loss = 0.0
        correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            ep_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        avg_loss = ep_loss / max(total, 1)
        acc = (correct / max(total, 1)) * 100
        print(f"  [Epoch {ep + 1}/{args.epochs}] Train Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")

    elapsed = time.time() - t0
    print(f"[*] Base training completed in {elapsed:.2f}s ({elapsed / args.epochs:.2f}s per epoch)")

    # 4. Save Model & Metadata
    out_file = Path(args.output)
    meta_file = Path(args.metadata)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    meta_file.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), str(out_file))
    sha256_digest = compute_sha256(out_file)
    metadata = {
        "architecture": "resnet18",
        "num_classes": num_classes,
        "classes": classes,
        "img_size": 224,
        "weights_file": str(out_file),
        "sha256": sha256_digest,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "samples_trained": len(train_ds),
        },
        "dataset_source": str(data_path),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[*] Checkpoint saved: {out_file} ({out_file.stat().st_size / (1024 * 1024):.2f} MB)")
    print(f"[*] SHA-256 Digest:   {sha256_digest}")
    print(f"[*] Metadata saved:   {meta_file}")
    print("=" * 65)


if __name__ == "__main__":
    main()
