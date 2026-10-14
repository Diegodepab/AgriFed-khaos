#!/usr/bin/env python3
"""Domain-Agnostic Base Model Trainer for AgriFed-KHAOS.

Trains a base vision classification backbone on any dataset with class subdirectories.
Outputs model weights and architecture metadata for federated orchestration.
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from agrifed_khaos.crypto import compute_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgriFed Domain-Agnostic Base Model Trainer")
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to training directory (class folders)"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="auto",
        help="Model architecture name, or 'auto' for ModelSelector decision (default: auto)",
    )
    parser.add_argument(
        "--target-client-tier",
        type=str,
        choices=["cpu-low", "cpu-high", "gpu"],
        default="cpu-low",
        help="Target client hardware tier for federated constraints (default: cpu-low)",
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
        "--balanced-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use inverse-frequency class weighting in CrossEntropyLoss to counter class skew",
    )
    parser.add_argument(
        "--threads", type=int, default=8, help="PyTorch CPU intra-op threads (default: 8)"
    )
    return parser.parse_args()


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
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
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

    # Estimate average image resolution from sample files
    from PIL import Image

    sample_res: list[tuple[int, int]] = []
    for img_path, _ in dataset.samples[:10]:
        try:
            with Image.open(img_path) as im:
                sample_res.append(im.size)
        except Exception:
            pass

    avg_res = None
    if sample_res:
        avg_w = int(sum(s[0] for s in sample_res) / len(sample_res))
        avg_h = int(sum(s[1] for s in sample_res) / len(sample_res))
        avg_res = (avg_w, avg_h)

    # 2. Select Architecture via Auto-FL Engine or User Specification
    from agrifed_khaos.hardware import ModelSelector
    from agrifed_khaos.models import build_model

    is_auto = args.arch.strip().lower() == "auto"
    selected_arch = "resnet18"
    selection_rationale = "User manually specified architecture."

    if is_auto:
        server_profile, best_candidate = ModelSelector.choose_best(
            task="classification",
            target_client_tier=args.target_client_tier,
            dataset_sample_count=total_samples,
            num_classes=num_classes,
            avg_resolution=avg_res,
        )
        selected_arch = best_candidate.architecture
        selection_rationale = best_candidate.rationale
        print("[*] Auto-FL Model Decision Engine Activated:")
        print(f"    • Selected Architecture:  {selected_arch} ({best_candidate.family.upper()})")
        print(f"    • Suitability Score:      {best_candidate.federated_suitability_score}/100")
        print(
            f"    • Checkpoint Footprint:   ~{best_candidate.checkpoint_size_mb:.1f} MB per round"
        )
        print(
            f"    • Dataset Characteristics: {total_samples} samples across {num_classes} classes"
        )
        if avg_res:
            print(f"    • Average Resolution:     {avg_res[0]}x{avg_res[1]} px")
        print(f"    • Architectural Decision: {best_candidate.rationale}")
    else:
        selected_arch = args.arch.strip()
        print(f"[*] Architecture specified manually by user: '{selected_arch}'")

    # Subsample if max_samples is set for manageable CPU runtime
    train_ds: Any = dataset
    if args.max_samples and args.max_samples < total_samples:
        from agrifed_khaos.data import create_stratified_subset

        train_ds = create_stratified_subset(dataset, args.max_samples, seed=42)
        print(
            f"[*] Stratified subsampling active: {len(train_ds)} samples "
            f"across {num_classes} classes"
        )

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 3. Initialize Model with ImageNet Pretrained Backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Base training device: {device.type.upper()}")
    model = build_model(
        architecture=selected_arch, num_classes=num_classes, pretrained=True
    ).to(device)
    head_params = [
        p for n, p in model.named_parameters() if any(k in n for k in ("fc", "classifier", "head"))
    ]
    bb_params = [
        p
        for n, p in model.named_parameters()
        if not any(k in n for k in ("fc", "classifier", "head"))
    ]
    if head_params and bb_params:
        param_groups = [
            {"params": bb_params, "lr": args.lr * 0.1},
            {"params": head_params, "lr": args.lr},
        ]
        print(f"[*] Differential Learning Rate: Backbone={args.lr * 0.1:.1e} | Head={args.lr:.1e}")
    else:
        param_groups = [{"params": model.parameters(), "lr": args.lr}]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    if args.balanced_loss:
        from agrifed_khaos.data import compute_class_weights

        class_weights = compute_class_weights(train_ds, num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(
            f"[*] Balanced Class-Weighted Loss Active: "
            f"weights={[round(w, 3) for w in class_weights.tolist()]}"
        )
    else:
        criterion = nn.CrossEntropyLoss()
        print("[*] Unweighted CrossEntropyLoss Active")

    # 4. Training Loop
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

        scheduler.step()
        avg_loss = ep_loss / max(total, 1)
        acc = (correct / max(total, 1)) * 100
        print(f"  [Epoch {ep + 1}/{args.epochs}] Train Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")

    elapsed = time.time() - t0
    print(f"[*] Base training completed in {elapsed:.2f}s ({elapsed / args.epochs:.2f}s per epoch)")

    # 5. Save Model & Metadata
    out_file = Path(args.output)
    meta_file = Path(args.metadata)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    meta_file.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), str(out_file))
    sha256_digest = compute_sha256(out_file)
    metadata = {
        "architecture": selected_arch,
        "num_classes": num_classes,
        "classes": classes,
        "img_size": 224,
        "target_client_tier": args.target_client_tier,
        "selection_mode": "auto" if is_auto else "manual",
        "selection_rationale": selection_rationale,
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
