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
        "--img-size",
        type=int,
        default=224,
        help="Input image resolution in pixels (default: 224)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Hardware accelerator to train on: auto, cpu, cuda (default: auto)",
    )
    parser.add_argument(
        "--threads", type=int, default=8, help="PyTorch CPU intra-op threads (default: 8)"
    )
    parser.add_argument(
        "--unfreeze-last-block",
        action="store_true",
        help="Train only the final feature stage + head (last_block_lr = lr * 0.2), frozen BN",
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=0.05, help="Label smoothing (default: 0.05)"
    )
    parser.add_argument(
        "--logit-adjust-tau",
        type=float,
        default=0.0,
        help=(
            "Logit-adjustment strength tau on log class priors (default: 0.0 = off; "
            "tau > 0 disables Effective-Number weights to avoid double-compensation)"
        ),
    )
    parser.add_argument(
        "--cb-beta",
        type=float,
        default=0.999,
        help="Effective-Number class-balanced weighting beta (default: 0.999)",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=float,
        default=0.5,
        help="Linear LR warmup length in epochs before cosine annealing (default: 0.5)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    from agrifed_khaos.data import resolve_dataset_dir

    try:
        data_path = resolve_dataset_dir(args.data_dir, split="train")
    except FileNotFoundError as exc:
        print(f"[!] Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # 1. Inspect classes dynamically from subfolders (Domain-Agnostic)
    transform = transforms.Compose(
        [
            # Pathology-preserving augmentation: crops keep >=75% of the leaf area so
            # small lesions are not cropped out; colour jitter stays gentle so
            # chlorosis/necrosis hues are not altered. Val/test pipelines (evaluate.py)
            # remain deterministic Resize + CenterCrop + ImageNet Normalize.
            transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
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
    print(f"[*] Base training device: {device.type.upper()}")
    model = build_model(
        architecture=selected_arch, num_classes=num_classes, pretrained=True
    ).to(device)
    def _is_head(n: str) -> bool:
        return any(k in n for k in ("fc", "classifier", "head", "norm_pre")) or n.startswith(
            "norm."
        )

    head_params = [p for n, p in model.named_parameters() if _is_head(n)]
    bb_params = [p for n, p in model.named_parameters() if not _is_head(n)]

    from agrifed_khaos.quality import find_last_block_prefix

    last_block_prefix = (
        find_last_block_prefix([n for n, _ in model.named_parameters() if not _is_head(n)])
        if args.unfreeze_last_block
        else None
    )
    if args.unfreeze_last_block and last_block_prefix is None:
        print("[!] WARNING: Could not locate final feature stage; training full model.")

    if last_block_prefix is not None:
        last_block_params = []
        for n, p in model.named_parameters():
            if _is_head(n):
                p.requires_grad = True
            elif n.startswith(last_block_prefix):
                p.requires_grad = True
                last_block_params.append(p)
            else:
                p.requires_grad = False
        param_groups = [
            {"params": last_block_params, "lr": args.lr * 0.2},
            {"params": head_params, "lr": args.lr},
        ]
        print(
            f"[*] Partial Unfreezing: '{last_block_prefix}*' (lr={args.lr * 0.2:.1e}) + "
            f"head (lr={args.lr:.1e}); rest of backbone frozen, BN stats frozen."
        )
    elif head_params and bb_params:
        param_groups = [
            {"params": bb_params, "lr": args.lr * 0.1},
            {"params": head_params, "lr": args.lr},
        ]
        print(f"[*] Differential Learning Rate: Backbone={args.lr * 0.1:.1e} | Head={args.lr:.1e}")
    else:
        param_groups = [{"params": model.parameters(), "lr": args.lr}]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    from agrifed_khaos.quality import build_warmup_cosine_scheduler

    steps_per_epoch = max(len(loader), 1)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=steps_per_epoch * max(args.epochs, 1),
        warmup_steps=int(round(args.warmup_epochs * steps_per_epoch)),
    )

    criterion: nn.Module
    if args.balanced_loss:
        from agrifed_khaos.data import get_dataset_targets
        from agrifed_khaos.quality import build_imbalance_loss

        class_counts = torch.bincount(
            torch.as_tensor([int(t) for t in get_dataset_targets(train_ds)], dtype=torch.long),
            minlength=num_classes,
        ).tolist()
        criterion = build_imbalance_loss(
            class_counts,
            label_smoothing=args.label_smoothing,
            logit_adjust_tau=args.logit_adjust_tau,
            beta=args.cb_beta,
            device=device,
        )
        cw = criterion.class_weight
        weights_desc = (
            f"Effective-Number weights (beta={args.cb_beta}): {[round(w, 3) for w in cw.tolist()]}"
            if cw is not None
            else "no class weights (pure Logit Adjustment)"
        )
        print(
            f"[*] Imbalance CE (tau={args.logit_adjust_tau}, "
            f"eps={args.label_smoothing}) + {weights_desc}"
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        print(f"[*] Unweighted CrossEntropyLoss Active (label_smoothing={args.label_smoothing})")

    # 4. Training Loop
    from agrifed_khaos.wrappers import freeze_bn_stats

    model.train()
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        if last_block_prefix is not None:
            freeze_bn_stats(model)
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
            scheduler.step()

            ep_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        avg_loss = ep_loss / max(total, 1)
        acc = (correct / max(total, 1)) * 100
        print(f"  [Epoch {ep + 1}/{args.epochs}] Train Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")

    elapsed = time.time() - t0
    print(f"[*] Base training completed in {elapsed:.2f}s ({elapsed / args.epochs:.2f}s per epoch)")

    # 5. Save Model Checkpoint Bundle & Auditable Metadata Mirror
    out_file = Path(args.output)
    meta_file = Path(args.metadata)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    meta_file.parent.mkdir(parents=True, exist_ok=True)

    from agrifed_khaos.models import save_checkpoint_bundle

    metadata = {
        "architecture": selected_arch,
        "num_classes": num_classes,
        "classes": classes,
        "img_size": args.img_size,
        "calibrated_thresholds": None,
        "temperature": None,
        "target_client_tier": args.target_client_tier,
        "selection_mode": "auto" if is_auto else "manual",
        "selection_rationale": selection_rationale,
        "weights_file": str(out_file),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "samples_trained": len(train_ds),
            "balanced_loss": args.balanced_loss,
            "label_smoothing": args.label_smoothing,
            "logit_adjust_tau": args.logit_adjust_tau,
            "cb_beta": args.cb_beta,
            "warmup_epochs": args.warmup_epochs,
            "unfreeze_last_block": last_block_prefix,
        },
        "dataset_source": str(data_path),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_checkpoint_bundle(out_file, model.state_dict(), metadata)
    sha256_digest = compute_sha256(out_file)
    metadata["sha256"] = sha256_digest
    meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[*] Checkpoint saved: {out_file} ({out_file.stat().st_size / (1024 * 1024):.2f} MB)")
    print(f"[*] SHA-256 Digest:   {sha256_digest}")
    print(f"[*] Metadata saved:   {meta_file}")
    print("=" * 65)
    val_cand = Path(args.data_dir).parent / "val"
    val_dir_str = str(val_cand) if val_cand.is_dir() else args.data_dir
    print("PROXIMOS PASOS OPERATIVOS:")
    print("  1. (Recomendado) Calibrar umbrales sobre partición de validación:")
    print(
        f"     python scripts/evaluate.py --model {out_file} "
        f"--data-dir {val_dir_str} --calibrate-threshold"
    )
    print("     [Nota: Calibrar sobre datos de validación separados previene umbrales optimistas]")
    print("")
    print("  2. Iniciar el servicio REST de inferencia:")
    print(f"     python scripts/serve.py --model {out_file}")
    print("     -> Expone endpoints: /predict, /download-model y /submit-update")
    print("")
    print("  3. ¿Cómo colabora el Cliente 2 (Cooperativa On-Premise)?")
    print(f"     El Cliente 2 descarga este modelo ({out_file.name}) desde el servidor maestro:")
    print(f"     curl -o models/{out_file.name} http://<IP_SERVIDOR>:8000/download-model")
    print("     Y entrena localmente sobre sus imágenes privadas sin compartirlas:")
    print(
        f"     python scripts/client_trainer.py --data-dir /ruta/fotos_privadas "
        f"--weights-in models/{out_file.name} --head-only --server-url http://<IP_SERVIDOR>:8000"
    )
    print("=" * 65)


if __name__ == "__main__":
    main()
