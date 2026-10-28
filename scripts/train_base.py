#!/usr/bin/env python3
"""Domain-Agnostic Base Model Trainer for AgriFed-KHAOS.

Trains a base vision classification backbone on any dataset with class subdirectories.
Outputs model weights and architecture metadata for federated orchestration.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from agrifed_khaos.crypto import compute_sha256

EARLY_STOPPING_PATIENCE = 5
PLATEAU_PATIENCE = 2  # Must stay strictly lower than EARLY_STOPPING_PATIENCE
PLATEAU_FACTOR = 0.1
GRAD_CLIP_MAX_NORM = 1.0
WEIGHT_DECAY = 1e-4


class EarlyStopping:
    """Track a higher-is-better metric, keep best weights and signal when to stop."""

    def __init__(self, patience: int = EARLY_STOPPING_PATIENCE, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = float("-inf")
        self.best_epoch = 0
        self.best_model_wts: dict[str, torch.Tensor] | None = None
        self.counter = 0

    def step(self, metric: float, model: nn.Module, epoch: int) -> bool:
        """Register this epoch's metric. Returns True if it improved on the best."""
        if metric > self.best_metric + self.min_delta:
            self.best_metric = metric
            self.best_epoch = epoch
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.counter = 0
            return True
        self.counter += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.counter >= self.patience


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    use_amp: bool,
) -> dict[str, float]:
    """Compute validation loss (unweighted CE), accuracy and macro-F1."""
    model.eval()
    loss_sum = 0.0
    total = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out = model(x)
        loss_sum += F.cross_entropy(out.float(), y, reduction="sum").item()
        preds = out.argmax(dim=1)
        idx = (y * num_classes + preds).cpu()
        confusion += torch.bincount(idx, minlength=num_classes * num_classes).view(
            num_classes, num_classes
        )
        total += x.size(0)

    tp = confusion.diag().float()
    support = confusion.sum(dim=1).float()
    predicted = confusion.sum(dim=0).float()
    denom = support + predicted
    f1 = torch.where(denom > 0, 2 * tp / denom.clamp(min=1), torch.zeros_like(tp))
    present = denom > 0
    macro_f1 = f1[present].mean().item() if present.any() else 0.0
    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": tp.sum().item() / max(total, 1),
        "macro_f1": macro_f1,
    }


class FocalLoss(nn.Module):
    """Multi-class Focal Loss with optional class-weighting and label smoothing."""

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()


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
    parser.add_argument(
        "--tl-strategy",
        type=str,
        choices=["partial", "head_only", "full"],
        default="partial",
        help="Transfer learning strategy: partial (default), head_only, full",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["ce", "weighted_ce", "focal"],
        default="ce",
        help="Loss function: ce (standard cross-entropy), weighted_ce, focal",
    )
    parser.add_argument(
        "--aug",
        type=str,
        choices=["basic", "randaugment", "none"],
        default="basic",
        help="Data augmentation policy: basic (default), randaugment, none",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=None,
        help="Fraction of dataset for training when flat (e.g. 0.8)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=None,
        help="Fraction of dataset for validation when flat (e.g. 0.1)",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=None,
        help="Fraction of dataset for test when flat (e.g. 0.1)",
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
    if args.aug == "none":
        transform = transforms.Compose(
            [
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        print("[*] Augmentation: None (Deterministic Resize + ImageNet Normalize)")
    elif args.aug == "randaugment":
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        print("[*] Augmentation: RandAugment Active (2 ops, mag=9, multi-scale crop)")
    else:  # "basic"
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        print("[*] Augmentation: Basic (Flips, gentle Jitter & RandomResizedCrop)")

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

    # Subsample or partition dataset
    train_ds: Any = dataset
    val_split_len: int | None = None
    test_split_len: int | None = None
    val_indices: list[int] = []

    if args.train_split is not None and args.train_split < 1.0:
        t_ratio = max(0.01, args.train_split)
        v_ratio = max(0.0, args.val_split if args.val_split is not None else 0.1)
        te_ratio = max(0.0, args.test_split if args.test_split is not None else 0.1)
        tot_ratio = t_ratio + v_ratio + te_ratio
        if tot_ratio > 0:
            t_ratio /= tot_ratio
            v_ratio /= tot_ratio
            te_ratio /= tot_ratio

        train_split_len = int(total_samples * t_ratio)
        val_split_len = int(total_samples * v_ratio)
        test_split_len = int(total_samples * te_ratio)
        rem = total_samples - (train_split_len + val_split_len + test_split_len)
        fractions = [
            (total_samples * t_ratio - train_split_len, 0),
            (total_samples * v_ratio - val_split_len, 1),
            (total_samples * te_ratio - test_split_len, 2),
        ]
        fractions.sort(key=lambda x: x[0], reverse=True)
        alloc = [train_split_len, val_split_len, test_split_len]
        for i in range(rem):
            alloc[fractions[i % 3][1]] += 1
        train_split_len, val_split_len, test_split_len = alloc
        if train_split_len == 0 and total_samples > 0:
            train_split_len = 1
            if test_split_len > 0:
                test_split_len -= 1
            elif val_split_len > 0:
                val_split_len -= 1

        splits = torch.utils.data.random_split(
            dataset,
            [train_split_len, val_split_len, test_split_len],
            generator=torch.Generator().manual_seed(42),
        )
        train_ds = splits[0]
        val_indices = list(splits[1].indices)
        print(
            f"[*] Dataset split applied: Train={train_split_len} "
            f"({train_split_len / total_samples * 100:.1f}%), "
            f"Val={val_split_len} ({val_split_len / total_samples * 100:.1f}%), "
            f"Test={test_split_len} ({test_split_len / total_samples * 100:.1f}%)"
        )

    if args.max_samples and args.max_samples < len(train_ds):
        from agrifed_khaos.data import create_stratified_subset

        train_ds = create_stratified_subset(train_ds, args.max_samples, seed=42)
        print(
            f"[*] Stratified subsampling active: {len(train_ds)} samples "
            f"across {num_classes} classes"
        )

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Validation set: deterministic (non-augmented) view, never the augmented train transform.
    eval_transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_loader: DataLoader | None = None
    val_source = ""
    if val_indices:
        eval_ds = datasets.ImageFolder(str(data_path), transform=eval_transform)
        val_loader = DataLoader(
            Subset(eval_ds, val_indices), batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        val_source = f"held-out split ({len(val_indices)} samples)"
    elif args.train_split is None:
        for val_name in ("val", "valid", "validation"):
            sibling = Path(data_path).parent / val_name
            if sibling.is_dir():
                try:
                    sibling_ds = datasets.ImageFolder(str(sibling), transform=eval_transform)
                except (FileNotFoundError, RuntimeError):
                    continue
                if sibling_ds.classes == classes and len(sibling_ds) > 0:
                    val_loader = DataLoader(
                        sibling_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
                    )
                    val_source = f"pre-split directory '{sibling}' ({len(sibling_ds)} samples)"
                    break
    if val_loader is not None:
        print(f"[*] Validation source: {val_source}")
    else:
        print(
            "[!] No validation data available: Early Stopping and LR plateau fall back to "
            "monitoring training loss (weaker overfitting protection)."
        )
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

    last_block_prefix = None
    if args.tl_strategy == "head_only":
        for n, p in model.named_parameters():
            p.requires_grad = _is_head(n)
        param_groups = [{"params": head_params, "lr": args.lr}]
        print(f"[*] TL Strategy: 'head_only' (Backbone frozen, head lr={args.lr:.1e})")
    elif args.tl_strategy == "full":
        for _, p in model.named_parameters():
            p.requires_grad = True
        if head_params and bb_params:
            param_groups = [
                {"params": bb_params, "lr": args.lr * 0.2},
                {"params": head_params, "lr": args.lr},
            ]
            print(
                f"[*] TL Strategy: 'full' (Backbone lr={args.lr * 0.2:.1e} | Head lr={args.lr:.1e})"
            )
        else:
            param_groups = [{"params": model.parameters(), "lr": args.lr}]
            print(f"[*] TL Strategy: 'full' (All params lr={args.lr:.1e})")
    else:  # "partial"
        last_block_prefix = find_last_block_prefix(
            [n for n, _ in model.named_parameters() if not _is_head(n)]
        )
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
                f"[*] TL Strategy: 'partial' ('{last_block_prefix}*' lr={args.lr * 0.2:.1e} + "
                f"head lr={args.lr:.1e}; rest of backbone frozen, BN stats frozen)"
            )
        elif head_params and bb_params:
            for _, p in model.named_parameters():
                p.requires_grad = True
            param_groups = [
                {"params": bb_params, "lr": args.lr * 0.1},
                {"params": head_params, "lr": args.lr},
            ]
            print(
                f"[*] TL Strategy: 'partial' (Diff LR: Backbone={args.lr * 0.1:.1e} | "
                f"Head={args.lr:.1e})"
            )
        else:
            param_groups = [{"params": model.parameters(), "lr": args.lr}]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

    # Linear warmup (per-step) hands over to ReduceLROnPlateau (per-epoch, metric driven).
    # The warmup scheduler is no longer stepped once warmup ends so it cannot overwrite
    # the LR reductions applied by the plateau scheduler.
    steps_per_epoch = max(len(loader), 1)
    warmup_steps = int(round(args.warmup_epochs * steps_per_epoch))
    warmup_scheduler = (
        torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda s: min(1.0, (s + 1) / max(warmup_steps, 1))
        )
        if warmup_steps > 0
        else None
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=PLATEAU_FACTOR,
        patience=PLATEAU_PATIENCE,
        threshold=1e-4,
        threshold_mode="abs",  # metric may be negative (-train_loss fallback)
        min_lr=1e-7,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    early_stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    monitor_name = "val_macro_f1" if val_loader is not None else "-train_loss"
    print(
        f"[*] Regularization: AdamW(wd={WEIGHT_DECAY}) | AMP={'on' if use_amp else 'off'} | "
        f"GradClip={GRAD_CLIP_MAX_NORM} | ReduceLROnPlateau(patience={PLATEAU_PATIENCE}, "
        f"factor={PLATEAU_FACTOR}) | EarlyStopping(patience={EARLY_STOPPING_PATIENCE}) "
        f"on {monitor_name}"
    )

    criterion: nn.Module
    if args.loss == "focal":
        from agrifed_khaos.data import get_dataset_targets

        class_counts = torch.bincount(
            torch.as_tensor([int(t) for t in get_dataset_targets(train_ds)], dtype=torch.long),
            minlength=num_classes,
        ).float()
        total_cnt = class_counts.sum()
        alpha_weights = (total_cnt / (class_counts * num_classes + 1e-6)).clamp(
            min=0.2, max=5.0
        ).to(device)
        criterion = FocalLoss(alpha=alpha_weights, gamma=2.0, label_smoothing=args.label_smoothing)
        print(
            f"[*] Loss: FocalLoss Active (gamma=2.0, eps={args.label_smoothing}, class-weighted)"
        )
    elif args.loss == "weighted_ce" or (
        args.loss == "ce" and args.balanced_loss and "--loss" not in sys.argv
    ):
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
            f"[*] Loss: Weighted CE (tau={args.logit_adjust_tau}, "
            f"eps={args.label_smoothing}) + {weights_desc}"
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        print(
            f"[*] Loss: Standard CrossEntropyLoss Active (label_smoothing={args.label_smoothing})"
        )

    # 4. Training Loop
    from agrifed_khaos.wrappers import freeze_bn_stats

    model.train()
    t0 = time.time()
    epochs_run = 0
    stopped_early = False
    last_val_metrics: dict[str, float] | None = None
    global_step = 0
    for ep in range(args.epochs):
        model.train()
        if last_block_prefix is not None:
            freeze_bn_stats(model)
        ep_loss = 0.0
        correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # clip true (unscaled) gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            if warmup_scheduler is not None and global_step < warmup_steps:
                warmup_scheduler.step()

            ep_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        epochs_run = ep + 1
        avg_loss = ep_loss / max(total, 1)
        acc = (correct / max(total, 1)) * 100
        log_line = (
            f"  [Epoch {ep + 1}/{args.epochs}] Train Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%"
        )

        if val_loader is not None:
            last_val_metrics = evaluate_model(model, val_loader, device, num_classes, use_amp)
            monitored = last_val_metrics["macro_f1"]
            log_line += (
                f" | Val Loss: {last_val_metrics['loss']:.4f} | "
                f"Val Acc: {last_val_metrics['accuracy'] * 100:.2f}% | "
                f"Val Macro-F1: {monitored:.4f}"
            )
        else:
            monitored = -avg_loss
        print(log_line)

        # Plateau scheduler only takes over after warmup; step it after every validation phase
        lrs_before = [g["lr"] for g in optimizer.param_groups]
        if global_step >= warmup_steps:
            plateau_scheduler.step(monitored)
        lrs_after = [g["lr"] for g in optimizer.param_groups]
        if lrs_after != lrs_before:
            print(
                f"  [LR] Plateau detected: learning rate reduced "
                f"{lrs_before[-1]:.2e} -> {lrs_after[-1]:.2e}"
            )

        if early_stopper.step(monitored, model, ep + 1):
            print(f"  [*] New best model at epoch {ep + 1} ({monitor_name}={monitored:.4f})")
        elif early_stopper.should_stop:
            stopped_early = True
            print(
                f"  [EarlyStopping] No improvement for {early_stopper.patience} epochs. "
                f"Stopping at epoch {ep + 1}; best epoch was {early_stopper.best_epoch}."
            )
            break

    elapsed = time.time() - t0
    if early_stopper.best_model_wts is not None:
        model.load_state_dict(early_stopper.best_model_wts)
        print(
            f"[*] Restored best weights from epoch {early_stopper.best_epoch} "
            f"({monitor_name}={early_stopper.best_metric:.4f})"
        )
    print(
        f"[*] Base training completed in {elapsed:.2f}s "
        f"({elapsed / max(epochs_run, 1):.2f}s per epoch, {epochs_run}/{args.epochs} epochs)"
    )

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
            "transfer_learning": args.tl_strategy,
            "loss_function": args.loss,
            "augmentation": args.aug,
            "balanced_loss": args.balanced_loss,
            "label_smoothing": args.label_smoothing,
            "logit_adjust_tau": args.logit_adjust_tau,
            "cb_beta": args.cb_beta,
            "warmup_epochs": args.warmup_epochs,
            "unfreeze_last_block": last_block_prefix,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip_max_norm": GRAD_CLIP_MAX_NORM,
            "mixed_precision": use_amp,
            "epochs_run": epochs_run,
            "early_stopping": {
                "patience": EARLY_STOPPING_PATIENCE,
                "monitor": monitor_name,
                "triggered": stopped_early,
                "best_epoch": early_stopper.best_epoch,
                "best_metric": (
                    early_stopper.best_metric
                    if early_stopper.best_metric != float("-inf")
                    else None
                ),
            },
            "val_metrics_last_epoch": last_val_metrics,
        },
        "dataset_source": str(data_path),
        "dataset_splits": {
            "train": args.train_split,
            "val": args.val_split,
            "test": args.test_split,
            "counts": {
                "train": len(train_ds),
                "val": val_split_len,
                "test": test_split_len,
            }
            if args.train_split is not None
            else None,
        },
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
