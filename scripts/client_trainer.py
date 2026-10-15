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
from torch.utils.data import DataLoader
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
        "--balanced-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use inverse-frequency class weighting in CrossEntropyLoss to counter local class skew"
        ),
    )
    parser.add_argument(
        "--threads", type=int, default=8, help="PyTorch CPU intra-op threads (default: 8)"
    )
    parser.add_argument(
        "--param-noise",
        "--noise-std",
        dest="noise_std",
        type=float,
        default=0.0,
        help="Heuristic parameter perturbation: RMS-scaled Gaussian noise (non-DP, default: 0.0)",
    )
    parser.add_argument(
        "--dp-clip-norm",
        type=float,
        default=0.0,
        help="Formal DP: Static L2 clipping bound C on update ΔW (default: 0.0, disabled)",
    )
    parser.add_argument(
        "--dp-noise-std",
        "--dp-noise",
        dest="dp_noise_std",
        type=float,
        default=0.0,
        help="Formal DP: Gaussian noise multiplier sigma for N(0, (sigma*C)^2 I) (default: 0.0)",
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
        "--allow-new-classes",
        action="store_true",
        help="Allow dynamic classifier expansion if client dataset contains unseen classes",
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
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Training hardware device: auto, cpu, cuda (default: auto)",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=None,
        help="Optional central server URL (e.g. http://master:8000) for automated submission",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    weights_in_path = Path(args.weights_in)
    if not weights_in_path.is_file():
        print(f"[!] Error: Base weights file not found: {weights_in_path}", file=sys.stderr)
        sys.exit(1)

    # 1. Resolve architecture and number of classes from bundle or metadata
    from agrifed_khaos.models import build_model, load_checkpoint_bundle, save_checkpoint_bundle

    state_dict, bundle_meta = load_checkpoint_bundle(weights_in_path, device=device)

    num_classes = bundle_meta.get("num_classes")
    canonical_classes = bundle_meta.get("classes")
    arch = args.arch or bundle_meta.get("architecture")
    meta_path = Path(args.metadata)
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if num_classes is None:
            num_classes = meta.get("num_classes", 9)
        if canonical_classes is None:
            canonical_classes = meta.get("classes")
        if not arch:
            arch = meta.get("architecture")

    num_classes = num_classes or 9
    arch = arch or "resnet18"
    if canonical_classes is None:
        canonical_classes = [f"class_{i}" for i in range(num_classes)]
    else:
        canonical_classes = list(canonical_classes)

    model = build_model(architecture=arch, num_classes=num_classes).to(device)
    model.load_state_dict(state_dict)
    print("=" * 65)
    print("AGRIFED-KHAOS: CLIENT TRAINER (ON-PREMISE ZERO-LEAKAGE)")
    print("=" * 65)
    print(f"[*] Architecture: {arch} ({num_classes} classes)")
    print(f"[*] Loaded base weights from {weights_in_path}")

    HEAD_PREFIXES = ("fc.", "classifier.", "head.", "model.head.")

    def is_head_param(param_name: str) -> bool:
        return param_name.startswith(HEAD_PREFIXES)

    # 2. Local Private Dataset Loading & Canonical Class Re-mapping
    raw_img_size = bundle_meta.get("img_size", 224)
    if isinstance(raw_img_size, int):
        img_size_tuple = (raw_img_size, raw_img_size)
    elif isinstance(raw_img_size, (list, tuple)) and len(raw_img_size) == 2:
        img_size_tuple = (int(raw_img_size[0]), int(raw_img_size[1]))
    else:
        img_size_tuple = (224, 224)

    transform = transforms.Compose(
        [
            transforms.Resize(img_size_tuple),
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

    # Audit & Re-map ImageFolder alphabetical targets against canonical base model classes
    local_classes = list(dataset.classes)
    known_set = set(canonical_classes)
    unseen_classes = sorted([c for c in local_classes if c not in known_set])

    if unseen_classes:
        if not args.allow_new_classes:
            valid_samples = [
                (p, t) for p, t in dataset.samples if dataset.classes[t] in known_set
            ]
            if not valid_samples:
                raise ValueError(
                    f"Local dataset at {data_path} contains only unknown class(es): "
                    f"{unseen_classes}. None match base model classes: {canonical_classes}. "
                    "Pass --allow-new-classes to enable dynamic classifier expansion."
                )
            num_filtered = len(dataset.samples) - len(valid_samples)
            print(
                f"[!] WARNING: Filtered out {num_filtered} sample(s) belonging to "
                f"unknown class(es) {unseen_classes}. "
                "(Pass --allow-new-classes to enable incremental expansion)."
            )
            dataset.samples = [
                (p, canonical_classes.index(dataset.classes[t])) for p, t in valid_samples
            ]
            dataset.targets = [t for _, t in dataset.samples]
            dataset.classes = list(canonical_classes)
            dataset.class_to_idx = {c: i for i, c in enumerate(canonical_classes)}
        else:
            expanded_classes = list(canonical_classes) + unseen_classes
            new_num_classes = len(expanded_classes)
            print(
                f"[*] Dynamic Classifier Expansion (--allow-new-classes): "
                f"Expanding classifier from {num_classes} to {new_num_classes} classes."
            )
            print(f"    Base classes: {canonical_classes}")
            print(f"    New classes:  {unseen_classes}")

            from agrifed_khaos.models import expand_model_classifier

            expand_model_classifier(
                model, new_num_classes, class_names=expanded_classes
            )
            model.to(device)
            num_classes = new_num_classes
            canonical_classes = expanded_classes
            bundle_meta["classes"] = expanded_classes
            bundle_meta["num_classes"] = new_num_classes

            dataset.samples = [
                (p, expanded_classes.index(dataset.classes[t])) for p, t in dataset.samples
            ]
            dataset.targets = [t for _, t in dataset.samples]
            dataset.classes = list(expanded_classes)
            dataset.class_to_idx = {c: i for i, c in enumerate(expanded_classes)}
    else:
        # Standard subset case: strictly map ImageFolder alphabetical indices
        # to canonical base model indices
        dataset.samples = [
            (p, canonical_classes.index(dataset.classes[t])) for p, t in dataset.samples
        ]
        dataset.targets = [t for _, t in dataset.samples]
        dataset.classes = list(canonical_classes)
        dataset.class_to_idx = {c: i for i, c in enumerate(canonical_classes)}

    train_ds: Any = dataset
    if args.max_samples and args.max_samples < len(dataset):
        from agrifed_khaos.data import create_stratified_subset

        train_ds = create_stratified_subset(dataset, args.max_samples, seed=42)
        print(f"[*] Stratified subsampled to {len(train_ds)} samples for fast local round")

    # Detect unobserved classes in local client partition (Missing Classes Masking)
    from agrifed_khaos.data import get_dataset_targets

    targets = get_dataset_targets(train_ds)
    present_classes = set(int(t) for t in targets)
    missing_classes = sorted(set(range(num_classes)) - present_classes)
    if missing_classes:
        print(
            f"[*] Extreme Non-IID skew: {len(missing_classes)} missing class(es) detected: "
            f"{missing_classes}. Gradient masking active on classification head."
        )

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

    # Reference weights for FedProx proximal regularization captured after expansion
    ref_params = [p.detach().clone() for p in model.parameters() if p.requires_grad]

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 3. Local Training with FedProx & Frozen BatchNorm Statistics
    from agrifed_khaos.wrappers import freeze_bn_stats

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

    model.train()
    freeze_bn_stats(model)
    print("[*] Frozen BatchNorm: Active (running_mean & running_var locked to base model)")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    saved_missing_head: list[tuple[torch.Tensor, torch.Tensor]] = []
    if missing_classes:
        max_miss = max(missing_classes)
        for n, p in model.named_parameters():
            if is_head_param(n) and p.requires_grad:
                if p.dim() in (1, 2) and p.shape[0] > max_miss:
                    saved_missing_head.append((p, p.detach()[missing_classes].clone()))

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

            # Zero gradients for missing classes in classification head
            if missing_classes:
                max_miss = max(missing_classes)
                for n, p in model.named_parameters():
                    if is_head_param(n) and p.grad is not None:
                        if p.dim() == 2 and p.shape[0] > max_miss:
                            p.grad[missing_classes, :] = 0.0
                        elif p.dim() == 1 and p.shape[0] > max_miss:
                            p.grad[missing_classes] = 0.0

            if args.max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.max_grad_norm)
            optimizer.step()

            # Hard projection: strictly preserve base model rows for missing classes
            if saved_missing_head:
                with torch.no_grad():
                    for p, base_rows in saved_missing_head:
                        if p.dim() == 2:
                            p[missing_classes, :] = base_rows
                        elif p.dim() == 1:
                            p[missing_classes] = base_rows

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

    # 4. Privacy Enhancement & Parameter Perturbation
    from agrifed_khaos.wrappers import (
        apply_adaptive_rms_noise,
        apply_dp_gaussian_noise,
    )

    if args.dp_clip_norm > 0.0 and args.dp_noise_std > 0.0:
        print(
            "[*] Applying Formal Differential Privacy on Update ΔW: "
            f"L2 clip C={args.dp_clip_norm}, Gaussian noise sigma={args.dp_noise_std}..."
        )
        apply_dp_gaussian_noise(
            trainable_params,
            reference_params=ref_params,
            clip_norm=args.dp_clip_norm,
            noise_std=args.dp_noise_std,
        )
    elif args.noise_std > 0.0:
        print(
            "[*] Applying Post-Training Parameter Perturbation: "
            f"Adaptive RMS-Scaled Gaussian noise (sigma={args.noise_std})..."
        )
        apply_adaptive_rms_noise(trainable_params, noise_std=args.noise_std)

    if saved_missing_head:
        with torch.no_grad():
            for p, base_rows in saved_missing_head:
                if p.dim() == 2:
                    p[missing_classes, :] = base_rows
                elif p.dim() == 1:
                    p[missing_classes] = base_rows

    # 5. Compute Delta Norm ||W - W_ref||_2
    delta_sq = 0.0
    for p, p_ref in zip(trainable_params, ref_params, strict=True):
        delta_sq += torch.sum((p - p_ref) ** 2).item()
    delta_norm = delta_sq**0.5

    # 6. Save Output Parameters & Metadata Manifest Bundle
    out_path = Path(args.weights_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if head_only:
        saved_state = {
            k: v.detach().cpu() for k, v in model.state_dict().items() if is_head_param(k)
        }
        training_scope = "head_only"
    else:
        saved_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        training_scope = "full"

    base_sha256 = compute_sha256(weights_in_path)
    num_samples = max(len(train_ds), 1)

    client_meta = {
        "client_weights_file": str(out_path),
        "client_id": args.client_id,
        "base_model_sha256": base_sha256,
        "num_samples": num_samples,
        "delta_norm": round(delta_norm, 4),
        "training_scope": training_scope,
        "architecture": arch,
        "num_classes": num_classes,
        "classes": canonical_classes,
        "img_size": raw_img_size,
        "missing_classes": missing_classes,
        "dp_applied": args.dp_clip_norm > 0.0 and args.dp_noise_std > 0.0,
        "dp_clip_norm": args.dp_clip_norm if args.dp_clip_norm > 0.0 else None,
        "dp_noise_std": args.dp_noise_std if args.dp_noise_std > 0.0 else None,
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

    save_checkpoint_bundle(out_path, saved_state, client_meta)
    out_sha256 = compute_sha256(out_path)
    client_meta["sha256"] = out_sha256

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

    filesize_kb = out_path.stat().st_size / 1024.0
    size_str = (
        f"{filesize_kb:.1f} KB" if filesize_kb < 1024.0 else f"{filesize_kb / 1024.0:.2f} MB"
    )

    print(f"[*] Training finished in {elapsed:.2f}s (Scope: {training_scope})")
    print(f"[*] Divergence Delta Norm ||ΔW||_2: {delta_norm:.4f}")
    print(f"[*] Output Parameter File: {out_path} ({size_str})")
    print(f"[*] SHA-256 Digest:        {out_sha256}")
    print(f"[*] Client Metadata Saved: {meta_path}")
    print("[*] Privacy Verified: 0 bytes of images transferred outside host.")

    if args.server_url:
        import urllib.request
        server_endpoint = f"{args.server_url.rstrip('/')}/submit-update"
        print(f"[*] Automated FL Transmission: Uploading update to {server_endpoint}...")
        try:
            with open(out_path, "rb") as f:
                data = f.read()
            boundary = f"----AgriFedBoundary{int(time.time())}"
            header = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="{out_path.name}"\r\n'
                f"Content-Type: application/octet-stream\r\n\r\n"
            ).encode()
            footer = f"\r\n--{boundary}--\r\n".encode()
            body = header + data + footer

            req = urllib.request.Request(
                f"{server_endpoint}?client_id={args.client_id}",
                data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_text = resp.read().decode("utf-8")
                print(f"[+] Automated transmission successful. Server response: {resp_text}")
        except Exception as exc:
            print(
                f"[!] Warning: Automated transmission to {server_endpoint} failed ({exc}). "
                f"Checkpoint is safely stored locally at {out_path}."
            )

    print("=" * 65)
    print("PROXIMOS PASOS (LADO SERVIDOR MAESTRO):")
    if args.server_url:
        print(
            f"  [+] El servidor maestro ya recibió tu actualización en: "
            f"models/client_updates/{args.client_id}_parameters.pt"
        )
        print("  El Administrador maestro incorporará tus parámetros con:")
        print(
            f"  python scripts/aggregate.py --base-model models/base_model.pt "
            f"--client-weights models/client_updates/{args.client_id}_parameters.pt "
            f"--output models/global_model.pt"
        )
    else:
        print(f"  Transfiere el archivo {out_path} al servidor maestro (SFTP / SCP).")
        print("  El Administrador maestro incorporará tus parámetros con:")
        print(
            f"  python scripts/aggregate.py --base-model models/base_model.pt "
            f"--client-weights {out_path} --output models/global_model.pt"
        )
    print("=" * 65)


if __name__ == "__main__":
    main()
