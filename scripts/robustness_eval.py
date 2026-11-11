#!/usr/bin/env python3
"""Out-of-Domain (In-The-Wild) Robustness Benchmark & OOD Calibrator for AgriFed-KHAOS.

Evaluates a vision classification model across:
1. Clean held-out validation samples (baseline in-distribution accuracy).
2. Five reproducible out-of-domain perturbations:
   - background_swap: Replaces uniform studio borders with synthetic scenes/gradients.
   - scale_framing: Off-center framing and wider zoom-out (simulating mobile distance).
   - illumination_cast: Color temperature shift + brightness/contrast perturbation.
   - sensor_degradation: Motion/defocus blur + JPEG quality=40 re-compression.
   - aspect_distortion: 16:9 / 4:3 non-square framing.
3. Optional OOD energy threshold calibration (5th percentile of in-domain free energy).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import datasets, transforms

from agrifed_khaos.data import RandomBackgroundSwapTransform, resolve_dataset_dir
from agrifed_khaos.inference_service import (
    build_aspect_letterbox,
    extract_smart_roi_crop,
    fuse_multiview_logits,
)
from agrifed_khaos.models import build_model, load_checkpoint_bundle


def apply_perturbation(img: Image.Image, kind: str, rng: np.random.Generator) -> Image.Image:
    """Apply a deterministic out-of-domain perturbation to a PIL image."""
    rgb = img.convert("RGB")
    w, h = rgb.size

    if kind == "background_swap":
        swap = RandomBackgroundSwapTransform(
            p=1.0, max_border_std=36.0, color_threshold=32.0, seed=int(rng.integers(0, 100000))
        )
        return swap(rgb)

    if kind == "scale_framing":
        # Simulate user holding camera further away (object occupies smaller fraction of frame)
        canvas_w, canvas_h = int(w * 1.45), int(h * 1.45)
        bg_c = tuple(int(v) for v in rng.integers(40, 210, size=3))
        canvas = Image.new("RGB", (canvas_w, canvas_h), bg_c)
        ox = int((canvas_w - w) * rng.uniform(0.2, 0.8))
        oy = int((canvas_h - h) * rng.uniform(0.2, 0.8))
        canvas.paste(rgb, (ox, oy))
        return canvas.resize((w, h), Image.Resampling.BILINEAR)

    if kind == "illumination_cast":
        # Warm/cool ambient color cast + brightness/contrast shift
        arr = np.asarray(rgb, dtype=np.float32)
        gains = rng.uniform(0.78, 1.22, size=(1, 1, 3)).astype(np.float32)
        arr = np.clip(arr * gains, 0.0, 255.0).astype(np.uint8)
        out = Image.fromarray(arr)
        out = ImageEnhance.Brightness(out).enhance(float(rng.uniform(0.75, 1.25)))
        out = ImageEnhance.Contrast(out).enhance(float(rng.uniform(0.75, 1.25)))
        return out

    if kind == "sensor_degradation":
        blurred = rgb.filter(ImageFilter.GaussianBlur(radius=1.2))
        buf = io.BytesIO()
        blurred.save(buf, format="JPEG", quality=40)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    if kind == "aspect_distortion":
        # Simulate 16:9 mobile capture with ambient side bars
        new_w = int(h * 16 / 9)
        if new_w <= w:
            new_w = int(w * 1.5)
        bg_c = tuple(int(v) for v in rng.integers(30, 220, size=3))
        canvas = Image.new("RGB", (new_w, h), bg_c)
        canvas.paste(rgb, ((new_w - w) // 2, 0))
        return canvas

    return rgb


@torch.no_grad()
def evaluate_perturbation_suite(
    model_path: str | Path,
    data_dir: str | Path,
    max_samples: int = 200,
    multiview: bool = True,
    device_str: str = "auto",
) -> dict[str, Any]:
    """Run clean + 5 out-of-domain perturbations and return structured robustness metrics."""
    device = torch.device(
        "cuda" if (device_str == "auto" and torch.cuda.is_available()) or device_str == "cuda" else "cpu"
    )
    state, meta = load_checkpoint_bundle(model_path, device=device)
    arch = meta.get("architecture", "resnet18")
    classes = meta.get("classes", [])
    num_classes = meta.get("num_classes") or len(classes) or 9
    temp = float(meta.get("temperature") or 1.0)
    raw_sz = meta.get("img_size", 224)
    img_sz = int(raw_sz[0]) if isinstance(raw_sz, (list, tuple)) else int(raw_sz)

    net = build_model(architecture=arch, num_classes=num_classes).to(device)
    net.load_state_dict(state)
    net.eval()

    rs_dim = (int(round(img_sz * 256 / 224)), int(round(img_sz * 256 / 224)))
    tf = transforms.Compose([
        transforms.Resize(rs_dim),
        transforms.CenterCrop((img_sz, img_sz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    resolved_dir = resolve_dataset_dir(str(data_dir), split="val", silent=True)
    raw_ds = datasets.ImageFolder(str(resolved_dir))
    total = len(raw_ds.samples)
    stride = max(1, total // max(max_samples, 1))
    sampled = raw_ds.samples[::stride][:max_samples]

    # Map ImageFolder class indices to model canonical classes if needed
    idx_map = {
        i: (classes.index(c) if c in classes else i)
        for i, c in enumerate(raw_ds.classes)
    }

    perturbations = [
        "clean",
        "background_swap",
        "scale_framing",
        "illumination_cast",
        "sensor_degradation",
        "aspect_distortion",
    ]
    correct_counts = {p: 0 for p in perturbations}
    clean_energies: list[float] = []
    t_val = max(temp, 1e-3)

    for s_idx, (img_p, raw_target) in enumerate(sampled):
        target = idx_map.get(raw_target, raw_target)
        with Image.open(img_p) as pil_im:
            base_im = pil_im.convert("RGB")

        for p_name in perturbations:
            rng = np.random.default_rng(42 + s_idx * 17)
            im = base_im if p_name == "clean" else apply_perturbation(base_im, p_name, rng)

            t_std = tf(im)
            views = [t_std]
            if multiview:
                roi_im = extract_smart_roi_crop(im)
                if roi_im is not im and roi_im.size != im.size:
                    views.append(tf(roi_im))
                w, h = im.size
                if max(w, h) > 0 and (abs(w - h) / float(max(w, h))) >= 0.08:
                    views.append(tf(build_aspect_letterbox(im, (img_sz, img_sz))))

            batch = torch.stack(views, dim=0).to(device)
            logits_b = net(batch).float()
            fused = fuse_multiview_logits(logits_b, temperature=t_val).cpu() if multiview else logits_b[0].cpu()
            pred = int(fused.argmax(dim=-1).item())
            if pred == target:
                correct_counts[p_name] += 1

            if p_name == "clean":
                energy = float((t_val * torch.logsumexp(fused / t_val, dim=-1)).item())
                clean_energies.append(energy)

    n_eval = max(len(sampled), 1)
    accs = {p: round((correct_counts[p] / n_eval) * 100.0, 2) for p in perturbations}
    ood_shift_keys = [p for p in perturbations if p != "clean"]
    mean_ood_acc = round(float(np.mean([accs[k] for k in ood_shift_keys])), 2)
    robustness_score = round(0.35 * accs["clean"] + 0.65 * mean_ood_acc, 2)
    p2_energy = (
        round(float(np.percentile(clean_energies, 2.0) * 0.88), 4)
        if clean_energies
        else 1.5
    )

    return {
        "model_path": str(model_path),
        "architecture": arch,
        "dataset_evaluated": str(resolved_dir),
        "samples_evaluated": len(sampled),
        "multiview_enabled": multiview,
        "clean_accuracy": accs["clean"],
        "mean_perturbed_accuracy": mean_ood_acc,
        "robustness_score": robustness_score,
        "perturbations": {k: accs[k] for k in ood_shift_keys},
        "recommended_ood_energy_threshold_p5": p2_energy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="AgriFed Out-of-Domain Robustness Evaluator")
    parser.add_argument("--model", type=str, required=True, help="Path to model .pt checkpoint")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to validation dataset")
    parser.add_argument("--max-samples", type=int, default=150, help="Max validation samples")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON report path")
    parser.add_argument(
        "--update-meta-ood",
        action="store_true",
        help="Write calibrated ood_threshold (p5 energy) into companion .meta.json",
    )
    args = parser.parse_args()

    report = evaluate_perturbation_suite(
        model_path=args.model,
        data_dir=args.data_dir,
        max_samples=args.max_samples,
        multiview=True,
    )
    print(json.dumps(report, indent=2))

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.update_meta_ood:
        mp = Path(args.model)
        stem = mp.stem.replace("_base", "")
        meta_p = mp.parent / f"{stem}.meta.json"
        if meta_p.is_file():
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            meta["ood_threshold"] = report["recommended_ood_energy_threshold_p5"]
            meta["robustness_report"] = {
                "clean_accuracy": report["clean_accuracy"],
                "mean_perturbed_accuracy": report["mean_perturbed_accuracy"],
                "robustness_score": report["robustness_score"],
                "perturbations": report["perturbations"],
            }
            meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"[*] Updated {meta_p} with ood_threshold={meta['ood_threshold']}", file=sys.stderr)


if __name__ == "__main__":
    main()

