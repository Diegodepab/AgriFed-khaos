#!/usr/bin/env python3
"""Parametrized Classification Dataset Builder for AgriFed-KHAOS.

Extracts balanced, context-padded image crops from detection annotations (e.g. OliveFruit_v1.1).
Supports configurable splits, per-class quotas, bounding box validation, and deterministic seeds.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

from PIL import Image


def find_default_raw_dir() -> str:
    candidates = [
        Path("../olivefruit_v1.1/OliveFruit_v1.1/dataset"),
        Path("/data/olivefruit_v1.1/OliveFruit_v1.1/dataset"),
        Path("/data/OliveFruit_v1.1/dataset"),
        Path("data/OliveFruit_v1.1/dataset"),
        Path("/home/diego/agriKhaos/olivefruit_v1.1/OliveFruit_v1.1/dataset"),
    ]
    for c in candidates:
        if c.is_dir():
            return str(c)
    return str(candidates[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AgriFed Domain-Agnostic Classification Crop Extractor"
    )
    default_raw = find_default_raw_dir()
    if Path(default_raw).is_dir():
        default_out = str(Path(default_raw).parent.parent / "classification")
    else:
        default_out = "data/classification"
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=default_raw,
        help="Path to raw dataset root containing split directories with labels.json and images/",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=default_out,
        help="Output root directory where ImageFolder splits will be constructed",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=1000,
        help="Target sample quota per class in train split (default: 1000)",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=400,
        help="Target sample quota per class in val split (default: 400)",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=400,
        help="Target sample quota per class in test split (default: 400)",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=55,
        help="Minimum bounding box width and height in pixels (default: 55)",
    )
    parser.add_argument(
        "--padding-pct",
        type=float,
        default=0.10,
        help="Context padding margin fraction added to each side of bbox (default: 0.10)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        help="JPEG quality for saved crop images (default: 92)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed for crop selection (default: 42)",
    )
    return parser.parse_args()


def process_split(
    split_name: str,
    raw_dir: Path,
    out_dir: Path,
    target_counts: dict[str, int],
    min_size: int = 55,
    padding_pct: float = 0.10,
    jpeg_quality: int = 92,
    seed: int = 42,
) -> dict[str, int]:
    split_dir = raw_dir / split_name
    labels_file = split_dir / "labels.json"
    images_dir = split_dir / "images"

    if not labels_file.is_file():
        print(f"[!] Warning: Labels file missing: {labels_file}, skipping '{split_name}'.")
        return {}
    if not images_dir.is_dir():
        print(f"[!] Warning: Images dir missing: {images_dir}, skipping '{split_name}'.")
        return {}

    with labels_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cat_map = {c["id"]: c["name"] for c in data.get("categories", [])}
    img_map = {img["id"]: img["file_name"] for img in data.get("images", [])}

    # Filter candidate annotations with bounding box sanity checks
    candidates: dict[str, list[dict]] = {}
    for a in data.get("annotations", []):
        cat_name = cat_map.get(a.get("category_id"))
        if not cat_name:
            continue

        bbox = a.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        x, y, w, h = bbox
        # Sanity check: bounding boxes must have non-negative coords and positive area
        if w < min_size or h < min_size or w <= 0 or h <= 0:
            continue

        candidates.setdefault(cat_name, []).append(a)

    rng = random.Random(seed)
    actual_counts: dict[str, int] = {}

    for cat_name, target_count in target_counts.items():
        annos = candidates.get(cat_name, [])
        rng.shuffle(annos)
        selected = annos[:target_count]
        cat_out = out_dir / split_name / cat_name
        shutil.rmtree(cat_out, ignore_errors=True)
        cat_out.mkdir(parents=True, exist_ok=True)

        saved = 0
        for idx, a in enumerate(selected):
            img_file = img_map.get(a["image_id"])
            if not img_file:
                continue
            img_path = images_dir / img_file
            if not img_path.exists():
                continue

            try:
                with Image.open(img_path) as im:
                    im_w, im_h = im.size
                    x, y, w, h = a["bbox"]

                    # Context padding calculation
                    pad_x = int(w * padding_pct)
                    pad_y = int(h * padding_pct)
                    x1 = max(0, int(x - pad_x))
                    y1 = max(0, int(y - pad_y))
                    x2 = min(im_w, int(x + w + pad_x))
                    y2 = min(im_h, int(y + h + pad_y))

                    # Bounding box geometry sanity verification
                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = im.crop((x1, y1, x2, y2))
                    out_path = cat_out / f"{split_name}_{cat_name}_{idx:05d}_{a['id']}.jpg"
                    crop.convert("RGB").save(out_path, quality=jpeg_quality)
                    saved += 1
            except Exception as e:
                print(f"[!] Error cropping {img_path}: {e}")

        actual_counts[cat_name] = saved
        print(
            f"[*] [{split_name}] Extracted {saved} crops for class '{cat_name}' "
            f"(target: {target_count}, candidates available: {len(annos)})"
        )

    return actual_counts


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if not raw_dir.is_dir():
        print(f"[!] Error: Raw dataset directory not found: {raw_dir}", file=sys.stderr)
        sys.exit(1)

    splits_config = {
        "train": {"sana": args.train_samples, "defectuosa": args.train_samples},
        "val": {"sana": args.val_samples, "defectuosa": args.val_samples},
        "test": {"sana": args.test_samples, "defectuosa": args.test_samples},
    }

    print("=" * 65)
    print("AGRIFED-KHAOS: CLASSIFICATION DATASET BUILDER")
    print("=" * 65)
    print(f"[*] Raw Dataset:      {raw_dir}")
    print(f"[*] Output Root:      {out_dir}")
    print(f"[*] Min BBox Size:    {args.min_size}px")
    print(f"[*] Padding Margin:   {args.padding_pct * 100:.1f}%")
    print(f"[*] Seed:             {args.seed}")

    total_extracted = 0
    for split_name, counts in splits_config.items():
        res = process_split(
            split_name=split_name,
            raw_dir=raw_dir,
            out_dir=out_dir,
            target_counts=counts,
            min_size=args.min_size,
            padding_pct=args.padding_pct,
            jpeg_quality=args.jpeg_quality,
            seed=args.seed,
        )
        total_extracted += sum(res.values())

    print(f"[*] Extraction finished! Total cropped images created: {total_extracted}")
    print("=" * 65)


if __name__ == "__main__":
    main()
