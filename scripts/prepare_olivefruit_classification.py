#!/usr/bin/env python3
"""Prepares a balanced ImageFolder classification dataset from OliveFruit_v1.1 annotations.

Extracts healthy (sana) and defective (defectuosa) olive fruit crops with 10% context margin.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from PIL import Image


def process_split(
    split_name: str,
    raw_dir: Path,
    out_dir: Path,
    target_counts: dict[str, int],
    min_size: int = 55,
    padding_pct: float = 0.10,
    seed: int = 42,
) -> dict[str, int]:
    labels_file = raw_dir / split_name / "labels.json"
    images_dir = raw_dir / split_name / "images"

    with labels_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cat_map = {c["id"]: c["name"] for c in data["categories"]}
    img_map = {img["id"]: img["file_name"] for img in data["images"]}

    # Filter candidate annotations
    candidates: dict[str, list[dict]] = {"sana": [], "defectuosa": []}
    for a in data["annotations"]:
        cat_name = cat_map.get(a["category_id"])
        if cat_name in candidates:
            bbox = a["bbox"]
            if bbox[2] >= min_size and bbox[3] >= min_size:
                candidates[cat_name].append(a)

    rng = random.Random(seed)
    actual_counts = {}

    for cat_name, target_count in target_counts.items():
        annos = candidates[cat_name]
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
                    pad_x = int(w * padding_pct)
                    pad_y = int(h * padding_pct)
                    x1 = max(0, int(x - pad_x))
                    y1 = max(0, int(y - pad_y))
                    x2 = min(im_w, int(x + w + pad_x))
                    y2 = min(im_h, int(y + h + pad_y))
                    crop = im.crop((x1, y1, x2, y2))
                    out_path = cat_out / f"{split_name}_{cat_name}_{idx:05d}_{a['id']}.jpg"
                    crop.convert("RGB").save(out_path, quality=92)
                    saved += 1
            except Exception as e:
                print(f"[!] Error cropping {img_path}: {e}")

        actual_counts[cat_name] = saved
        print(
            f"[*] [{split_name}] Extracted {saved} crops for class '{cat_name}' "
            f"(target: {target_count})"
        )

    return actual_counts

def main() -> None:
    raw_dir = Path("/home/diego/agriKhaos/olivefruit_v1.1/OliveFruit_v1.1/dataset")
    out_dir = Path("/home/diego/agriKhaos/olivefruit_v1.1/classification")

    splits_config = {
        "train": {"sana": 1000, "defectuosa": 1000},
        "val": {"sana": 400, "defectuosa": 400},
        "test": {"sana": 400, "defectuosa": 400},
    }

    print("=" * 65)
    print("OLIVEFRUIT V1.1: IMAGEFOLDER CLASSIFICATION BUILDER")
    print("=" * 65)
    print(f"[*] Raw Dataset:  {raw_dir}")
    print(f"[*] Output Root:  {out_dir}")

    total_extracted = 0
    for split_name, counts in splits_config.items():
        res = process_split(split_name, raw_dir, out_dir, counts)
        total_extracted += sum(res.values())

    print(f"[*] Extraction finished! Total cropped images created: {total_extracted}")
    print("=" * 65)

if __name__ == "__main__":
    main()

