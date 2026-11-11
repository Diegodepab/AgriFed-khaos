"""COCO format dataset parser, bounding box utilities, and dataset converters.

Provides complete native support for Object Detection datasets in COCO JSON format:
- Recursive discovery of COCO splits (train/val/test/unannotated) and labels.json files
- Robust parsing of categories, images, and bounding box annotations
- Fast conversion from COCO JSON to Ultralytics YOLO format (images, labels, yaml)
- Automated crop extraction for transforming detection datasets into classification ImageFolder
- Full dataset profiling, health checks, and metadata generation without external C-extensions
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from agrifed_khaos.data import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)

# Common filenames used for COCO annotation files
COCO_ANNOTATION_FILENAMES = frozenset({
    "labels.json",
    "annotations.json",
    "instances.json",
    "instances_train.json",
    "instances_val.json",
    "instances_test.json",
    "instances_train2017.json",
    "instances_val2017.json",
    "instances_test2017.json",
    "train.json",
    "val.json",
    "test.json",
    "_annotations.coco.json",
})

# Ignored JSON files that are not COCO datasets
IGNORED_JSON_NAMES = frozenset({
    "metadata.json",
    "catalog.json",
    "history.json",
    "package.json",
    "tsconfig.json",
    "report.json",
    "evaluation_report.json",
})


@dataclass
class COCOCategory:
    """COCO category definition."""

    id: int
    name: str
    supercategory: str | None = None


@dataclass
class COCOImage:
    """COCO image record."""

    id: int
    file_name: str
    width: int
    height: int
    license: int | None = None


@dataclass
class COCOAnnotation:
    """COCO bounding box and segmentation annotation."""

    id: int
    image_id: int
    category_id: int
    # bbox in COCO format: [x_min, y_min, width, height] in absolute pixels
    bbox: tuple[float, float, float, float]
    area: float
    iscrowd: int = 0
    segmentation: Any = None
    derived_label: str | None = None


@dataclass
class COCOSplitInfo:
    """Information for a single split (train, val, test)."""

    split_name: str
    json_path: Path
    images_dir: Path
    total_images: int
    total_annotations: int
    categories: dict[int, str]
    category_box_counts: dict[str, int]
    category_image_counts: dict[str, int]
    unannotated_images_count: int = 0


@dataclass
class COCODatasetProfile:
    """Comprehensive profile of a COCO format dataset."""

    valid: bool
    dataset_path: Path
    splits: dict[str, COCOSplitInfo] = field(default_factory=dict)
    all_categories: list[str] = field(default_factory=list)
    total_images_all_splits: int = 0
    total_annotations_all_splits: int = 0
    min_resolution: tuple[int, int] = (0, 0)
    max_resolution: tuple[int, int] = (0, 0)
    health_score: int = 100
    corrupted_count: int = 0
    missing_images_count: int = 0
    error: str | None = None


def is_coco_json_file(file_path: Path) -> bool:
    """Check if a file is a valid COCO annotation JSON file."""
    if not file_path.is_file() or file_path.name.lower() in IGNORED_JSON_NAMES:
        return False
    if file_path.suffix.lower() != ".json":
        return False

    try:
        # Check first 4KB for COCO signature keys to avoid loading gigantic files if not COCO
        with open(file_path, "rb") as f:
            head = f.read(4096).decode("utf-8", errors="ignore")
        if '"categories"' not in head and '"annotations"' not in head and '"images"' not in head:
            return False

        # Load file safely
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return False

        has_categories = "categories" in data and isinstance(data["categories"], list)
        has_images = "images" in data and isinstance(data["images"], list)
        has_annotations = "annotations" in data and isinstance(data["annotations"], list)

        return (has_categories or has_annotations) and has_images
    except Exception:
        return False


def is_coco_dataset(dir_path: Path | str) -> bool:
    """Return True if dir_path contains a valid COCO dataset."""
    p = Path(dir_path)
    if not p.is_dir():
        return False

    # Check immediate directory and up to depth 3
    candidate_jsons: list[Path] = []
    try:
        for root, dirs, files in os.walk(p):
            # Prune hidden / macosx directories
            dirs[:] = [d for d in dirs if not d.startswith((".", "_")) and d.lower() != "__macosx"]
            rel_depth = len(Path(root).relative_to(p).parts)
            if rel_depth > 3:
                dirs.clear()
                continue
            for f in files:
                if f.lower().endswith(".json") and f.lower() not in IGNORED_JSON_NAMES:
                    candidate_jsons.append(Path(root) / f)
    except OSError:
        return False

    for c_json in candidate_jsons:
        if is_coco_json_file(c_json):
            # Verify there are image files either adjacent, in images/, or in parent/sibling dirs
            img_dir = _find_images_dir_for_json(c_json)
            if img_dir and any(
                f.suffix.lower() in IMAGE_EXTENSIONS for f in img_dir.iterdir() if f.is_file()
            ):
                return True

    return False


def _find_images_dir_for_json(json_path: Path) -> Path | None:
    """Find the images directory corresponding to a COCO annotation JSON."""
    parent = json_path.parent

    # 1. Check parent / "images"
    cand_images = parent / "images"
    if cand_images.is_dir():
        return cand_images

    # 2. Check parent directory itself (if it contains images)
    try:
        has_imgs = any(
            f.suffix.lower() in IMAGE_EXTENSIONS for f in parent.iterdir() if f.is_file()
        )
        if has_imgs:
            return parent
    except OSError:
        pass

    # 3. Check grandparent / "images" / split_name
    split_name = parent.name
    grandparent = parent.parent
    cand_gp_images = grandparent / "images" / split_name
    if cand_gp_images.is_dir():
        return cand_gp_images

    cand_gp_images_direct = grandparent / "images"
    if cand_gp_images_direct.is_dir():
        return cand_gp_images_direct

    # 4. Check sibling directories named train/val/test or images
    for sibling in parent.iterdir():
        if sibling.is_dir() and sibling.name.lower() in ("images", "img", "photos"):
            return sibling

    return None


def discover_coco_splits(dir_path: Path | str) -> dict[str, tuple[Path, Path]]:
    """Discover all splits (train, val, test, etc.) in a COCO dataset.

    Returns:
        Mapping of split_name -> (json_path, images_dir)
    """
    p = Path(dir_path)
    splits: dict[str, tuple[Path, Path]] = {}
    if not p.is_dir():
        return splits

    # Collect all COCO JSON files
    coco_jsons: list[Path] = []
    for root, dirs, files in os.walk(p):
        dirs[:] = [d for d in dirs if not d.startswith((".", "_")) and d.lower() != "__macosx"]
        for f in files:
            if f.lower().endswith(".json") and f.lower() not in IGNORED_JSON_NAMES:
                full_f = Path(root) / f
                if is_coco_json_file(full_f):
                    coco_jsons.append(full_f)

    for c_json in sorted(coco_jsons, key=lambda x: str(x)):
        img_d = _find_images_dir_for_json(c_json)
        if not img_d:
            continue

        # Inferred split name
        lower_parts = [part.lower() for part in c_json.parts]
        stem = c_json.stem.lower()

        split_name = "train"
        if "test_unannotated" in lower_parts or "unannotated" in stem:
            split_name = "test_unannotated"
        elif any(k in lower_parts or k in stem for k in ("val", "valid", "validation")):
            split_name = "val"
        elif any(k in lower_parts or k in stem for k in ("test", "testing", "eval")):
            split_name = "test"
        elif any(k in lower_parts or k in stem for k in ("train", "training")):
            split_name = "train"
        else:
            split_name = c_json.parent.name.lower() if c_json.parent != p else "train"

        # Disambiguate if multiple
        final_key = split_name
        counter = 2
        while final_key in splits:
            final_key = f"{split_name}_{counter}"
            counter += 1

        splits[final_key] = (c_json, img_d)

    return splits


def parse_coco_split(
    json_path: Path,
    images_dir: Path,
    split_name: str = "train",
) -> COCOSplitInfo:
    """Parse a single COCO JSON file and compute stats."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # Categories
    categories: dict[int, str] = {}
    for cat in data.get("categories", []):
        cat_id = int(cat.get("id", 0))
        cat_name = str(cat.get("name", f"class_{cat_id}")).strip()
        categories[cat_id] = cat_name

    # Images
    images = data.get("images", [])
    total_images = len(images)

    # Annotations
    annotations = data.get("annotations", [])
    total_annotations = len(annotations)

    category_box_counts: dict[str, int] = {cat_name: 0 for cat_name in categories.values()}
    cat_images_map: dict[str, set[int]] = {cat_name: set() for cat_name in categories.values()}
    annotated_image_ids: set[int] = set()

    for ann in annotations:
        cat_id = ann.get("category_id")
        cat_name = categories.get(cat_id, str(cat_id))
        category_box_counts[cat_name] = category_box_counts.get(cat_name, 0) + 1

        img_id = ann.get("image_id")
        if img_id is not None:
            annotated_image_ids.add(img_id)
            if cat_name in cat_images_map:
                cat_images_map[cat_name].add(img_id)

    category_image_counts: dict[str, int] = {
        cat_name: len(img_set) for cat_name, img_set in cat_images_map.items()
    }
    unannotated_count = max(0, total_images - len(annotated_image_ids))

    return COCOSplitInfo(
        split_name=split_name,
        json_path=json_path,
        images_dir=images_dir,
        total_images=total_images,
        total_annotations=total_annotations,
        categories=categories,
        category_box_counts=category_box_counts,
        category_image_counts=category_image_counts,
        unannotated_images_count=unannotated_count,
    )


def compute_detection_geometry_profile(
    images_data: list[dict[str, Any]],
    classes: list[str],
) -> dict[str, Any]:
    """Compute geometric and spatial metrics for object detection datasets.

    Calculates:
    - COCO official size intervals (<32², 32²-96², ≥96² px).
    - Aspect ratio (W/H) statistics and orientative orientation brackets.
    - Box density per image (mean, median, max, p90).
    - Unannotated / background image count and percentage.
    - Vectorized intra-image pairwise IoU overlap (pairs with IoU > 0.50).
    - Lightweight sample preview metadata with normalized boxes.
    """
    import numpy as np

    total_images = len(images_data)
    if total_images == 0:
        return {
            "total_images": 0,
            "total_boxes": 0,
            "box_sizes_coco": {
                "small": {"count": 0, "pct": 0.0},
                "medium": {"count": 0, "pct": 0.0},
                "large": {"count": 0, "pct": 0.0},
            },
            "aspect_ratio": {
                "mean": 0.0,
                "median": 0.0,
                "p10": 0.0,
                "p90": 0.0,
                "vertical_lt_075": 0,
                "square_075_133": 0,
                "horizontal_gt_133": 0,
            },
            "density": {"mean": 0.0, "median": 0.0, "max": 0, "p90": 0.0},
            "unannotated_images": {"count": 0, "pct": 0.0},
            "intra_image_overlap": {
                "overlapping_pairs_gt_050": 0,
                "images_with_high_overlap_pct": 0.0,
            },
            "sample_preview": None,
        }

    boxes_per_image: list[int] = []
    all_areas_px: list[float] = []
    all_aspect_ratios: list[float] = []
    overlapping_pairs_gt_050 = 0
    images_with_overlap = 0
    sample_preview: dict[str, Any] | None = None

    for img_rec in images_data:
        w_img = max(1, int(img_rec.get("width", 640)))
        h_img = max(1, int(img_rec.get("height", 640)))
        boxes = img_rec.get("boxes", [])
        k = len(boxes)
        boxes_per_image.append(k)

        # Prioritize first sample with 1 to 12 annotations for visual preview
        if sample_preview is None and 1 <= k <= 12 and img_rec.get("image_path"):
            sample_preview = {
                "image_path": str(img_rec["image_path"]),
                "image_width": w_img,
                "image_height": h_img,
                "boxes": [
                    {
                        "class_name": str(b.get("class_name", "object")),
                        "class_id": int(b.get("class_id", 0)),
                        "box_xywhn": [float(v) for v in b.get("box_xywhn", [0.5, 0.5, 0.2, 0.2])],
                    }
                    for b in boxes
                ],
            }

        if k == 0:
            continue

        img_xyxy: list[list[float]] = []
        for b in boxes:
            if "box_xyxy" in b and len(b["box_xyxy"]) == 4:
                x1, y1, x2, y2 = [float(v) for v in b["box_xyxy"]]
            elif "box_xywhn" in b and len(b["box_xywhn"]) == 4:
                xc, yc, bw, bh = [float(v) for v in b["box_xywhn"]]
                x1 = (xc - bw / 2.0) * w_img
                y1 = (yc - bh / 2.0) * h_img
                x2 = (xc + bw / 2.0) * w_img
                y2 = (yc + bh / 2.0) * h_img
            else:
                continue

            bw_px = max(1.0, x2 - x1)
            bh_px = max(1.0, y2 - y1)
            area = bw_px * bh_px
            all_areas_px.append(area)
            all_aspect_ratios.append(bw_px / bh_px)
            img_xyxy.append([x1, y1, x2, y2])

        # Pairwise IoU vectorization via NumPy for images with 2+ boxes
        if len(img_xyxy) >= 2:
            arr = np.array(img_xyxy, dtype=np.float32)
            x1 = np.maximum(arr[:, None, 0], arr[None, :, 0])
            y1 = np.maximum(arr[:, None, 1], arr[None, :, 1])
            x2 = np.minimum(arr[:, None, 2], arr[None, :, 2])
            y2 = np.minimum(arr[:, None, 3], arr[None, :, 3])

            inter_w = np.maximum(0.0, x2 - x1)
            inter_h = np.maximum(0.0, y2 - y1)
            inter_area = inter_w * inter_h

            areas = (arr[:, 2] - arr[:, 0]) * (arr[:, 3] - arr[:, 1])
            union_area = areas[:, None] + areas[None, :] - inter_area
            iou_mat = np.where(union_area > 0, inter_area / union_area, 0.0)

            i_idx, j_idx = np.triu_indices(len(img_xyxy), k=1)
            pair_ious = iou_mat[i_idx, j_idx]
            n_over = int(np.sum(pair_ious > 0.50))
            if n_over > 0:
                overlapping_pairs_gt_050 += n_over
                images_with_overlap += 1

    total_boxes = len(all_areas_px)
    bpi_arr = np.array(boxes_per_image, dtype=np.float32) if boxes_per_image else np.array([0], dtype=np.float32)
    unannotated_cnt = int(np.sum(bpi_arr == 0))

    # Official COCO area intervals (pixels²):
    # Small: < 32² (1024), Medium: 32²..96² (1024..9216), Large: >= 96² (9216)
    small_cnt = sum(1 for a in all_areas_px if a < 1024)
    med_cnt = sum(1 for a in all_areas_px if 1024 <= a < 9216)
    large_cnt = sum(1 for a in all_areas_px if a >= 9216)

    # Orientative aspect ratio distributions
    v_cnt = sum(1 for r in all_aspect_ratios if r < 0.75)
    s_cnt = sum(1 for r in all_aspect_ratios if 0.75 <= r <= 1.33)
    h_cnt = sum(1 for r in all_aspect_ratios if r > 1.33)

    ar_mean = float(np.mean(all_aspect_ratios)) if all_aspect_ratios else 1.0
    ar_med = float(np.median(all_aspect_ratios)) if all_aspect_ratios else 1.0
    ar_p10 = float(np.percentile(all_aspect_ratios, 10)) if all_aspect_ratios else 1.0
    ar_p90 = float(np.percentile(all_aspect_ratios, 90)) if all_aspect_ratios else 1.0

    return {
        "total_images": total_images,
        "total_boxes": total_boxes,
        "box_sizes_coco": {
            "small": {"count": small_cnt, "pct": round((small_cnt / total_boxes * 100.0), 1) if total_boxes else 0.0},
            "medium": {"count": med_cnt, "pct": round((med_cnt / total_boxes * 100.0), 1) if total_boxes else 0.0},
            "large": {"count": large_cnt, "pct": round((large_cnt / total_boxes * 100.0), 1) if total_boxes else 0.0},
        },
        "aspect_ratio": {
            "mean": round(ar_mean, 2),
            "median": round(ar_med, 2),
            "p10": round(ar_p10, 2),
            "p90": round(ar_p90, 2),
            "vertical_lt_075": v_cnt,
            "square_075_133": s_cnt,
            "horizontal_gt_133": h_cnt,
        },
        "density": {
            "mean": round(float(np.mean(bpi_arr)), 2),
            "median": round(float(np.median(bpi_arr)), 1),
            "max": int(np.max(bpi_arr)),
            "p90": round(float(np.percentile(bpi_arr, 90)), 1),
        },
        "unannotated_images": {
            "count": unannotated_cnt,
            "pct": round((unannotated_cnt / total_images * 100.0), 1) if total_images else 0.0,
        },
        "intra_image_overlap": {
            "overlapping_pairs_gt_050": overlapping_pairs_gt_050,
            "images_with_high_overlap_pct": round((images_with_overlap / total_images * 100.0), 1) if total_images else 0.0,
        },
        "sample_preview": sample_preview,
    }


def inspect_coco_dataset(
    dir_path: Path | str,
    preferred_split: str = "train",
    max_health_scans: int = 500,
) -> dict[str, Any]:
    """Inspect and analyze a COCO dataset, returning standardized AgriFed inspector payload."""
    p = Path(dir_path).resolve()
    splits_dict = discover_coco_splits(p)

    if not splits_dict:
        return {
            "valid": False,
            "format": "coco_detection",
            "dataset_path": str(p),
            "original_path": str(dir_path),
            "auto_detected": False,
            "error": f"No se detectaron archivos de anotación COCO JSON válidos en {dir_path}",
        }

    # Select primary split (preferred_split if available, else first split)
    prim_key = preferred_split if preferred_split in splits_dict else next(iter(splits_dict.keys()))
    prim_json, prim_img_d = splits_dict[prim_key]

    parsed_splits: dict[str, COCOSplitInfo] = {}
    all_categories_set: set[str] = set()
    total_imgs = 0
    total_anns = 0

    for s_name, (s_json, s_img_d) in splits_dict.items():
        s_info = parse_coco_split(s_json, s_img_d, split_name=s_name)
        parsed_splits[s_name] = s_info
        all_categories_set.update(s_info.categories.values())
        total_imgs += s_info.total_images
        total_anns += s_info.total_annotations

    primary_info = parsed_splits[prim_key]

    # Calculate global category bounding box occurrences across all parsed splits
    global_cat_box_counts: dict[str, int] = collections.defaultdict(int)
    for s_info in parsed_splits.values():
        for cat_name, cnt in s_info.category_box_counts.items():
            global_cat_box_counts[cat_name] += cnt

    # Filter out phantom categories (such as Roboflow's dataset project slug with 0 annotations across all splits)
    if total_anns > 0:
        active_categories = {c for c in all_categories_set if global_cat_box_counts.get(c, 0) > 0}
        classes = sorted(active_categories) if active_categories else sorted(all_categories_set)
    else:
        classes = sorted(all_categories_set)

    filtered_image_counts = {
        k: v for k, v in primary_info.category_image_counts.items() if k in classes
    }
    filtered_box_counts = {
        k: v for k, v in primary_info.category_box_counts.items() if k in classes
    }

    # Health scanning on sample of images
    corrupted_count = 0
    missing_count = 0
    min_w, min_h = float("inf"), float("inf")
    max_w, max_h = 0, 0
    seen_hashes: set[str] = set()
    duplicates_count = 0

    # Read image entries from primary json
    with open(prim_json, encoding="utf-8") as f:
        prim_data = json.load(f)
    prim_images = prim_data.get("images", [])

    sample_images = prim_images[:max_health_scans]
    for img_rec in sample_images:
        fname = img_rec.get("file_name", "")
        img_p = prim_img_d / fname
        if not img_p.is_file():
            missing_count += 1
            continue

        try:
            hasher = hashlib.md5()
            with img_p.open("rb") as f_img:
                while chunk := f_img.read(65536):
                    hasher.update(chunk)
            h = hasher.hexdigest()
            if h in seen_hashes:
                duplicates_count += 1
            else:
                seen_hashes.add(h)

            w = img_rec.get("width")
            h_val = img_rec.get("height")
            if not w or not h_val:
                with Image.open(img_p) as pil_img:
                    w, h_val = pil_img.size

            min_w = min(min_w, w)
            max_w = max(max_w, w)
            min_h = min(min_h, h_val)
            max_h = max(max_h, h_val)
        except Exception:
            corrupted_count += 1

    if min_w == float("inf"):
        min_w, min_h = 0, 0

    health_score = max(0, 100 - (corrupted_count * 5) - (duplicates_count * 2))

    # Architecture recommendation based on dataset volume and target
    if total_anns > 5000:
        rec_arch = "yolov8s"
        rec_reason = (
            f"Macro-dataset de detección ({total_anns:,} anotaciones): "
            "Recomendado Ultralytics YOLOv8s / YOLO11n por balance mAP y velocidad."
        )
    else:
        rec_arch = "yolov8n"
        rec_reason = (
            f"Dataset de detección ({total_anns:,} anotaciones): "
            "Ultralytics YOLOv8n ultra-ligero para rápida convergencia y despliegue edge."
        )

    # Format splits summary for UI
    split_dist: dict[str, int] = {k: v.total_images for k, v in parsed_splits.items()}
    split_pcts: dict[str, float] = {}
    if total_imgs > 0:
        split_pcts = {k: round((cnt / total_imgs) * 100.0, 1) for k, cnt in split_dist.items()}

    from agrifed_khaos.metadata import suggest_dataset_metadata

    suggested = suggest_dataset_metadata(
        dataset_path=str(p),
        classes=classes,
        total_images=total_imgs,
        image_counts=filtered_image_counts,
        source_name=p.name,
        is_yolo=True,  # Treats as object detection narrative
        total_boxes=total_anns,
    )

    cat_map = {c["id"]: c["name"] for c in prim_data.get("categories", [])}
    anns_by_img: dict[Any, list[dict[str, Any]]] = {}
    for ann in prim_data.get("annotations", []):
        i_id = ann.get("image_id")
        if i_id not in anns_by_img:
            anns_by_img[i_id] = []
        anns_by_img[i_id].append(ann)

    images_data: list[dict[str, Any]] = []
    for img_rec in prim_images:
        i_id = img_rec.get("id")
        fname = img_rec.get("file_name", "")
        w_px = max(1, int(img_rec.get("width") or 640))
        h_px = max(1, int(img_rec.get("height") or 640))
        img_anns = anns_by_img.get(i_id, [])
        b_list: list[dict[str, Any]] = []
        for ann in img_anns:
            bbox = ann.get("bbox", [])
            if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                continue
            cid = ann.get("category_id", 0)
            cname = cat_map.get(cid, str(cid))
            x, y, bw, bh = [float(v) for v in bbox]
            xc = (x + bw / 2.0) / float(w_px)
            yc = (y + bh / 2.0) / float(h_px)
            norm_w = bw / float(w_px)
            norm_h = bh / float(h_px)
            b_list.append({
                "class_name": cname,
                "class_id": cid,
                "box_xyxy": [x, y, x + bw, y + bh],
                "box_xywhn": [round(xc, 6), round(yc, 6), round(norm_w, 6), round(norm_h, 6)],
            })
        images_data.append({
            "image_path": str(prim_img_d / fname),
            "width": w_px,
            "height": h_px,
            "boxes": b_list,
        })

    detection_profile = compute_detection_geometry_profile(images_data, classes)

    result = {
        "valid": True,
        "format": "coco_detection",
        "dataset_task": "object_detection",
        "dataset_path": str(prim_json.parent),
        "original_path": str(dir_path),
        "auto_detected": prim_json.parent != p,
        "classes": classes,
        "image_counts": filtered_image_counts,
        "box_counts": filtered_box_counts,
        "total_images": total_imgs,
        "total_boxes": total_anns,
        "primary_split": prim_key,
        "detection_profile": detection_profile,
        "splits": {
            k: {
                "json_path": str(v.json_path),
                "images_dir": str(v.images_dir),
                "images": v.total_images,
                "annotations": v.total_annotations,
                "unannotated": v.unannotated_images_count,
            }
            for k, v in parsed_splits.items()
        },
        "is_pre_split": len(parsed_splits) > 1,
        "split_distribution": split_dist,
        "split_percentages": split_pcts,
        "unlabeled_count": primary_info.unannotated_images_count,
        "recommended_arch": rec_arch,
        "recommendation_details": {
            "recommended_arch": rec_arch,
            "reason": rec_reason,
        },
        "health_profile": {
            "health_score": health_score,
            "corrupted_count": corrupted_count,
            "duplicates_count": duplicates_count,
            "missing_images_count": missing_count,
            "resolution_warning": (max_w > min_w * 3) if min_w > 0 else False,
            "total_scanned": len(sample_images),
            "min_resolution": [int(min_w), int(min_h)],
            "max_resolution": [int(max_w), int(max_h)],
        },
        "available_training_modes": ["detection", "crop_classification"],
    }
    result.update(suggested)
    return result


def coco_to_yolo(
    coco_json_path: Path | str,
    images_dir: Path | str,
    output_dir: Path | str,
    split_name: str = "train",
    class_mapping: dict[int, str] | None = None,
    symlink_images: bool = False,
) -> Path:
    """Convert a COCO JSON split into Ultralytics YOLO dataset format.

    Creates:
      <output_dir>/images/<split_name>/*.jpg
      <output_dir>/labels/<split_name>/*.txt
      <output_dir>/data.yaml
    """
    c_json = Path(coco_json_path)
    img_in = Path(images_dir)
    out_dir = Path(output_dir)

    out_images = out_dir / "images" / split_name
    out_labels = out_dir / "labels" / split_name
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    with open(c_json, encoding="utf-8") as f:
        data = json.load(f)

    # Categories to zero-indexed IDs
    categories = data.get("categories", [])
    if class_mapping is None:
        cat_id_to_yolo: dict[int, int] = {}
        yolo_names: dict[int, str] = {}
        for yolo_id, cat in enumerate(sorted(categories, key=lambda c: c.get("id", 0))):
            orig_id = int(cat["id"])
            cat_id_to_yolo[orig_id] = yolo_id
            yolo_names[yolo_id] = str(cat.get("name", f"class_{yolo_id}")).strip()
    else:
        cat_id_to_yolo = {orig_id: idx for idx, orig_id in enumerate(sorted(class_mapping.keys()))}
        yolo_names = {idx: class_mapping[orig_id] for orig_id, idx in cat_id_to_yolo.items()}

    # Image metadata lookup
    images_map: dict[int, dict[str, Any]] = {}
    for img in data.get("images", []):
        images_map[img["id"]] = img

    # Group annotations by image_id
    anns_by_img: dict[int, list[dict[str, Any]]] = {}
    for ann in data.get("annotations", []):
        img_id = ann.get("image_id")
        if img_id is not None:
            anns_by_img.setdefault(img_id, []).append(ann)

    # Convert per image
    for img_id, img_info in images_map.items():
        fname = img_info.get("file_name", "")
        if not fname:
            continue

        src_img = img_in / fname
        if not src_img.is_file():
            continue

        dest_img = out_images / fname
        if not dest_img.exists():
            if symlink_images:
                try:
                    dest_img.symlink_to(src_img.resolve())
                except OSError:
                    shutil.copy2(src_img, dest_img)
            else:
                shutil.copy2(src_img, dest_img)

        # Labels
        img_w = float(img_info.get("width") or 0)
        img_h = float(img_info.get("height") or 0)
        if img_w <= 0 or img_h <= 0:
            with Image.open(src_img) as pil_i:
                img_w, img_h = float(pil_i.width), float(pil_i.height)

        label_lines: list[str] = []
        for ann in anns_by_img.get(img_id, []):
            cat_id = ann.get("category_id")
            if cat_id not in cat_id_to_yolo:
                continue
            y_cls = cat_id_to_yolo[cat_id]
            bbox = ann.get("bbox")
            if not bbox or len(bbox) < 4:
                continue

            x_min, y_min, w_box, h_box = bbox
            if w_box <= 0 or h_box <= 0:
                continue

            # Convert to YOLO center x, center y, w, h normalized
            x_c = (x_min + w_box / 2.0) / img_w
            y_c = (y_min + h_box / 2.0) / img_h
            w_norm = w_box / img_w
            h_norm = h_box / img_h

            x_c = max(0.0, min(1.0, x_c))
            y_c = max(0.0, min(1.0, y_c))
            w_norm = max(0.0, min(1.0, w_norm))
            h_norm = max(0.0, min(1.0, h_norm))

            label_lines.append(f"{y_cls} {x_c:.6f} {y_c:.6f} {w_norm:.6f} {h_norm:.6f}")

        label_path = out_labels / f"{Path(fname).stem}.txt"
        content = "\n".join(label_lines) + ("\n" if label_lines else "")
        label_path.write_text(content, encoding="utf-8")

    # Generate data.yaml if not present
    yaml_path = out_dir / "data.yaml"
    if not yaml_path.exists():
        yaml_lines = [
            f"path: {out_dir.resolve()}",
            f"train: images/{split_name}",
            f"val: images/{split_name}",
            "names:",
        ]
        for y_id, name in yolo_names.items():
            yaml_lines.append(f"  {y_id}: {name}")
        yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    return yaml_path


def coco_to_classification_crops(
    coco_json_path: Path | str,
    images_dir: Path | str,
    output_dir: Path | str,
    split_name: str = "train",
    min_box_size: int = 24,
    padding_pct: float = 0.05,
    max_crops_per_class: int | None = 2000,
    target_categories: list[str] | None = None,
) -> dict[str, int]:
    """Extract object bounding boxes from COCO annotations as classification crops.

    Optimized for high-throughput batch extraction:
    - Pre-groups annotations by image to decode each source image only once.
    - Balanced class sampling (defaults to 2,000 crops/class) for instantaneous generation.
    - Creates standard ImageFolder structure:
      <output_dir>/<split_name>/<category_name>/<crop_uuid>.jpg

    Returns:
        dict mapping category_name -> number of crops generated
    """
    c_json = Path(coco_json_path)
    img_in = Path(images_dir)
    out_dir = Path(output_dir)

    with open(c_json, encoding="utf-8") as f:
        data = json.load(f)

    # Category map
    annotations = data.get("annotations", [])
    used_cat_ids = {
        int(a["category_id"]) for a in annotations if a.get("category_id") is not None
    }

    cat_map: dict[int, str] = {}
    for cat in data.get("categories", []):
        cat_id = int(cat["id"])
        c_name = str(cat.get("name", f"class_{cat_id}")).strip()
        # Filter out phantom categories that have 0 annotations in this split
        if cat_id not in used_cat_ids:
            continue
        if target_categories is None or c_name in target_categories:
            cat_map[cat_id] = c_name

    images_map: dict[int, str] = {
        img["id"]: img.get("file_name", "") for img in data.get("images", [])
    }

    crops_count: dict[str, int] = {name: 0 for name in cat_map.values()}

    # Group annotations by image_id for single-pass image decoding
    anns_by_img: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for ann in annotations:
        cat_id = ann.get("category_id")
        if cat_id in cat_map:
            anns_by_img[ann.get("image_id")].append(ann)

    for img_id, anns in anns_by_img.items():
        # Filter annotations belonging to classes that still need crops
        active_anns = [
            a
            for a in anns
            if max_crops_per_class is None
            or crops_count[cat_map[a["category_id"]]] < max_crops_per_class
        ]
        if not active_anns:
            continue

        fname = images_map.get(img_id, "")
        if not fname:
            continue

        src_p = img_in / fname
        if not src_p.is_file():
            continue

        try:
            with Image.open(src_p) as pil_img:
                img_w, img_h = pil_img.size

                for ann in active_anns:
                    cat_name = cat_map[ann["category_id"]]
                    if (
                        max_crops_per_class is not None
                        and crops_count[cat_name] >= max_crops_per_class
                    ):
                        continue

                    bbox = ann.get("bbox")
                    if not bbox or len(bbox) < 4:
                        continue

                    x_min, y_min, w_box, h_box = bbox
                    if w_box < min_box_size or h_box < min_box_size:
                        continue

                    pad_w = w_box * padding_pct
                    pad_h = h_box * padding_pct

                    crop_x1 = max(0, int(x_min - pad_w))
                    crop_y1 = max(0, int(y_min - pad_h))
                    crop_x2 = min(img_w, int(x_min + w_box + pad_w))
                    crop_y2 = min(img_h, int(y_min + h_box + pad_h))

                    if (crop_x2 - crop_x1) < min_box_size or (crop_y2 - crop_y1) < min_box_size:
                        continue

                    crop = pil_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                    cat_dir = out_dir / split_name / cat_name
                    cat_dir.mkdir(parents=True, exist_ok=True)
                    crop_fname = f"crop_{ann.get('id', 0)}_{crops_count[cat_name] + 1}.jpg"
                    crop_dest = cat_dir / crop_fname
                    crop.save(crop_dest, format="JPEG", quality=85)
                    crops_count[cat_name] += 1
        except Exception:
            continue

        if max_crops_per_class is not None and all(
            cnt >= max_crops_per_class for cnt in crops_count.values()
        ):
            break

    # Clean up any empty directories that may have been created
    split_dir = out_dir / split_name
    if split_dir.is_dir():
        for d in list(split_dir.iterdir()):
            if d.is_dir() and not any(d.iterdir()):
                try:
                    d.rmdir()
                except OSError:
                    pass

    return {k: v for k, v in crops_count.items() if v > 0}


def prepare_detection_dataset(
    data_dir: Path | str,
    output_yolo_dir: Path | str | None = None,
    symlink_images: bool = True,
) -> tuple[Path, list[str]]:
    """Resolve or convert any detection dataset (COCO JSON or YOLO format) into an Ultralytics-ready data.yaml.

    Returns:
        tuple (data_yaml_path, list_of_class_names)
    """
    root_p = Path(data_dir).resolve()

    # 1. Check if root_p or immediate children already have a valid data.yaml
    yaml_candidates = list(root_p.glob("*.yaml")) + list(root_p.glob("*.yml"))
    if not yaml_candidates and (root_p / "dataset").is_dir():
        yaml_candidates = list((root_p / "dataset").glob("*.yaml")) + list((root_p / "dataset").glob("*.yml"))

    for y_cand in yaml_candidates:
        try:
            import yaml

            y_data = yaml.safe_load(y_cand.read_text(encoding="utf-8"))
            if isinstance(y_data, dict) and "names" in y_data and ("train" in y_data or "val" in y_data):
                raw_names = y_data["names"]
                if isinstance(raw_names, list):
                    classes = [str(n) for n in raw_names]
                elif isinstance(raw_names, dict):
                    classes = [str(raw_names[k]) for k in sorted(raw_names.keys(), key=lambda x: int(x))]
                else:
                    classes = []
                if classes:
                    return y_cand.resolve(), classes
        except Exception:
            pass

    # 2. Check if it's already a YOLO format directory (has images/ and labels/)
    from agrifed_khaos.yolo import discover_yolo_directories, load_yolo_class_names

    img_d, lbl_d, yaml_p = discover_yolo_directories(root_p)
    if img_d and lbl_d:
        if yaml_p and yaml_p.is_file():
            class_map = load_yolo_class_names(yaml_p)
            classes = [class_map[k] for k in sorted(class_map.keys())]
            return yaml_p.resolve(), classes
        else:
            parent_dir = img_d.parent if img_d.parent != root_p else root_p
            yaml_path = parent_dir / "data.yaml"
            class_map = load_yolo_class_names(None)
            yaml_lines = [
                f"path: {parent_dir.resolve()}",
                f"train: {img_d.relative_to(parent_dir)}",
                f"val: {img_d.relative_to(parent_dir)}",
                "names:",
            ]
            for cid, cname in class_map.items():
                yaml_lines.append(f"  {cid}: {cname}")
            yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
            classes = [class_map[k] for k in sorted(class_map.keys())]
            return yaml_path.resolve(), classes

    # 3. Check if it's a COCO format dataset
    splits_dict = discover_coco_splits(root_p)
    if not splits_dict:
        raise ValueError(
            f"No se detectó un dataset de detección válido (COCO JSON o Ultralytics YOLO) en '{data_dir}'."
        )

    if output_yolo_dir is None:
        cache_base = Path("data/yolo_cache")
        cache_base.mkdir(parents=True, exist_ok=True)
        out_dir = cache_base / f"{root_p.name}_yolo"
    else:
        out_dir = Path(output_yolo_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect categories and ensure only categories with actual annotations are mapped
    unified_categories: dict[int, str] = {}
    all_cat_defs: dict[int, str] = {}
    used_cat_ids: set[int] = set()

    for s_raw, (s_json, _) in splits_dict.items():
        if "unannotated" in s_raw:
            continue
        try:
            with open(s_json, encoding="utf-8") as f:
                s_data = json.load(f)
                for cat in s_data.get("categories", []):
                    cid = int(cat["id"])
                    cname = str(cat.get("name", f"class_{cid}")).strip()
                    all_cat_defs[cid] = cname
                for ann in s_data.get("annotations", []):
                    used_cat_ids.add(int(ann.get("category_id", 0)))
        except Exception:
            pass

    if used_cat_ids:
        unified_categories = {
            cid: all_cat_defs[cid] for cid in sorted(used_cat_ids) if cid in all_cat_defs
        }
    else:
        unified_categories = all_cat_defs

    sorted_cat_ids = sorted(unified_categories.keys())
    yolo_class_names = [unified_categories[orig_id] for orig_id in sorted_cat_ids]

    converted_splits: set[str] = set()
    for s_raw, (s_json, s_img_d) in splits_dict.items():
        if "unannotated" in s_raw:
            continue
        canonical_split = "train"
        if any(v in s_raw.lower() for v in ("val", "valid", "validation")):
            canonical_split = "val"
        elif any(v in s_raw.lower() for v in ("test", "eval")):
            canonical_split = "test"
        elif "train" in s_raw.lower():
            canonical_split = "train"
        else:
            canonical_split = s_raw

        coco_to_yolo(
            coco_json_path=s_json,
            images_dir=s_img_d,
            output_dir=out_dir,
            split_name=canonical_split,
            class_mapping=unified_categories,
            symlink_images=symlink_images,
        )
        converted_splits.add(canonical_split)

    train_ref = "images/train" if "train" in converted_splits else f"images/{next(iter(converted_splits))}"
    val_ref = "images/val" if "val" in converted_splits else ("images/test" if "test" in converted_splits else train_ref)
    yaml_lines = [
        f"path: {out_dir.resolve()}",
        f"train: {train_ref}",
        f"val: {val_ref}",
    ]
    if "test" in converted_splits:
        yaml_lines.append("test: images/test")
    yaml_lines.append("names:")
    for y_id, c_name in enumerate(yolo_class_names):
        yaml_lines.append(f"  {y_id}: {c_name}")

    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    return yaml_path.resolve(), yolo_class_names
