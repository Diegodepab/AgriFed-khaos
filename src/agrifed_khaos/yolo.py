"""YOLO dataset parser, bounding box utilities, and PyTorch YOLODataLoader.

Provides full support for Ultralytics YOLOv5/YOLOv8/YOLOv11 datasets:
- Discovery of images/ and labels/ directory pairs and data.yaml
- Parsing bounding boxes (class_id, x, y, w, h) and segmentation polygons
- PyTorch YOLOImageDataset and batched YOLODataLoader with yolo_collate_fn
- Dynamic dataset inspection returning class distribution, box count, and masks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.functional as TF
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from agrifed_khaos.data import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)

# Standard 80 COCO classes for automatic mapping when data.yaml is not present
COCO_80_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


@dataclass
class YOLOBBox:
    """A single YOLO detection bounding box (or polygon mask)."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    class_name: str | None = None
    confidence: float = 1.0
    segmentation: list[tuple[float, float]] | None = None

    def to_xyxy(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        """Convert normalized (x_center, y_center, w, h) to absolute pixel (x1, y1, x2, y2)."""
        x_c = self.x_center * img_w
        y_c = self.y_center * img_h
        w = self.width * img_w
        h = self.height * img_h
        return (x_c - w / 2.0, y_c - h / 2.0, x_c + w / 2.0, y_c + h / 2.0)

    def to_tensor(self) -> torch.Tensor:
        """Return [class_id, x_center, y_center, width, height] as float32 tensor."""
        return torch.tensor(
            [float(self.class_id), self.x_center, self.y_center, self.width, self.height],
            dtype=torch.float32,
        )


def parse_yolo_label(file_path: Path) -> list[YOLOBBox]:
    """Parse a single YOLO .txt label file into a list of YOLOBBox items.

    Handles:
    - Standard 5-column detection: <class_id> <x_center> <y_center> <width> <height>
    - Segmentation polygon (>5 columns): <class_id> <x1> <y1> <x2> <y2> ...
    """
    if not file_path.is_file():
        return []

    boxes: list[YOLOBBox] = []
    try:
        content = file_path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.warning("Could not read YOLO label %s: %s", file_path, exc)
        return []

    if not content:
        return []

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        try:
            class_id = int(float(parts[0]))
            coords = [float(p) for p in parts[1:]]
        except ValueError:
            continue

        if len(coords) == 4:
            # Standard YOLO bbox: x_center, y_center, width, height
            x_c, y_c, w, h = coords
            x_c = max(0.0, min(1.0, x_c))
            y_c = max(0.0, min(1.0, y_c))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
            boxes.append(YOLOBBox(class_id=class_id, x_center=x_c, y_center=y_c, width=w, height=h))

        elif len(coords) >= 6 and len(coords) % 2 == 0:
            # YOLO polygon segmentation: x1, y1, x2, y2, ...
            points = [(max(0.0, min(1.0, coords[i])), max(0.0, min(1.0, coords[i + 1])))
                      for i in range(0, len(coords), 2)]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            x_c = (x_min + x_max) / 2.0
            y_c = (y_min + y_max) / 2.0
            w = max(0.0, x_max - x_min)
            h = max(0.0, y_max - y_min)
            boxes.append(
                YOLOBBox(
                    class_id=class_id,
                    x_center=x_c,
                    y_center=y_c,
                    width=w,
                    height=h,
                    segmentation=points,
                )
            )

    return boxes


def discover_yolo_directories(
    root_path: Path,
    preferred_split: str = "train",
) -> tuple[Path | None, Path | None, Path | None]:
    """Find images_dir, labels_dir and optional data.yaml in root_path or subdirectories."""
    root_p = Path(root_path)
    if not root_p.is_dir():
        return None, None, None

    # Search for data.yaml or *.yaml
    yaml_candidates = list(root_p.glob("*.yaml")) + list(root_p.glob("*.yml"))
    yaml_path = yaml_candidates[0] if yaml_candidates else None

    # Check potential images and labels pairs in priority order
    split_variations = [
        preferred_split,
        f"{preferred_split}2017",
        f"{preferred_split}128",
        "",
    ]

    # 1. Direct candidates inside root_p / images and root_p / labels
    base_img = root_p / "images"
    base_lbl = root_p / "labels"

    if base_img.is_dir() and base_lbl.is_dir():
        for s in split_variations:
            img_sub = base_img / s if s else base_img
            lbl_sub = base_lbl / s if s else base_lbl
            if img_sub.is_dir() and lbl_sub.is_dir():
                has_imgs = any(
                    f.suffix.lower() in IMAGE_EXTENSIONS
                    for f in img_sub.iterdir()
                    if f.is_file()
                )
                if has_imgs:
                    return img_sub, lbl_sub, yaml_path

    # 2. Search nested directory (e.g. coco128/images, coco128/labels)
    for child in root_p.iterdir():
        if child.is_dir() and not child.name.startswith((".", "_")):
            c_img = child / "images"
            c_lbl = child / "labels"
            if c_img.is_dir() and c_lbl.is_dir():
                for s in split_variations:
                    img_sub = c_img / s if s else c_img
                    lbl_sub = c_lbl / s if s else c_lbl
                    if img_sub.is_dir() and lbl_sub.is_dir():
                        has_imgs = any(
                            f.suffix.lower() in IMAGE_EXTENSIONS
                            for f in img_sub.iterdir()
                            if f.is_file()
                        )
                        if has_imgs:
                            c_yaml = list(child.glob("*.yaml")) or yaml_candidates
                            return img_sub, lbl_sub, c_yaml[0] if c_yaml else yaml_path

    return None, None, yaml_path


def load_yolo_class_names(yaml_path: Path | None, max_class_id: int = 79) -> dict[int, str]:
    """Extract class names mapping from data.yaml or fallback to COCO / class_N."""
    class_map: dict[int, str] = {}
    if yaml_path and yaml_path.is_file():
        try:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                names = data.get("names")
                if isinstance(names, list):
                    for idx, name in enumerate(names):
                        class_map[idx] = str(name)
                elif isinstance(names, dict):
                    for k, v in names.items():
                        try:
                            class_map[int(k)] = str(v)
                        except ValueError:
                            pass
        except Exception as exc:
            logger.warning("Error reading YOLO yaml %s: %s", yaml_path, exc)

    # If no mapping was found in yaml:
    if not class_map:
        if max_class_id < len(COCO_80_CLASSES):
            for i, name in enumerate(COCO_80_CLASSES):
                class_map[i] = name
        else:
            for i in range(max_class_id + 1):
                class_map[i] = f"class_{i}"

    return class_map


def is_yolo_dataset(dir_path: Path) -> bool:
    """Return True if dir_path contains a valid YOLO dataset structure."""
    img_d, lbl_d, _ = discover_yolo_directories(dir_path)
    return img_d is not None and lbl_d is not None


def inspect_yolo_dataset(
    dir_path: Path,
    preferred_split: str = "train",
) -> dict[str, Any]:
    """Inspect and analyze a YOLO dataset, returning boxes, classes, and mask information."""
    img_d, lbl_d, yaml_p = discover_yolo_directories(dir_path, preferred_split=preferred_split)
    if not img_d or not lbl_d:
        return {
            "valid": False,
            "format": "yolo_detection",
            "error": f"No se encontró estructura YOLO (images/ y labels/) en {dir_path}",
        }

    image_files = sorted(
        [f for f in img_d.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda x: x.name,
    )
    total_images = len(image_files)
    if total_images == 0:
        return {
            "valid": False,
            "format": "yolo_detection",
            "error": f"No se encontraron imágenes en el directorio YOLO: {img_d}",
        }

    total_boxes = 0
    images_with_boxes = 0
    empty_images = 0
    has_segmentation_masks = False
    raw_class_box_counts: dict[int, int] = {}
    raw_class_img_counts: dict[int, int] = {}

    for img_p in image_files:
        lbl_p = lbl_d / f"{img_p.stem}.txt"
        boxes = parse_yolo_label(lbl_p)
        if boxes:
            images_with_boxes += 1
            total_boxes += len(boxes)
            seen_in_img: set[int] = set()
            for b in boxes:
                raw_class_box_counts[b.class_id] = raw_class_box_counts.get(b.class_id, 0) + 1
                seen_in_img.add(b.class_id)
                if b.segmentation is not None:
                    has_segmentation_masks = True
            for c_id in seen_in_img:
                raw_class_img_counts[c_id] = raw_class_img_counts.get(c_id, 0) + 1
        else:
            empty_images += 1

    max_c_id = max(raw_class_box_counts.keys()) if raw_class_box_counts else 0
    class_map = load_yolo_class_names(yaml_p, max_class_id=max_c_id)

    named_box_counts: dict[str, int] = {
        class_map.get(cid, f"class_{cid}"): count
        for cid, count in sorted(raw_class_box_counts.items())
    }
    named_img_counts: dict[str, int] = {
        class_map.get(cid, f"class_{cid}"): count
        for cid, count in sorted(raw_class_img_counts.items())
    }
    classes_list = sorted(named_box_counts.keys())

    return {
        "valid": True,
        "format": "yolo_detection",
        "task": "Detección de Objetos (YOLO)",
        "dataset_path": str(img_d),
        "labels_path": str(lbl_d),
        "yaml_path": str(yaml_p) if yaml_p else None,
        "total_images": total_images,
        "total_boxes": total_boxes,
        "images_with_boxes": images_with_boxes,
        "empty_images": empty_images,
        "has_segmentation_masks": has_segmentation_masks,
        "classes": classes_list,
        "class_box_counts": named_box_counts,
        "image_counts": named_img_counts,
        "recommended_arch": "yolov8n",
        "recommendation_details": {
            "recommended_arch": "yolov8n",
            "reason": (
                f"Dataset YOLO detectado con {total_boxes} cajas anotadas y "
                f"{len(classes_list)} clases en {total_images} imágenes."
            ),
        },
    }


class YOLOImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]):
    """PyTorch Dataset loading images and normalized YOLO bounding boxes."""

    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path | None = None,
        img_size: int = 640,
        class_names: dict[int, str] | list[str] | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = (
            Path(labels_dir)
            if labels_dir
            else self.images_dir.parent / "labels" / self.images_dir.name
        )
        self.img_size = img_size

        if isinstance(class_names, list):
            self.class_names: dict[int, str] = {i: name for i, name in enumerate(class_names)}
        elif isinstance(class_names, dict):
            self.class_names = class_names
        else:
            self.class_names = {}

        self.samples = sorted(
            [
                f
                for f in self.images_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            ],
            key=lambda x: x.name,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        img_path = self.samples[idx]

        with Image.open(img_path) as raw_im:
            rgb_im = raw_im.convert("RGB")
            orig_w, orig_h = rgb_im.size

            # Letterbox / resize to target square img_size
            resized = rgb_im.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
            img_tensor = TF.to_tensor(resized)

        # Parse matching labels
        label_file = self.labels_dir / f"{img_path.stem}.txt"
        boxes = parse_yolo_label(label_file)

        if boxes:
            target_list = [b.to_tensor() for b in boxes]
            targets_tensor = torch.stack(target_list, dim=0)  # Shape [N, 5]
        else:
            targets_tensor = torch.zeros((0, 5), dtype=torch.float32)

        meta: dict[str, Any] = {
            "image_id": img_path.stem,
            "path": str(img_path),
            "orig_size": (orig_w, orig_h),
            "boxes_count": len(boxes),
            "has_mask": any(b.segmentation is not None for b in boxes),
        }

        return img_tensor, targets_tensor, meta


def yolo_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]],
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """Custom YOLO collate function.

    Packs images into [B, 3, H, W] and concatenates targets into [Total_Boxes, 6]
    where each target row is [batch_idx, class_id, x_center, y_center, width, height].
    """
    images, targets, metas = zip(*batch, strict=False)
    stacked_images = torch.stack(images, dim=0)

    batched_targets: list[torch.Tensor] = []
    for b_idx, t in enumerate(targets):
        if t.shape[0] > 0:
            batch_col = torch.full((t.shape[0], 1), fill_value=float(b_idx), dtype=torch.float32)
            b_targets = torch.cat([batch_col, t], dim=1)  # [N_i, 6]
            batched_targets.append(b_targets)

    if batched_targets:
        stacked_targets = torch.cat(batched_targets, dim=0)
    else:
        stacked_targets = torch.zeros((0, 6), dtype=torch.float32)

    return stacked_images, stacked_targets, list(metas)


def build_yolo_dataloader(
    data_dir: str | Path,
    split: str = "train",
    batch_size: int = 16,
    num_workers: int = 0,
    shuffle: bool = True,
    img_size: int = 640,
) -> tuple[DataLoader[Any], dict[str, Any]]:
    """Factory creating a ready-to-use PyTorch DataLoader with YOLO collate function."""
    dir_p = Path(data_dir)
    img_d, lbl_d, yaml_p = discover_yolo_directories(dir_p, preferred_split=split)
    if not img_d or not lbl_d:
        raise FileNotFoundError(f"No se encontró estructura válida de YOLO en '{data_dir}'")

    class_map = load_yolo_class_names(yaml_p)
    dataset = YOLOImageDataset(
        images_dir=img_d,
        labels_dir=lbl_d,
        img_size=img_size,
        class_names=class_map,
    )

    loader = DataLoader(
        dataset,
        batch_size=max(1, batch_size),
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=yolo_collate_fn,
    )

    info = {
        "images_dir": str(img_d),
        "labels_dir": str(lbl_d),
        "total_images": len(dataset),
        "img_size": img_size,
        "classes": list(class_map.values()),
    }

    return loader, info
