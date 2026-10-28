"""Tests for YOLO dataset parsing, YOLODataLoader, and YOLO inspection in KHAOS Hub."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from agrifed_khaos.catalog import ModelCatalogManager
from agrifed_khaos.models import build_model, save_checkpoint_bundle
from agrifed_khaos.yolo import (
    YOLOBBox,
    YOLOImageDataset,
    build_yolo_dataloader,
    discover_yolo_directories,
    inspect_yolo_dataset,
    is_yolo_dataset,
    parse_yolo_label,
    yolo_collate_fn,
)
from scripts.serve import create_app


@pytest.fixture
def dummy_yolo_dataset(tmp_path: Path) -> Path:
    """Create a minimal YOLO detection dataset with images and labels."""
    root = tmp_path / "mock_yolo_dataset"
    img_dir = root / "images" / "train"
    lbl_dir = root / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create data.yaml
    yaml_content = "names:\n  0: aceituna_sana\n  1: aceituna_defecto\nnc: 2\n"
    (root / "data.yaml").write_text(yaml_content, encoding="utf-8")

    # 2. Create valid JPEG image bytes via PIL
    im = Image.new("RGB", (64, 64), color="darkgreen")
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    valid_jpeg = buf.getvalue()

    for i in range(1, 4):
        (img_dir / f"img_{i:03d}.jpg").write_bytes(valid_jpeg)

    # 3. Create sample labels
    # img_001: 2 standard bounding boxes
    (lbl_dir / "img_001.txt").write_text(
        "0 0.50 0.50 0.30 0.40\n1 0.20 0.30 0.15 0.25\n",
        encoding="utf-8",
    )
    # img_002: 1 segmentation polygon (>5 tokens)
    (lbl_dir / "img_002.txt").write_text(
        "0 0.10 0.10 0.20 0.10 0.20 0.30 0.10 0.30\n",
        encoding="utf-8",
    )
    # img_003: background image without boxes (empty label file)
    (lbl_dir / "img_003.txt").write_text("", encoding="utf-8")

    return root


def test_parse_yolo_label_detection_and_masks(tmp_path: Path) -> None:
    """Ensure standard 5-column bboxes and polygon masks are parsed properly."""
    lbl_file = tmp_path / "test_label.txt"
    lbl_file.write_text(
        "# Comments should be ignored\n"
        "0 0.5 0.5 0.2 0.4\n"
        "1 0.1 0.2 0.3 0.2 0.3 0.5 0.1 0.5\n",
        encoding="utf-8",
    )

    boxes = parse_yolo_label(lbl_file)
    assert len(boxes) == 2

    # Standard bbox
    b1: YOLOBBox = boxes[0]
    assert b1.class_id == 0
    assert b1.x_center == 0.5
    assert b1.y_center == 0.5
    assert b1.width == 0.2
    assert b1.height == 0.4
    assert b1.segmentation is None

    # Polygon segmentation bbox
    b2: YOLOBBox = boxes[1]
    assert b2.class_id == 1
    assert b2.segmentation is not None
    assert len(b2.segmentation) == 4
    assert b2.x_center == 0.2
    assert b2.y_center == 0.35
    assert b2.width == pytest.approx(0.2, abs=1e-3)
    assert b2.height == pytest.approx(0.3, abs=1e-3)


def test_discover_yolo_and_inspect_dataset(dummy_yolo_dataset: Path) -> None:
    """Test YOLO directory discovery and dataset inspection metrics."""
    assert is_yolo_dataset(dummy_yolo_dataset) is True

    img_d, lbl_d, yaml_p = discover_yolo_directories(dummy_yolo_dataset, preferred_split="train")
    assert img_d is not None and img_d.is_dir()
    assert lbl_d is not None and lbl_d.is_dir()
    assert yaml_p is not None and yaml_p.is_file()

    info = inspect_yolo_dataset(dummy_yolo_dataset, preferred_split="train")
    assert info["valid"] is True
    assert info["format"] == "yolo_detection"
    assert info["total_images"] == 3
    assert info["total_boxes"] == 3
    assert info["images_with_boxes"] == 2
    assert info["empty_images"] == 1
    assert info["has_segmentation_masks"] is True
    assert "aceituna_sana" in info["classes"]
    assert "aceituna_defecto" in info["classes"]
    assert info["class_box_counts"]["aceituna_sana"] == 2
    assert info["class_box_counts"]["aceituna_defecto"] == 1


def test_yolo_image_dataset_and_collate_fn(dummy_yolo_dataset: Path) -> None:
    """Test PyTorch YOLOImageDataset and yolo_collate_fn batch assembly."""
    img_d = dummy_yolo_dataset / "images" / "train"
    lbl_d = dummy_yolo_dataset / "labels" / "train"

    dataset = YOLOImageDataset(images_dir=img_d, labels_dir=lbl_d, img_size=256)
    assert len(dataset) == 3

    # Sample 0 has 2 boxes
    img, targets, meta = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 256, 256)
    assert targets.shape == (2, 5)
    assert meta["boxes_count"] == 2

    # Sample 2 has 0 boxes
    _, targets_empty, meta_empty = dataset[2]
    assert targets_empty.shape == (0, 5)
    assert meta_empty["boxes_count"] == 0

    # Test collate function
    batch = [dataset[0], dataset[1], dataset[2]]
    images_tensor, targets_tensor, metas = yolo_collate_fn(batch)
    assert images_tensor.shape == (3, 3, 256, 256)
    # Sample 0 has 2 boxes, Sample 1 has 1 box, Sample 2 has 0 boxes -> total 3 boxes
    assert targets_tensor.shape == (3, 6)
    # First column must be batch index [0, 0, 1]
    assert targets_tensor[:, 0].tolist() == [0.0, 0.0, 1.0]
    assert len(metas) == 3


def test_build_yolo_dataloader_factory(dummy_yolo_dataset: Path) -> None:
    """Test build_yolo_dataloader factory function."""
    loader, info = build_yolo_dataloader(
        data_dir=dummy_yolo_dataset,
        split="train",
        batch_size=2,
        shuffle=False,
        img_size=128,
    )

    assert info["total_images"] == 3
    assert "aceituna_sana" in info["classes"]
    assert len(loader) == 2  # 3 samples with batch_size 2 -> 2 batches

    first_batch = next(iter(loader))
    imgs, targets, _ = first_batch
    assert imgs.shape[0] == 2
    assert targets.shape[1] == 6


def test_api_inspect_yolo_dataset(dummy_yolo_dataset: Path, tmp_path: Path) -> None:
    """Test /api/master/datasets/inspect endpoint detecting a YOLO dataset."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    m = build_model("resnet18", num_classes=2, pretrained=False)
    save_checkpoint_bundle(models_dir / "m.pt", m.state_dict(), {"classes": ["c1", "c2"]})
    meta_path = models_dir / "meta.json"
    meta_path.write_text(json.dumps({"classes": ["c1", "c2"]}))

    app = create_app(str(models_dir / "m.pt"), str(meta_path))
    cm = ModelCatalogManager(catalog_path=models_dir / "cat.json", models_dir=models_dir)
    app.state.catalog = cm
    client = TestClient(app)

    res = client.post(
        "/api/master/datasets/inspect",
        json={"dataset_path": str(dummy_yolo_dataset)},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["valid"] is True
    assert data["format"] == "yolo_detection"
    assert data["task"] == "Detección de Objetos (YOLO)"
    assert data["total_images"] == 3
    assert data["total_boxes"] == 3
    assert data["has_segmentation_masks"] is True
    assert "aceituna_sana" in data["classes"]
