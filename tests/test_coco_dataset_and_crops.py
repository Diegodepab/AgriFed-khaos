"""Tests for COCO dataset discovery, parsing, crop extraction, and API integration."""

import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from agrifed_khaos.coco import (
    coco_to_classification_crops,
    coco_to_yolo,
    discover_coco_splits,
    inspect_coco_dataset,
    is_coco_dataset,
    is_coco_json_file,
)
from scripts.serve import create_app


def _create_synthetic_coco_dataset(root: Path) -> tuple[Path, Path]:
    """Create a minimal synthetic COCO dataset for testing."""
    split_dir = root / "dataset" / "train"
    img_dir = split_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create 2 synthetic images
    img1 = Image.new("RGB", (200, 200), color=(100, 150, 200))
    img1_p = img_dir / "leaf_001.jpg"
    img1.save(img1_p)

    img2 = Image.new("RGB", (300, 300), color=(50, 100, 150))
    img2_p = img_dir / "leaf_002.jpg"
    img2.save(img2_p)

    # 2. Create labels.json
    coco_data = {
        "info": {"description": "Synthetic Test Olive Dataset"},
        "categories": [
            {"id": 1, "name": "aceituna_sana"},
            {"id": 2, "name": "aceituna_defectuosa"},
        ],
        "images": [
            {"id": 1, "file_name": "leaf_001.jpg", "width": 200, "height": 200},
            {"id": 2, "file_name": "leaf_002.jpg", "width": 300, "height": 300},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [20.0, 30.0, 50.0, 60.0],
                "area": 3000.0,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "bbox": [100.0, 110.0, 40.0, 40.0],
                "area": 1600.0,
                "iscrowd": 0,
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 1,
                "bbox": [50.0, 60.0, 70.0, 80.0],
                "area": 5600.0,
                "iscrowd": 0,
            },
        ],
    }

    json_path = split_dir / "labels.json"
    json_path.write_text(json.dumps(coco_data, indent=2), encoding="utf-8")
    return root, json_path


def test_coco_detection_and_profiling(tmp_path: Path) -> None:
    """Verify is_coco_dataset, split discovery, and inspection profiling."""
    root, json_path = _create_synthetic_coco_dataset(tmp_path / "coco_test")

    assert is_coco_json_file(json_path) is True
    assert is_coco_dataset(root) is True

    splits = discover_coco_splits(root)
    assert "train" in splits
    discovered_json, discovered_imgs = splits["train"]
    assert discovered_json == json_path
    assert discovered_imgs.is_dir()

    profile = inspect_coco_dataset(root)
    assert profile["valid"] is True
    assert profile["format"] == "coco_detection"
    assert profile["dataset_task"] == "object_detection"
    assert profile["classes"] == ["aceituna_defectuosa", "aceituna_sana"]
    assert profile["total_boxes"] == 3
    assert profile["total_images"] == 2
    assert profile["box_counts"]["aceituna_sana"] == 2
    assert profile["box_counts"]["aceituna_defectuosa"] == 1
    assert profile["image_counts"]["aceituna_sana"] == 2
    assert profile["image_counts"]["aceituna_defectuosa"] == 1
    assert profile["health_profile"]["health_score"] == 100


def test_coco_to_yolo_conversion(tmp_path: Path) -> None:
    """Verify conversion of COCO JSON to Ultralytics YOLO dataset format."""
    root, json_path = _create_synthetic_coco_dataset(tmp_path / "coco_yolo")
    imgs_dir = json_path.parent / "images"
    out_yolo = tmp_path / "yolo_converted"

    yaml_p = coco_to_yolo(json_path, imgs_dir, out_yolo, split_name="train")
    assert yaml_p.is_file()
    assert (out_yolo / "images" / "train" / "leaf_001.jpg").is_file()
    assert (out_yolo / "labels" / "train" / "leaf_001.txt").is_file()

    # Verify label line content
    lines = (out_yolo / "labels" / "train" / "leaf_001.txt").read_text().splitlines()
    assert len(lines) == 2
    # Check normalized coordinate ranges (0 <= x <= 1)
    for line in lines:
        parts = line.split()
        assert len(parts) == 5
        for coord in parts[1:]:
            val = float(coord)
            assert 0.0 <= val <= 1.0


def test_coco_to_classification_crops(tmp_path: Path) -> None:
    """Verify extracting bounding box crops into PyTorch ImageFolder structure."""
    root, json_path = _create_synthetic_coco_dataset(tmp_path / "coco_crops")
    imgs_dir = json_path.parent / "images"
    out_crops = tmp_path / "crops_out"

    res = coco_to_classification_crops(
        coco_json_path=json_path,
        images_dir=imgs_dir,
        output_dir=out_crops,
        split_name="train",
        min_box_size=10,
        padding_pct=0.1,
    )

    assert res["aceituna_sana"] == 2
    assert res["aceituna_defectuosa"] == 1

    sana_dir = out_crops / "train" / "aceituna_sana"
    defect_dir = out_crops / "train" / "aceituna_defectuosa"
    assert sana_dir.is_dir()
    assert defect_dir.is_dir()
    assert len(list(sana_dir.glob("*.jpg"))) == 2
    assert len(list(defect_dir.glob("*.jpg"))) == 1


def test_api_inspect_coco_and_convert_crops_endpoint(tmp_path: Path) -> None:
    """Test POST /api/master/datasets/inspect and /convert-crops endpoints via FastAPI."""
    root, _ = _create_synthetic_coco_dataset(tmp_path / "api_coco")

    from agrifed_khaos.models import build_model, save_checkpoint_bundle

    model_path = tmp_path / "model.pt"
    m = build_model("resnet18", num_classes=2, pretrained=False)
    save_checkpoint_bundle(
        model_path,
        m.state_dict(),
        {"architecture": "resnet18", "num_classes": 2, "classes": ["c1", "c2"]},
    )
    meta_path = tmp_path / "meta.json"
    meta_path.write_text('{"classes": ["c1", "c2"]}')

    app = create_app(str(model_path), str(meta_path))
    client = TestClient(app)

    # 1. Inspect COCO dataset
    r_insp = client.post("/api/master/datasets/inspect", json={"dataset_path": str(root)})
    assert r_insp.status_code == 200
    data_insp = r_insp.json()
    assert data_insp["valid"] is True
    assert data_insp["format"] == "coco_detection"
    assert data_insp["total_boxes"] == 3
    assert "detection_profile" in data_insp
    assert "box_sizes_coco" in data_insp["detection_profile"]
    assert "density" in data_insp["detection_profile"]
    assert "aceituna_sana" in data_insp["classes"]

    # 2. Convert to crops endpoint
    r_conv = client.post(
        "/api/master/datasets/convert-crops",
        json={
            "dataset_path": str(root),
            "min_box_size": 10,
            "padding_pct": 0.05,
        },
    )
    assert r_conv.status_code == 200
    data_conv = r_conv.json()
    assert data_conv["valid"] is True
    # The returned inspection on the new cropped dataset should now be standard classification!
    assert data_conv["classes"] == ["aceituna_defectuosa", "aceituna_sana"]
    assert data_conv["total_images"] == 3
    assert "converted_crops" in data_conv
    assert data_conv["converted_crops"]["aceituna_sana"] == 2


def test_auto_fl_with_coco_detection_auto_crops(tmp_path: Path) -> None:
    """Test Auto-FL seamlessly auto-extracting crops on the fly when given a COCO dataset."""
    root, _ = _create_synthetic_coco_dataset(tmp_path / "autofl_coco")

    from agrifed_khaos.models import build_model, save_checkpoint_bundle

    model_path = tmp_path / "model.pt"
    m = build_model("resnet18", num_classes=2, pretrained=False)
    save_checkpoint_bundle(
        model_path,
        m.state_dict(),
        {"architecture": "resnet18", "num_classes": 2, "classes": ["c1", "c2"]},
    )
    meta_path = tmp_path / "meta.json"
    meta_path.write_text('{"classes": ["c1", "c2"]}')

    app = create_app(str(model_path), str(meta_path))
    client = TestClient(app)

    # 1. Directly call Auto-FL with the raw COCO root directory
    r_autofl = client.post(
        "/api/master/models/auto-fl",
        json={
            "title": "Modelo Aceituna COCO Auto-FL",
            "slug": "modelo-aceituna-coco",
            "dataset_path": str(root),
            "epochs": 1,
            "architecture": "resnet18",
        },
    )
    assert r_autofl.status_code == 202
    data = r_autofl.json()
    assert data["status"] == "accepted"
    # Classes should NOT be ['images'], but the true COCO categories
    assert "aceituna_sana" in data["classes_detected"]
    assert "aceituna_defectuosa" in data["classes_detected"]
    assert len(data["classes_detected"]) == 2

    # 2. Test rejection when a directory has only 1 dummy class and is not COCO
    dummy_single = tmp_path / "single_class_ds"
    img_dir = dummy_single / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    im = Image.new("RGB", (32, 32), color="blue")
    im.save(img_dir / "sample.jpg")

    r_reject = client.post(
        "/api/master/models/auto-fl",
        json={
            "title": "Modelo Dummy",
            "dataset_path": str(dummy_single),
        },
    )
    assert r_reject.status_code == 400
    assert "solo contiene 1 clase" in r_reject.json()["detail"]


def test_coco_with_roboflow_phantom_zero_annotation_category(tmp_path: Path) -> None:
    """Verify that datasets exported by Roboflow with a dummy 0-annotation category 0 are cleansed."""
    root = tmp_path / "roboflow_coco"
    split_dir = root / "train"
    split_dir.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (100, 100), color=(120, 150, 100))
    img.save(split_dir / "test_olive.jpg")

    coco_data = {
        "categories": [
            {"id": 0, "name": "diseases-usdB", "supercategory": "none"},
            {"id": 1, "name": "Anthracnose", "supercategory": "diseases-usdB"},
            {"id": 2, "name": "Tuberculosis", "supercategory": "diseases-usdB"},
        ],
        "images": [{"id": 1, "file_name": "test_olive.jpg", "width": 100, "height": 100}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 40, 40], "area": 1600, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [50, 50, 30, 30], "area": 900, "iscrowd": 0},
        ],
    }
    json_path = split_dir / "_annotations.coco.json"
    json_path.write_text(json.dumps(coco_data), encoding="utf-8")

    # 1. Inspect dataset
    profile = inspect_coco_dataset(root)
    assert profile["valid"] is True
    assert "diseases-usdB" not in profile["classes"]
    assert profile["classes"] == ["Anthracnose", "Tuberculosis"]
    assert "diseases-usdB" not in profile["box_counts"]

    # 2. Convert crops
    out_crops = tmp_path / "crops_out"
    crops_res = coco_to_classification_crops(
        coco_json_path=json_path,
        images_dir=split_dir,
        output_dir=out_crops,
        split_name="train",
        min_box_size=10,
    )
    assert "diseases-usdB" not in crops_res
    assert (out_crops / "train" / "Anthracnose").is_dir()
    assert (out_crops / "train" / "Tuberculosis").is_dir()
    assert not (out_crops / "train" / "diseases-usdB").exists()

    # 3. Prepare detection
    from agrifed_khaos.coco import prepare_detection_dataset
    yaml_p, class_names = prepare_detection_dataset(root, output_yolo_dir=tmp_path / "yolo_out")
    assert "diseases-usdB" not in class_names
    assert class_names == ["Anthracnose", "Tuberculosis"]


def test_compute_detection_geometry_profile_and_preview_endpoint(tmp_path: Path) -> None:
    """Test geometric profiling (COCO sizes, aspect ratio, density, IoU overlap) and preview endpoint."""
    from agrifed_khaos.coco import compute_detection_geometry_profile
    from starlette.testclient import TestClient
    from scripts.serve import create_app
    from PIL import Image

    # Create dummy images and boxes:
    # Box 1: Small (< 32x32 = 1024) -> 20x20 = 400
    # Box 2: Medium (32..96) -> 50x50 = 2500
    # Box 3: Large (>= 96) -> 120x100 = 12000
    # Box 4: High overlap with Box 3 (IoU > 0.50) -> 110x95
    img1_file = tmp_path / "img1.jpg"
    img2_file = tmp_path / "img2.jpg"
    Image.new("RGB", (640, 640), color="green").save(img1_file)
    Image.new("RGB", (640, 640), color="blue").save(img2_file)

    images_data = [
        {
            "image_path": str(img1_file),
            "width": 640,
            "height": 640,
            "boxes": [
                {"class_name": "leaf", "box_xyxy": [10, 10, 30, 30], "class_id": 0},          # 20x20 = 400 (Small)
                {"class_name": "spot", "box_xyxy": [50, 50, 100, 100], "class_id": 1},        # 50x50 = 2500 (Medium)
                {"class_name": "disease", "box_xyxy": [200, 200, 320, 300], "class_id": 2},    # 120x100 = 12000 (Large)
                {"class_name": "disease", "box_xyxy": [205, 205, 320, 300], "class_id": 2},    # Overlaps Box 3 heavily (IoU > 0.8)
            ],
        },
        {
            "image_path": str(img2_file),
            "width": 640,
            "height": 640,
            "boxes": [], # Unannotated / background image
        },
    ]

    profile = compute_detection_geometry_profile(images_data, ["leaf", "spot", "disease"])

    assert profile["total_images"] == 2
    assert profile["total_boxes"] == 4
    # Check COCO sizes
    assert profile["box_sizes_coco"]["small"]["count"] == 1
    assert profile["box_sizes_coco"]["medium"]["count"] == 1
    assert profile["box_sizes_coco"]["large"]["count"] == 2
    # Check density
    assert profile["density"]["max"] == 4
    assert profile["density"]["mean"] == 2.0
    # Check unannotated
    assert profile["unannotated_images"]["count"] == 1
    assert profile["unannotated_images"]["pct"] == 50.0
    # Check intra-image IoU overlap (> 0.50)
    assert profile["intra_image_overlap"]["overlapping_pairs_gt_050"] >= 1
    assert profile["intra_image_overlap"]["images_with_high_overlap_pct"] == 50.0
    # Check sample preview
    assert profile["sample_preview"] is not None
    assert profile["sample_preview"]["image_path"] == str(img1_file)
    assert len(profile["sample_preview"]["boxes"]) == 4

    # Test preview-image endpoint
    from agrifed_khaos.models import build_model, save_checkpoint_bundle
    model_path = tmp_path / "model.pt"
    m = build_model("resnet18", num_classes=2, pretrained=False)
    save_checkpoint_bundle(
        model_path,
        m.state_dict(),
        {"architecture": "resnet18", "num_classes": 2, "classes": ["c1", "c2"]},
    )
    meta_path = tmp_path / "meta.json"
    meta_path.write_text('{"classes": ["c1", "c2"]}')

    app = create_app(str(model_path), str(meta_path))
    client = TestClient(app)

    # 1. Valid project image preview
    res_valid = client.get("/api/master/datasets/preview-image", params={"image_path": "src/agrifed_khaos/static/logo.png"})
    assert res_valid.status_code == 200

    # 2. Path traversal security check outside repo
    res_traversal = client.get("/api/master/datasets/preview-image", params={"image_path": "/etc/passwd"})
    assert res_traversal.status_code == 403

    # 3. Missing file
    res_missing = client.get("/api/master/datasets/preview-image", params={"image_path": "data/non_existent_image_12345.jpg"})
    assert res_missing.status_code == 404

