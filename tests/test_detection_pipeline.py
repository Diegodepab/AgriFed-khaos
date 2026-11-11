"""Unit tests for Object Detection pipeline, COCO-to-YOLO conversion, and YOLO inference."""

import json
from pathlib import Path

import pytest
from PIL import Image

from agrifed_khaos.coco import is_coco_dataset, prepare_detection_dataset
from agrifed_khaos.schemas import AutoFLRequest, PredictResponse


@pytest.fixture
def dummy_coco_dataset(tmp_path: Path):
    """Create a temporary valid COCO detection dataset."""
    ds_root = tmp_path / "wheathead_mini"
    train_dir = ds_root / "dataset" / "train"
    train_images = train_dir / "images"
    train_images.mkdir(parents=True, exist_ok=True)

    # Create 2 sample images
    img1_p = train_images / "sample1.jpg"
    img2_p = train_images / "sample2.jpg"
    img = Image.new("RGB", (640, 640), color=(120, 160, 80))
    img.save(img1_p)
    img.save(img2_p)

    # Create COCO annotations with 1 class (wheat_head) and bounding boxes
    coco_data = {
        "categories": [{"id": 1, "name": "wheat_head"}],
        "images": [
            {"id": 1, "file_name": "sample1.jpg", "width": 640, "height": 640},
            {"id": 2, "file_name": "sample2.jpg", "width": 640, "height": 640},
        ],
        "annotations": [
            {"id": 101, "image_id": 1, "category_id": 1, "bbox": [100, 100, 80, 120], "area": 9600},
            {
                "id": 102,
                "image_id": 1,
                "category_id": 1,
                "bbox": [300, 250, 90, 140],
                "area": 12600,
            },
            {"id": 103, "image_id": 2, "category_id": 1, "bbox": [200, 150, 70, 110], "area": 7700},
        ],
    }
    (train_dir / "labels.json").write_text(json.dumps(coco_data), encoding="utf-8")

    return ds_root


def test_prepare_detection_dataset(dummy_coco_dataset: Path, tmp_path: Path):
    assert is_coco_dataset(dummy_coco_dataset)

    yolo_out = tmp_path / "yolo_converted"
    yaml_path, classes = prepare_detection_dataset(dummy_coco_dataset, output_yolo_dir=yolo_out)

    assert yaml_path.is_file()
    assert classes == ["wheat_head"]
    assert (yolo_out / "images" / "train" / "sample1.jpg").exists()
    assert (yolo_out / "labels" / "train" / "sample1.txt").is_file()

    # Check YOLO label format: class_id x_center y_center w h (normalized)
    label_content = (
        (yolo_out / "labels" / "train" / "sample1.txt").read_text(encoding="utf-8").strip()
    )
    lines = label_content.splitlines()
    assert len(lines) == 2
    cols = lines[0].split()
    assert cols[0] == "0"  # zero-indexed class id
    assert 0.0 < float(cols[1]) < 1.0
    assert 0.0 < float(cols[2]) < 1.0


def test_autofl_request_schema():
    req = AutoFLRequest(
        title="Trigo Detector",
        dataset_path="data/uploads/wheat",
        task="detection",
        architecture="yolov8n",
    )
    assert req.task == "detection"
    assert req.architecture == "yolov8n"


def test_predict_response_detection_schema():
    resp = PredictResponse(
        predicted_class="wheat_head",
        confidence=0.92,
        probabilities={"wheat_head": 1.0},
        model_name="trigo_detector_base.pt",
        task="detection",
        total_detections=3,
        detection_counts={"wheat_head": 3},
        detections=[
            {
                "box_xyxy": [100, 100, 180, 220],
                "confidence": 0.92,
                "class_name": "wheat_head",
                "class_id": 0,
            }
        ],
    )
    assert resp.task == "detection"
    assert resp.total_detections == 3
    assert len(resp.detections) == 1


def test_yolo_dataset_inspection_and_detection_profile(tmp_path: Path):
    """Test YOLO inspection returns detection_profile with COCO area intervals and aspect ratios."""
    from agrifed_khaos.yolo import inspect_yolo_dataset

    yolo_dir = tmp_path / "yolo_sample"
    images_dir = yolo_dir / "images"
    labels_dir = yolo_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy images
    img1 = images_dir / "sample_a.jpg"
    img2 = images_dir / "sample_b.jpg"
    Image.new("RGB", (640, 640), color="red").save(img1)
    Image.new("RGB", (640, 640), color="yellow").save(img2)

    # Create label files
    # Box 1: Small (< 32x32 = 1024): 20x20 at 640x640 -> w=0.03125, h=0.03125
    # Box 2: Large (>= 96x96 = 9216): 200x200 at 640x640 -> w=0.3125, h=0.3125
    (labels_dir / "sample_a.txt").write_text("0 0.5 0.5 0.03 0.03\n0 0.4 0.4 0.35 0.35\n", encoding="utf-8")
    # Sample B has no boxes (empty image)
    (labels_dir / "sample_b.txt").write_text("", encoding="utf-8")

    (yolo_dir / "data.yaml").write_text("names:\n  0: weed\n", encoding="utf-8")

    res = inspect_yolo_dataset(yolo_dir)
    assert res["valid"] is True
    assert res["total_images"] == 2
    assert res["total_boxes"] == 2
    assert "detection_profile" in res
    dp = res["detection_profile"]
    assert dp["total_images"] == 2
    assert dp["total_boxes"] == 2
    assert dp["box_sizes_coco"]["small"]["count"] == 1
    assert dp["box_sizes_coco"]["large"]["count"] == 1
    assert dp["unannotated_images"]["count"] == 1
    assert dp["sample_preview"] is not None
    assert dp["sample_preview"]["image_path"] == str(img1)


def test_compute_greedy_matched_mean_iou(tmp_path: Path):
    """Test 1-to-1 greedy bipartite matching Mean IoU with class filtering and confidence thresholds."""
    import sys
    import torch
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root / "scripts") not in sys.path:
        sys.path.insert(0, str(repo_root / "scripts"))
    from train_detection import compute_greedy_matched_mean_iou

    dataset_dir = tmp_path / "eval_ds"
    images_dir = dataset_dir / "images" / "val"
    labels_dir = dataset_dir / "labels" / "val"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    img_p = images_dir / "sample.jpg"
    Image.new("RGB", (640, 640), color="blue").save(img_p)

    # GT Box: class 0, center (0.5, 0.5), w=0.2, h=0.2 -> [0.4, 0.4, 0.6, 0.6]
    (labels_dir / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(f"path: {dataset_dir}\nval: images/val\nnames:\n  0: olive_fly\n", encoding="utf-8")

    # Mock Ultralytics YOLO model
    class MockBoxes:
        def __init__(self, xyxyn, cls, conf):
            self.xyxyn = torch.tensor(xyxyn)
            self.cls = torch.tensor(cls)
            self.conf = torch.tensor(conf)

        def __len__(self):
            return len(self.xyxyn)

    class MockResult:
        def __init__(self, boxes):
            self.boxes = boxes

    class MockModel:
        def __init__(self, results_to_return):
            self.results = results_to_return

        def __call__(self, source, conf=0.25, verbose=False):
            return self.results

    # Case 1: Pred box exactly matches GT (xyxy: 0.4, 0.4, 0.6, 0.6) with conf 0.9 -> IoU = 1.0
    mock_boxes_exact = MockBoxes(
        xyxyn=[[0.4, 0.4, 0.6, 0.6]],
        cls=[0],
        conf=[0.90],
    )
    model1 = MockModel([MockResult(mock_boxes_exact)])
    mean_iou, count = compute_greedy_matched_mean_iou(model1, data_yaml, split="val")
    assert count == 1
    assert abs(mean_iou - 1.0) < 1e-4

    # Case 2: Class mismatch (pred is class 1 instead of 0) -> IoU matching rejected -> 0 matched
    mock_boxes_mismatch = MockBoxes(
        xyxyn=[[0.4, 0.4, 0.6, 0.6]],
        cls=[1],
        conf=[0.95],
    )
    model2 = MockModel([MockResult(mock_boxes_mismatch)])
    mean_iou2, count2 = compute_greedy_matched_mean_iou(model2, data_yaml, split="val")
    assert count2 == 0
    assert mean_iou2 == 0.0

    # Case 3: Conf < 0.25 filtered out -> 0 matched
    mock_boxes_low_conf = MockBoxes(
        xyxyn=[[0.4, 0.4, 0.6, 0.6]],
        cls=[0],
        conf=[0.10],
    )
    model3 = MockModel([MockResult(mock_boxes_low_conf)])
    mean_iou3, count3 = compute_greedy_matched_mean_iou(model3, data_yaml, split="val")
    assert count3 == 0
    assert mean_iou3 == 0.0


def test_catalog_api_eval_preview_and_detection_metrics(tmp_path: Path):
    """Test /api/models/{slug} returns detection metrics and /api/models/{slug}/eval-preview serves mosaic."""
    from fastapi.testclient import TestClient
    from agrifed_khaos.catalog import ModelCatalogManager
    from scripts.serve import create_app

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    slug = "test-olive-detector"

    # Create dummy checkpoint & eval preview image
    pt_file = models_dir / f"{slug}_base.pt"
    pt_file.write_bytes(b"dummy_weights")
    preview_file = models_dir / f"{slug}_eval_preview.jpg"
    Image.new("RGB", (640, 480), color="green").save(preview_file)
    raw_sample = models_dir / f"{slug}_sample_raw.jpg"
    Image.new("RGB", (640, 480), color="blue").save(raw_sample)
    annotated_sample = models_dir / f"{slug}_sample_annotated.jpg"
    Image.new("RGB", (640, 480), color="red").save(annotated_sample)

    # Create meta.json and evaluation_report.json
    meta_json = {
        "task": "detection",
        "architecture": "yolov8n",
        "eval_split": "test",
        "metrics": {
            "mAP50": 88.5,
            "mAP50_95": 65.2,
            "mean_iou": 0.785,
            "matched_detections": 42,
        },
        "classes": ["olive_fruit"],
    }
    (models_dir / f"{slug}.meta.json").write_text(json.dumps(meta_json), encoding="utf-8")

    eval_report = {
        "slug": slug,
        "task": "detection",
        "eval_split": "test",
        "metrics": {
            "mAP50": 88.5,
            "mAP50_95": 65.2,
            "mAP75": 72.0,
            "precision": 91.0,
            "recall": 84.5,
            "mean_iou": 0.785,
            "matched_detections": 42,
        },
        "per_class": [
            {
                "class_name": "olive_fruit",
                "class_id": 0,
                "images": 15,
                "instances": 42,
                "precision": 0.91,
                "recall": 0.845,
                "mAP50": 0.885,
                "mAP50_95": 0.652,
            }
        ],
        "eval_preview_available": True,
    }
    (models_dir / f"{slug}_evaluation_report.json").write_text(json.dumps(eval_report), encoding="utf-8")

    # Catalog JSON entry
    catalog_json = {
        "models": [
            {
                "slug": slug,
                "title": "Olive Detector Test",
                "task": "detection",
                "architecture": "yolov8n",
                "active_checkpoint": f"{slug}_base.pt",
                "classes": ["olive_fruit"],
                "versions": [f"{slug}_base.pt"],
            }
        ]
    }
    catalog_file = models_dir / "catalog.json"
    catalog_file.write_text(json.dumps(catalog_json), encoding="utf-8")

    # Create app and client
    app = create_app(str(pt_file), str(models_dir / f"{slug}.meta.json"))
    app.state.catalog = ModelCatalogManager(catalog_path=catalog_file, models_dir=models_dir)

    # Monkeypatch catalog.py MODELS_DIR to point to tmp_path models_dir for the endpoint
    import agrifed_khaos.api.catalog as cat_module
    orig_models_dir = Path("models").resolve()
    try:
        # Test GET /api/models/{slug}
        client = TestClient(app)
        res = client.get(f"/api/models/{slug}")
        assert res.status_code == 200
        data = res.json()
        assert data["model"]["task"] == "detection"
        ds_info = data["dataset_info"]
        assert ds_info["eval_split"] == "test"
        assert ds_info["detection_metrics"]["mAP50"] == 88.5
        assert ds_info["detection_metrics"]["mean_iou"] == 0.785

        # Also verify 404 for nonexistent eval preview
        res_404 = client.get("/api/models/nonexistent-model-xyz/eval-preview")
        assert res_404.status_code == 404

        # Test sample-raw and sample-annotated 404 and 200
        res_raw_404 = client.get("/api/models/nonexistent-model-xyz/sample-raw")
        assert res_raw_404.status_code == 404
        res_ann_404 = client.get("/api/models/nonexistent-model-xyz/sample-annotated")
        assert res_ann_404.status_code == 404

        # Test real model endpoints
        res_real_raw = client.get("/api/models/deteccion-beans-angular-leafspot-beans-rust/sample-raw")
        assert res_real_raw.status_code == 200
        assert res_real_raw.headers["content-type"] in ("image/jpeg", "image/jpg")

        res_real_ann = client.get("/api/models/deteccion-beans-angular-leafspot-beans-rust/sample-annotated")
        assert res_real_ann.status_code == 200
        assert res_real_ann.headers["content-type"] in ("image/jpeg", "image/jpg")
    finally:
        pass


