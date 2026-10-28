"""Integration tests for Master MLOps Studio endpoints:
- POST/GET /api/master/recommend-arch
- GET /api/master/models/{slug}/telemetry
- GET /api/master/federated/updates
- POST /api/master/models/{slug}/gradcam
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from starlette.testclient import TestClient

from agrifed_khaos.crypto import compute_sha256
from agrifed_khaos.models import build_model, save_checkpoint_bundle
from scripts.serve import create_app


@pytest.fixture
def master_test_app(tmp_path: Path) -> tuple[TestClient, Path]:
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy model checkpoint
    model = build_model("resnet18", num_classes=3, pretrained=False)
    model_path = models_dir / "global_model.pt"
    bundle_meta = {
        "architecture": "resnet18",
        "num_classes": 3,
        "classes": ["sana", "roña", "mildiu"],
        "img_size": [64, 64],
        "temperature": 1.25,
        "calibrated_thresholds": {"sana": 0.45, "roña": 0.55, "mildiu": 0.50},
    }
    save_checkpoint_bundle(model_path, model.state_dict(), bundle_meta)

    # Create candidate model checkpoint for A/B testing
    model_b = build_model("resnet18", num_classes=3, pretrained=False)
    model_b_path = models_dir / "candidate_model.pt"
    save_checkpoint_bundle(model_b_path, model_b.state_dict(), bundle_meta)

    # Create catalog.json
    catalog_path = models_dir / "catalog.json"
    catalog_data = {
        "models": [
            {
                "slug": "test-olive-model",
                "title": "Test Olive Pathology Model",
                "domain": "Agricultura",
                "crop": "Olivo",
                "target_element": "Hoja",
                "purpose": "Detección de patologías",
                "description": "Modelo de prueba para telemetría",
                "input_guide": "Foto clara de la hoja",
                "active_checkpoint": "global_model.pt",
                "versions": ["global_model.pt", "candidate_model.pt"],
                "version_history": [
                    {
                        "version_id": "v1",
                        "checkpoint": "global_model.pt",
                        "created_at": "2026-10-16 11:13:37",
                        "parent_version": None,
                        "origin": "train_base",
                        "evaluation_report": "test-olive-model_evaluation_report.json",
                        "status": "active",
                        "metrics_summary": {"overall_accuracy": 89.33},
                    },
                    {
                        "version_id": "v2",
                        "checkpoint": "candidate_model.pt",
                        "created_at": "2026-10-26 10:00:00",
                        "parent_version": "v1",
                        "origin": "federated_aggregation",
                        "evaluation_report": None,
                        "status": "candidate",
                        "metrics_summary": {},
                    },
                ],
                "architecture": "resnet18",
                "num_classes": 3,
                "classes": ["sana", "roña", "mildiu"],
                "thumbnail_url": None,
                "example_images": [],
                "calibrated_thresholds": {"sana": 0.45, "roña": 0.55, "mildiu": 0.50},
                "temperature": 1.25,
                "updated_at": "2026-10-26 12:00:00",
                "is_deleted": False,
                "deleted_at": None,
                "purge_after": None,
            }
        ]
    }
    import json
    catalog_path.write_text(json.dumps(catalog_data, indent=2), encoding="utf-8")

    # Create an evaluation report JSON
    eval_rep_path = models_dir / "test-olive-model_evaluation_report.json"
    eval_rep_data = {
        "dataset_evaluated": "/data/test",
        "total_samples": 300,
        "overall_accuracy": 89.33,
        "cross_entropy_loss": 0.281,
        "macro_avg": {"precision": 0.895, "recall": 0.893, "f1_score": 0.894},
        "per_class": {
            "sana": {"precision": 0.92, "recall": 0.90, "f1_score": 0.91, "support": 100},
            "roña": {"precision": 0.88, "recall": 0.89, "f1_score": 0.885, "support": 100},
            "mildiu": {"precision": 0.885, "recall": 0.89, "f1_score": 0.887, "support": 100},
        },
        "confusion_matrix_raw": [
            [90, 6, 4],
            [5, 89, 6],
            [3, 8, 89],
        ],
    }
    eval_rep_path.write_text(json.dumps(eval_rep_data, indent=2), encoding="utf-8")

    # Create client_updates directory with a sample update
    updates_dir = models_dir / "client_updates"
    updates_dir.mkdir(parents=True, exist_ok=True)
    sample_update = updates_dir / "client_farm_1_parameters.pt"
    save_checkpoint_bundle(
        sample_update,
        {"fc.weight": torch.randn(3, 512)},
        {"classes": ["sana", "roña", "mildiu"]},
    )
    meta_update = updates_dir / "client_farm_1_parameters.meta.json"
    meta_update.write_text(
        json.dumps(
            {
                "client_id": "client_farm_1",
                "sha256": "abcdef1234567890",
                "base_model_sha256": "1234567890abcdef",
                "training_scope": "head_only",
                "num_samples": 45,
                "hmac_signature": "mock_hmac_sig",
            }
        ),
        encoding="utf-8",
    )

    app = create_app(model_path=str(model_path), metadata_path=str(models_dir / "metadata.json"))
    client = TestClient(app)
    return client, tmp_path


def test_recommend_architecture_get_and_post(master_test_app: tuple[TestClient, Path]) -> None:
    client, _ = master_test_app

    # Test GET
    res_get = client.get(
        "/api/master/recommend-arch?task=classification&target_client_tier=cpu-low&dataset_sample_count=800"
    )
    assert res_get.status_code == 200
    data_get = res_get.json()
    assert "profile" in data_get
    assert "recommendations" in data_get
    assert data_get["total_candidates"] >= 2
    # Verify recommended models structure
    first_cand = data_get["recommendations"][0]
    assert "architecture" in first_cand
    assert "federated_suitability_score" in first_cand
    assert "rationale" in first_cand

    # Test POST with JSON body
    post_payload = {
        "task": "classification",
        "target_client_tier": "gpu",
        "dataset_sample_count": 5000,
        "num_classes": 10,
        "network_speed_mbit": 100.0,
    }
    res_post = client.post("/api/master/recommend-arch", json=post_payload)
    assert res_post.status_code == 200
    data_post = res_post.json()
    assert data_post["profile"]["device_type"] in {"cpu", "cuda", "mps"}
    assert len(data_post["recommendations"]) == data_get["total_candidates"]


def test_get_model_telemetry(master_test_app: tuple[TestClient, Path]) -> None:
    client, _ = master_test_app

    # 1. Telemetry for existing model
    res = client.get("/api/master/models/test-olive-model/telemetry")
    assert res.status_code == 200
    data = res.json()
    assert data["slug"] == "test-olive-model"
    assert data["architecture"] == "resnet18"
    assert data["num_classes"] == 3
    assert data["classes"] == ["sana", "roña", "mildiu"]
    assert data["temperature"] == 1.25
    assert data["calibrated_thresholds"]["sana"] == 0.45

    # Check metrics and confusion matrix
    assert data["has_eval_report"] is True
    assert data["metrics"]["overall_accuracy"] == 89.33
    assert data["metrics"]["cross_entropy_loss"] == 0.281
    assert data["confusion_matrix"]["matrix"] == [
        [90, 6, 4],
        [5, 89, 6],
        [3, 8, 89],
    ]

    # 2. Telemetry for non-existent model -> 404
    res_404 = client.get("/api/master/models/non-existent-slug/telemetry")
    assert res_404.status_code == 404


def test_get_federated_updates(master_test_app: tuple[TestClient, Path]) -> None:
    client, _ = master_test_app

    res = client.get("/api/master/federated/updates")
    assert res.status_code == 200
    data = res.json()
    assert "updates" in data
    assert "total" in data
    assert isinstance(data["updates"], list)


def test_gradcam_realtime_overlay(master_test_app: tuple[TestClient, Path]) -> None:
    client, _ = master_test_app

    # Create synthetic test image
    img = Image.new("RGB", (64, 64), color=(120, 180, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    # Post image to gradcam endpoint
    res = client.post(
        "/api/master/models/test-olive-model/gradcam",
        files={"file": ("leaf.jpg", buf.getvalue(), "image/jpeg")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["slug"] == "test-olive-model"
    assert data["architecture"] == "resnet18"
    assert 0 <= data["predicted_class_index"] < 3
    assert data["predicted_class_name"] in ["sana", "roña", "mildiu"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["cam_overlay_base64"].startswith("data:image/jpeg;base64,")


def test_master_ui_contains_inspector_and_fl_inbox(
    master_test_app: tuple[TestClient, Path],
) -> None:
    client, _ = master_test_app
    res = client.get("/master")
    assert res.status_code == 200
    html = res.text
    # Check inspector modal elements
    assert "modelInspectorModal" in html
    assert "tabBtnGradcam" in html
    assert "metricAccuracy" in html
    assert "inspectorConfusionMatrixContainer" in html
    # Check Auto-FL accordion
    assert "autoFlPanel" in html
    assert "computeArchRecommendation" in html
    # Check Federated Updates inbox
    assert "flUpdatesTbody" in html
    assert "flUpdatesCountBadge" in html


def test_compare_ab_success_and_ground_truth(
    master_test_app: tuple[TestClient, Path],
) -> None:
    client, _ = master_test_app

    # Create synthetic test image
    img = Image.new("RGB", (64, 64), color=(80, 160, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")

    # Post to compare-ab with v1 and v2, with ground truth matching
    res = client.post(
        "/api/master/models/test-olive-model/compare-ab"
        "?version_a=v1&version_b=v2&ground_truth_class=sana&include_gradcam=true",
        files={"file": ("leaf.jpg", buf.getvalue(), "image/jpeg")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["slug"] == "test-olive-model"
    assert "version_a" in data
    assert "version_b" in data
    assert "agreement" in data
    assert "summary" in data

    va = data["version_a"]
    assert va["version_id"] == "v1"
    assert va["checkpoint"] == "global_model.pt"
    assert va["predicted_class_name"] in ["sana", "roña", "mildiu"]
    assert 0.0 <= va["confidence"] <= 1.0
    assert va["cam_overlay_base64"] is not None
    assert va["cam_overlay_base64"].startswith("data:image/jpeg;base64,")
    assert va["aggregate_metrics"] is not None
    assert va["aggregate_metrics"]["overall_accuracy"] == 89.33

    vb = data["version_b"]
    assert vb["version_id"] == "v2"
    assert vb["checkpoint"] == "candidate_model.pt"
    assert vb["cam_overlay_base64"] is not None
    # v2 has no eval report
    assert vb["aggregate_metrics"] is None

    # Check ground truth match boolean
    assert isinstance(va["matches_ground_truth"], bool)
    assert isinstance(vb["matches_ground_truth"], bool)


def test_compare_ab_missing_checkpoint_fails_gracefully(
    master_test_app: tuple[TestClient, Path],
) -> None:
    client, _ = master_test_app

    img = Image.new("RGB", (64, 64), color=(80, 160, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")

    # Request comparison with non-existent checkpoint
    res = client.post(
        "/api/master/models/test-olive-model/compare-ab"
        "?version_a=v1&version_b=non_existent_weights.pt",
        files={"file": ("leaf.jpg", buf.getvalue(), "image/jpeg")},
    )
    assert res.status_code == 404
    assert "no encontrado" in res.json()["detail"]


def test_version_traceability_in_telemetry(
    master_test_app: tuple[TestClient, Path],
) -> None:
    client, _ = master_test_app

    res = client.get("/api/master/models/test-olive-model/telemetry")
    assert res.status_code == 200
    data = res.json()
    assert "version_history" in data
    assert len(data["version_history"]) == 2
    v1 = data["version_history"][0]
    assert v1["version_id"] == "v1"
    assert v1["checkpoint"] == "global_model.pt"
    assert v1["status"] == "active"
    assert data["evaluated_checkpoint"] == "global_model.pt"


def test_dataset_upload_data_quality_profiler(
    master_test_app: tuple[TestClient, Path],
) -> None:
    """Test Data Quality Profiler detects corruption, duplicates, and resolution variance."""
    import zipfile
    client, _ = master_test_app

    # Create dummy images
    img_clean_1 = Image.new("RGB", (224, 224), color="blue")
    buf1 = io.BytesIO()
    img_clean_1.save(buf1, format="JPEG")

    img_clean_2 = Image.new("RGB", (224, 224), color="green")
    buf2 = io.BytesIO()
    img_clean_2.save(buf2, format="JPEG")

    # High-resolution image (variance test)
    img_4k = Image.new("RGB", (3840, 2160), color="red")
    buf_4k = io.BytesIO()
    img_4k.save(buf_4k, format="JPEG")

    # Corrupted file
    corrupt_bytes = b"corrupted header binary garbage data not an image"

    # Build zip archive with subfolders
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("dataset/sano/img1.jpg", buf1.getvalue())
        zf.writestr("dataset/sano/img1_dup.jpg", buf1.getvalue())  # Duplicate exact hash
        zf.writestr("dataset/sano/img_huge.jpg", buf_4k.getvalue())  # Resolution variance
        zf.writestr("dataset/enfermo/img2.jpg", buf2.getvalue())
        zf.writestr("dataset/enfermo/broken.jpg", corrupt_bytes)  # Corrupted

    zip_buf.seek(0)

    # Post to upload endpoint
    res = client.post(
        "/api/master/datasets/upload",
        files={"file": ("dataset_test.zip", zip_buf.getvalue(), "application/zip")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["valid"] is True
    assert "health_profile" in data
    hp = data["health_profile"]
    assert hp["corrupted_count"] == 1
    assert hp["duplicates_count"] == 1
    assert hp["resolution_warning"] is True
    assert hp["health_score"] < 70  # Red zone because of corruption
    assert hp["total_scanned"] == 5


def test_train_base_cli_mlops_arguments(tmp_path: Path) -> None:
    import subprocess
    import sys

    # Create dummy classification dataset
    ds_dir = tmp_path / "dataset"
    (ds_dir / "cls1").mkdir(parents=True)
    (ds_dir / "cls2").mkdir(parents=True)

    img = Image.new("RGB", (64, 64), color="blue")
    img.save(ds_dir / "cls1" / "a.jpg")
    img.save(ds_dir / "cls2" / "b.jpg")

    out_pt = tmp_path / "out.pt"
    out_meta = tmp_path / "out.json"

    cmd = [
        sys.executable,
        "scripts/train_base.py",
        "--data-dir",
        str(ds_dir),
        "--output",
        str(out_pt),
        "--metadata",
        str(out_meta),
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--lr",
        "0.001",
        "--arch",
        "resnet18",
        "--target-client-tier",
        "cpu-low",
        "--tl-strategy",
        "head_only",
        "--loss",
        "focal",
        "--aug",
        "basic",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"train_base.py failed: {res.stderr}"
    assert out_pt.exists()
    assert out_meta.exists()


def test_federated_updates_strict_model_isolation(
    master_test_app: tuple[TestClient, Path],
) -> None:
    client, tmp_path = master_test_app
    models_dir = tmp_path / "models"

    # 1. Add a second distinct model to the catalog
    catalog_path = models_dir / "catalog.json"
    catalog_data = json.loads(catalog_path.read_text(encoding="utf-8"))

    mango_model = build_model("resnet18", num_classes=2, pretrained=False)
    mango_chk = models_dir / "mango_model.pt"
    save_checkpoint_bundle(
        mango_chk, mango_model.state_dict(), {"classes": ["sano", "antracnosis"]}
    )

    catalog_data["models"].append({
        "slug": "test-mango-model",
        "title": "Test Mango Model",
        "domain": "Agricultura",
        "crop": "Mango",
        "target_element": "Fruto",
        "purpose": "Detección",
        "description": "Modelo de mango",
        "active_checkpoint": "mango_model.pt",
        "versions": ["mango_model.pt"],
        "architecture": "resnet18",
        "num_classes": 2,
        "classes": ["sano", "antracnosis"],
    })
    catalog_path.write_text(json.dumps(catalog_data, indent=2), encoding="utf-8")

    # 2. Setup updates directory with model partition folders
    updates_dir = models_dir / "client_updates"
    updates_dir.mkdir(parents=True, exist_ok=True)
    olive_up_dir = updates_dir / "test-olive-model"
    olive_up_dir.mkdir(parents=True, exist_ok=True)
    mango_up_dir = updates_dir / "test-mango-model"
    mango_up_dir.mkdir(parents=True, exist_ok=True)

    # 2.a Olive update inside test-olive-model/
    olive_pt = olive_up_dir / "coop_olive_parameters.pt"
    save_checkpoint_bundle(
        olive_pt, {"fc.weight": torch.randn(3, 512), "fc.bias": torch.randn(3)}, {}
    )
    (olive_up_dir / "coop_olive_parameters.meta.json").write_text(
        json.dumps({
            "client_id": "coop_olive",
            "sha256": compute_sha256(olive_pt),
            "base_model_sha256": compute_sha256(models_dir / "global_model.pt"),
            "training_scope": "head_only",
            "num_samples": 120,
            "target_model": "test-olive-model",
        }),
        encoding="utf-8",
    )

    # 2.b Mango update inside test-mango-model/
    mango_pt = mango_up_dir / "coop_mango_parameters.pt"
    save_checkpoint_bundle(
        mango_pt, {"fc.weight": torch.randn(2, 512), "fc.bias": torch.randn(2)}, {}
    )
    (mango_up_dir / "coop_mango_parameters.meta.json").write_text(
        json.dumps({
            "client_id": "coop_mango",
            "sha256": compute_sha256(mango_pt),
            "base_model_sha256": compute_sha256(mango_chk),
            "training_scope": "head_only",
            "num_samples": 95,
            "target_model": "test-mango-model",
        }),
        encoding="utf-8",
    )

    # 2.c Olive update loose in root with explicit prefix
    prefixed_pt = updates_dir / "test-olive-model_extra_parameters.pt"
    save_checkpoint_bundle(
        prefixed_pt, {"fc.weight": torch.randn(3, 512), "fc.bias": torch.randn(3)}, {}
    )
    (updates_dir / "test-olive-model_extra_parameters.meta.json").write_text(
        json.dumps({
            "client_id": "coop_olive_extra",
            "sha256": compute_sha256(prefixed_pt),
            "base_model_sha256": compute_sha256(models_dir / "global_model.pt"),
            "training_scope": "head_only",
            "num_samples": 50,
        }),
        encoding="utf-8",
    )

    # 3. Query updates for Olive model -> Must isolate strictly
    res_olive = client.get("/api/master/federated/updates?target_model=test-olive-model")
    assert res_olive.status_code == 200
    data_olive = res_olive.json()
    olive_files = [u["filename"] for u in data_olive["updates"]]
    
    assert "coop_olive_parameters.pt" in olive_files
    assert "test-olive-model_extra_parameters.pt" in olive_files
    # State leakage check: Mango update must NEVER appear in Olive drawer!
    assert "coop_mango_parameters.pt" not in olive_files

    # 4. Query updates for Mango model -> Must isolate strictly
    res_mango = client.get("/api/master/federated/updates?target_model=test-mango-model")
    assert res_mango.status_code == 200
    data_mango = res_mango.json()
    mango_files = [u["filename"] for u in data_mango["updates"]]

    assert "coop_mango_parameters.pt" in mango_files
    # Olive updates must NEVER appear in Mango drawer!
    assert "coop_olive_parameters.pt" not in mango_files
    assert "test-olive-model_extra_parameters.pt" not in mango_files


def test_model_telemetry_rich_metrics_suite(
    master_test_app: tuple[TestClient, Path],
) -> None:
    client, tmp_path = master_test_app
    models_dir = tmp_path / "models"

    # Inject mock evaluation_report for test-olive-model
    rep_path = models_dir / "test-olive-model_evaluation_report.json"
    rep_path.write_text(json.dumps({
        "dataset_evaluated": "/data/test",
        "total_samples": 300,
        "overall_accuracy": 94.5,
        "cross_entropy_loss": 0.185,
        "macro_avg": {
            "precision": 0.942,
            "recall": 0.938,
            "f1_score": 0.940,
        },
        "per_class": {
            "sana": {"precision": 0.96, "recall": 0.95, "f1_score": 0.955, "support": 100},
            "roña": {"precision": 0.93, "recall": 0.92, "f1_score": 0.925, "support": 100},
            "mildiu": {"precision": 0.94, "recall": 0.94, "f1_score": 0.940, "support": 100},
        },
        "confusion_matrix_raw": [[95, 3, 2], [4, 92, 4], [2, 4, 94]],
    }), encoding="utf-8")

    res = client.get("/api/master/models/test-olive-model/telemetry")
    assert res.status_code == 200
    d = res.json()

    assert d["has_eval_report"] is True
    metrics = d["metrics"]
    assert metrics["overall_accuracy"] == 94.5
    assert metrics["cross_entropy_loss"] == 0.185
    assert metrics["macro_precision"] == 0.942
    assert metrics["macro_recall"] == 0.938
    assert metrics["macro_f1"] == 0.940
    assert d["temperature"] == 1.25


def test_model_evaluate_endpoint_validation(
    master_test_app: tuple[TestClient, Path],
) -> None:
    client, tmp_path = master_test_app
    models_dir = tmp_path / "models"

    # Non-existent model -> 404
    res_404 = client.post("/api/master/models/non-existent-slug/evaluate")
    assert res_404.status_code == 404

    # Existing model without compatible test split -> 400
    res_400 = client.post("/api/master/models/test-olive-model/evaluate")
    assert res_400.status_code == 400
    assert "No se encontró un conjunto de test o validación compatible" in res_400.json()["detail"]

    # Provide explicit dummy valid dir
    val_dir = models_dir / "val_set"
    (val_dir / "sana").mkdir(parents=True)
    (val_dir / "roña").mkdir(parents=True)
    (val_dir / "mildiu").mkdir(parents=True)
    img = Image.new("RGB", (32, 32), color="green")
    img.save(val_dir / "sana" / "s.jpg")
    img.save(val_dir / "roña" / "r.jpg")
    img.save(val_dir / "mildiu" / "m.jpg")

    res_ok = client.post(
        f"/api/master/models/test-olive-model/evaluate?val_dir={val_dir}&batch_size=4"
    )
    assert res_ok.status_code == 200
    d = res_ok.json()
    assert d["status"] == "started"
    assert d["action"] == "evaluate_model:test-olive-model"
    assert d["slug"] == "test-olive-model"





