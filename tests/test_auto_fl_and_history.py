"""Tests for Active Model Deletion Lock lifting, Auto-FL, and Observability history.json."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from agrifed_khaos.catalog import (
    LogicalModelEntry,
    ModelCatalogManager,
    append_model_history,
    get_model_history,
)
from agrifed_khaos.models import build_model, save_checkpoint_bundle
from scripts.serve import create_app


@pytest.fixture
def auto_fl_test_env(tmp_path: Path) -> dict[str, Any]:
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create base checkpoint
    active_path = models_dir / "global_model.pt"
    m = build_model("resnet18", num_classes=2, pretrained=False)
    save_checkpoint_bundle(active_path, m.state_dict(), {"classes": ["sana", "dano"]})

    meta_path = models_dir / "metadata.json"
    meta_path.write_text(json.dumps({"classes": ["sana", "dano"], "arch": "resnet18"}))

    # 2. Bootstrap catalog
    catalog_path = models_dir / "catalog.json"
    cm = ModelCatalogManager(catalog_path=catalog_path, models_dir=models_dir)
    entry = LogicalModelEntry(
        slug="test-walnut-model",
        title="Detección de Defectos en Nueces",
        domain="Agricultura",
        active_checkpoint="global_model.pt",
        versions=["global_model.pt"],
        classes=["sana", "dano"],
    )
    cm.save_catalog([entry])

    # 3. Create dummy dataset directory with class subfolders
    dataset_dir = tmp_path / "data" / "nueces_dataset"
    (dataset_dir / "optima").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "defecto").mkdir(parents=True, exist_ok=True)
    # Put a dummy byte file in each
    (dataset_dir / "optima" / "img1.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 200)
    (dataset_dir / "defecto" / "img2.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 200)

    # 4. Create FastAPI app
    app = create_app(str(active_path), str(meta_path))
    app.state.catalog = cm
    client = TestClient(app)

    return {
        "models_dir": models_dir,
        "active_path": active_path,
        "meta_path": meta_path,
        "dataset_dir": dataset_dir,
        "catalog_manager": cm,
        "app": app,
        "client": client,
    }


def test_active_model_deletion_lock_lifted_and_transitions_to_none(
    auto_fl_test_env: dict[str, Any],
) -> None:
    """Ensure deleting or archiving an active model is permitted and sets active state to None."""
    client: TestClient = auto_fl_test_env["client"]
    cm: ModelCatalogManager = auto_fl_test_env["catalog_manager"]

    # 1. Soft-delete active model
    res_del = client.post("/api/master/models/test-walnut-model/delete")
    assert res_del.status_code == 200
    del_data = res_del.json()
    assert del_data["status"] == "deleted"
    assert del_data["active_checkpoint"] is None

    # Check catalog entry
    entry = cm.get_by_slug("test-walnut-model", include_deleted=True)
    assert entry is not None
    assert entry.is_deleted is True
    assert entry.active_checkpoint is None

    # 2. Restore model
    res_restore = client.post("/api/master/models/test-walnut-model/restore")
    assert res_restore.status_code == 200
    res_entry = cm.get_by_slug("test-walnut-model", include_deleted=False)
    assert res_entry is not None
    assert res_entry.active_checkpoint == "global_model.pt"

    # 3. Archive active model
    res_arch = client.post("/api/master/models/test-walnut-model/archive")
    assert res_arch.status_code == 200
    assert res_arch.json()["status"] == "archived"
    assert res_arch.json()["active_checkpoint"] is None

    archived_entry = cm.get_by_slug("test-walnut-model", include_deleted=False)
    assert archived_entry is not None
    assert archived_entry.active_checkpoint is None


def test_ui_and_predict_handle_active_checkpoint_none(auto_fl_test_env: dict[str, Any]) -> None:
    """Ensure catalog endpoints and inference return 503 when active_checkpoint is null."""
    client: TestClient = auto_fl_test_env["client"]

    # Archive model so active_checkpoint becomes None
    client.post("/api/master/models/test-walnut-model/archive")

    # 1. Public catalog list should not crash and include model with size 0 and null checkpoint
    res_cat = client.get("/api/models")
    assert res_cat.status_code == 200
    models = res_cat.json()["models"]
    walnut_m = next((m for m in models if m["slug"] == "test-walnut-model"), None)
    assert walnut_m is not None
    assert walnut_m["active_checkpoint"] is None
    assert walnut_m["is_active"] is False

    # 2. Model detail endpoint should return model with null download_url
    res_det = client.get("/api/models/test-walnut-model")
    assert res_det.status_code == 200
    det_data = res_det.json()
    assert det_data["model"]["active_checkpoint"] is None
    assert det_data["is_active"] is False
    assert det_data["download_url"] is None

    # 3. Direct inference on out-of-service model must return 503 rather than crash
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color="green").save(buf, format="JPEG")
    valid_img = buf.getvalue()
    res_pred = client.post(
        "/predict",
        files={"file": ("sample.jpg", valid_img, "image/jpeg")},
        params={"model_slug": "test-walnut-model"},
    )
    assert res_pred.status_code == 503
    assert "fuera de servicio" in res_pred.json()["detail"].lower()


def test_auto_fl_endpoint_validation_and_launch(auto_fl_test_env: dict[str, Any]) -> None:
    """Validate Auto-FL input checking and asynchronous pipeline acceptance."""
    client: TestClient = auto_fl_test_env["client"]
    dataset_dir: Path = auto_fl_test_env["dataset_dir"]

    # 1. Missing title
    r1 = client.post(
        "/api/master/models/auto-fl",
        json={"title": "", "dataset_path": str(dataset_dir)},
    )
    assert r1.status_code == 400
    assert "título" in r1.json()["detail"].lower()

    # 2. Non-existent dataset directory
    r2 = client.post(
        "/api/master/models/auto-fl",
        json={"title": "Nueces Auto", "dataset_path": "/invalid/nonexistent/path"},
    )
    assert r2.status_code == 400
    assert "no existe" in r2.json()["detail"].lower()

    # 3. Dataset without class subdirectories
    empty_dir = dataset_dir.parent / "empty_dataset"
    empty_dir.mkdir(parents=True, exist_ok=True)
    r3 = client.post(
        "/api/master/models/auto-fl",
        json={"title": "Nueces Auto", "dataset_path": str(empty_dir)},
    )
    assert r3.status_code == 400
    assert "subcarpetas de clases" in r3.json()["detail"].lower()

    # 4. Valid Auto-FL launch
    r4 = client.post(
        "/api/master/models/auto-fl",
        json={
            "title": "Clasificador Nueces Auto-FL",
            "slug": "clasificador-nueces-auto",
            "domain": "Agricultura",
            "dataset_path": str(dataset_dir),
            "epochs": 2,
            "architecture": "resnet18",
            "target_client_tier": "cpu-low",
        },
    )
    assert r4.status_code == 202
    data = r4.json()
    assert data["status"] == "accepted"
    assert data["slug"] == "clasificador-nueces-auto"
    assert "optima" in data["classes_detected"]
    assert "defecto" in data["classes_detected"]


def test_model_history_foundation(auto_fl_test_env: dict[str, Any]) -> None:
    """Test models/<slug>_history.json lineage append, metrics extraction, and API retrieval."""
    models_dir: Path = auto_fl_test_env["models_dir"]
    client: TestClient = auto_fl_test_env["client"]
    slug = "test-walnut-model"

    # 1. Record initial training version
    rec1 = append_model_history(
        models_dir=models_dir,
        slug=slug,
        version="v1",
        checkpoint_name="walnut_v1.pt",
        accuracy=0.885,
        macro_f1=0.860,
        loss=0.342,
        metadata={"origin": "train_base"},
    )
    assert rec1["version"] == "v1"
    assert rec1["accuracy"] == 0.885

    # 2. Record promoted version after federated aggregation
    rec2 = append_model_history(
        models_dir=models_dir,
        slug=slug,
        version="v2",
        checkpoint_name="walnut_v2.pt",
        metrics={
            "post_aggregation_acc": 0.932,
            "post_aggregation_loss": 0.198,
            "macro_f1": 0.915,
        },
        metadata={"origin": "federated_aggregation"},
    )
    assert rec2["version"] == "v2"
    assert rec2["accuracy"] == 0.932
    assert rec2["loss"] == 0.198
    assert rec2["macro_f1"] == 0.915

    # 3. Retrieve via file helper
    history = get_model_history(models_dir=models_dir, slug=slug)
    assert len(history) == 2
    assert history[0]["version"] == "v1"
    assert history[1]["version"] == "v2"

    # 4. Retrieve via REST endpoint
    res = client.get(f"/api/master/models/{slug}/history")
    assert res.status_code == 200
    api_history = res.json()["history"]
    assert len(api_history) == 2
    assert api_history[1]["accuracy"] == 0.932


def test_master_model_create_page_and_dataset_inspection(
    auto_fl_test_env: dict[str, Any],
) -> None:
    """Verify dedicated model creation studio page and dataset inspection endpoint."""
    client: TestClient = auto_fl_test_env["client"]
    dataset_dir: Path = auto_fl_test_env["dataset_dir"]

    # 1. GET /master/models/new and GET /master/create render HTML studio
    res_page = client.get("/master/models/new")
    assert res_page.status_code == 200
    assert "Estudio de Creación de Modelos" in res_page.text
    assert "btnModeAutoFl" in res_page.text
    assert "datasetPathInput" in res_page.text
    assert "previewTitle" in res_page.text
    assert "previewDomainBadge" in res_page.text
    assert "btnLaunchPipeline" in res_page.text

    res_alias = client.get("/master/create")
    assert res_alias.status_code == 200
    assert "Estudio de Creación de Modelos" in res_alias.text

    # 2. Inspect valid dataset directory
    res_valid = client.post(
        "/api/master/datasets/inspect",
        json={"dataset_path": str(dataset_dir)},
    )
    assert res_valid.status_code == 200
    valid_data = res_valid.json()
    assert valid_data["valid"] is True
    assert valid_data["total_images"] == 2
    assert "optima" in valid_data["classes"]
    assert "defecto" in valid_data["classes"]
    assert valid_data["image_counts"]["optima"] == 1
    assert valid_data["image_counts"]["defecto"] == 1
    assert "recommended_arch" in valid_data

    # 3. Inspect directory without images (like subfolders without valid extensions)
    no_img_dir = dataset_dir.parent / "no_images_dataset"
    (no_img_dir / "class_a").mkdir(parents=True, exist_ok=True)
    (no_img_dir / "class_a" / "notes.txt").write_text("not an image")
    res_no_img = client.post(
        "/api/master/datasets/inspect",
        json={"dataset_path": str(no_img_dir)},
    )
    assert res_no_img.status_code == 200
    no_img_data = res_no_img.json()
    assert no_img_data["valid"] is False
    assert no_img_data["total_images"] == 0
    assert "extensiones soportadas" in no_img_data["error"]

    # 4. Auto-FL rejects directory with no valid images
    res_auto_reject = client.post(
        "/api/master/models/auto-fl",
        json={
            "title": "Modelo Sin Imagenes",
            "dataset_path": str(no_img_dir),
        },
    )
    assert res_auto_reject.status_code == 400
    assert "no contiene imágenes válidas" in res_auto_reject.json()["detail"].lower()

    # 5. Clear master job logs
    res_clear = client.post("/api/master/jobs/clear")
    assert res_clear.status_code == 200
    assert res_clear.json()["status"] == "cleared"


def test_dataset_upload_zip_and_fetch_url(auto_fl_test_env: dict[str, Any]) -> None:
    """Test dataset ingestion via ZIP upload, multi-file folder upload, and HTTPS URL download."""
    client: TestClient = auto_fl_test_env["client"]

    # 1. Create a valid in-memory zip archive with classes
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        img_buf = io.BytesIO()
        Image.new("RGB", (32, 32), color="green").save(img_buf, format="JPEG")
        sample_img = img_buf.getvalue()

        zf.writestr("clase_sana/sample1.jpg", sample_img)
        zf.writestr("clase_sana/sample2.jpg", sample_img)
        zf.writestr("clase_defecto/sample3.jpg", sample_img)

    zip_bytes = zip_buf.getvalue()

    # 2. Upload .zip archive
    res_upload_zip = client.post(
        "/api/master/datasets/upload",
        files={"file": ("dataset_test.zip", zip_bytes, "application/zip")},
    )
    assert res_upload_zip.status_code == 200
    zip_data = res_upload_zip.json()
    assert zip_data["valid"] is True
    assert zip_data["total_images"] == 3
    assert "clase_sana" in zip_data["classes"]
    assert "clase_defecto" in zip_data["classes"]
    assert zip_data["image_counts"]["clase_sana"] == 2
    assert zip_data["image_counts"]["clase_defecto"] == 1
    assert Path(zip_data["dataset_path"]).is_dir()

    # 3. Upload multi-file folder with paths
    res_upload_folder = client.post(
        "/api/master/datasets/upload",
        files=[
            ("files", ("img_a.jpg", sample_img, "image/jpeg")),
            ("files", ("img_b.jpg", sample_img, "image/jpeg")),
        ],
        data={
            "paths": ["folder_dataset/clase_1/img_a.jpg", "folder_dataset/clase_2/img_b.jpg"]
        },
    )
    assert res_upload_folder.status_code == 200
    folder_data = res_upload_folder.json()
    assert folder_data["valid"] is True
    assert folder_data["total_images"] == 2
    assert "clase_1" in folder_data["classes"]
    assert "clase_2" in folder_data["classes"]

    # 4. Reject invalid archive formats
    res_bad_ext = client.post(
        "/api/master/datasets/upload",
        files={"file": ("dataset.pdf", b"%PDF-1.4...", "application/pdf")},
    )
    assert res_bad_ext.status_code == 400

    # 5. Fetch URL security checks (invalid protocol and SSRF blocked hosts)
    res_bad_scheme = client.post(
        "/api/master/datasets/fetch-url",
        json={"url": "ftp://example.com/dataset.zip"},
    )
    assert res_bad_scheme.status_code == 400
    assert "solo se admiten enlaces http o https" in res_bad_scheme.json()["detail"].lower()

    res_ssrf = client.post(
        "/api/master/datasets/fetch-url",
        json={"url": "http://169.254.169.254/latest/meta-data/"},
    )
    assert res_ssrf.status_code == 400
    assert "host no permitido" in res_ssrf.json()["detail"].lower()

    # 6. Fetch URL success with mocked streaming download
    class MockStreamResponse:
        status_code = 200
        reason_phrase = "OK"

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def aiter_bytes(self, chunk_size: int = 65536):
            yield zip_bytes

    class MockAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        def stream(self, method: str, url: str):
            return MockStreamResponse()

    with patch("httpx.AsyncClient", side_effect=MockAsyncClient):
        res_fetch = client.post(
            "/api/master/datasets/fetch-url",
            json={"url": "https://data.example.org/datasets/walnut_archive.zip"},
        )
        assert res_fetch.status_code == 200
        fetch_data = res_fetch.json()
        assert fetch_data["valid"] is True
        assert fetch_data["total_images"] == 3
        assert "clase_sana" in fetch_data["classes"]
        assert Path(fetch_data["dataset_path"]).is_dir()


def test_nested_dataset_auto_resolution(auto_fl_test_env: dict[str, Any], tmp_path: Path) -> None:
    """Ensure root directories with wrapper/non-image folders auto-resolve to class splits."""
    client: TestClient = auto_fl_test_env["client"]

    # Create root directory structure like /data/EnfermedadesHoja_v1.0
    repo_root = tmp_path / "EnfermedadesHoja_v1.0"
    (repo_root / "annotations").mkdir(parents=True, exist_ok=True)
    (repo_root / "preprocess").mkdir(parents=True, exist_ok=True)
    (repo_root / "detection" / "train" / "images").mkdir(parents=True, exist_ok=True)
    (repo_root / "dataset" / "train" / "repilo").mkdir(parents=True, exist_ok=True)
    (repo_root / "dataset" / "train" / "sano").mkdir(parents=True, exist_ok=True)
    (repo_root / "dataset" / "val" / "repilo").mkdir(parents=True, exist_ok=True)
    (repo_root / "dataset" / "val" / "sano").mkdir(parents=True, exist_ok=True)

    dummy_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 100
    (repo_root / "dataset" / "train" / "repilo" / "img1.jpg").write_bytes(dummy_jpeg)
    (repo_root / "dataset" / "train" / "repilo" / "img2.jpg").write_bytes(dummy_jpeg)
    (repo_root / "dataset" / "train" / "sano" / "img3.jpg").write_bytes(dummy_jpeg)
    (repo_root / "dataset" / "val" / "repilo" / "img4.jpg").write_bytes(dummy_jpeg)
    (repo_root / "dataset" / "val" / "sano" / "img5.jpg").write_bytes(dummy_jpeg)

    # 1. Inspect passing the repo root
    res = client.post(
        "/api/master/datasets/inspect",
        json={"dataset_path": str(repo_root)},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["valid"] is True
    assert data["auto_detected"] is True
    assert data["original_path"] == str(repo_root)
    assert data["dataset_path"] == str(repo_root / "dataset" / "train")
    assert sorted(data["classes"]) == ["repilo", "sano"]
    assert data["total_images"] == 3
    assert data["image_counts"]["repilo"] == 2
    assert data["image_counts"]["sano"] == 1

    # 2. Inspect passing the intermediate dataset/ folder
    res_ds = client.post(
        "/api/master/datasets/inspect",
        json={"dataset_path": str(repo_root / "dataset")},
    )
    assert res_ds.status_code == 200
    ds_data = res_ds.json()
    assert ds_data["valid"] is True
    assert ds_data["dataset_path"] == str(repo_root / "dataset" / "train")


def test_find_dataset_class_dir_and_split_selection(tmp_path: Path) -> None:
    """Test standalone find_dataset_class_dir and resolve_dataset_dir functions."""
    from agrifed_khaos.data import find_dataset_class_dir, resolve_dataset_dir

    root = tmp_path / "complex_repo"
    (root / "dataset" / "train" / "cat").mkdir(parents=True, exist_ok=True)
    (root / "dataset" / "train" / "dog").mkdir(parents=True, exist_ok=True)
    (root / "dataset" / "val" / "cat").mkdir(parents=True, exist_ok=True)
    (root / "dataset" / "val" / "dog").mkdir(parents=True, exist_ok=True)

    dummy_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 50
    (root / "dataset" / "train" / "cat" / "c.jpg").write_bytes(dummy_jpeg)
    (root / "dataset" / "train" / "dog" / "d.jpg").write_bytes(dummy_jpeg)
    (root / "dataset" / "val" / "cat" / "c.jpg").write_bytes(dummy_jpeg)
    (root / "dataset" / "val" / "dog" / "d.jpg").write_bytes(dummy_jpeg)

    # Train split preferred by default
    assert find_dataset_class_dir(root) == (root / "dataset" / "train")
    # Val split explicitly preferred
    assert find_dataset_class_dir(root, preferred_split="val") == (root / "dataset" / "val")

    # resolve_dataset_dir resolves train
    assert resolve_dataset_dir(root, split="train") == (root / "dataset" / "train")


def test_tailwind_warning_suppression_in_templates() -> None:
    """Ensure HTML templates contain the script interceptor to silence Tailwind warnings."""
    templates_dir = Path("src/agrifed_khaos/templates")
    for html_file in templates_dir.glob("*.html"):
        content = html_file.read_text(encoding="utf-8")
        if "cdn.tailwindcss.com" in content:
            assert "cdn.tailwindcss.com" in content
            assert "console.warn" in content
            assert "origWarn" in content


def test_available_datasets_and_expert_stats_endpoints(
    auto_fl_test_env: dict[str, Any],
) -> None:
    """Ensure /api/expert/stats alias and /api/master/datasets/available return 200 OK."""
    client: TestClient = auto_fl_test_env["client"]

    # 1. Expert stats alias
    res_stats = client.get("/api/expert/stats?model_slug=test-walnut-model")
    assert res_stats.status_code == 200
    stats_data = res_stats.json()
    assert "pending" in stats_data
    assert "verified_total" in stats_data

    # 2. Available datasets
    res_avail = client.get("/api/master/datasets/available")
    assert res_avail.status_code == 200
    avail_data = res_avail.json()
    assert "datasets" in avail_data
    assert "total" in avail_data




