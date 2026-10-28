"""Tests for Batch Inference (ZIP upload) endpoint and UX contracts."""

import csv
import io
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from agrifed_khaos.catalog import LogicalModelEntry, ModelCatalogManager
from agrifed_khaos.models import build_model, save_checkpoint_bundle
from scripts.serve import create_app


def _create_dummy_checkpoint(path: Path) -> None:
    model = build_model("resnet18", num_classes=3, pretrained=False)
    save_checkpoint_bundle(
        path,
        model.state_dict(),
        {
            "architecture": "resnet18",
            "num_classes": 3,
            "classes": ["Healthy", "aculus_olearius", "olive_peacock_spot"],
            "img_size": [32, 32],
            "epoch": 5,
        },
    )


def test_batch_zip_inference_end_to_end(tmp_path: Path) -> None:
    """End-to-end test for POST /api/predict/batch."""
    model_path = tmp_path / "test_model.pt"
    _create_dummy_checkpoint(model_path)

    catalog_path = tmp_path / "catalog.json"
    cat_mgr = ModelCatalogManager(catalog_path=catalog_path, models_dir=tmp_path)
    entry = LogicalModelEntry(
        slug="olivo-test",
        title="Modelo Test Olivo",
        domain="Olivo",
        active_checkpoint=model_path.name,
    )
    cat_mgr.save_catalog([entry])

    meta_path = tmp_path / "meta.json"
    meta_path.write_text('{"classes": ["Healthy", "aculus_olearius", "olive_peacock_spot"]}')

    app = create_app(str(model_path), str(meta_path))
    app.state.catalog = cat_mgr
    client = TestClient(app)

    # 1. Reject non-zip files
    r_bad = client.post(
        "/api/predict/batch",
        files={"file": ("test.txt", io.BytesIO(b"not a zip"), "text/plain")},
    )
    assert r_bad.status_code == 400
    assert "zip" in r_bad.json()["detail"].lower()

    # 2. Reject zip with no images
    empty_zip_buf = io.BytesIO()
    with zipfile.ZipFile(empty_zip_buf, "w") as zf:
        zf.writestr("notes.txt", "hello world")
    empty_zip_buf.seek(0)

    r_no_img = client.post(
        "/api/predict/batch",
        files={"file": ("empty.zip", empty_zip_buf, "application/zip")},
    )
    assert r_no_img.status_code == 400
    assert "no supported images" in r_no_img.json()["detail"].lower()

    # 3. Valid ZIP with 3 synthetic test images
    valid_zip_buf = io.BytesIO()
    with zipfile.ZipFile(valid_zip_buf, "w") as zf:
        for idx in range(3):
            img = Image.new("RGB", (64, 64), color=(idx * 40, 100, 120))
            b = io.BytesIO()
            img.save(b, format="JPEG")
            zf.writestr(f"sample_leaf_{idx}.jpg", b.getvalue())
    valid_zip_buf.seek(0)

    r_success = client.post(
        "/api/predict/batch?model_slug=olivo-test",
        files={"file": ("batch_test.zip", valid_zip_buf, "application/zip")},
    )
    assert r_success.status_code == 200
    assert r_success.headers["content-type"] == "application/zip"

    # Verify returned ZIP archive structure and report.csv
    out_buf = io.BytesIO(r_success.content)
    with zipfile.ZipFile(out_buf, "r") as zf:
        namelist = zf.namelist()
        assert "report.csv" in namelist

        # Read and check report.csv content
        csv_bytes = zf.read("report.csv").decode("utf-8")
        reader = list(csv.DictReader(csv_bytes.splitlines()))
        assert len(reader) == 3

        for row in reader:
            assert "filename" in row
            assert "predicted_class" in row
            assert "confidence" in row
            assert "ood_flag" in row
            assert row["filename"].startswith("sample_leaf_")
            # Image should exist in the subfolder named after predicted_class
            subfolder_file = f"{row['predicted_class']}/{row['filename']}"
            assert subfolder_file in namelist


def test_model_detail_template_batch_and_reset_ux_elements(tmp_path: Path) -> None:
    """Verify that model_detail.html contains the individual reset and batch inference UX."""
    model_path = tmp_path / "model.pt"
    _create_dummy_checkpoint(model_path)
    app = create_app(str(model_path), str(tmp_path / "meta.json"))
    client = TestClient(app)

    r = client.get("/models/default")
    assert r.status_code == 200
    html = r.text

    # Mode switcher
    assert "setInferenceMode" in html
    assert "btnModeIndividual" in html
    assert "btnModeBatch" in html
    assert "inferModeIndividual" in html
    assert "inferModeBatch" in html

    # Batch UI components
    assert "batchZipInput" in html
    assert "btnRunBatch" in html
    assert "Procesando imágenes, esto puede tardar..." in html
    assert "/api/predict/batch" in html

    # Individual Reset UX
    assert "resetInferenceResultState" in html
    assert "btnDownloadReport" in html
    assert "downloadDiagnosisReport" in html
