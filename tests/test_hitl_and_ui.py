"""Tests for HITL Queue and Multi-Role Web Endpoints (/expert, /master, /api/*)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch
from PIL import Image
from starlette.testclient import TestClient

from agrifed_khaos.hitl import HITLQueueManager
from agrifed_khaos.models import build_model, save_checkpoint_bundle
from scripts.serve import create_app


@pytest.fixture
def hitl_env(tmp_path: Path) -> tuple[Path, Path, Path]:
    model_path = tmp_path / "global_model.pt"
    meta_path = tmp_path / "metadata.json"
    hitl_dir = tmp_path / "hitl_queue"

    model = build_model("resnet18", num_classes=3, pretrained=False)
    bundle_meta = {
        "architecture": "resnet18",
        "num_classes": 3,
        "classes": ["sana", "roña", "mildiu"],
        "img_size": [64, 64],
        "calibrated_thresholds": {"sana": 0.5, "roña": 0.5, "mildiu": 0.5},
    }
    save_checkpoint_bundle(model_path, model.state_dict(), bundle_meta)
    return model_path, meta_path, hitl_dir


def test_hitl_queue_manager_enqueue_verify_reject(tmp_path: Path) -> None:
    """Test HITLQueueManager enqueue, uncertainty ordering, verify and reject logic."""
    import asyncio

    mgr = HITLQueueManager(root_dir=tmp_path / "hitl_queue")

    async def _run() -> None:
        # Enqueue item 1 (high confidence 0.95)
        item1 = await mgr.enqueue(
            image_bytes=b"dummy_img_bytes_1",
            filename="leaf1.jpg",
            model_name="test_model.pt",
            predicted_class="sana",
            confidence=0.95,
            probabilities={"sana": 0.95, "roña": 0.05},
        )
        assert item1.model_name == "test_model.pt"

        # Enqueue item 2 (low confidence 0.42 - uncertainty sampling should prioritize this)
        item2 = await mgr.enqueue(
            image_bytes=b"dummy_img_bytes_2",
            filename="leaf2.jpg",
            model_name="test_model.pt",
            predicted_class="roña",
            confidence=0.42,
            probabilities={"sana": 0.40, "roña": 0.42, "mildiu": 0.18},
        )

        # Enqueue item 3 (to reject)
        item3 = await mgr.enqueue(
            image_bytes=b"dummy_img_bytes_3",
            filename="random.jpg",
            model_name="test_model.pt",
            predicted_class="sana",
            confidence=0.50,
            probabilities={"sana": 0.50, "roña": 0.50},
        )

        # 1. List pending with uncertainty sampling: least confident (item2: 0.42) MUST be first
        pending = await mgr.list_pending(sort_by_uncertainty=True)
        assert len(pending) == 3
        assert pending[0]["item_id"] == item2.item_id
        assert pending[0]["confidence"] == 0.42
        assert pending[1]["item_id"] == item3.item_id

        # 2. Verify item2 with ground truth class
        ver_res = await mgr.verify(item2.item_id, verified_class="mildiu", reviewer="dr_agronomo")
        assert ver_res["status"] == "verified"
        assert ver_res["verified_class"] == "mildiu"

        # Check directory structure
        ver_img = mgr.get_image_path(item2.item_id)
        assert ver_img is not None and "verified/mildiu" in str(ver_img)

        # 3. Reject item3
        rej_res = await mgr.reject(item3.item_id, reviewer="dr_agronomo", reason="not a leaf")
        assert rej_res["status"] == "rejected"
        rej_img = mgr.get_image_path(item3.item_id)
        assert rej_img is not None and "rejected" in str(rej_img)

        # 4. Check statistics
        stats = await mgr.stats()
        assert stats["pending"] == 1
        assert stats["verified_total"] == 1
        assert stats["rejected"] == 1
        assert stats["verified_by_class"].get("mildiu") == 1

    asyncio.run(_run())


def test_serve_multi_role_pages_and_api(hitl_env: tuple[Path, Path, Path]) -> None:
    """Test GET /, GET /expert, GET /master HTML routes and associated APIs."""
    model_path, meta_path, hitl_dir = hitl_env

    app = create_app(str(model_path), str(meta_path))
    # Override hitl queue root to test dir
    app.state.hitl = HITLQueueManager(root_dir=hitl_dir)
    client = TestClient(app)

    # 1. Verify GET / returns HTML for Agnostic Client
    resp_client = client.get("/", headers={"accept": "text/html"})
    assert resp_client.status_code == 200
    assert "Diagnóstico Agrícola Inteligente" in resp_client.text
    assert "🌾 Cliente" in resp_client.text

    # 2. Verify GET /expert returns HTML for Expert Agronomist
    resp_expert = client.get("/expert")
    assert resp_expert.status_code == 200
    assert "Panel de Validación Agronómica" in resp_expert.text
    assert "Cola de Fotografías por Validar" in resp_expert.text

    # 3. Verify GET /master returns HTML for Master Administrator
    resp_master = client.get("/master")
    assert resp_master.status_code == 200
    assert "Centro de Control MLOps" in resp_master.text
    assert "Job Runner" in resp_master.text

    # 4. Test API Models Listing
    resp_models = client.get("/api/models")
    assert resp_models.status_code == 200
    data_models = resp_models.json()
    assert "models" in data_models
    assert len(data_models["models"]) >= 1

    # 5. Predict with save_for_training=True (default) and verify HITL enqueue
    dummy_img = Image.new("RGB", (64, 64), color="green")
    buf = io.BytesIO()
    dummy_img.save(buf, format="JPEG")
    buf.seek(0)

    resp_pred = client.post(
        "/predict?save_for_training=true",
        files={"file": ("field_sample.jpg", buf, "image/jpeg")},
    )
    assert resp_pred.status_code == 200
    pred_data = resp_pred.json()
    assert pred_data.get("saved_for_training") is True
    assert "predicted_class" in pred_data

    # Check pending feedback in API
    resp_pending = client.get("/api/feedback/pending")
    assert resp_pending.status_code == 200
    pending_data = resp_pending.json()
    assert pending_data["count"] >= 1
    item = pending_data["items"][0]
    item_id = item["item_id"]

    # Expert verifies the item
    resp_verify = client.post(f"/api/feedback/verify/{item_id}?verified_class=mildiu")
    assert resp_verify.status_code == 200
    assert resp_verify.json()["status"] == "verified"


def test_serve_model_selection_and_lru_cache(tmp_path: Path) -> None:
    """Test model selection via query parameter and LRU caching without global mutation."""
    model1_path = tmp_path / "model1.pt"
    model2_path = tmp_path / "model2.pt"

    m1 = build_model("resnet18", num_classes=2, pretrained=False)
    m2 = build_model("resnet18", num_classes=2, pretrained=False)
    # Set distinct constant biases so outputs are recognizable
    with torch.no_grad():
        m1.fc.weight.zero_()  # type: ignore[union-attr]
        m1.fc.bias.copy_(torch.tensor([10.0, -10.0]))  # type: ignore[union-attr]
        m2.fc.weight.zero_()  # type: ignore[union-attr]
        m2.fc.bias.copy_(torch.tensor([-10.0, 10.0]))  # type: ignore[union-attr]

    meta1 = {
        "architecture": "resnet18",
        "num_classes": 2,
        "classes": ["clase_a", "clase_b"],
        "img_size": [64, 64],
    }
    meta2 = {
        "architecture": "resnet18",
        "num_classes": 2,
        "classes": ["clase_x", "clase_y"],
        "img_size": [64, 64],
    }
    save_checkpoint_bundle(model1_path, m1.state_dict(), meta1)
    save_checkpoint_bundle(model2_path, m2.state_dict(), meta2)

    app = create_app(str(model1_path), str(tmp_path / "meta.json"))
    client = TestClient(app)

    # 1. Prediction without param uses default active model (model1 -> clase_a)
    buf1 = io.BytesIO()
    Image.new("RGB", (64, 64), color="red").save(buf1, format="JPEG")
    buf1.seek(0)
    r1 = client.post(
        "/predict?save_for_training=false",
        files={"file": ("t.jpg", buf1, "image/jpeg")},
    )
    assert r1.status_code == 200
    assert r1.json()["predicted_class"] == "clase_a"

    # 2. Prediction specifying model2 -> clase_y (from model2 biases)
    buf2 = io.BytesIO()
    Image.new("RGB", (64, 64), color="red").save(buf2, format="JPEG")
    buf2.seek(0)
    r2 = client.post(
        f"/predict?model_name={model2_path.name}&save_for_training=false",
        files={"file": ("t.jpg", buf2, "image/jpeg")},
    )
    assert r2.status_code == 200
    assert r2.json()["predicted_class"] == "clase_y"

    # 3. Global active model remains model1 (not mutated)
    r_health = client.get("/health")
    assert r_health.json()["classes"] == ["clase_a", "clase_b"]


def test_master_job_manager_status_and_lock() -> None:
    """Test JobManager FIFO queue execution, non-blocking enqueuing, cancellation, and status polling."""
    import asyncio
    import sys

    from scripts.serve import JobManager

    jm = JobManager()

    async def _test_flow() -> None:
        assert not jm.is_running
        status = jm.get_status()
        assert status["is_running"] is False
        assert status["queue_length"] == 0

        # Launch quick command
        test_cmd_1 = [
            sys.executable,
            "-c",
            "import time; print('hello from job 1'); time.sleep(0.2)",
        ]
        res1 = await jm.enqueue_or_run("job_1", test_cmd_1)
        assert res1["status"] == "started"
        assert res1["queue_position"] == 0
        assert jm.is_running
        st_running = jm.get_status()
        assert st_running["is_running"] is True
        assert st_running["action"] == "job_1"

        # Attempt concurrent run -> must NOT fail or throw 409; must enqueue smoothly
        test_cmd_2 = [
            sys.executable,
            "-c",
            "import time; print('hello from job 2'); time.sleep(0.1)",
        ]
        res2 = await jm.enqueue_or_run("job_2", test_cmd_2)
        assert res2["status"] == "queued"
        assert res2["queue_position"] == 1

        test_cmd_3 = [
            sys.executable,
            "-c",
            "print('hello from job 3')",
        ]
        res3 = await jm.enqueue_or_run("job_3", test_cmd_3)
        assert res3["status"] == "queued"
        assert res3["queue_position"] == 2

        st_queued = jm.get_status()
        assert st_queued["queue_length"] == 2

        # Cancel job_3 from queue
        cancel_res3 = await jm.cancel_job(res3["job_id"])
        assert cancel_res3["cancelled"] is True
        assert cancel_res3["type"] == "queued_removed"
        assert jm.get_status()["queue_length"] == 1

        # Wait for job 1 and then job 2 to finish sequentially
        for _ in range(40):
            await asyncio.sleep(0.05)
            if not jm.is_running and len(jm.queue) == 0:
                break

        assert not jm.is_running
        assert jm.last_exit_code == 0
        logs = "\n".join(jm.logs)
        assert "hello from job 1" in logs
        assert "hello from job 2" in logs

        # Test cancel running job
        slow_cmd = [
            sys.executable,
            "-c",
            "import time; time.sleep(5)",
        ]
        res4 = await jm.enqueue_or_run("job_4_slow", slow_cmd)
        assert res4["status"] == "started"
        assert jm.is_running

        cancel_res4 = await jm.cancel_job("job_4_slow")
        assert cancel_res4["cancelled"] is True
        assert cancel_res4["type"] == "running_terminated"

        for _ in range(20):
            await asyncio.sleep(0.05)
            if not jm.is_running:
                break
        assert not jm.is_running

    asyncio.run(_test_flow())


def test_master_admin_endpoints(hitl_env: tuple[Path, Path, Path]) -> None:
    """Test /api/master/models, /api/master/set-active, and /api/master/train-hitl."""
    model_path, meta_path, hitl_dir = hitl_env

    app = create_app(str(model_path), str(meta_path))
    app.state.hitl = HITLQueueManager(root_dir=hitl_dir)
    client = TestClient(app)

    # 1. GET /api/master/models
    r_models = client.get("/api/master/models")
    assert r_models.status_code == 200
    data_models = r_models.json()
    assert "models" in data_models

    # 2. POST /api/master/set-active
    r_set = client.post(f"/api/master/set-active?model_name={model_path.name}")
    assert r_set.status_code == 200
    assert r_set.json()["status"] == "activated"
    assert r_set.json()["active_model"] == model_path.name

    # Also test JSON body
    r_set_body = client.post("/api/master/set-active", json={"model_name": model_path.name})
    assert r_set_body.status_code == 200
    assert r_set_body.json()["active_model"] == model_path.name

    # 3. POST /api/master/train-hitl when verified/ is empty -> 400
    r_empty_hitl = client.post("/api/master/train-hitl")
    assert r_empty_hitl.status_code == 400

    # Create dummy verified image to test 202 Accepted
    ver_dir = hitl_dir / "verified" / "sano"
    ver_dir.mkdir(parents=True, exist_ok=True)
    dummy_file = ver_dir / "sample.jpg"
    dummy_file.write_bytes(b"dummy_image_data")

    r_hitl = client.post("/api/master/train-hitl")
    assert r_hitl.status_code == 202
    assert r_hitl.json()["status"] == "accepted"


def test_master_soft_delete_trash_and_purge(hitl_env: tuple[Path, Path, Path]) -> None:
    """Test soft-delete to trash, active/sole model safeguards, restore, and permanent purge."""
    model_path, meta_path, hitl_dir = hitl_env
    models_dir = model_path.parent

    # Create a secondary checkpoint for testing multiple models
    model2_path = models_dir / "test_sec_model.pt"
    m2 = build_model("resnet18", num_classes=2, pretrained=False)
    save_checkpoint_bundle(model2_path, m2.state_dict(), {"classes": ["c1", "c2"]})

    app = create_app(str(model_path), str(meta_path))
    client = TestClient(app)

    # Bootstrap catalog with two models
    from agrifed_khaos.catalog import LogicalModelEntry, ModelCatalogManager
    cat_mgr = ModelCatalogManager(catalog_path=models_dir / "catalog.json", models_dir=models_dir)
    e1 = LogicalModelEntry(
        slug="primary-model",
        title="Primary Olive Model",
        domain="Agricultura",
        active_checkpoint=model_path.name,
        versions=[model_path.name],
        classes=["sana", "roña"],
    )
    e2 = LogicalModelEntry(
        slug="secondary-model",
        title="Secondary Avocado Model",
        domain="Agricultura",
        active_checkpoint=model2_path.name,
        versions=[model2_path.name],
        classes=["c1", "c2"],
    )
    cat_mgr.save_catalog([e1, e2])
    app.state.catalog = cat_mgr

    # 1. Soft-delete secondary-model -> moves to trash quarantine (200 OK)
    r_del_sec = client.post("/api/master/models/secondary-model/delete?retention_days=7")
    assert r_del_sec.status_code == 200
    assert r_del_sec.json()["status"] == "deleted"
    assert r_del_sec.json()["purge_after"] is not None

    # 2. Model is hidden from public catalog and detail
    r_catalog = client.get("/api/models")
    slugs = [m["slug"] for m in r_catalog.json()["models"]]
    assert "secondary-model" not in slugs
    assert "primary-model" in slugs

    r_detail = client.get("/api/models/secondary-model")
    assert r_detail.status_code == 404

    # 3. Check Trash list endpoint
    r_trash = client.get("/api/master/trash")
    assert r_trash.status_code == 200
    trash_data = r_trash.json()
    assert trash_data["total"] == 1
    assert trash_data["trash"][0]["slug"] == "secondary-model"
    assert trash_data["trash"][0]["days_remaining"] >= 6.0

    # 4. Deleting active model is allowed without lock; transitions active state to None
    r_del_active = client.post("/api/master/models/primary-model/delete")
    assert r_del_active.status_code == 200
    assert r_del_active.json()["status"] == "deleted"
    assert r_del_active.json()["active_checkpoint"] is None

    # 5. Restore primary-model and secondary-model from trash
    r_res_pri = client.post("/api/master/models/primary-model/restore")
    assert r_res_pri.status_code == 200
    r_restore = client.post("/api/master/models/secondary-model/restore")
    assert r_restore.status_code == 200
    assert r_restore.json()["status"] == "restored"

    # Test Archive endpoint on secondary model
    r_arch = client.post("/api/master/models/secondary-model/archive")
    assert r_arch.status_code == 200
    assert r_arch.json()["status"] == "archived"
    assert r_arch.json()["active_checkpoint"] is None

    # Now it is back in active catalog
    r_catalog_after = client.get("/api/models")
    slugs_after = [m["slug"] for m in r_catalog_after.json()["models"]]
    assert "secondary-model" in slugs_after

    # 7. Test Hard Purge
    # Soft delete again first
    client.post("/api/master/models/secondary-model/delete")

    # Mismatched confirmation must fail with 400
    r_purge_fail = client.post("/api/master/models/secondary-model/purge?confirm_slug=wrong-slug")
    assert r_purge_fail.status_code == 400
    assert "Doble confirmación requerida" in r_purge_fail.json()["detail"]

    # Valid confirmation purges permanently
    assert model2_path.exists()
    r_purge = client.post("/api/master/models/secondary-model/purge?confirm_slug=secondary-model")
    assert r_purge.status_code == 200
    assert r_purge.json()["status"] == "purged"
    assert not model2_path.exists()  # unshared checkpoint physically deleted from disk!


def test_design_system_and_refactored_ui_architecture(
    hitl_env: tuple[Path, Path, Path],
) -> None:
    """Verify shared design system assets, fixed HTML DOM structure, and studio contracts."""
    model_p, meta_p, _ = hitl_env
    app = create_app(model_path=str(model_p), metadata_path=str(meta_p))
    client = TestClient(app)

    # 1. Static Design System Assets (CSS & JS)
    r_css = client.get("/static/khaos-ui.css")
    assert r_css.status_code == 200
    assert "text/css" in r_css.headers["content-type"]
    assert "--kh-brand-600" in r_css.text
    assert "prefers-reduced-motion" in r_css.text

    r_js = client.get("/static/khaos-ui.js")
    assert r_js.status_code == 200
    assert "javascript" in r_js.headers["content-type"]
    assert "renderCatalogCardHtml" in r_js.text
    assert "Referencia ilustrativa" in r_js.text

    # 2. Checkpoints endpoint contract for Model Creation Studio
    r_chk = client.get("/api/master/checkpoints")
    assert r_chk.status_code == 200
    assert "checkpoints" in r_chk.json()

    # 3. Model Creation Studio 6-stage progressive structure & valid client tiers
    r_studio = client.get("/master/models/new")
    assert r_studio.status_code == 200
    assert "/static/khaos-ui.css" in r_studio.text
    assert "/static/khaos-ui.js" in r_studio.text
    assert 'id="liveCatalogCardHost"' in r_studio.text
    assert 'id="cropInput"' in r_studio.text
    assert 'id="targetElementInput"' in r_studio.text
    assert 'id="purposeInput"' in r_studio.text
    assert 'id="thumbnailUrlInput"' in r_studio.text
    assert 'value="cpu-high"' in r_studio.text
    assert 'value="gpu"' in r_studio.text
    assert 'value="cpu-mid"' not in r_studio.text

    # 4. Master Admin page: single trashModelsTbody (no duplicate malformed HTML) + High-Craft UI
    r_master = client.get("/master")
    assert r_master.status_code == 200
    assert r_master.text.count('id="trashModelsTbody"') == 1
    assert "fonts.googleapis.com/css2?family=Inter" in r_master.text
    assert "border-slate-200/60" in r_master.text
    assert "ring-emerald-600/10" in r_master.text

    # 5. Public catalog: mobile filter drawer toggle & accessible dialog attributes
    r_cat = client.get("/", headers={"Accept": "text/html"})
    assert r_cat.status_code == 200
    assert 'id="btnMobileFiltersToggle"' in r_cat.text
    assert 'role="dialog"' in r_cat.text
    assert "Referencia ilustrativa" in r_cat.text

    # 6. Expert workspace: HITL filter/sort controls & structured rejection reasons
    r_exp = client.get("/expert")
    assert r_exp.status_code == 200
    assert 'id="hitlClassFilterSelect"' in r_exp.text
    assert 'id="hitlSortSelect"' in r_exp.text
    assert "calidad_insuficiente" in r_exp.text
    assert "sintoma_ambiguo" in r_exp.text
    assert "renderDetectionCard" in r_exp.text
    assert "toggleDrawMode" in r_exp.text
    assert "submitDetectionValidation" in r_exp.text
    assert "confirmEmptyDetectionItem" in r_exp.text

    # 7. Modality (Detection vs Classification) UI Badges & Filters
    assert 'id="archModalityBadge"' in r_studio.text
    assert "Detección de Objetos (Bounding Boxes)" in r_studio.text
    assert "Clasificación de Imagen (Diagnóstico Global)" in r_studio.text
    assert 'id="taskFilterDetection"' in r_master.text
    assert 'id="taskFilterClassification"' in r_master.text
    assert 'id="facetTaskSelect"' in r_cat.text
    assert "Modalidad de Visión" in r_cat.text


def test_hitl_detection_queue_and_yolo_export(tmp_path: Path) -> None:
    """Test object detection HITL flow: enqueue with detections, verify_detection (YOLO), negative images, and stats."""
    import asyncio

    mgr = HITLQueueManager(root_dir=tmp_path / "hitl_queue")

    async def _run() -> None:
        # 1. Enqueue detection sample with boxes
        raw_boxes = [
            {"class_name": "Tuberculosis", "box_xywhn": [0.5, 0.4, 0.2, 0.3], "confidence": 0.88},
            {"class_name": "BlackScale", "box_xywhn": [0.8, 0.7, 0.15, 0.25], "confidence": 0.65},
        ]
        item1 = await mgr.enqueue(
            image_bytes=b"fake_jpeg_bytes_det_1",
            filename="olive_det_1.jpg",
            model_name="yolo_olivo.pt",
            predicted_class="Tuberculosis",
            confidence=0.88,
            probabilities={"Tuberculosis": 0.88, "BlackScale": 0.65},
            model_slug="yolo-olivo",
            detections=raw_boxes,
        )
        assert item1.detections is not None
        assert len(item1.detections) == 2
        assert item1.detections[0].class_name == "Tuberculosis"
        assert item1.detections[0].box_xywhn == [0.5, 0.4, 0.2, 0.3]

        # 2. Enqueue second detection sample for negative image verification
        item2 = await mgr.enqueue(
            image_bytes=b"fake_jpeg_bytes_det_2",
            filename="olive_det_2.jpg",
            model_name="yolo_olivo.pt",
            predicted_class="Sin detecciones",
            confidence=0.10,
            probabilities={},
            model_slug="yolo-olivo",
            detections=[],
        )

        # 3. Verify item1 with detection boxes
        verified_classes = ["Tuberculosis", "BlackScale", "Anthracnose"]
        res1 = await mgr.verify_detection(
            item_id=item1.item_id,
            boxes=[
                {"class_name": "Tuberculosis", "box_xywhn": [0.51, 0.41, 0.22, 0.31]},
                {"class_name": "Anthracnose", "box_xywhn": [0.1, 0.2, 0.05, 0.05]},
            ],
            class_names=verified_classes,
            reviewer="expert_dr_olive",
            notes="Added undetected Anthracnose lesion",
        )
        assert res1["status"] == "verified"
        assert res1["task"] == "detection"
        assert len(res1["verified_boxes"]) == 2

        # Check YOLO filesystem structure
        v_dir = tmp_path / "hitl_queue" / "yolo-olivo" / "verified"
        assert (v_dir / "images" / f"{item1.item_id}.jpg").is_file()
        label_file1 = v_dir / "labels" / f"{item1.item_id}.txt"
        assert label_file1.is_file()
        lines = label_file1.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        # Tuberculosis is class 0, Anthracnose is class 2
        assert lines[0].startswith("0 0.510000 0.410000 0.220000 0.310000")
        assert lines[1].startswith("2 0.100000 0.200000 0.050000 0.050000")

        # Check dataset.yaml and data.yaml
        assert (v_dir / "dataset.yaml").is_file()
        yaml_content = (v_dir / "dataset.yaml").read_text(encoding="utf-8")
        assert "train: images" in yaml_content
        assert "0: Tuberculosis" in yaml_content
        assert "1: BlackScale" in yaml_content
        assert "2: Anthracnose" in yaml_content

        # 4. Verify item2 as negative (background) image with empty boxes
        res2 = await mgr.verify_detection(
            item_id=item2.item_id,
            boxes=[],
            class_names=verified_classes,
            reviewer="expert_dr_olive",
            notes="Confirmed background leaf with no diseases",
        )
        assert res2["status"] == "verified"
        assert res2["verified_boxes"] == []
        label_file2 = v_dir / "labels" / f"{item2.item_id}.txt"
        assert label_file2.is_file()
        assert label_file2.stat().st_size == 0  # 0 bytes empty label for YOLO negative

        # 5. Check queue statistics
        stats = await mgr.stats(model_slug="yolo-olivo")
        assert stats["pending"] == 0
        assert stats["verified_total"] == 2
        assert stats["verified_negative_count"] == 1
        assert stats["verified_by_class"]["Tuberculosis"] == 1
        assert stats["verified_by_class"]["Anthracnose"] == 1

    asyncio.run(_run())


def test_api_feedback_verify_detection_and_retraining(hitl_env: tuple[Path, Path, Path]) -> None:
    """Test API endpoint POST /api/feedback/verify/{id} for detection and POST /api/master/train-hitl."""
    import asyncio

    model_p, meta_p, hitl_dir = hitl_env
    app = create_app(str(model_p), str(meta_p))
    mgr = HITLQueueManager(root_dir=hitl_dir)
    app.state.hitl = mgr
    client = TestClient(app)

    # Enqueue a detection sample directly
    item = mgr.enqueue_sync(
        image_bytes=b"dummy_olive_bytes",
        filename="det_api_test.jpg",
        model_name="test_det",
        predicted_class="Tuberculosis",
        confidence=0.75,
        probabilities={"Tuberculosis": 0.75},
        model_slug="test-det-model",
        detections=[{"class_name": "Tuberculosis", "box_xywhn": [0.4, 0.4, 0.2, 0.2]}],
    )

    # 1. Verify via JSON POST payload
    verify_resp = client.post(
        f"/api/feedback/verify/{item.item_id}",
        json={
            "boxes": [
                {"class_name": "Tuberculosis", "box_xywhn": [0.42, 0.42, 0.25, 0.25]}
            ],
            "reviewer": "qa_tester",
            "notes": "Verified bounding box",
        },
    )
    assert verify_resp.status_code == 200, verify_resp.text
    data = verify_resp.json()
    assert data["status"] == "verified"
    assert data["task"] == "detection"
    assert len(data["verified_boxes"]) == 1

    # 2. Trigger HITL retraining for detection
    train_resp = client.post(
        "/api/master/train-hitl",
        params={"model_slug": "test-det-model"},
    )
    assert train_resp.status_code == 202
    train_data = train_resp.json()
    assert train_data["pipeline"] == ["train_detection.py"]


def test_three_button_client_workspace_end_to_end(hitl_env: tuple[Path, Path, Path]) -> None:
    """Test the 3-button Client Studio: class folder uploads, local retrain (Btn 1), local vs global test (Btn 2), and federation (Btn 3)."""
    from agrifed_khaos.catalog import LogicalModelEntry, ModelCatalogManager

    model_p, meta_p, hitl_dir = hitl_env
    models_root = model_p.parent

    # Register test model in catalog inside tmp_path
    catalog_path = models_root / "catalog.json"
    cm = ModelCatalogManager(catalog_path=str(catalog_path), models_dir=str(models_root))
    cm.upsert_entry(
        LogicalModelEntry(
            slug="test-model",
            title="Test Olive Model",
            domain="Olivo",
            architecture="resnet18",
            classes=["sana", "repilo"],
            num_classes=2,
            active_checkpoint=model_p.name,
            versions=[model_p.name],
        )
    )

    app = create_app(str(model_p), str(meta_p))
    app.state.hitl = HITLQueueManager(root_dir=hitl_dir)
    app.state.catalog = cm
    client = TestClient(app)

    # 0. Verify PWA Web App manifest
    manifest_res = client.get("/manifest.json")
    assert manifest_res.status_code == 200
    assert manifest_res.json()["display"] == "standalone"

    # 1. Initial Client Workspace (Cliente 1)
    ws_res = client.get("/api/expert/models/test-model/client-workspace?client_id=cliente_01")
    assert ws_res.status_code == 200
    ws_data = ws_res.json()
    assert ws_data["slug"] == "test-model"
    assert ws_data["client_id"] == "cliente_01"
    assert ws_data["local_model"]["exists"] is False

    # 2. Upload photos into class folders ("sana" and "repilo")
    img_sana = io.BytesIO()
    Image.new("RGB", (64, 64), color=(30, 180, 40)).save(img_sana, format="JPEG")
    img_sana.seek(0)

    img_repilo = io.BytesIO()
    Image.new("RGB", (64, 64), color=(180, 40, 30)).save(img_repilo, format="JPEG")
    img_repilo.seek(0)

    up1 = client.post(
        "/api/expert/models/test-model/upload-class-samples",
        data={"client_id": "cliente_01", "class_name": "sana"},
        files=[("files", ("hoja_sana.jpg", img_sana, "image/jpeg"))],
    )
    assert up1.status_code == 200
    assert up1.json()["uploaded_count"] == 1

    up2 = client.post(
        "/api/expert/models/test-model/upload-class-samples",
        data={"client_id": "cliente_01", "class_name": "repilo"},
        files=[("files", ("hoja_repilo.jpg", img_repilo, "image/jpeg"))],
    )
    assert up2.status_code == 200
    assert up2.json()["uploaded_count"] == 1

    # 3. Button 1: Retrain Local Model for cliente_01
    retrain_res = client.post(
        "/api/expert/models/test-model/retrain-local?client_id=cliente_01&epochs=1"
    )
    assert retrain_res.status_code == 200, retrain_res.text
    retrain_data = retrain_res.json()
    assert retrain_data["success"] is True
    assert retrain_data["local_model"]["num_samples"] >= 2

    # 4. Button 2: Test & Compare Local Model vs Global Model
    test_img = io.BytesIO()
    Image.new("RGB", (64, 64), color=(30, 180, 40)).save(test_img, format="JPEG")
    test_img.seek(0)

    cmp_res = client.post(
        "/api/expert/models/test-model/test-local",
        data={"client_id": "cliente_01"},
        files={"file": ("test_leaf.jpg", test_img, "image/jpeg")},
    )
    assert cmp_res.status_code == 200, cmp_res.text
    cmp_data = cmp_res.json()
    assert cmp_data["has_local_model"] is True
    assert "predicted_class" in cmp_data["global_model"]["result"]
    assert "predicted_class" in cmp_data["local_model"]["result"]

    # 5. Button 3: Federate to Central Service, promote version, and sync back to client
    fed_res = client.post(
        "/api/expert/models/test-model/federate?client_id=cliente_01&include_other_ready_clients=true"
    )
    assert fed_res.status_code == 200, fed_res.text
    fed_data = fed_res.json()
    assert fed_data["success"] is True
    assert fed_data["new_version_id"].startswith("v")
    assert fed_data["client_synced"] is True


def test_user_auth_profile_and_admin_privacy_crud(hitl_env: tuple[Path, Path, Path]) -> None:
    """Test SQLite user DB, roles (admin/expert/user), header profile update, and admin email privacy."""
    from agrifed_khaos.catalog import LogicalModelEntry, ModelCatalogManager

    model_p, meta_p, hitl_dir = hitl_env
    models_root = model_p.parent
    cm = ModelCatalogManager(catalog_path=str(models_root / "catalog.json"), models_dir=str(models_root))
    cm.upsert_entry(
        LogicalModelEntry(
            slug="test-model",
            title="Test Olive Model",
            domain="Olivo",
            architecture="resnet18",
            classes=["sana", "repilo"],
            num_classes=2,
            active_checkpoint=model_p.name,
            versions=[model_p.name],
        )
    )

    app = create_app(str(model_p), str(meta_p))
    app.state.hitl = HITLQueueManager(root_dir=hitl_dir)
    app.state.catalog = cm
    client = TestClient(app)

    # 1. Verify /expert HTML no longer has the synthetic Cliente 1..10 selector
    exp_html = client.get("/models/test-model/expert").text
    assert "clientNodeSelector" not in exp_html
    assert "tenClientsVisualGrid" not in exp_html
    assert "workspaceOwnerBadge" in exp_html

    # 2. Unauthenticated admin call must be rejected with 401
    unauth_list = client.get("/api/admin/users")
    assert unauth_list.status_code == 401

    # 3. Login as seeded admin
    login_admin = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "admin123"},
    )
    assert login_admin.status_code == 200
    admin_me = client.get("/api/auth/me").json()
    assert admin_me["authenticated"] is True
    assert admin_me["user"]["role"] == "admin"
    assert admin_me["user"]["email"] == "admin@khaos.local"  # Self sees own email

    # 4. Admin lists users -> MUST NOT contain 'email' key for any user
    users_res = client.get("/api/admin/users")
    assert users_res.status_code == 200
    users_list = users_res.json()["users"]
    assert len(users_list) >= 3
    for u in users_list:
        assert "email" not in u

    # 5. Admin creates a new Expert user (validating required full_name, email, role)
    create_res = client.post(
        "/api/admin/users",
        json={
            "username": "dra_elena",
            "full_name": "Dra. Elena Gómez",
            "email": "elena.gomez@olivar.es",
            "role": "expert",
        },
    )
    assert create_res.status_code == 201, create_res.text
    created_payload = create_res.json()
    assert "email" not in created_payload["user"]
    elena_id = created_payload["user"]["id"]
    first_pwd = created_payload["generated_password"]
    assert len(first_pwd) >= 8

    # 6. Duplicate email or username must be rejected with 400
    dup_email = client.post(
        "/api/admin/users",
        json={
            "username": "otra_elena",
            "full_name": "Otra Elena",
            "email": "elena.gomez@olivar.es",
            "role": "user",
        },
    )
    assert dup_email.status_code == 400

    # 7. Admin generates a new password for Dra. Elena without seeing her email
    reset_res = client.post(f"/api/admin/users/{elena_id}/reset-password", json={})
    assert reset_res.status_code == 200
    reset_data = reset_res.json()
    assert "email" not in reset_data["user"]
    new_pwd = reset_data["generated_password"]
    assert new_pwd != first_pwd

    # 8. Login as Dra. Elena with the admin-generated password
    client.post("/api/auth/logout")
    login_elena = client.post(
        "/api/auth/login",
        json={"username": "dra_elena", "password": new_pwd},
    )
    assert login_elena.status_code == 200

    # 9. Dra. Elena cannot access Admin user list (403 Forbidden)
    forbidden_res = client.get("/api/admin/users")
    assert forbidden_res.status_code == 403

    # 10. Dra. Elena updates her own username and password from the Header Profile endpoint
    prof_res = client.patch(
        "/api/auth/profile",
        json={
            "username": "elena_experta",
            "current_password": new_pwd,
            "new_password": "MiNuevaPasswordSegura99!",
        },
    )
    assert prof_res.status_code == 200, prof_res.text
    prof_user = prof_res.json()["user"]
    assert prof_user["username"] == "elena_experta"
    assert prof_user["email"] == "elena.gomez@olivar.es"  # Own profile shows email

    # 11. Workspace in /expert automatically uses her authenticated username ('elena_experta')
    ws_elena = client.get("/api/expert/models/test-model/client-workspace")
    assert ws_elena.status_code == 200
    assert ws_elena.json()["client_id"] == "elena_experta"


def test_login_page_and_rbac_view_protection(hitl_env: tuple[Path, Path, Path]) -> None:
    """Test dedicated /login screen and role-based page route access rules."""
    model_p, meta_p, hitl_dir = hitl_env
    models_dir = model_p.parent
    from agrifed_khaos.catalog import LogicalModelEntry, ModelCatalogManager

    cm = ModelCatalogManager(catalog_path=str(models_dir / "catalog.json"), models_dir=str(models_dir))
    cm.upsert_entry(
        LogicalModelEntry(
            slug="test-rbac-model",
            title="RBAC Test Model",
            domain="Olivo",
            architecture="resnet18",
            classes=["sana", "repilo"],
            num_classes=2,
            active_checkpoint=model_p.name,
            versions=[model_p.name],
        )
    )

    app = create_app(str(model_p), str(meta_p))
    app.state.hitl = HITLQueueManager(root_dir=hitl_dir)
    app.state.catalog = cm

    # 1. Unauthenticated client with browser user-agent
    browser_headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
    client = TestClient(app, headers=browser_headers)

    # 1a. GET /login returns 200 with dedicated production login view (no demo cards)
    r_login = client.get("/login")
    assert r_login.status_code == 200
    assert "Iniciar Sesión" in r_login.text
    assert "Crear Cuenta" in r_login.text
    assert "formLogin" in r_login.text
    assert "formRegister" in r_login.text
    assert "roleCard_admin" not in r_login.text
    assert "PERFILES DE ACCESO DISPONIBLES" not in r_login.text
    assert "Matriz de Accesos por Rol" not in r_login.text

    # 1b. Unauthenticated browser accessing /expert, /master or /accounts is redirected to /login
    r_unauth_exp = client.get("/expert", follow_redirects=False)
    assert r_unauth_exp.status_code == 303
    assert r_unauth_exp.headers["location"] == "/login?next=/expert"

    r_unauth_master = client.get("/master", follow_redirects=False)
    assert r_unauth_master.status_code == 303
    assert r_unauth_master.headers["location"] == "/login?next=/master"

    r_unauth_accounts = client.get("/accounts", follow_redirects=False)
    assert r_unauth_accounts.status_code == 303
    assert r_unauth_accounts.headers["location"] == "/login?next=/accounts"

    # 1c. Agnostic views are accessible without authentication, with header mount slot
    r_root = client.get("/")
    assert r_root.status_code == 200
    assert "khHeaderProfileMount" in r_root.text

    r_consumer = client.get("/models/test-rbac-model")
    assert r_consumer.status_code == 200
    assert "khHeaderProfileMount" in r_consumer.text

    # 2. Login as 'usuario' (Agnostic User)
    login_user = client.post("/api/auth/login", json={"username": "usuario", "password": "usuario123"})
    assert login_user.status_code == 200

    # User can access models and consume them
    assert client.get("/").status_code == 200
    assert client.get("/models/test-rbac-model").status_code == 200

    # User CANNOT access /expert, /master or /accounts (403 Forbidden with Access Denied page)
    r_user_exp = client.get("/expert")
    assert r_user_exp.status_code == 403
    assert "Acceso Restringido" in r_user_exp.text
    assert "@usuario" in r_user_exp.text

    r_user_master = client.get("/master")
    assert r_user_master.status_code == 403
    assert "Acceso Restringido" in r_user_master.text

    r_user_acc = client.get("/accounts")
    assert r_user_acc.status_code == 403
    assert "Acceso Restringido" in r_user_acc.text

    # 3. Login as 'experto' (Agronomist Expert)
    client.post("/api/auth/logout")
    login_expert = client.post("/api/auth/login", json={"username": "experto", "password": "experto123"})
    assert login_expert.status_code == 200

    # Expert can access models and /expert
    assert client.get("/").status_code == 200
    assert client.get("/models/test-rbac-model").status_code == 200
    r_exp_page = client.get("/expert")
    assert r_exp_page.status_code == 200

    # Expert CANNOT access /master or /accounts (403 Forbidden)
    r_exp_master = client.get("/master")
    assert r_exp_master.status_code == 403
    assert "Acceso Restringido" in r_exp_master.text

    r_exp_acc = client.get("/accounts")
    assert r_exp_acc.status_code == 403
    assert "Acceso Restringido" in r_exp_acc.text

    # 4. Login as 'admin' (Administrator)
    client.post("/api/auth/logout")
    login_adm = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    assert login_adm.status_code == 200

    # Admin can access everything
    assert client.get("/").status_code == 200
    assert client.get("/models/test-rbac-model").status_code == 200
    assert client.get("/expert").status_code == 200
    assert client.get("/master").status_code == 200
    assert client.get("/accounts").status_code == 200


def test_user_registration_approval_workflow_and_policies(
    hitl_env: tuple[Path, Path, Path],
) -> None:
    """Test user self-registration, admin pending approval, banning, and registration policies."""
    model_p, meta_p, hitl_dir = hitl_env
    app = create_app(str(model_p), str(meta_p))
    app.state.hitl = HITLQueueManager(root_dir=hitl_dir)
    client = TestClient(app)

    # 1. Self-register a new user (default policy requires admin approval)
    reg_res = client.post(
        "/api/auth/register",
        json={
            "full_name": "Dr. Carlos Agrónomo",
            "username": "carlos_agri",
            "email": "carlos@finca-olivos.es",
            "password": "PasswordSegura2026!",
        },
    )
    assert reg_res.status_code == 201, reg_res.text
    reg_data = reg_res.json()
    assert reg_data["success"] is True
    assert reg_data["requires_approval"] is True
    assert "pendiente de aprobación" in reg_data["message"]

    # 2. Attempt login as pending user -> must be rejected with 401 informing pending status
    pend_login = client.post(
        "/api/auth/login",
        json={"username": "carlos_agri", "password": "PasswordSegura2026!"},
    )
    assert pend_login.status_code == 401
    assert "pendiente de validación" in pend_login.json()["detail"]

    # 3. Admin logs in and checks pending accounts
    client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    pending_res = client.get("/api/admin/accounts/pending")
    assert pending_res.status_code == 200
    pending_list = pending_res.json()["pending_users"]
    carlos_entry = next((u for u in pending_list if u["username"] == "carlos_agri"), None)
    assert carlos_entry is not None
    assert carlos_entry["status"] == "pending"
    carlos_id = carlos_entry["id"]

    # 4. Admin approves Carlos and assigns role 'expert'
    approve_res = client.post(
        f"/api/admin/accounts/pending/{carlos_id}/approve",
        json={"role": "expert"},
    )
    assert approve_res.status_code == 200
    assert approve_res.json()["user"]["role"] == "expert"
    assert approve_res.json()["user"]["status"] == "active"

    # 5. Carlos can now log in successfully and access /expert
    client.post("/api/auth/logout")
    carlos_login = client.post(
        "/api/auth/login",
        json={"username": "carlos_agri", "password": "PasswordSegura2026!"},
    )
    assert carlos_login.status_code == 200
    carlos_user = carlos_login.json()["user"]
    assert carlos_user["role"] == "expert"

    # Carlos has access to /expert
    assert client.get("/expert").status_code == 200

    # 6. Admin bans Carlos account
    client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    ban_res = client.post(f"/api/admin/accounts/{carlos_id}/ban")
    assert ban_res.status_code == 200
    assert ban_res.json()["user"]["status"] == "banned"

    # 7. Carlos can no longer log in (401 Account Banned)
    client.post("/api/auth/logout")
    banned_login = client.post(
        "/api/auth/login",
        json={"username": "carlos_agri", "password": "PasswordSegura2026!"},
    )
    assert banned_login.status_code == 401
    assert "suspendida o bloqueada" in banned_login.json()["detail"]

    # 8. Admin configures banned email wildcard domain pattern
    client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    add_ban_res = client.post(
        "/api/admin/banned-entries",
        json={"pattern": "@tempmail.com", "reason": "Dominio temporal no verificado"},
    )
    assert add_ban_res.status_code == 201

    # Attempting to register with banned domain is blocked with 400
    client.post("/api/auth/logout")
    spammer_reg = client.post(
        "/api/auth/register",
        json={
            "full_name": "Spam User",
            "username": "spammer99",
            "email": "hacker@tempmail.com",
            "password": "Password123!",
        },
    )
    assert spammer_reg.status_code == 400
    assert "no está permitido" in spammer_reg.json()["detail"]

    # 9. Admin modifies policy to allow auto-approval
    client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    pol_update = client.post(
        "/api/admin/policies",
        json={"require_admin_approval": False},
    )
    assert pol_update.status_code == 200
    assert pol_update.json()["policies"]["require_admin_approval"] is False

    # New user registers and is active immediately
    client.post("/api/auth/logout")
    auto_reg = client.post(
        "/api/auth/register",
        json={
            "full_name": "Ana Directa",
            "username": "ana_directa",
            "email": "ana@cooperativa.es",
            "password": "PasswordDirecta123!",
        },
    )
    assert auto_reg.status_code == 201
    assert auto_reg.json()["requires_approval"] is False

    # Ana logs in immediately without waiting
    ana_login = client.post(
        "/api/auth/login",
        json={"username": "ana_directa", "password": "PasswordDirecta123!"},
    )
    assert ana_login.status_code == 200


def test_hitl_multi_expert_voting_and_consensus(hitl_env: tuple[Path, Path, Path]) -> None:
    """Test multi-expert voting traceability, dispute detection, consensus, and consolidation."""
    model_path, meta_path, hitl_dir = hitl_env
    app = create_app(str(model_path), str(meta_path))
    hitl = HITLQueueManager(root_dir=hitl_dir)
    app.state.hitl = hitl

    client = TestClient(app)

    # 1. Enqueue a pending item
    dummy_img = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 200
    meta = hitl.enqueue_sync(
        image_bytes=dummy_img,
        filename="hoja_ambigua.jpg",
        model_name="mangofruit.pt",
        model_slug="mangofruit-v1-0",
        predicted_class="antracnosis",
        confidence=0.54,
    )
    item_id = meta.item_id

    # Initial check: 0 votes -> consensus_status == "none"
    pending_res = client.get("/api/feedback/pending?model_slug=mangofruit-v1-0")
    assert pending_res.status_code == 200
    items = pending_res.json()["items"]
    target_item = next(it for it in items if it["item_id"] == item_id)
    assert target_item["votes_count"] == 0
    assert target_item["consensus_status"] == "none"

    # 2. Expert 1 votes "sana"
    vote1_res = client.post(
        f"/api/feedback/vote/{item_id}",
        json={
            "reviewer": "agronomo_jaen",
            "decision": "verified",
            "class_name": "sana",
            "notes": "Lesión superficial no fúngica",
        },
    )
    assert vote1_res.status_code == 200
    assert vote1_res.json()["votes_count"] == 1

    # Check consensus after 1 vote -> consensus on "sana"
    pending_res = client.get("/api/feedback/pending?model_slug=mangofruit-v1-0")
    target_item = next(it for it in pending_res.json()["items"] if it["item_id"] == item_id)
    assert target_item["votes_count"] == 1
    assert target_item["consensus_status"] == "consensus"
    assert "sana" in target_item["consensus_label"]

    # 3. Expert 2 votes "antracnosis" -> Dispute / En Discusión
    vote2_res = client.post(
        f"/api/feedback/vote/{item_id}",
        json={
            "reviewer": "patologo_cordoba",
            "decision": "verified",
            "class_name": "antracnosis",
            "notes": "Borde necrótico característico",
        },
    )
    assert vote2_res.status_code == 200
    assert vote2_res.json()["votes_count"] == 2

    pending_res = client.get("/api/feedback/pending?model_slug=mangofruit-v1-0")
    target_item = next(it for it in pending_res.json()["items"] if it["item_id"] == item_id)
    assert target_item["votes_count"] == 2
    assert target_item["consensus_status"] == "disputed"
    assert "En Discusión" in target_item["consensus_label"]

    # 4. Expert 3 votes "antracnosis", and Expert 1 updates their vote to "antracnosis"
    client.post(
        f"/api/feedback/vote/{item_id}",
        json={
            "reviewer": "experto_sevilla",
            "decision": "verified",
            "class_name": "antracnosis",
        },
    )
    client.post(
        f"/api/feedback/vote/{item_id}",
        json={
            "reviewer": "agronomo_jaen",
            "decision": "verified",
            "class_name": "antracnosis",
            "notes": "Rectifico tras analizar con lupa",
        },
    )

    pending_res = client.get("/api/feedback/pending?model_slug=mangofruit-v1-0")
    target_item = next(it for it in pending_res.json()["items"] if it["item_id"] == item_id)
    assert target_item["votes_count"] == 3
    assert target_item["consensus_status"] == "consensus"
    assert "antracnosis" in target_item["consensus_label"]

    # 5. Consolidate to verified dataset
    ver_res = client.post(
        f"/api/feedback/verify/{item_id}?verified_class=antracnosis",
        json={"reviewer": "agronomo_jaen"},
    )
    assert ver_res.status_code == 200
    assert ver_res.json()["status"] == "verified"
    assert ver_res.json()["verified_class"] == "antracnosis"
    # Preserves vote traceability in verified item metadata
    assert len(ver_res.json()["votes"]) == 3









