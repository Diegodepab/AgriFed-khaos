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
    """Test JobManager execution, conflict rejection (HTTP 409), and status polling."""
    import asyncio

    from scripts.serve import JobManager

    jm = JobManager()

    async def _test_flow() -> None:
        assert not jm.is_running
        status = jm.get_status()
        assert status["is_running"] is False

        # Launch quick command
        test_cmd = [
            "python3",
            "-c",
            "import time; print('hello from test'); time.sleep(0.2)",
        ]
        await jm.run_command("test_echo", test_cmd)
        assert jm.is_running
        st_running = jm.get_status()
        assert st_running["is_running"] is True
        assert st_running["action"] == "test_echo"

        # Attempt concurrent run -> must raise RuntimeError
        with pytest.raises(RuntimeError, match="already actively running"):
            await jm.run_command("conflict", ["echo", "fail"])

        # Wait for termination
        for _ in range(20):
            await asyncio.sleep(0.05)
            if not jm.is_running:
                break

        assert not jm.is_running
        assert jm.last_exit_code == 0
        logs = "\n".join(jm.logs)
        assert "hello from test" in logs

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



