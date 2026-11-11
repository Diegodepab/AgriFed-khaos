"""Automated tests for Expert Model Personal Versioning and Pinning (Lock) in AgriFed-KHAOS.

Tests cover:
1. POST /api/expert/models/{slug}/saved-versions (Snapshot creation with custom name and weights)
2. GET /api/expert/models/{slug}/saved-versions (Listing expert versions sorted by date)
3. PUT /api/expert/models/{slug}/saved-versions/{id}/pin (Toggling pinned/locked state)
4. POST /api/expert/models/{slug}/saved-versions/{id}/activate (Switching active version)
5. GET /api/expert/models/{slug}/saved-versions/{id}/download-pt (.pt export)
6. DELETE /api/expert/models/{slug}/saved-versions/{id} (Version deletion)
"""

from __future__ import annotations

import io
import json

import pytest
import torch
from fastapi.testclient import TestClient

from agrifed_khaos.crypto import compute_sha256
from agrifed_khaos.models import build_model, load_checkpoint_bundle, save_checkpoint_bundle
from agrifed_khaos.wrappers import freeze_bn_stats
from scripts.serve import create_app


@pytest.fixture(scope="module")
def sample_expert_model_env(tmp_path_factory):
    """Create test model and directory environment."""
    work_dir = tmp_path_factory.mktemp("expert_versioning_env")
    ckpt_path = work_dir / "base_olive_test.pt"
    meta_path = work_dir / "base_olive_test.meta.json"

    arch = "resnet18"
    classes = ["aceituna", "defectuosa", "desenfocada", "envero", "negra", "sana", "verde"]
    num_classes = len(classes)

    model = build_model(architecture=arch, num_classes=num_classes)
    model.eval()
    freeze_bn_stats(model)

    meta = {
        "architecture": arch,
        "num_classes": num_classes,
        "classes": classes,
        "input_size": [224, 224],
        "training_scope": "full",
        "task": "classification",
        "active_checkpoint": "base_olive_test.pt",
    }

    save_checkpoint_bundle(ckpt_path, model.state_dict(), meta)
    meta_path.write_text(json.dumps(meta, indent=2))

    # Also generate a catalog.json in work_dir so catalog manager discovers it
    catalog_path = work_dir / "catalog.json"
    catalog_data = {
        "models": [
            {
                "slug": "olive-test-model",
                "title": "Olive Test Model",
                "domain": "Olivo",
                "task": "classification",
                "architecture": arch,
                "classes": classes,
                "active_checkpoint": "base_olive_test.pt",
                "checkpoints": ["base_olive_test.pt"],
            }
        ]
    }
    catalog_path.write_text(json.dumps(catalog_data, indent=2))

    return {
        "ckpt_path": ckpt_path,
        "meta_path": meta_path,
        "work_dir": work_dir,
        "arch": arch,
        "classes": classes,
        "num_classes": num_classes,
        "sha256": compute_sha256(ckpt_path),
        "slug": "olive-test-model",
    }


def test_expert_save_and_list_version_snapshots(sample_expert_model_env, monkeypatch):
    """Test creating snapshots with custom semantic names and listing them."""
    app = create_app(
        str(sample_expert_model_env["ckpt_path"]),
        str(sample_expert_model_env["meta_path"]),
    )
    client = TestClient(app)
    slug = sample_expert_model_env["slug"]

    # Read base head weights
    base_state, _ = load_checkpoint_bundle(sample_expert_model_env["ckpt_path"], device="cpu")
    w = base_state["fc.weight"].numpy().tolist()
    b = base_state["fc.bias"].numpy().tolist()

    # 1. Save version snapshot 1
    save_payload_1 = {
        "custom_name": "v1.1 • Muestreo Parcela Norte (45 fotos)",
        "description": "Fotos de hojas jóvenes con síntomas tempranos de repilo.",
        "base_model_sha256": sample_expert_model_env["sha256"],
        "num_samples": 45,
        "accuracy": 92.5,
        "final_loss": 0.2150,
        "delta_norm": 0.0420,
        "classes": sample_expert_model_env["classes"],
        "head_weights": {"weight": w, "bias": b},
        "is_pinned": True,
    }

    res1 = client.post(f"/api/expert/models/{slug}/saved-versions", json=save_payload_1)
    assert res1.status_code == 201, res1.text
    data1 = res1.json()
    assert data1["custom_name"] == "v1.1 • Muestreo Parcela Norte (45 fotos)"
    assert data1["num_samples"] == 45
    assert data1["is_pinned"] is True
    assert data1["is_active"] is True
    assert "version_id" in data1

    # 2. Save version snapshot 2
    save_payload_2 = {
        "custom_name": "v1.2 • Parcela Sur Soleada (80 fotos)",
        "description": "Muestreo bajo luz directa a mediodía.",
        "base_model_sha256": sample_expert_model_env["sha256"],
        "num_samples": 80,
        "accuracy": 95.0,
        "final_loss": 0.1520,
        "delta_norm": 0.0610,
        "classes": sample_expert_model_env["classes"],
        "head_weights": {"weight": w, "bias": b},
        "is_pinned": False,
    }

    res2 = client.post(f"/api/expert/models/{slug}/saved-versions", json=save_payload_2)
    assert res2.status_code == 201, res2.text
    data2 = res2.json()
    assert data2["is_pinned"] is False
    assert data2["is_active"] is True
    v2_id = data2["version_id"]

    # 3. List saved versions
    list_res = client.get(f"/api/expert/models/{slug}/saved-versions")
    assert list_res.status_code == 200
    versions = list_res.json()
    assert len(versions) >= 2

    # Verify latest is at index 0 and marked active
    assert versions[0]["version_id"] == v2_id
    assert versions[0]["is_active"] is True


def test_expert_toggle_pin_version(sample_expert_model_env):
    """Test toggling pinned/locked state on an expert version."""
    app = create_app(
        str(sample_expert_model_env["ckpt_path"]),
        str(sample_expert_model_env["meta_path"]),
    )
    client = TestClient(app)
    slug = sample_expert_model_env["slug"]

    # Save a version
    base_state, _ = load_checkpoint_bundle(sample_expert_model_env["ckpt_path"], device="cpu")
    w = base_state["fc.weight"].numpy().tolist()
    b = base_state["fc.bias"].numpy().tolist()

    save_res = client.post(
        f"/api/expert/models/{slug}/saved-versions",
        json={
            "custom_name": "Test Pinning Model",
            "base_model_sha256": sample_expert_model_env["sha256"],
            "num_samples": 20,
            "head_weights": {"weight": w, "bias": b},
            "is_pinned": False,
        },
    )
    v_id = save_res.json()["version_id"]

    # Toggle to True
    pin_res = client.put(
        f"/api/expert/models/{slug}/saved-versions/{v_id}/pin",
        json={"is_pinned": True},
    )
    assert pin_res.status_code == 200
    assert pin_res.json()["is_pinned"] is True

    # Verify in list
    v_list = client.get(f"/api/expert/models/{slug}/saved-versions").json()
    target = next(v for v in v_list if v["version_id"] == v_id)
    assert target["is_pinned"] is True

    # Toggle to False
    unpin_res = client.put(
        f"/api/expert/models/{slug}/saved-versions/{v_id}/pin",
        json={"is_pinned": False},
    )
    assert unpin_res.status_code == 200
    assert unpin_res.json()["is_pinned"] is False


def test_expert_activate_version(sample_expert_model_env):
    """Test switching active version among saved snapshots."""
    app = create_app(
        str(sample_expert_model_env["ckpt_path"]),
        str(sample_expert_model_env["meta_path"]),
    )
    client = TestClient(app)
    slug = sample_expert_model_env["slug"]

    base_state, _ = load_checkpoint_bundle(sample_expert_model_env["ckpt_path"], device="cpu")
    w = base_state["fc.weight"].numpy().tolist()
    b = base_state["fc.bias"].numpy().tolist()

    res_a = client.post(
        f"/api/expert/models/{slug}/saved-versions",
        json={
            "custom_name": "Version A",
            "base_model_sha256": sample_expert_model_env["sha256"],
            "head_weights": {"weight": w, "bias": b},
        },
    )
    v_a = res_a.json()["version_id"]

    res_b = client.post(
        f"/api/expert/models/{slug}/saved-versions",
        json={
            "custom_name": "Version B",
            "base_model_sha256": sample_expert_model_env["sha256"],
            "head_weights": {"weight": w, "bias": b},
        },
    )
    v_b = res_b.json()["version_id"]

    # Version B is initially active
    list1 = client.get(f"/api/expert/models/{slug}/saved-versions").json()
    rec_a = next(v for v in list1 if v["version_id"] == v_a)
    rec_b = next(v for v in list1 if v["version_id"] == v_b)
    assert rec_b["is_active"] is True
    assert rec_a["is_active"] is False

    # Activate Version A
    act_res = client.post(f"/api/expert/models/{slug}/saved-versions/{v_a}/activate")
    assert act_res.status_code == 200

    list2 = client.get(f"/api/expert/models/{slug}/saved-versions").json()
    rec_a2 = next(v for v in list2 if v["version_id"] == v_a)
    rec_b2 = next(v for v in list2 if v["version_id"] == v_b)
    assert rec_a2["is_active"] is True
    assert rec_b2["is_active"] is False


def test_expert_download_pt_checkpoint_bundle(sample_expert_model_env):
    """Test exporting and downloading full PyTorch .pt bundle for a saved version."""
    app = create_app(
        str(sample_expert_model_env["ckpt_path"]),
        str(sample_expert_model_env["meta_path"]),
    )
    client = TestClient(app)
    slug = sample_expert_model_env["slug"]

    # Slightly modify weights so we can verify exact weight injection
    base_state, _ = load_checkpoint_bundle(sample_expert_model_env["ckpt_path"], device="cpu")
    w = (base_state["fc.weight"] + 0.1234).numpy().tolist()
    b = (base_state["fc.bias"] - 0.4321).numpy().tolist()

    save_res = client.post(
        f"/api/expert/models/{slug}/saved-versions",
        json={
            "custom_name": "Exportable Test Version",
            "version_tag": "v1.99-custom",
            "base_model_sha256": sample_expert_model_env["sha256"],
            "num_samples": 77,
            "accuracy": 88.8,
            "head_weights": {"weight": w, "bias": b},
        },
    )
    v_id = save_res.json()["version_id"]

    # Download .pt bundle
    dl_res = client.get(f"/api/expert/models/{slug}/saved-versions/{v_id}/download-pt")
    assert dl_res.status_code == 200, dl_res.text
    assert dl_res.headers["content-type"] == "application/octet-stream"
    assert "attachment" in dl_res.headers.get("content-disposition", "")

    # Load downloaded bytes with torch
    buf = io.BytesIO(dl_res.content)
    ckpt = torch.load(buf, map_location="cpu", weights_only=False)

    assert "state_dict" in ckpt
    assert "metadata" in ckpt
    meta = ckpt["metadata"]
    assert meta["expert_version_tag"] == "v1.99-custom"
    assert meta["expert_custom_name"] == "Exportable Test Version"
    assert meta["num_samples"] == 77
    assert meta["accuracy"] == 88.8

    # Verify injected head weights
    state = ckpt["state_dict"]
    assert torch.allclose(state["fc.weight"], torch.tensor(w, dtype=torch.float32), atol=1e-5)
    assert torch.allclose(state["fc.bias"], torch.tensor(b, dtype=torch.float32), atol=1e-5)


def test_expert_delete_version_snapshot(sample_expert_model_env):
    """Test deleting a saved version snapshot."""
    app = create_app(
        str(sample_expert_model_env["ckpt_path"]),
        str(sample_expert_model_env["meta_path"]),
    )
    client = TestClient(app)
    slug = sample_expert_model_env["slug"]

    base_state, _ = load_checkpoint_bundle(sample_expert_model_env["ckpt_path"], device="cpu")
    w = base_state["fc.weight"].numpy().tolist()
    b = base_state["fc.bias"].numpy().tolist()

    res = client.post(
        f"/api/expert/models/{slug}/saved-versions",
        json={
            "custom_name": "Temporary Version to Delete",
            "base_model_sha256": sample_expert_model_env["sha256"],
            "head_weights": {"weight": w, "bias": b},
        },
    )
    v_id = res.json()["version_id"]

    # Delete version
    del_res = client.delete(f"/api/expert/models/{slug}/saved-versions/{v_id}")
    assert del_res.status_code == 200
    assert del_res.json()["deleted_id"] == v_id

    # Verify not in list
    v_list = client.get(f"/api/expert/models/{slug}/saved-versions").json()
    assert not any(v["version_id"] == v_id for v in v_list)

    # Deleting again returns 404
    del_again = client.delete(f"/api/expert/models/{slug}/saved-versions/{v_id}")
    assert del_again.status_code == 404


def test_expert_delete_client_workspace_permanent(sample_expert_model_env):
    """Test permanently deleting a client workspace without temporary trash storage."""
    app = create_app(
        str(sample_expert_model_env["ckpt_path"]),
        str(sample_expert_model_env["meta_path"]),
    )
    client = TestClient(app)
    slug = sample_expert_model_env["slug"]

    # Delete client workspace
    del_res = client.delete(f"/api/expert/models/{slug}/client-workspace?client_id=experto_01")
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True


def test_expert_move_class_sample(sample_expert_model_env):
    """Test moving a sample photo between class folders in the client workspace."""
    from PIL import Image

    app = create_app(
        str(sample_expert_model_env["ckpt_path"]),
        str(sample_expert_model_env["meta_path"]),
    )
    client = TestClient(app)
    slug = sample_expert_model_env["slug"]

    # 1. Create a dummy test image
    img = Image.new("RGB", (32, 32), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    # 2. Upload image to class "sana"
    upload_res = client.post(
        f"/api/expert/models/{slug}/upload-class-samples",
        data={"class_name": "sana", "client_id": "test_user"},
        files={"files": ("test_olive.jpg", buf.getvalue(), "image/jpeg")},
    )
    assert upload_res.status_code == 200
    assert upload_res.json()["uploaded_count"] == 1

    # 3. Find the uploaded filename from client workspace
    ws_res = client.get(f"/api/expert/models/{slug}/client-workspace?client_id=test_user")
    assert ws_res.status_code == 200
    sana_cls = next(c for c in ws_res.json()["classes"] if c["name"] == "sana")
    assert len(sana_cls["samples"]) >= 1
    fn = sana_cls["samples"][0]["filename"]

    # 4. Move sample from "sana" to "defectuosa"
    move_res = client.post(
        f"/api/expert/models/{slug}/move-class-sample?from_class=sana&to_class=defectuosa&filename={fn}&client_id=test_user"
    )
    assert move_res.status_code == 200
    assert move_res.json()["moved_to"] == "defectuosa"

    # 5. Verify sample is now in "defectuosa"
    ws_after = client.get(f"/api/expert/models/{slug}/client-workspace?client_id=test_user").json()
    defec_cls = next(c for c in ws_after["classes"] if c["name"] == "defectuosa")
    assert any(s["filename"] == fn for s in defec_cls["samples"])


def test_expert_clear_client_samples_frees_storage(sample_expert_model_env):
    """Test clearing all client local samples to free storage while preserving workspace."""
    from PIL import Image

    app = create_app(
        str(sample_expert_model_env["ckpt_path"]),
        str(sample_expert_model_env["meta_path"]),
    )
    client = TestClient(app)
    slug = sample_expert_model_env["slug"]

    # 1. Upload sample
    img = Image.new("RGB", (32, 32), color=(50, 100, 150))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    client.post(
        f"/api/expert/models/{slug}/upload-class-samples",
        data={"class_name": "envero", "client_id": "test_user_free"},
        files={"files": ("sample_to_clear.jpg", buf.getvalue(), "image/jpeg")},
    )

    # Verify sample is counted
    ws_before = client.get(f"/api/expert/models/{slug}/client-workspace?client_id=test_user_free").json()
    assert ws_before["total_samples"] >= 1

    # 2. Clear all samples to free space
    clear_res = client.delete(f"/api/expert/models/{slug}/clear-samples?client_id=test_user_free")
    assert clear_res.status_code == 200
    assert clear_res.json()["success"] is True
    assert clear_res.json()["removed"] >= 1

    # 3. Verify total samples is 0 and storage is liberated
    ws_after = client.get(f"/api/expert/models/{slug}/client-workspace?client_id=test_user_free").json()
    assert ws_after["total_samples"] == 0


def test_expert_benchmark_test_samples_and_evaluation(sample_expert_model_env):
    """Test retrieving server test benchmark samples and running real-time evaluation comparing Base vs Local."""
    from PIL import Image

    work_dir = sample_expert_model_env["work_dir"]
    slug = sample_expert_model_env["slug"]

    # 1. Prepare simulated test dataset folder structure
    ds_dir = work_dir / "dataset"
    test_dir = ds_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    for cls_name in ["sana", "defectuosa"]:
        cls_p = test_dir / cls_name
        cls_p.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (64, 64), color=(120, 180, 70))
        img.save(cls_p / f"{cls_name}_01.jpg", format="JPEG")

    # Update metadata to reference dataset source
    meta_path = work_dir / f"{slug}.meta.json"
    meta_content = {
        "architecture": sample_expert_model_env["arch"],
        "num_classes": sample_expert_model_env["num_classes"],
        "classes": sample_expert_model_env["classes"],
        "dataset_source": str(ds_dir / "train"),
    }
    meta_path.write_text(json.dumps(meta_content, indent=2))

    app = create_app(
        str(sample_expert_model_env["ckpt_path"]),
        str(sample_expert_model_env["meta_path"]),
    )
    client = TestClient(app)

    # 2. Test GET /benchmark-test-samples
    samples_res = client.get(f"/api/expert/models/{slug}/benchmark-test-samples")
    assert samples_res.status_code == 200
    s_data = samples_res.json()
    assert s_data["success"] is True
    assert s_data["total_samples"] >= 2
    assert len(s_data["samples"]) >= 2
    sample0 = s_data["samples"][0]
    assert "true_class" in sample0
    assert "image_url" in sample0

    # 3. Test GET /benchmark-image
    img_res = client.get(sample0["image_url"])
    assert img_res.status_code == 200
    assert img_res.headers["content-type"] in ("image/jpeg", "image/png", "application/octet-stream")

    # 4. Test POST /run-benchmark-evaluation with client head weights
    base_state, _ = load_checkpoint_bundle(sample_expert_model_env["ckpt_path"], device="cpu")
    w = base_state["fc.weight"].numpy().tolist()
    b = base_state["fc.bias"].numpy().tolist()

    eval_res = client.post(
        f"/api/expert/models/{slug}/run-benchmark-evaluation?client_id=test_bench_user",
        json={"head_weights": {"weight": w, "bias": b}},
    )
    assert eval_res.status_code == 200
    e_data = eval_res.json()
    assert e_data["success"] is True
    assert e_data["total_samples"] >= 2
    assert "base_accuracy" in e_data
    assert "local_accuracy" in e_data
    assert "retention_rate" in e_data
    assert "summary" in e_data
    assert len(e_data["samples"]) >= 2
    for item in e_data["samples"]:
        assert item["status"] in ("STABLE", "IMPROVED", "REGRESSED", "BOTH_WRONG")

    # 5. Test model with no test directory returns graceful response
    res_no_test = client.post(
        "/api/expert/models/non-existent-model/run-benchmark-evaluation",
        json={},
    )
    # _resolve_model_entry_and_base may raise 404 or return success: false
    assert res_no_test.status_code in (404, 200)
    if res_no_test.status_code == 200:
        assert res_no_test.json()["success"] is False

