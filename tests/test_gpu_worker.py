"""Tests for the standalone GPU Compute Worker (agrifed-gpu) and Master GPU bridge."""

from __future__ import annotations

import io
import json
import tarfile
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agrifed_khaos.api.mlops import router as mlops_router
from agrifed_khaos.gpu_worker import (
    GPU_CACHE_DIR,
    MODELS_DIR,
    _safe_extract,
    create_gpu_worker_app,
    prune_dataset_cache,
)


def test_gpu_worker_health_endpoint() -> None:
    """Verify /health reports service status, GPU telemetry, and FIFO queue state."""
    app = create_gpu_worker_app()
    client = TestClient(app)

    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["service"] == "agrifed-gpu"
    assert "cuda_available" in data
    assert "device_name" in data
    assert "vram_total_gb" in data
    assert data["is_busy"] is False
    assert data["queue_length"] == 0


def test_gpu_worker_upload_dataset_and_fifo_queue(tmp_path: Path) -> None:
    """Verify dataset archive upload and FIFO queue behavior on /jobs/train."""
    app = create_gpu_worker_app()
    client = TestClient(app)

    # Create a tiny in-memory .tar dataset
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w") as tf:
        info = tarfile.TarInfo(name="class_a/sample.txt")
        payload = b"dummy"
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    tar_buf.seek(0)

    up_resp = client.post(
        "/jobs/upload-dataset",
        data={"dataset_id": "test_ds_gpu_01"},
        files={"archive": ("test_ds_gpu_01.tar", tar_buf, "application/x-tar")},
    )
    assert up_resp.status_code == 200
    up_data = up_resp.json()
    assert up_data["status"] == "uploaded"
    assert (GPU_CACHE_DIR / "test_ds_gpu_01" / "class_a" / "sample.txt").is_file()

    try:
        # Mock JobManager.enqueue_or_run to verify command construction
        with patch.object(
            app.state.job_manager,
            "enqueue_or_run",
            new=AsyncMock(
                side_effect=[
                    {
                        "status": "started",
                        "job_id": "j1",
                        "action": "gpu_train:wheat-yolo",
                        "queue_position": 0,
                    },
                    {
                        "status": "queued",
                        "job_id": "j2",
                        "action": "gpu_train:leaf-cls",
                        "queue_position": 1,
                    },
                ]
            ),
        ) as mock_enqueue:
            r1 = client.post(
                "/jobs/train",
                json={
                    "slug": "wheat-yolo",
                    "task": "detection",
                    "dataset_path": "test_ds_gpu_01",
                    "architecture": "yolov8n",
                    "epochs": 5,
                },
            )
            assert r1.status_code == 200
            assert r1.json()["status"] == "started"
            assert r1.json()["task"] == "detection"

            r2 = client.post(
                "/jobs/train",
                json={
                    "slug": "leaf-cls",
                    "task": "classification",
                    "dataset_path": "test_ds_gpu_01",
                    "architecture": "resnet18",
                    "epochs": 5,
                },
            )
            assert r2.status_code == 200
            assert r2.json()["status"] == "queued"
            assert mock_enqueue.call_count == 2
    finally:
        import shutil

        shutil.rmtree(GPU_CACHE_DIR / "test_ds_gpu_01", ignore_errors=True)


def test_gpu_worker_artifact_retrieval() -> None:
    """Verify artifact listing, download, and path traversal protection."""
    app = create_gpu_worker_app()
    client = TestClient(app)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    slug = "testgpuart"
    pt_file = MODELS_DIR / f"{slug}_base.pt"
    meta_file = MODELS_DIR / f"{slug}.meta.json"
    pt_file.write_bytes(b"FAKE_PT_WEIGHTS")
    meta_file.write_text(json.dumps({"arch": "yolov8n", "task": "detection"}), encoding="utf-8")

    try:
        list_resp = client.get(f"/jobs/{slug}/artifacts")
        assert list_resp.status_code == 200
        list_data = list_resp.json()
        assert list_data["slug"] == slug
        assert len(list_data["artifacts"]) >= 2
        assert list_data["metadata"]["arch"] == "yolov8n"

        dl_resp = client.get(f"/jobs/{slug}/artifacts/{slug}_base.pt")
        assert dl_resp.status_code == 200
        assert dl_resp.content == b"FAKE_PT_WEIGHTS"

        # Unauthorized file access must return 403
        forbidden_resp = client.get(f"/jobs/{slug}/artifacts/other_model_base.pt")
        assert forbidden_resp.status_code == 403
    finally:
        pt_file.unlink(missing_ok=True)
        meta_file.unlink(missing_ok=True)


def test_gpu_worker_dataset_exists_and_delete() -> None:
    """Verify GET and DELETE on /datasets/{dataset_id}."""
    app = create_gpu_worker_app()
    client = TestClient(app)

    test_ds = GPU_CACHE_DIR / "ds_probe_test"
    test_ds.mkdir(parents=True, exist_ok=True)
    (test_ds / "dummy.txt").write_text("hello", encoding="utf-8")

    try:
        r_get = client.get("/datasets/ds_probe_test")
        assert r_get.status_code == 200
        assert r_get.json()["cached"] is True

        r_del = client.delete("/datasets/ds_probe_test")
        assert r_del.status_code == 200
        assert r_del.json()["deleted"] is True
        assert not test_ds.exists()
    finally:
        if test_ds.exists():
            import shutil

            shutil.rmtree(test_ds, ignore_errors=True)


def test_gpu_worker_safe_extract_zip_traversal(tmp_path: Path) -> None:
    """Verify safe extract rejects paths attempting directory traversal."""
    from fastapi import HTTPException

    target = tmp_path / "target"
    target.mkdir()

    # Create zip with ../../escape.txt
    bad_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("../../escape.txt", "pwn")

    with pytest.raises(HTTPException) as exc_info:
        _safe_extract(bad_zip, "bad.zip", target)
    assert exc_info.value.status_code == 400


def test_gpu_worker_lru_pruning(tmp_path: Path) -> None:
    """Verify prune_dataset_cache evicts the least-recently used datasets."""
    d1 = GPU_CACHE_DIR / "ds_old"
    d2 = GPU_CACHE_DIR / "ds_new"
    d1.mkdir(parents=True, exist_ok=True)
    d2.mkdir(parents=True, exist_ok=True)
    (d1 / "data.bin").write_bytes(b"0" * 1024)
    (d2 / "data.bin").write_bytes(b"0" * 1024)
    (d1 / ".last_used").write_text("1000", encoding="utf-8")
    (d2 / ".last_used").write_text("2000", encoding="utf-8")

    try:
        # Budget of 1500 bytes: d1 (older) should be pruned, d2 kept
        pruned = prune_dataset_cache(max_bytes=1500, protected=set())
        assert "ds_old" in pruned
        assert not d1.exists()
        assert d2.exists()
    finally:
        import shutil

        shutil.rmtree(d1, ignore_errors=True)
        shutil.rmtree(d2, ignore_errors=True)


def test_master_gpu_worker_configuration_endpoint() -> None:
    """Verify Master API can configure and probe AGRIFED_GPU_WORKER_URL."""
    app = FastAPI()
    app.include_router(mlops_router)
    client = TestClient(app)

    # Clear configuration
    r_clear = client.post("/api/master/gpu-worker", json={"gpu_worker_url": ""})
    assert r_clear.status_code == 200
    assert r_clear.json()["configured"] is False
