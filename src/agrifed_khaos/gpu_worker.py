"""Standalone GPU Compute Worker API for AgriFed-KHAOS (agrifed-gpu).

Designed for *shared* GPU nodes (e.g. one NVIDIA Tesla T4 serving several teams):

- True zero idle VRAM: this control-plane process never imports torch nor creates a
  CUDA context. Telemetry comes from ``nvidia-smi``. Every job runs in a child
  subprocess, so the OS reclaims 100% of its VRAM the instant the job exits.
- Strict FIFO queue (concurrency=1) via ``JobManager``: two AgriFed jobs never
  compete for the GPU.
- Good neighbour on a shared host:
    * VRAM admission gate (``scripts/gpu_vram_gate.py``): a job waits until the
      physical GPU has enough free memory instead of OOM-ing another service.
    * Per-process VRAM cap (``AGRIFED_GPU_MEM_FRACTION``) applied inside the
      training process, and AutoBatch sized to that cap.
    * CPU niceness for training processes so latency-sensitive neighbours win.
    * Bounded disk usage: dataset cache with an LRU size budget and explicit
      artifact / dataset deletion endpoints used by the dispatcher.
- Optional bearer-token auth (``AGRIFED_GPU_TOKEN``) for every endpoint except
  ``/health``.
"""

from __future__ import annotations

import collections
import importlib.metadata
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agrifed_khaos.jobs import JobManager

logger = logging.getLogger("agrifed_khaos.gpu_worker")

WORKSPACE_ROOT = Path(os.environ.get("AGRIFED_WORKSPACE", ".")).resolve()
MODELS_DIR = WORKSPACE_ROOT / "models"
DATA_DIR = WORKSPACE_ROOT / "data"
GPU_CACHE_DIR = DATA_DIR / "gpu_worker_cache"
YOLO_CACHE_DIR = DATA_DIR / "yolo_cache"
LAST_USED_MARKER = ".last_used"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _safe_id(raw: str) -> str:
    return "".join(c for c in raw if c.isalnum() or c in ("-", "_")).strip("_-")[:120]


class CountingDeque(collections.deque):  # type: ignore[type-arg]
    """Bounded log buffer that also tracks a monotonic sequence number.

    Clients poll with ``since=<seq>``; the sequence keeps growing even after old
    lines are evicted, so streaming never stalls once the buffer is full.
    """

    def __init__(self, maxlen: int) -> None:
        super().__init__(maxlen=maxlen)
        self.total = 0

    def append(self, item: Any) -> None:  # noqa: D102
        self.total += 1
        super().append(item)

    def extend(self, items: Any) -> None:  # noqa: D102
        for it in items:
            self.append(it)


class GPUTrainJobRequest(BaseModel):
    """Payload to enqueue a training job on the remote GPU worker."""

    slug: str = Field(..., min_length=1, max_length=120)
    task: str = Field(default="classification", description="'classification' or 'detection'")
    dataset_path: str = Field(..., description="dataset_id in the worker cache or path under data/")
    architecture: str = Field(default="resnet18")
    epochs: int = Field(default=15, ge=1, le=500)
    batch_size: int = Field(default=16, ge=0, le=256, description="0 = AutoBatch (detection)")
    lr: float = Field(default=0.001, gt=0.0, le=1.0)
    img_size: int = Field(default=640, ge=32, le=2048)
    target_client_tier: str = Field(default="gpu")
    transfer_learning: str = Field(default="partial")
    loss_function: str = Field(default="focal")
    augmentation: str = Field(default="in_the_wild")
    train_split: float | None = Field(default=None)
    val_split: float | None = Field(default=None)
    test_split: float | None = Field(default=None)
    min_free_vram_mb: int | None = Field(
        default=None, ge=0, description="Override the VRAM admission threshold for this job"
    )
    cleanup_dataset_after_run: bool = Field(
        default=False,
        description="Remove the cached dataset after the job (otherwise LRU-pruned).",
    )


def probe_gpu_telemetry() -> dict[str, Any]:
    """Query GPU telemetry through nvidia-smi only (never creates a CUDA context)."""
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = None
    cuda_build = (
        torch_version.split("+cu", 1)[1] if torch_version and "+cu" in torch_version else None
    )
    cuda_version = (
        f"{cuda_build[:-1]}.{cuda_build[-1]}" if cuda_build and len(cuda_build) >= 3 else None
    )

    info: dict[str, Any] = {
        "cuda_available": False,
        "device_count": 0,
        "device_name": "cpu",
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "vram_total_gb": 0.0,
        "vram_allocated_mb": 0.0,
        "vram_reserved_mb": 0.0,
        "vram_system_used_mb": None,
        "vram_system_free_mb": None,
        "gpu_utilization_pct": None,
    }
    if not shutil.which("nvidia-smi"):
        return info
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("nvidia-smi query failed: %s", exc)
        return info
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    if proc.returncode != 0 or not lines:
        return info
    parts = [p.strip() for p in lines[0].split(",")]
    try:
        info.update(
            {
                "cuda_available": True,
                "device_count": len(lines),
                "device_name": parts[0],
                "vram_total_gb": round(float(parts[1]) / 1024.0, 2),
                "vram_system_used_mb": round(float(parts[2]), 1),
                "vram_system_free_mb": round(float(parts[3]), 1),
                "gpu_utilization_pct": float(parts[4]) if len(parts) > 4 else None,
            }
        )
    except (IndexError, ValueError) as exc:
        logger.debug("Unexpected nvidia-smi output %r: %s", lines[0], exc)
    return info


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for f in path.rglob("*"):
        try:
            if f.is_file() and not f.is_symlink():
                total += f.stat().st_size
        except OSError:
            continue
    return total


def _touch_last_used(ds_dir: Path) -> None:
    try:
        (ds_dir / LAST_USED_MARKER).write_text(str(time.time()), encoding="utf-8")
    except OSError:
        pass


def _last_used(ds_dir: Path) -> float:
    marker = ds_dir / LAST_USED_MARKER
    if marker.is_file():
        try:
            return float(marker.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            pass
    try:
        return ds_dir.stat().st_mtime
    except OSError:
        return 0.0


def _remove_dataset(ds_dir: Path) -> None:
    shutil.rmtree(ds_dir, ignore_errors=True)
    # Derived YOLO conversion cache (symlinks + labels) for this dataset
    if YOLO_CACHE_DIR.is_dir():
        for derived in YOLO_CACHE_DIR.glob(f"{ds_dir.name}*_yolo"):
            shutil.rmtree(derived, ignore_errors=True)


def prune_dataset_cache(max_bytes: int, protected: set[str]) -> list[str]:
    """Evict least-recently-used cached datasets until the cache fits ``max_bytes``."""
    if not GPU_CACHE_DIR.is_dir():
        return []
    entries = [d for d in GPU_CACHE_DIR.iterdir() if d.is_dir()]
    sizes = {d: _dir_size_bytes(d) for d in entries}
    total = sum(sizes.values())
    removed: list[str] = []
    for d in sorted(entries, key=_last_used):
        if total <= max_bytes:
            break
        if d.name in protected:
            continue
        _remove_dataset(d)
        total -= sizes[d]
        removed.append(d.name)
    return removed


def _safe_extract(archive_path: Path, filename: str, target_dir: Path) -> None:
    """Extract .zip/.tar archives rejecting absolute paths, '..' and links outside target."""
    target_root = target_dir.resolve()

    def _inside(member_name: str) -> bool:
        dest = (target_root / member_name).resolve()
        return dest == target_root or target_root in dest.parents

    if filename.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            for name in zf.namelist():
                if not _inside(name):
                    raise HTTPException(status_code=400, detail="Archive contains unsafe paths")
            zf.extractall(target_root)
        return

    if filename.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive_path, "r:*") as tf:
            members = tf.getmembers()
            for m in members:
                if not _inside(m.name) or m.isdev():
                    raise HTTPException(status_code=400, detail="Archive contains unsafe paths")
                if (m.issym() or m.islnk()) and not _inside(str(Path(m.name).parent / m.linkname)):
                    raise HTTPException(status_code=400, detail="Archive contains unsafe links")
            if sys.version_info >= (3, 12) or hasattr(tarfile, "data_filter"):
                tf.extractall(target_root, members=members, filter="data")
            else:  # pragma: no cover - older interpreters
                tf.extractall(target_root, members=members)
        return

    raise HTTPException(status_code=400, detail="Unsupported archive format. Use .zip or .tar")


def create_gpu_worker_app() -> FastAPI:
    """Create the FastAPI application for the standalone `agrifed-gpu` service."""
    app = FastAPI(
        title="AgriFed-KHAOS GPU Compute Worker",
        description=(
            "Shared-GPU training microservice with strict FIFO queuing, VRAM admission "
            "control and zero idle VRAM footprint."
        ),
        version="1.1.0",
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    GPU_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    jm = JobManager(queue_file=MODELS_DIR / "gpu_job_queue.json")
    jm.logs = CountingDeque(maxlen=2000)
    app.state.job_manager = jm
    app.state.job_datasets = {}  # job_id -> dataset dir name (protects it from LRU pruning)

    expected_token = os.environ.get("AGRIFED_GPU_TOKEN", "").strip()

    async def require_token(request: Request) -> None:
        if not expected_token:
            return
        header = request.headers.get("authorization", "")
        if header != f"Bearer {expected_token}":
            raise HTTPException(status_code=401, detail="Unauthorized")

    auth = [Depends(require_token)]

    def _job_state(job_id: str) -> dict[str, Any]:
        if jm.active_job_id == job_id:
            return {"job_id": job_id, "state": "running", "action": jm.action, "exit_code": None}
        for idx, item in enumerate(jm.queue):
            if item.job_id == job_id:
                return {
                    "job_id": job_id,
                    "state": "queued",
                    "action": item.action,
                    "position": idx + 1,
                    "exit_code": None,
                }
        for rec in reversed(jm.history):
            if rec.get("job_id") == job_id:
                code = rec.get("exit_code")
                return {
                    "job_id": job_id,
                    "state": "succeeded" if code == 0 else "failed",
                    "action": rec.get("action"),
                    "exit_code": code,
                    "finished_at": rec.get("finished_at"),
                }
        return {"job_id": job_id, "state": "unknown", "exit_code": None}

    def _protected_datasets() -> set[str]:
        live = {jm.active_job_id} | {q.job_id for q in jm.queue}
        return {ds for jid, ds in app.state.job_datasets.items() if jid in live}

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "agrifed-gpu",
            **probe_gpu_telemetry(),
            "is_busy": jm.is_running,
            "active_job": jm.action if jm.is_running else None,
            "queue_length": len(jm.queue),
            "queued_jobs": [
                {"job_id": q.job_id, "action": q.action, "enqueued_at": q.created_at}
                for q in jm.queue
            ],
            "auth_required": bool(expected_token),
        }

    @app.get("/jobs/status", dependencies=auth)
    async def jobs_status(
        since: int = Query(default=0, ge=0, description="Return log lines with seq > since"),
    ) -> dict[str, Any]:
        logs = list(jm.logs)
        total = getattr(jm.logs, "total", len(logs))
        first_seq = total - len(logs)  # sequence number of logs[0] minus one
        start = max(0, since - first_seq)
        return {
            "is_running": jm.is_running,
            "action": jm.action,
            "active_job_id": jm.active_job_id,
            "started_at": jm.started_at,
            "exit_code": jm.last_exit_code,
            "queue_length": len(jm.queue),
            "queue": [
                {
                    "job_id": q.job_id,
                    "action": q.action,
                    "enqueued_at": q.created_at,
                    "position": idx + 1,
                }
                for idx, q in enumerate(jm.queue)
            ],
            "total_log_lines": total,
            "logs": logs[start:],
        }

    @app.get("/jobs/by-id/{job_id}", dependencies=auth)
    async def job_by_id(job_id: str) -> dict[str, Any]:
        return _job_state(job_id)

    @app.post("/jobs/cancel", dependencies=auth)
    async def cancel_job(job_id: str = Query(..., min_length=1)) -> dict[str, Any]:
        return await jm.cancel_job(job_id)

    @app.get("/datasets/{dataset_id}", dependencies=auth)
    async def dataset_exists(dataset_id: str) -> dict[str, Any]:
        safe = _safe_id(dataset_id)
        ds_dir = GPU_CACHE_DIR / safe
        if not safe or not ds_dir.is_dir():
            raise HTTPException(status_code=404, detail="Dataset not cached")
        _touch_last_used(ds_dir)
        return {"dataset_id": safe, "dataset_path": str(ds_dir), "cached": True}

    @app.delete("/datasets/{dataset_id}", dependencies=auth)
    async def delete_dataset(dataset_id: str) -> dict[str, Any]:
        safe = _safe_id(dataset_id)
        if safe in _protected_datasets():
            raise HTTPException(status_code=409, detail="Dataset in use by a queued/running job")
        ds_dir = GPU_CACHE_DIR / safe
        existed = ds_dir.is_dir()
        if existed:
            _remove_dataset(ds_dir)
        return {"dataset_id": safe, "deleted": existed}

    @app.post("/jobs/upload-dataset", dependencies=auth)
    async def upload_dataset(
        dataset_id: str = Form(..., description="Unique identifier for the dataset cache"),
        archive: UploadFile = File(  # noqa: B008
            ..., description=".zip or .tar dataset archive"
        ),
    ) -> dict[str, Any]:
        safe = _safe_id(dataset_id)
        if not safe:
            raise HTTPException(status_code=400, detail="Invalid dataset_id")
        if safe in _protected_datasets():
            raise HTTPException(status_code=409, detail="Dataset in use by a queued/running job")

        target_dir = GPU_CACHE_DIR / safe
        tmp_dir = GPU_CACHE_DIR / f".{safe}.partial"
        tmp_archive = GPU_CACHE_DIR / f".{safe}.upload"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        filename = (archive.filename or "dataset.tar").lower()
        try:
            with tmp_archive.open("wb") as f_out:
                while chunk := await archive.read(4 * 1024 * 1024):
                    f_out.write(chunk)
            _safe_extract(tmp_archive, filename, tmp_dir)
            if target_dir.exists():
                _remove_dataset(target_dir)
            tmp_dir.rename(target_dir)  # atomic publish: never a half-extracted dataset
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        finally:
            tmp_archive.unlink(missing_ok=True)

        _touch_last_used(target_dir)
        budget = int(_env_float("AGRIFED_GPU_CACHE_MAX_GB", 30.0) * 1024**3)
        evicted = prune_dataset_cache(budget, protected=_protected_datasets() | {safe})
        return {
            "status": "uploaded",
            "dataset_id": safe,
            "dataset_path": str(target_dir),
            "evicted": evicted,
        }

    @app.post("/jobs/train", dependencies=auth)
    async def enqueue_train_job(req: GPUTrainJobRequest) -> dict[str, Any]:
        safe_slug = _safe_id(req.slug)
        if not safe_slug:
            raise HTTPException(status_code=400, detail="Invalid model slug")

        # Datasets must live inside the worker's data/ tree (cache or synced folder)
        cached = GPU_CACHE_DIR / _safe_id(req.dataset_path)
        candidate = Path(req.dataset_path)
        if not candidate.is_absolute():
            candidate = WORKSPACE_ROOT / candidate
        candidate = candidate.resolve()
        if cached.is_dir() and _safe_id(req.dataset_path):
            ds_dir = cached
        elif candidate.is_dir() and (candidate == DATA_DIR or DATA_DIR in candidate.parents):
            ds_dir = candidate
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{req.dataset_path}' not found in the GPU worker data area.",
            )
        _touch_last_used(ds_dir)

        out_weights = MODELS_DIR / f"{safe_slug}_base.pt"
        out_meta = MODELS_DIR / f"{safe_slug}.meta.json"
        gpu_present = probe_gpu_telemetry()["cuda_available"]
        is_detection = req.task.lower() == "detection" or req.architecture.lower().startswith(
            "yolo"
        )
        workers = str(max(0, _env_int("AGRIFED_GPU_WORKERS", 4)))

        if is_detection:
            default_arch = os.environ.get("AGRIFED_GPU_DET_DEFAULT_ARCH", "yolo11s")
            arch = req.architecture if req.architecture.lower().startswith("yolo") else default_arch
            train_cmd = [
                sys.executable,
                str(WORKSPACE_ROOT / "scripts" / "train_detection.py"),
                "--data-dir", str(ds_dir),
                "--output", str(out_weights),
                "--metadata", str(out_meta),
                "--epochs", str(req.epochs),
                "--batch-size", str(req.batch_size),  # 0 = AutoBatch sized to the VRAM cap
                "--img-size", str(req.img_size),
                "--lr", str(req.lr),
                "--arch", arch,
                "--patience", str(max(10, req.epochs // 4)),
                "--workers", workers,
                "--device", "0" if gpu_present else "cpu",
            ]  # fmt: skip
        else:
            arch = req.architecture
            train_cmd = [
                sys.executable,
                str(WORKSPACE_ROOT / "scripts" / "train_base.py"),
                "--data-dir", str(ds_dir),
                "--output", str(out_weights),
                "--metadata", str(out_meta),
                "--epochs", str(req.epochs),
                "--batch-size", str(max(1, req.batch_size)),
                "--lr", str(req.lr),
                "--arch", arch,
                "--target-client-tier", req.target_client_tier,
                "--tl-strategy", req.transfer_learning,
                "--loss", req.loss_function,
                "--aug", req.augmentation,
                "--device", "cuda" if gpu_present else "cpu",
                "--workers", workers,
                "--max-samples", "0",
            ]  # fmt: skip
            if req.train_split is not None:
                train_cmd += [
                    "--train-split", str(req.train_split),
                    "--val-split", str(req.val_split if req.val_split is not None else 0.1),
                    "--test-split", str(req.test_split if req.test_split is not None else 0.1),
                ]  # fmt: skip

        nice_level = _env_int("AGRIFED_GPU_NICE", 10)
        if nice_level > 0 and shutil.which("nice"):
            train_cmd = ["nice", "-n", str(nice_level), *train_cmd]

        pipeline: list[list[str]] = []
        if gpu_present:
            min_free = (
                req.min_free_vram_mb
                if req.min_free_vram_mb is not None
                else _env_int("AGRIFED_GPU_MIN_FREE_MB", 6000)
            )
            pipeline.append(
                [
                    sys.executable,
                    str(WORKSPACE_ROOT / "scripts" / "gpu_vram_gate.py"),
                    "--min-free-mb",
                    str(min_free),
                    "--timeout-s",
                    str(_env_int("AGRIFED_GPU_GATE_TIMEOUT_S", 7200)),
                ]  # fmt: skip
            )
        pipeline.append(train_cmd)

        async def _after_gpu_job() -> None:
            if req.cleanup_dataset_after_run and ds_dir.parent == GPU_CACHE_DIR:
                _remove_dataset(ds_dir)

        job_res = await jm.enqueue_or_run(
            action=f"gpu_train:{safe_slug}",
            cmd=pipeline,
            on_success=_after_gpu_job,
        )
        if ds_dir.parent == GPU_CACHE_DIR and job_res.get("job_id"):
            app.state.job_datasets[job_res["job_id"]] = ds_dir.name
        return {
            **job_res,
            "slug": safe_slug,
            "task": "detection" if is_detection else "classification",
            "architecture": arch,
            "device": "cuda" if gpu_present else "cpu",
            "target_weights": out_weights.name,
            "target_metadata": out_meta.name,
        }

    def _artifact_paths(slug: str) -> list[Path]:
        safe = _safe_id(slug)
        return [
            MODELS_DIR / f"{safe}_base.pt",
            MODELS_DIR / f"{safe}.meta.json",
            MODELS_DIR / f"{safe}_evaluation_report.json",
            MODELS_DIR / f"{safe}_eval_preview.jpg",
        ]

    @app.get("/jobs/{slug}/artifacts", dependencies=auth)
    async def list_job_artifacts(slug: str) -> dict[str, Any]:
        safe_slug = _safe_id(slug)
        artifacts = [
            {"filename": f.name, "size_bytes": f.stat().st_size, "modified_at": f.stat().st_mtime}
            for f in _artifact_paths(safe_slug)
            if f.is_file()
        ]
        if not artifacts:
            raise HTTPException(status_code=404, detail=f"No artifacts found for '{safe_slug}'")
        meta_content: dict[str, Any] = {}
        meta_file = MODELS_DIR / f"{safe_slug}.meta.json"
        if meta_file.is_file():
            try:
                meta_content = json.loads(meta_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta_content = {}
        return {"slug": safe_slug, "artifacts": artifacts, "metadata": meta_content}

    @app.get("/jobs/{slug}/artifacts/{filename}", dependencies=auth)
    async def download_job_artifact(slug: str, filename: str) -> FileResponse:
        allowed = {p.name: p for p in _artifact_paths(slug)}
        if filename not in allowed:
            raise HTTPException(
                status_code=403, detail="Artifact filename not allowed for this slug."
            )
        target_path = allowed[filename]
        if not target_path.is_file():
            raise HTTPException(status_code=404, detail=f"Artifact '{filename}' not found.")
        return FileResponse(
            path=str(target_path), filename=filename, media_type="application/octet-stream"
        )

    @app.delete("/jobs/{slug}/artifacts", dependencies=auth)
    async def delete_job_artifacts(slug: str) -> dict[str, Any]:
        removed = []
        for f in _artifact_paths(slug):
            if f.is_file():
                f.unlink(missing_ok=True)
                removed.append(f.name)
        return {"slug": _safe_id(slug), "deleted": removed}

    return app


app = create_gpu_worker_app()
