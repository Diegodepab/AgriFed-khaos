#!/usr/bin/env python3
"""Lightweight REST Inference Server and MLOps Portal for AgriFed-KHAOS.

Supports 3 dedicated web interfaces and endpoints on port 8000:
- GET /: Agnostic Client (model selection, disease prediction, confidence & probabilities)
- GET /expert: Expert Agronomist (HITL queue review, uncertainty sampling, on-premise FL tutorial)
- GET /master: Administrator / MLOps (model manager, async training/aggregation/calibration runner)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import collections
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.parse
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from PIL import Image
from pydantic import BaseModel
from starlette.background import BackgroundTask
from torchvision import transforms

from agrifed_khaos.catalog import (
    LogicalModelEntry,
    ModelCatalogManager,
    ModelVersionRecord,
    append_model_history,
    get_model_history,
    slugify,
)
from agrifed_khaos.crypto import compute_sha256
from agrifed_khaos.federated_ops import (
    FederatedAggregationRequest,
    FederatedOperationRecord,
    FederatedOperationsManager,
    audit_client_update,
    resolve_meta_path,
    resolve_validation_directory_for_model,
)
from agrifed_khaos.hardware import HardwareProfile, ModelSelector
from agrifed_khaos.hitl import HITLQueueManager
from agrifed_khaos.models import (
    GradCAM,
    build_model,
    get_cam_target_layer,
    load_checkpoint_bundle,
)
from agrifed_khaos.ui import (
    render_agnostic_client_page,
    render_expert_page,
    render_master_admin_page,
    render_model_consumer_page,
    render_model_create_page,
)

# Modo pruebas: sin restricciones artificiales de resolución de imagen
Image.MAX_IMAGE_PIXELS = None
MAX_UPLOAD_SIZE_BYTES = 15 * 1024 * 1024  # Base de prueba para predict
MAX_UPDATE_SIZE_BYTES = 50 * 1024 * 1024
MAX_MANIFEST_SIZE_BYTES = 1 * 1024 * 1024

_client_locks: dict[str, asyncio.Lock] = {}
_client_locks_guard = asyncio.Lock()
_profiler_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ds_profiler")


class AutoFLRequest(BaseModel):
    """Payload schema for Zero-to-Model Auto-FL training request."""

    title: str
    slug: str | None = None
    domain: str = "General"
    crop: str = ""
    target_element: str = ""
    purpose: str = ""
    thumbnail_url: str | None = None
    description: str = ""
    input_guide: str = "Fotografía clara de la muestra con iluminación adecuada."
    dataset_path: str
    epochs: int = 3
    architecture: str = "auto"
    batch_size: int = 32
    lr: float = 1e-3
    target_client_tier: str = "cpu-low"
    transfer_learning: str = "partial"
    loss_function: str = "ce"
    augmentation: str = "basic"
    train_split: float | None = 0.8
    val_split: float | None = 0.1
    test_split: float | None = 0.1


class DatasetInspectRequest(BaseModel):
    """Payload schema for inspecting dataset directories."""

    dataset_path: str
    source_name: str | None = None


class DatasetDeduplicateRequest(BaseModel):
    """Payload schema for deduplicating and cleaning a dataset directory."""

    dataset_path: str
    remove_duplicates: bool = True
    remove_corrupted: bool = False


class DatasetFetchURLRequest(BaseModel):
    """Payload schema for fetching a dataset archive from a remote URL."""

    url: str


async def _get_client_lock(client_id: str) -> asyncio.Lock:
    """Acquire or allocate an async lock dedicated to a client_id to serialize writes."""
    async with _client_locks_guard:
        if client_id not in _client_locks:
            _client_locks[client_id] = asyncio.Lock()
        return _client_locks[client_id]


def _atomic_write(path: Path, content: bytes) -> None:
    """Replace one file atomically so readers never observe a partial write."""
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp_path.write_bytes(content)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


class JobManager:
    """Manages at most one asynchronous heavy training/aggregation job at a time."""

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.process: asyncio.subprocess.Process | None = None
        self.action: str | None = None
        self.started_at: str | None = None
        self.last_exit_code: int | None = None
        self.logs: collections.deque[str] = collections.deque(maxlen=200)

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.returncode is None

    async def run_command(
        self,
        action: str,
        cmd: list[str],
        on_success: Any | None = None,
    ) -> None:
        async with self.lock:
            if self.is_running:
                raise RuntimeError(
                    f"A job ('{self.action}') is already actively running. Wait for completion."
                )
            self.action = action
            self.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
            self.last_exit_code = None
            self.logs.clear()
            self.logs.append(f"[*] Starting job '{action}': {' '.join(cmd)}")

            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

        # Background task to stream output non-blocking
        asyncio.create_task(self._monitor_stream(self.process, on_success=on_success))

    async def _monitor_stream(
        self,
        proc: asyncio.subprocess.Process,
        on_success: Any | None = None,
    ) -> None:
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                self.logs.append(text)
        exit_code = await proc.wait()
        self.last_exit_code = exit_code
        self.logs.append(f"[*] Job '{self.action}' terminated with exit code {exit_code}")
        if exit_code == 0 and on_success is not None:
            try:
                if asyncio.iscoroutinefunction(on_success):
                    await on_success()
                else:
                    on_success()
            except Exception as exc:
                self.logs.append(f"[!] Error in post-job callback: {exc}")

    async def run_pipeline(self, action: str, commands: list[list[str]]) -> None:
        """Run multiple commands sequentially, streaming logs."""
        async with self.lock:
            if self.is_running:
                raise RuntimeError(
                    f"A job ('{self.action}') is already actively running. Wait for completion."
                )
            self.action = action
            self.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
            self.last_exit_code = None
            self.logs.clear()
            self.logs.append(f"[*] Starting pipeline '{action}' with {len(commands)} steps")

        await self._monitor_pipeline(commands)

    async def _monitor_pipeline(self, commands: list[list[str]]) -> None:
        for idx, cmd in enumerate(commands, start=1):
            self.logs.append(f"[*] Step {idx}/{len(commands)}: {' '.join(cmd)}")
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            self.process = proc
            assert proc.stdout is not None
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    self.logs.append(text)
            exit_code = await proc.wait()
            if exit_code != 0:
                self.last_exit_code = exit_code
                self.logs.append(f"[!] Step {idx} failed with exit code {exit_code}")
                return
        self.last_exit_code = 0
        self.logs.append(f"[*] Pipeline '{self.action}' terminated successfully with exit code 0")

    def get_status(self) -> dict[str, Any]:
        return {
            "is_running": self.is_running,
            "action": self.action,
            "started_at": self.started_at,
            "last_exit_code": self.last_exit_code,
            "pid": self.process.pid if self.is_running and self.process else None,
            "logs": list(self.logs),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgriFed REST Inference Server & Web Portal")
    parser.add_argument(
        "--model", type=str, default="models/global_model.pt", help="Path to model weights"
    )
    parser.add_argument(
        "--metadata", type=str, default="models/metadata.json", help="Path to metadata json"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Listen host (default: 0.0.0.0)"
    )
    parser.add_argument("--port", type=int, default=8000, help="Listen port (default: 8000)")
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Model architecture (default: resolved from metadata.json or 'resnet18')",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference hardware device: auto, cpu, cuda (default: auto)",
    )
    parser.add_argument(
        "--ssl-keyfile",
        type=str,
        default=os.getenv("AGRIFED_SSL_KEYFILE"),
        help="Path to SSL private key file for HTTPS (default: $AGRIFED_SSL_KEYFILE)",
    )
    parser.add_argument(
        "--ssl-certfile",
        type=str,
        default=os.getenv("AGRIFED_SSL_CERTFILE"),
        help="Path to SSL certificate file for HTTPS (default: $AGRIFED_SSL_CERTFILE)",
    )
    return parser.parse_args()


def compute_ood_guardrail(
    logits: torch.Tensor | None,
    probs: torch.Tensor,
    num_classes: int,
    temperature: float = 1.0,
    threshold: float | None = None,
    entropy_threshold: float = 0.65,
) -> dict[str, Any]:
    """Compute Energy-based Out-of-Distribution (OoD) score and Shannon Entropy guardrails.

    Energy formulation (Liu et al., NeurIPS 2020):
        Free Energy: E(x; T) = -T * torch.logsumexp(logits / T, dim=-1)
        Energy Score: S(x; T) = -E(x; T) = T * torch.logsumexp(logits / T, dim=-1)

    Shannon Entropy:
        H(p) = -sum(p_i * log(p_i + 1e-9))
        Normalized: H_norm = H(p) / log(max(num_classes, 2))
    """
    t_val = max(float(temperature), 1e-3)

    if logits is not None:
        logits_1d = logits.squeeze(0) if logits.dim() > 1 else logits
        free_energy = float((-t_val * torch.logsumexp(logits_1d / t_val, dim=-1)).item())
    else:
        # Fallback approximation from probabilities if raw logits are not supplied
        free_energy = float(
            (-t_val * torch.logsumexp(torch.log(probs + 1e-9) / t_val, dim=-1)).item()
        )

    energy_score = -free_energy

    # Shannon Entropy
    probs_1d = probs.squeeze(0) if probs.dim() > 1 else probs
    entropy = float(-(probs_1d * torch.log(probs_1d + 1e-9)).sum().item())
    max_entropy = math.log(max(num_classes, 2))
    norm_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    # Effective threshold resolution
    if threshold is not None:
        eff_thresh = float(threshold)
    else:
        env_thresh = os.getenv("AGRIFED_OOD_ENERGY_THRESHOLD")
        if env_thresh:
            eff_thresh = float(env_thresh)
        else:
            # Dynamic empirical threshold based on class count baseline
            eff_thresh = math.log(max(num_classes, 2)) + 0.35

    low_energy = energy_score < eff_thresh
    high_entropy = norm_entropy >= entropy_threshold
    is_ood = bool(low_energy or high_entropy)

    if eff_thresh > 0:
        energy_deficit = max(
            0.0, min(1.0, (eff_thresh - energy_score) / max(eff_thresh, 1e-3))
        )
    else:
        energy_deficit = 1.0 if low_energy else 0.0

    ood_score = max(energy_deficit, norm_entropy)

    return {
        "ood_flag": is_ood,
        "ood_score": round(float(ood_score), 4),
        "energy": round(float(free_energy), 4),
        "free_energy": round(float(free_energy), 4),
        "energy_score": round(float(energy_score), 4),
        "entropy": round(float(entropy), 4),
        "normalized_entropy": round(float(norm_entropy), 4),
        "ood_threshold": round(float(eff_thresh), 4),
    }


def create_app(
    model_path: str,
    metadata_path: str = "models/metadata.json",
    architecture: str | None = None,
    device_name: str = "auto",
) -> FastAPI:
    app = FastAPI(
        title="AgriFed-KHAOS Inference API & MLOps Portal",
        description="Zero-dependency REST service and UI for vision model inference & diagnosis",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if device_name == "cuda" or (device_name == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model_p = Path(model_path)
    meta_p = Path(metadata_path)

    active_model_p: Path | None = None
    active_explicitly_cleared: bool = False
    last_mtime: float = -1.0
    bundle_meta: dict[str, Any] = {}
    classes: list[str] = []
    arch: str = architecture or "resnet18"
    num_classes: int = 9
    calibrated_thresholds: Any = None
    temperature: float = 1.0
    model_sha256: str = ""
    img_size_tuple: tuple[int, int] = (224, 224)
    resize_tuple: tuple[int, int] = (256, 256)
    transform = transforms.Compose(
        [
            transforms.Resize(resize_tuple),
            transforms.CenterCrop(img_size_tuple),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    net: torch.nn.Module = build_model(architecture=arch, num_classes=num_classes).to(device)

    # LRU Cache for Loaded Inference Models (Max 2 models in RAM to prevent OOM)
    # Mapping of str(path.resolve()) -> dict(net=..., mtime=..., meta=..., classes=...)
    model_cache: collections.OrderedDict[str, dict[str, Any]] = collections.OrderedDict()
    MAX_LRU_MODELS = 2

    # Queue manager for Human-In-The-Loop feedback & Active Learning
    hitl_manager = HITLQueueManager()

    # Data space model catalog and registry manager
    resolved_models_dir = (
        Path(model_path).parent if Path(model_path).parent.is_dir() else Path("models")
    )
    catalog_manager = ModelCatalogManager(
        catalog_path=resolved_models_dir / "catalog.json",
        models_dir=resolved_models_dir,
    )
    app.state.catalog = catalog_manager

    fl_ops_manager = FederatedOperationsManager(
        storage_path=resolved_models_dir / "federated_operations.json"
    )
    app.state.fl_ops = fl_ops_manager

    def _get_fl_ops_manager() -> FederatedOperationsManager:
        return getattr(app.state, "fl_ops", fl_ops_manager)

    def _get_catalog_manager() -> ModelCatalogManager:
        return getattr(app.state, "catalog", catalog_manager)

    # Heavy job runner for /master operations
    job_manager = JobManager()

    def _resolve_and_reload_model(force: bool = False) -> None:
        nonlocal active_model_p, last_mtime, bundle_meta, classes, arch, num_classes
        nonlocal calibrated_thresholds, temperature, model_sha256, img_size_tuple, transform, net
        nonlocal active_explicitly_cleared

        if active_explicitly_cleared and not force:
            return
        if force:
            active_explicitly_cleared = False

        cm = _get_catalog_manager()
        active_entries = cm.load_catalog(include_deleted=False)
        operational = [e for e in active_entries if e.active_checkpoint]
        if active_entries and not operational and not force:
            active_model_p = None
            active_explicitly_cleared = True
            return

        candidate_p = Path(model_path)
        if not candidate_p.is_file():
            base_cand = Path("models/base_model.pt")
            if base_cand.is_file():
                candidate_p = base_cand
            else:
                return
        else:
            if (
                candidate_p == Path("models/base_model.pt")
                and Path("models/global_model.pt").is_file()
            ):
                candidate_p = Path("models/global_model.pt")

        current_mtime = candidate_p.stat().st_mtime
        if not force and active_model_p == candidate_p and current_mtime == last_mtime:
            return

        state, b_meta = load_checkpoint_bundle(candidate_p, device=device)
        sha = hashlib.sha256()
        with candidate_p.open("rb") as handle:
            while chunk := handle.read(65536):
                sha.update(chunk)
        model_sha256 = sha.hexdigest()

        meta: dict[str, Any] = {}
        if meta_p.is_file():
            try:
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
            except Exception:
                meta = {}

        classes = b_meta.get("classes") or meta.get("classes") or []
        arch = (
            architecture
            or b_meta.get("architecture")
            or meta.get("architecture")
            or "resnet18"
        )
        num_classes = (
            b_meta.get("num_classes")
            or (len(classes) if classes else None)
            or meta.get("num_classes")
            or 9
        )
        calibrated_thresholds = b_meta.get("calibrated_thresholds") or meta.get(
            "calibrated_thresholds"
        )
        temperature = float(b_meta.get("temperature") or meta.get("temperature") or 1.0)

        net = build_model(architecture=arch, num_classes=num_classes).to(device)
        if state:
            net.load_state_dict(state)
            net.eval()

        raw_img_size = b_meta.get("img_size") or meta.get("img_size") or 224
        if isinstance(raw_img_size, int):
            img_size_tuple = (raw_img_size, raw_img_size)
        elif isinstance(raw_img_size, (list, tuple)) and len(raw_img_size) == 2:
            img_size_tuple = (int(raw_img_size[0]), int(raw_img_size[1]))
        else:
            img_size_tuple = (224, 224)

        resize_dim = (
            int(round(img_size_tuple[0] * 256 / 224)),
            int(round(img_size_tuple[1] * 256 / 224)),
        )
        transform = transforms.Compose(
            [
                transforms.Resize(resize_dim),
                transforms.CenterCrop(img_size_tuple),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        active_model_p = candidate_p
        last_mtime = current_mtime

    _resolve_and_reload_model()

    def _get_model_for_inference(target_model_name: str | None = None) -> tuple[
        torch.nn.Module,
        list[str],
        int,
        Any,
        float,
        transforms.Compose,
        str,
    ]:
        """Fetch model, metadata and transforms via LRU cache without mutating global active."""
        _resolve_and_reload_model()
        if not target_model_name:
            if not active_model_p or net is None:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Modelo temporalmente fuera de servicio: "
                        "No hay ningún modelo activo en el sistema."
                    ),
                )
            return (
                net,
                classes,
                num_classes,
                calibrated_thresholds,
                temperature,
                transform,
                active_model_p.name,
            )

        # Check if target_model_name is a catalog slug
        target_checkpoint = target_model_name
        catalog_entry = _get_catalog_manager().get_by_slug(
            target_model_name, include_deleted=True
        )
        if catalog_entry:
            if catalog_entry.is_deleted:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"El modelo '{target_model_name}' está en la papelera "
                        f"y no admite inferencias."
                    ),
                )
            if not catalog_entry.active_checkpoint:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Modelo temporalmente fuera de servicio: "
                        f"El modelo '{catalog_entry.title}' no tiene ningún "
                        "checkpoint activo asignado."
                    ),
                )
            target_checkpoint = catalog_entry.active_checkpoint

        # Validate path
        safe_name = Path(target_checkpoint).name
        base_dir = active_model_p.parent if active_model_p else Path("models")
        target_path = base_dir / safe_name
        if not target_path.is_file():
            target_path = Path("models") / safe_name
        if not target_path.is_file():
            if not active_model_p:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Modelo temporalmente fuera de servicio: "
                        f"Checkpoint '{target_checkpoint}' no disponible en disco "
                        "y no hay modelo activo por defecto."
                    ),
                )
            # If not found, fall back gracefully to default
            return (
                net,
                classes,
                num_classes,
                calibrated_thresholds,
                temperature,
                transform,
                active_model_p.name,
            )

        resolved_key = str(target_path.resolve())
        mtime = target_path.stat().st_mtime

        # Check in LRU cache
        if resolved_key in model_cache:
            entry = model_cache[resolved_key]
            if entry["mtime"] == mtime:
                model_cache.move_to_end(resolved_key)
                return (
                    entry["net"],
                    entry["classes"],
                    entry["num_classes"],
                    entry["calibrated_thresholds"],
                    entry["temperature"],
                    entry["transform"],
                    safe_name,
                )

        # Load fresh model into cache
        try:
            state, b_meta = load_checkpoint_bundle(target_path, device=device)
            m_classes = b_meta.get("classes") or classes
            m_arch = b_meta.get("architecture") or "resnet18"
            m_num = b_meta.get("num_classes") or len(m_classes) or 9
            m_cal = b_meta.get("calibrated_thresholds")
            m_temp = float(b_meta.get("temperature") or 1.0)

            m_net = build_model(architecture=m_arch, num_classes=m_num).to(device)
            if state:
                m_net.load_state_dict(state)
                m_net.eval()

            raw_size = b_meta.get("img_size", 224)
            if isinstance(raw_size, int):
                sz_tuple = (raw_size, raw_size)
            elif isinstance(raw_size, (list, tuple)) and len(raw_size) == 2:
                sz_tuple = (int(raw_size[0]), int(raw_size[1]))
            else:
                sz_tuple = (224, 224)

            rs_dim = (int(round(sz_tuple[0] * 256 / 224)), int(round(sz_tuple[1] * 256 / 224)))
            m_tf = transforms.Compose(
                [
                    transforms.Resize(rs_dim),
                    transforms.CenterCrop(sz_tuple),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

            m_ood_thresh = b_meta.get("ood_threshold") or b_meta.get("ood_energy_threshold")

            # Evict if exceeding LRU capacity
            while len(model_cache) >= MAX_LRU_MODELS:
                model_cache.popitem(last=False)

            model_cache[resolved_key] = {
                "net": m_net,
                "mtime": mtime,
                "classes": m_classes,
                "num_classes": m_num,
                "calibrated_thresholds": m_cal,
                "temperature": m_temp,
                "transform": m_tf,
                "ood_threshold": float(m_ood_thresh) if m_ood_thresh is not None else None,
            }
            return m_net, m_classes, m_num, m_cal, m_temp, m_tf, safe_name
        except Exception:
            # Fall back to active
            return (
                net,
                classes,
                num_classes,
                calibrated_thresholds,
                temperature,
                transform,
                Path(str(active_model_p)).name if active_model_p else "default",
            )

    def _decode_probs_row(
        probs: torch.Tensor,
        infer_classes: list[str],
        infer_num_classes: int,
        infer_cal: Any,
        filename: str | None,
        actual_model_name: str,
        logits: torch.Tensor | None = None,
        ood_threshold: float | None = None,
        ood_temperature: float = 1.0,
    ) -> dict[str, Any]:
        probs_list: list[float] = probs.tolist()
        pred_idx = int(probs.argmax().item())
        confidence = float(probs_list[pred_idx])

        if infer_cal:
            if infer_num_classes == 2 and any(
                k in infer_cal
                for k in ("calibrated_youden", "calibrated_f1", "optimal_threshold")
            ):
                tau_pos: float | None = None
                if "calibrated_youden" in infer_cal:
                    opt = infer_cal["calibrated_youden"].get("optimal_threshold")
                    if opt is not None:
                        tau_pos = float(opt)
                elif "calibrated_f1" in infer_cal:
                    opt = infer_cal["calibrated_f1"].get("optimal_threshold")
                    if opt is not None:
                        tau_pos = float(opt)
                elif "optimal_threshold" in infer_cal:
                    opt = infer_cal.get("optimal_threshold")
                    if opt is not None:
                        tau_pos = float(opt)

                if tau_pos is not None:
                    pred_idx = 1 if probs_list[1] >= tau_pos else 0
                    confidence = float(probs_list[pred_idx])
            else:
                best_score = -1.0
                best_c = 0
                for c in range(infer_num_classes):
                    c_name = infer_classes[c] if c < len(infer_classes) else f"class_{c}"
                    tau_c = 0.50
                    if isinstance(infer_cal, dict):
                        val = infer_cal.get(c_name)
                        if val is None:
                            val = infer_cal.get(str(c))
                        if val is None:
                            val = infer_cal.get(c)
                        if isinstance(val, dict) and "optimal_threshold" in val:
                            val = val["optimal_threshold"]
                        if val is not None and isinstance(val, (int, float)):
                            tau_c = float(val)
                    elif isinstance(infer_cal, (list, tuple)) and c < len(infer_cal):
                        tau_c = float(infer_cal[c])

                    score_c = probs_list[c] / max(tau_c, 1e-6)
                    if score_c > best_score:
                        best_score = score_c
                        best_c = c

                pred_idx = best_c
                confidence = float(probs_list[pred_idx])

        pred_class = (
            infer_classes[pred_idx]
            if infer_classes and pred_idx < len(infer_classes)
            else f"class_{pred_idx}"
        )
        prob_dict = {
            infer_classes[i] if i < len(infer_classes) else f"class_{i}": round(
                float(probs_list[i]), 4
            )
            for i in range(infer_num_classes)
        }

        ood_metrics = compute_ood_guardrail(
            logits=logits,
            probs=probs,
            num_classes=infer_num_classes,
            temperature=ood_temperature,
            threshold=ood_threshold,
        )

        return {
            "predicted_class": pred_class,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "filename": filename,
            "model_name": actual_model_name,
            "threshold_applied": "calibrated" if infer_cal else "nominal_argmax",
            "ood_flag": ood_metrics["ood_flag"],
            "ood_score": ood_metrics["ood_score"],
            "energy": ood_metrics["energy"],
            "free_energy": ood_metrics["free_energy"],
            "energy_score": ood_metrics["energy_score"],
            "entropy": ood_metrics["entropy"],
            "normalized_entropy": ood_metrics["normalized_entropy"],
            "ood_threshold": ood_metrics["ood_threshold"],
        }

    def _infer_single(
        img_bytes: bytes,
        filename: str | None,
        model_name: str | None = None,
        ood_threshold: float | None = None,
    ) -> dict[str, Any]:
        (
            infer_net,
            infer_classes,
            infer_num_classes,
            infer_cal,
            infer_temp,
            infer_tf,
            actual_model_name,
        ) = _get_model_for_inference(model_name)

        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Image.DecompressionBombError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Decompression bomb detected: image exceeds pixel limit ({e})",
            ) from e
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image format or corrupted payload: {e}",
            ) from e

        tensor = infer_tf(img).unsqueeze(0).to(device)

        with torch.inference_mode():
            logits = infer_net(tensor)
            probs = F.softmax(logits / max(infer_temp, 1e-3), dim=1)[0].cpu()

        eff_thresh = ood_threshold
        if eff_thresh is None:
            for v in model_cache.values():
                if v.get("net") is infer_net:
                    eff_thresh = v.get("ood_threshold")
                    break

        return _decode_probs_row(
            probs,
            infer_classes,
            infer_num_classes,
            infer_cal,
            filename,
            actual_model_name,
            logits=logits[0].cpu(),
            ood_threshold=eff_thresh,
            ood_temperature=infer_temp if infer_temp > 0 else 1.0,
        )

    # =========================================================================
    # UI ROUTING (3 Distinct Views for Roles)
    # =========================================================================

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> Any:
        accept_header = request.headers.get("accept", "")
        # If client explicitly asks for application/json without html, return API root
        if "application/json" in accept_header and "text/html" not in accept_header:
            _resolve_and_reload_model()
            return {
                "name": "AgriFed-KHAOS REST Inference Server",
                "status": "online",
                "active_model": str(active_model_p) if active_model_p else str(model_p),
                "architecture": arch,
                "classes": classes,
                "endpoints": {
                    "health": "GET /health",
                    "interactive_docs": "GET /docs",
                    "predict": "POST /predict",
                    "predict_batch": "POST /predict/batch",
                    "download_model": "GET /download-model",
                    "submit_update": "POST /submit-update",
                },
            }
        return HTMLResponse(render_agnostic_client_page())

    @app.get("/models/{slug}", response_class=HTMLResponse)
    def model_consumer_view(slug: str) -> HTMLResponse:
        return HTMLResponse(render_model_consumer_page(slug))

    @app.get("/models/{slug}/expert", response_class=HTMLResponse)
    def model_expert_view(slug: str) -> HTMLResponse:
        return HTMLResponse(render_expert_page(slug))

    @app.get("/expert", response_class=HTMLResponse)
    def expert_view() -> HTMLResponse:
        return HTMLResponse(render_expert_page())

    @app.get("/master", response_class=HTMLResponse)
    def master_view() -> HTMLResponse:
        return HTMLResponse(render_master_admin_page())

    @app.get("/master/models/new", response_class=HTMLResponse)
    @app.get("/master/create", response_class=HTMLResponse)
    def master_model_create_view() -> HTMLResponse:
        return HTMLResponse(render_model_create_page())

    @app.get("/static/{asset_name}")
    def serve_static_asset(asset_name: str) -> FileResponse:
        """Serve shared static design-system CSS and JS assets."""
        safe_name = Path(asset_name).name
        if safe_name != asset_name or safe_name.startswith("."):
            raise HTTPException(status_code=404, detail="Static asset not found")
        static_dir = Path(__file__).resolve().parent.parent / "src" / "agrifed_khaos" / "static"
        target = static_dir / safe_name
        if not target.is_file():
            raise HTTPException(status_code=404, detail="Static asset not found")
        media_type = "text/plain; charset=utf-8"
        if safe_name.endswith(".css"):
            media_type = "text/css; charset=utf-8"
        elif safe_name.endswith(".js"):
            media_type = "application/javascript; charset=utf-8"
        return FileResponse(str(target), media_type=media_type)

    # =========================================================================
    # API: MODEL MANAGEMENT & DISCOVERY (Data Space Model Hub)
    # =========================================================================

    @app.get("/api/models")
    def list_models(
        q: str | None = None,
        domain: str | None = None,
        page: int = 1,
        page_size: int = 12,
    ) -> dict[str, Any]:
        """Search, filter, and paginate logical models in the Data Space catalog."""
        _resolve_and_reload_model()
        cm = _get_catalog_manager()
        cm.sync_with_filesystem(default_classes=classes)
        catalog_res = cm.search_and_paginate(
            q=q, domain=domain, page=page, page_size=page_size
        )

        enriched_items: list[dict[str, Any]] = []
        for m in catalog_res["models"]:
            active_cp = m.get("active_checkpoint")
            sz = 0
            is_active = False
            cp_file_str = ""
            if active_cp:
                cp_file = Path("models") / active_cp
                sz = cp_file.stat().st_size if cp_file.is_file() else 0
                is_active = (
                    active_model_p is not None
                    and cp_file.is_file()
                    and cp_file.resolve() == active_model_p.resolve()
                )
                cp_file_str = str(cp_file)
            item = dict(m)
            item["name"] = active_cp or m.get("title", "")
            item["path"] = cp_file_str
            item["size_bytes"] = sz
            item["is_active"] = is_active
            item["has_calibration"] = bool(m.get("calibrated_thresholds"))
            enriched_items.append(item)

        # Fallback only when catalog file does not exist or has never been initialized
        # (e.g. running under isolated test fixtures in tmp_path without a catalog.json)
        all_catalog_entries = cm.load_catalog(include_deleted=True)
        if not all_catalog_entries and not cm.catalog_path.is_file():
            models_dir = Path("models")
            if models_dir.is_dir():
                for p in sorted(models_dir.glob("*.pt")):
                    name = p.name
                    if (
                        name.endswith("_momentum.pt")
                        or name == "server_momentum.pt"
                        or name.startswith("client")
                        or "parameters" in name
                        or name.endswith("_candidate.pt")
                        or name.endswith(".backup")
                    ):
                        continue
                    enriched_items.append(
                        {
                            "name": p.name,
                            "title": p.name.replace(".pt", "").replace("_", " ").title(),
                            "slug": slugify(p.name),
                            "domain": "General",
                            "crop": "",
                            "target_element": "",
                            "purpose": "",
                            "description": "Modelo de inferencia federado",
                            "active_checkpoint": p.name,
                            "versions": [p.name],
                            "architecture": arch,
                            "num_classes": num_classes,
                            "classes": classes,
                            "thumbnail_url": None,
                            "example_images": [],
                            "path": str(p),
                            "size_bytes": p.stat().st_size,
                            "is_active": active_model_p is not None
                            and p.resolve() == active_model_p.resolve(),
                            "has_calibration": bool(calibrated_thresholds),
                        }
                    )
            if not enriched_items and active_model_p and active_model_p.is_file():
                enriched_items.append(
                    {
                        "name": active_model_p.name,
                        "title": active_model_p.name.replace(".pt", "").replace("_", " ").title(),
                        "slug": slugify(active_model_p.name),
                        "domain": "General",
                        "crop": "",
                        "target_element": "",
                        "purpose": "",
                        "description": "Modelo activo de inferencia",
                        "active_checkpoint": active_model_p.name,
                        "versions": [active_model_p.name],
                        "architecture": arch,
                        "num_classes": num_classes,
                        "classes": classes,
                        "thumbnail_url": None,
                        "example_images": [],
                        "path": str(active_model_p),
                        "size_bytes": active_model_p.stat().st_size,
                        "is_active": True,
                        "has_calibration": bool(calibrated_thresholds),
                    }
                )

        return {
            "models": enriched_items,
            "items": enriched_items,
            "total": catalog_res["total"] if catalog_res["total"] > 0 else len(enriched_items),
            "page": catalog_res["page"],
            "page_size": catalog_res["page_size"],
            "total_pages": catalog_res["total_pages"],
            "domains": catalog_res["domains"],
            "active_model": active_model_p.name if active_model_p else None,
        }

    @app.get("/api/models/{slug}")
    def get_model_detail(slug: str) -> dict[str, Any]:
        """Fetch full details for a single logical model by slug or filename."""
        _resolve_and_reload_model()
        cm = _get_catalog_manager()
        cm.sync_with_filesystem(default_classes=classes)
        entry = cm.get_by_slug(slug, include_deleted=True)
        if entry and entry.is_deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{slug}' is in trash quarantine and not available.",
            )
        if not entry:
            if active_model_p and (
                active_model_p.name == slug or slugify(active_model_p.name) == slug
            ):
                entry = LogicalModelEntry(
                    slug=slugify(active_model_p.name),
                    title=active_model_p.name.replace(".pt", "").title(),
                    domain="General",
                    description="Modelo activo de inferencia",
                    active_checkpoint=active_model_p.name,
                    versions=[active_model_p.name],
                    architecture=arch,
                    num_classes=num_classes,
                    classes=classes,
                )
            else:
                raise HTTPException(status_code=404, detail=f"Model '{slug}' not found")

        sz = 0
        is_active = False
        if entry.active_checkpoint:
            cp_file = Path("models") / entry.active_checkpoint
            if (
                not cp_file.is_file()
                and active_model_p
                and active_model_p.name == entry.active_checkpoint
            ):
                cp_file = active_model_p
            sz = cp_file.stat().st_size if cp_file.is_file() else 0
            is_active = (
                active_model_p is not None
                and cp_file.is_file()
                and cp_file.resolve() == active_model_p.resolve()
            )

        return {
            "model": entry.model_dump(),
            "size_bytes": sz,
            "is_active": is_active,
            "has_calibration": bool(entry.calibrated_thresholds),
            "download_url": (
                f"/download-model?model_slug={entry.slug}"
                if entry.active_checkpoint
                else None
            ),
        }

    @app.get("/api/models/{slug}/thumbnail")
    def get_model_thumbnail(slug: str) -> FileResponse:
        """Serve representative thumbnail image for a logical model if available."""
        _resolve_and_reload_model()
        cm = _get_catalog_manager()
        entry = cm.get_by_slug(slug, include_deleted=False)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Model '{slug}' not found")

        candidates: list[Path] = []
        if entry.thumbnail_url:
            clean_url = entry.thumbnail_url.strip()
            if not clean_url.startswith("http://") and not clean_url.startswith("https://"):
                clean_path = clean_url.lstrip("/")
                candidates.append(Path(clean_path))
                candidates.append(Path("models") / clean_path)

        candidates.extend(
            [
                Path("models/thumbnails") / f"{slug}.jpg",
                Path("models/thumbnails") / f"{slug}.jpeg",
                Path("models/thumbnails") / f"{slug}.png",
                Path("models/thumbnails") / f"{slug}.webp",
                Path("models/thumbnails") / f"{entry.active_checkpoint}.jpg",
                Path("models/thumbnails") / f"{entry.active_checkpoint}.png",
            ]
        )

        for cand in candidates:
            if cand.is_file():
                suffix = cand.suffix.lower()
                media_type = (
                    "image/png"
                    if suffix == ".png"
                    else ("image/webp" if suffix == ".webp" else "image/jpeg")
                )
                return FileResponse(str(cand), media_type=media_type)

        raise HTTPException(
            status_code=404,
            detail=f"No thumbnail image available for model '{slug}'",
        )

    @app.post("/api/models/save")
    async def save_model_card(request: Request) -> dict[str, Any]:
        """Create or update a logical model card in the catalog."""
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise HTTPException(status_code=400, detail="Invalid JSON payload")
            slug = body.get("slug") or slugify(body.get("title", ""))
            if not slug:
                raise HTTPException(status_code=400, detail="Model slug or title is required")

            entry = LogicalModelEntry(
                slug=slug,
                title=body.get("title", slug.replace("-", " ").title()),
                domain=body.get("domain", "General"),
                crop=body.get("crop", ""),
                target_element=body.get("target_element", ""),
                purpose=body.get("purpose", ""),
                thumbnail_url=body.get("thumbnail_url") or None,
                description=body.get("description", ""),
                input_guide=body.get("input_guide", ""),
                active_checkpoint=body.get("active_checkpoint", "global_model.pt"),
                versions=body.get("versions", [body.get("active_checkpoint", "global_model.pt")]),
                architecture=body.get("architecture", "resnet18"),
                num_classes=int(body.get("num_classes", len(body.get("classes", [])) or 9)),
                classes=body.get("classes", []),
                calibrated_thresholds=body.get("calibrated_thresholds"),
            )
            _get_catalog_manager().upsert_entry(entry)
            return {"status": "saved", "model": entry.model_dump()}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to save model card: {e}") from e

    @app.post("/api/models/activate")
    def activate_model(model_name: str = Query(...)) -> dict[str, Any]:
        """Set the default active global model for the server."""
        nonlocal model_path
        safe_name = Path(model_name).name
        candidate = Path("models") / safe_name
        if not candidate.is_file():
            if (resolved_models_dir / safe_name).is_file():
                candidate = resolved_models_dir / safe_name
            elif Path(model_path).name == safe_name and Path(model_path).is_file():
                candidate = Path(model_path)
            elif Path(model_name).is_file():
                candidate = Path(model_name)
            else:
                raise HTTPException(status_code=404, detail=f"Model '{safe_name}' not found")

        model_path = str(candidate)
        _resolve_and_reload_model(force=True)
        return {
            "status": "activated",
            "active_model": safe_name,
            "architecture": arch,
            "num_classes": num_classes,
        }

    # =========================================================================
    # API: HITL / HUMAN-IN-THE-LOOP ACTIVE LEARNING
    # =========================================================================

    @app.get("/api/feedback/pending")
    async def get_pending_feedback(model_slug: str | None = Query(None)) -> dict[str, Any]:
        """Fetch pending inference samples sorted by least confidence (uncertainty sampling)."""
        active_hitl = getattr(app.state, "hitl", hitl_manager)
        items = await active_hitl.list_pending(sort_by_uncertainty=True, model_slug=model_slug)
        return {"items": items, "count": len(items)}

    @app.get("/api/feedback/stats")
    @app.get("/api/expert/stats")
    async def get_feedback_stats(model_slug: str | None = Query(None)) -> dict[str, Any]:
        active_hitl = getattr(app.state, "hitl", hitl_manager)
        return await active_hitl.stats(model_slug=model_slug)

    @app.get("/api/feedback/image/{item_id}")
    def get_feedback_image(item_id: str) -> FileResponse:
        active_hitl = getattr(app.state, "hitl", hitl_manager)
        img_p = active_hitl.get_image_path(item_id)
        if not img_p or not img_p.is_file():
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(str(img_p))

    @app.post("/api/feedback/verify/{item_id}")
    async def verify_feedback(
        item_id: str,
        verified_class: str = Query(...),
        reviewer: str = "expert",
        notes: str | None = None,
    ) -> dict[str, Any]:
        active_hitl = getattr(app.state, "hitl", hitl_manager)
        try:
            return await active_hitl.verify(
                item_id, verified_class=verified_class, reviewer=reviewer, notes=notes
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    @app.post("/api/feedback/reject/{item_id}")
    async def reject_feedback(
        item_id: str,
        reviewer: str = "expert",
        reason: str | None = None,
    ) -> dict[str, Any]:
        active_hitl = getattr(app.state, "hitl", hitl_manager)
        try:
            return await active_hitl.reject(item_id, reviewer=reviewer, reason=reason)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    # =========================================================================
    # API: MASTER ASYNC JOB RUNNER
    # =========================================================================

    @app.get("/api/master/status")
    def get_master_job_status() -> dict[str, Any]:
        return job_manager.get_status()

    @app.post("/api/master/jobs/clear")
    def clear_master_job_logs() -> dict[str, Any]:
        """Clear execution logs from job manager terminal."""
        job_manager.logs.clear()
        return {"status": "cleared", "logs": []}

    @app.post("/api/master/run")
    async def run_master_job(action: str = Query(...)) -> dict[str, Any]:
        if job_manager.is_running:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Job '{job_manager.action}' is currently running. "
                    "Please wait for it to finish."
                ),
            )

        cmd: list[str] = [sys.executable]
        if action == "train_hitl":
            active_hitl = getattr(app.state, "hitl", hitl_manager)
            verified_candidates = [
                active_hitl.verified_dir,
                active_hitl.root_dir / "verified",
            ]
            if active_hitl.root_dir.is_dir():
                for v in active_hitl.root_dir.glob("*/verified"):
                    if v not in verified_candidates:
                        verified_candidates.append(v)

            chosen_verified: Path | None = None
            for cand in verified_candidates:
                if cand.is_dir() and any(
                    f.is_file()
                    for class_dir in cand.iterdir()
                    if class_dir.is_dir()
                    for f in class_dir.iterdir()
                ):
                    chosen_verified = cand
                    break

            if not chosen_verified:
                raise HTTPException(
                    status_code=400,
                    detail="No verified images found in hitl_queue to train on.",
                )
            base_model_file = str(active_model_p) if active_model_p else "models/global_model.pt"
            cmd.extend(
                [
                    "scripts/client_trainer.py",
                    "--data-dir",
                    str(chosen_verified),
                    "--weights-in",
                    base_model_file,
                    "--weights-out",
                    "models/client_updates/hitl_verified_parameters.pt",
                    "--epochs",
                    "2",
                    "--unfreeze-last-block",
                    "--allow-new-classes",
                    "--client-id",
                    "hitl_expert_curated",
                ]
            )
        elif action == "aggregate":
            base_model_file = str(active_model_p) if active_model_p else "models/base_model.pt"
            base_p = Path(base_model_file)
            if not base_p.is_file():
                base_p = Path("models/base_model.pt")

            cm = _get_catalog_manager()
            target_slug = "patologias-foliares-olivo"
            entry = cm.get_by_slug(target_slug)

            # Audit and select only strictly eligible parameter updates
            candidate_files: list[Path] = []
            updates_root = Path("models/client_updates")
            if updates_root.is_dir():
                candidate_files.extend(
                    [p for p in updates_root.glob("*.pt") if not p.name.startswith(".")]
                )
            for p in Path("models").glob("*_parameters.pt"):
                if p not in candidate_files and not p.name.startswith("."):
                    candidate_files.append(p)

            eligible_updates: list[Path] = []
            for p in candidate_files:
                audit = audit_client_update(
                    p,
                    target_base_checkpoint=base_p,
                    target_arch=entry.architecture if entry else "resnet18",
                )
                if audit.is_eligible:
                    eligible_updates.append(p)

            if not eligible_updates:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "No se encontraron actualizaciones aptas en models/client_updates/. "
                        "Los artefactos fueron rechazados por fallo de integridad o linaje."
                    ),
                )

            # Resolve validation directory
            val_dir, _ = resolve_validation_directory_for_model(
                entry.classes if entry else classes
            )

            candidate_out = Path("models") / f"{target_slug}_candidate.pt"
            cmd.extend(
                [
                    "scripts/aggregate.py",
                    "--base-model",
                    str(base_p),
                    "--output",
                    str(candidate_out),
                    "--allow-new-classes",
                ]
            )
            if val_dir:
                cmd.extend(["--val-dir", str(val_dir), "--max-acc-drop", "0.03"])

            cmd.append("--client-weights")
            cmd.extend(str(p) for p in eligible_updates)
        elif action == "calibrate":
            target = str(active_model_p) if active_model_p else "models/global_model.pt"
            # Try val or test directory
            val_cand = Path("data/EnfermedadesHoja_v1.0/dataset/val")
            if not val_cand.is_dir():
                val_cand = Path("data/EnfermedadesHoja_v1.0/dataset/test")
            cmd.extend(
                [
                    "scripts/evaluate.py",
                    "--model",
                    target,
                    "--data-dir",
                    str(val_cand) if val_cand.is_dir() else "models/hitl_queue/verified",
                    "--calibrate-threshold",
                ]
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown master action: '{action}'")

        await job_manager.run_command(action, cmd)
        return {"status": "started", "action": action, "command": cmd}

    train_hitl_lock = asyncio.Lock()

    @app.get("/api/master/models")
    @app.get("/api/master/checkpoints")
    def get_master_models() -> dict[str, Any]:
        """List valid inference models and physical checkpoints in models/ directory."""
        res = list_models()
        models_list = res.get("models", [])
        checkpoints: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for m in models_list:
            cp_name = m.get("active_checkpoint") or m.get("name")
            if cp_name and cp_name not in seen_names:
                seen_names.add(cp_name)
                checkpoints.append({
                    "name": cp_name,
                    "size_mb": round(m.get("size_bytes", 0) / (1024 * 1024), 2),
                    "size_bytes": m.get("size_bytes", 0),
                    "architecture": m.get("architecture", "resnet18"),
                    "classes": m.get("classes", []),
                    "is_active": m.get("is_active", False),
                })
        models_dir = Path("models")
        if models_dir.is_dir():
            for p in sorted(models_dir.glob("*.pt")):
                name = p.name
                if name.endswith("_momentum.pt") or name == "server_momentum.pt":
                    continue
                if name.startswith("client") or "parameters" in name:
                    continue
                if name.endswith("_candidate.pt") or name.endswith(".backup"):
                    continue
                if name not in seen_names:
                    seen_names.add(name)
                    sz = p.stat().st_size
                    checkpoints.append({
                        "name": name,
                        "size_mb": round(sz / (1024 * 1024), 2),
                        "size_bytes": sz,
                        "architecture": arch,
                        "classes": classes,
                        "is_active": active_model_p is not None
                        and p.resolve() == active_model_p.resolve(),
                    })
        res["checkpoints"] = checkpoints
        return res

    @app.post("/api/master/set-active")
    async def set_active_model(
        request: Request,
        model_name: str | None = Query(None),
    ) -> dict[str, Any]:
        """Set default active in-memory model for / inferences."""
        target = model_name
        if not target and "application/json" in request.headers.get("content-type", ""):
            try:
                body = await request.json()
                if isinstance(body, dict):
                    target = body.get("model_name")
            except Exception:
                pass
        if not target:
            raise HTTPException(status_code=400, detail="Missing required parameter 'model_name'")
        return activate_model(model_name=target)

    @app.post("/api/master/train-hitl", status_code=202)
    async def train_hitl_endpoint() -> dict[str, Any]:
        """Protected by asyncio.Lock, launch background retraining on verified HITL dataset."""
        if train_hitl_lock.locked() or job_manager.is_running:
            raise HTTPException(
                status_code=409,
                detail="A training or aggregation job is already in progress.",
            )

        active_hitl = getattr(app.state, "hitl", hitl_manager)
        verified_candidates = [
            active_hitl.verified_dir,
            active_hitl.root_dir / "verified",
        ]
        if active_hitl.root_dir.is_dir():
            for v in active_hitl.root_dir.glob("*/verified"):
                if v not in verified_candidates:
                    verified_candidates.append(v)

        chosen_verified: Path | None = None
        for cand in verified_candidates:
            if cand.is_dir() and any(
                f.is_file()
                for class_dir in cand.iterdir()
                if class_dir.is_dir()
                for f in class_dir.iterdir()
            ):
                chosen_verified = cand
                break

        if not chosen_verified:
            raise HTTPException(
                status_code=400,
                detail=f"No verified images found in {active_hitl.root_dir} to train on.",
            )

        base_model_file = str(active_model_p) if active_model_p else "models/global_model.pt"
        client_update_out = "models/client_updates/hitl_verified_parameters.pt"

        cmd_train = [
            sys.executable,
            "scripts/client_trainer.py",
            "--data-dir",
            str(chosen_verified),
            "--weights-in",
            base_model_file,
            "--weights-out",
            client_update_out,
            "--epochs",
            "2",
            "--unfreeze-last-block",
            "--allow-new-classes",
            "--client-id",
            "hitl_expert_curated",
        ]

        cmd_agg = [
            sys.executable,
            "scripts/aggregate.py",
            "--base-model",
            base_model_file,
            "--output",
            "models/global_model.pt",
            "--allow-new-classes",
            "--client-weights",
            client_update_out,
        ]

        await train_hitl_lock.acquire()

        async def _run_with_lock() -> None:
            try:
                await job_manager.run_pipeline("train_hitl_pipeline", [cmd_train, cmd_agg])
            finally:
                train_hitl_lock.release()

        asyncio.create_task(_run_with_lock())

        return {
            "status": "accepted",
            "message": "Re-entrenamiento HITL y agregación federada iniciados en segundo plano.",
            "pipeline": ["client_trainer.py", "aggregate.py"],
        }

    @app.get("/api/master/trash")
    def get_trash_models() -> dict[str, Any]:
        """List models currently held in the soft-delete trash quarantine."""
        cm = _get_catalog_manager()
        cm.auto_purge_expired()
        trash_items = cm.list_trash()
        total_bytes = sum(t.get("size_bytes", 0) for t in trash_items)
        return {
            "trash": trash_items,
            "total": len(trash_items),
            "total_bytes": total_bytes,
        }

    @app.post("/api/master/models/{slug}/delete")
    async def soft_delete_model_endpoint(
        slug: str,
        retention_days: int = Query(7, ge=1, le=90),
    ) -> dict[str, Any]:
        """Move a model to trash quarantine (soft delete) with automatic retention period."""
        cm = _get_catalog_manager()
        entry = cm.get_by_slug(slug, include_deleted=True)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Modelo '{slug}' no encontrado.")
        if entry.is_deleted:
            return {
                "status": "already_deleted",
                "slug": slug,
                "deleted_at": entry.deleted_at,
                "purge_after": entry.purge_after,
            }

        # Gracefully transition active state to None if this model was active
        nonlocal active_model_p, active_explicitly_cleared
        if active_model_p and entry.active_checkpoint and (
            entry.active_checkpoint == active_model_p.name
            or (Path("models") / entry.active_checkpoint).resolve() == active_model_p.resolve()
        ):
            active_model_p = None
            active_explicitly_cleared = True

        updated_entry = cm.soft_delete_model(slug, retention_days=retention_days)
        job_manager.logs.append(
            f"[*] [AUDIT] Modelo '{slug}' ({updated_entry.title}) movido a papelera. "
            f"Cuarentena: {retention_days} días (Auto-purga: {updated_entry.purge_after}). "
            f"Estado activo transicionado a None."
        )

        return {
            "status": "deleted",
            "action": "soft_delete",
            "slug": slug,
            "active_checkpoint": None,
            "deleted_at": updated_entry.deleted_at,
            "purge_after": updated_entry.purge_after,
            "retention_days": retention_days,
        }

    @app.post("/api/master/models/{slug}/archive")
    async def archive_model_endpoint(slug: str) -> dict[str, Any]:
        """Archive a model's active version, transitioning its active state to None."""
        cm = _get_catalog_manager()
        entry = cm.get_by_slug(slug, include_deleted=True)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Modelo '{slug}' no encontrado.")

        nonlocal active_model_p, active_explicitly_cleared
        if active_model_p and entry.active_checkpoint and (
            entry.active_checkpoint == active_model_p.name
            or (Path("models") / entry.active_checkpoint).resolve() == active_model_p.resolve()
        ):
            active_model_p = None
            active_explicitly_cleared = True

        updated_entry = cm.archive_model(slug)
        job_manager.logs.append(
            f"[*] [AUDIT] Modelo '{slug}' ({updated_entry.title}) archivado. "
            f"Estado activo transicionado a None."
        )
        return {
            "status": "archived",
            "slug": slug,
            "active_checkpoint": None,
            "message": f"Modelo '{updated_entry.title}' archivado correctamente.",
        }

    @app.post("/api/master/models/{slug}/restore")
    async def restore_model_endpoint(slug: str) -> dict[str, Any]:
        """Restore a soft-deleted model back to active catalog."""
        cm = _get_catalog_manager()
        entry = cm.get_by_slug(slug, include_deleted=True)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Modelo '{slug}' no encontrado.")
        if not entry.is_deleted:
            return {"status": "already_active", "slug": slug}

        restored_entry = cm.restore_model(slug)
        nonlocal active_model_p, active_explicitly_cleared
        if active_model_p is None and restored_entry.active_checkpoint:
            active_explicitly_cleared = False
            _resolve_and_reload_model(force=True)

        job_manager.logs.append(
            f"[*] [AUDIT] Modelo '{slug}' ({restored_entry.title}) "
            "restaurado con éxito al catálogo activo."
        )

        return {
            "status": "restored",
            "slug": slug,
            "active_checkpoint": restored_entry.active_checkpoint,
            "model": restored_entry.model_dump(),
        }

    @app.post("/api/master/models/{slug}/purge")
    async def purge_model_endpoint(
        slug: str,
        confirm_slug: str = Query(
            ..., description="Confirmation matching the slug to be permanently destroyed"
        ),
    ) -> dict[str, Any]:
        """Permanently destroy a model and free unshared disk space (Hard Delete)."""
        if confirm_slug.strip() != slug.strip():
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Doble confirmación requerida: 'confirm_slug' ('{confirm_slug}') "
                    f"no coincide exactamente con el slug del modelo ('{slug}')."
                ),
            )

        cm = _get_catalog_manager()
        entry = cm.get_by_slug(slug, include_deleted=True)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Modelo '{slug}' no encontrado.")

        # Gracefully transition active state to None if purging the currently active model
        nonlocal active_model_p, active_explicitly_cleared
        if active_model_p and entry.active_checkpoint and (
            entry.active_checkpoint == active_model_p.name
            or (Path("models") / entry.active_checkpoint).resolve() == active_model_p.resolve()
        ):
            active_model_p = None
            active_explicitly_cleared = True

        result = cm.purge_model(slug, remove_files=True)
        job_manager.logs.append(
            f"[*] [AUDIT] PURGA PERMANENTE: Modelo '{slug}' destruido. "
            f"Archivos físicos eliminados: {result['deleted_files']}, "
            f"Espacio liberado: {result['freed_bytes'] // 1024} KB"
        )

        return {
            "status": "purged",
            "slug": slug,
            "details": result,
        }

    @app.get("/api/master/models/{slug}/history")
    def get_model_history_endpoint(slug: str) -> dict[str, Any]:
        """Fetch history iterations (accuracy, loss, F1) for a model."""
        cm = _get_catalog_manager()
        history = get_model_history(cm.models_dir, slug)
        return {"slug": slug, "history": history, "total": len(history)}

    _inspect_cache: dict[tuple[str, int], dict[str, Any]] = {}
    _available_ds_cache: dict[str, Any] = {"ts": 0.0, "payload": None}

    def _count_images_scandir(dir_path: Path, valid_exts: set[str]) -> int:
        cnt = 0
        try:
            with os.scandir(dir_path) as it:
                for entry in it:
                    if entry.is_file(follow_symlinks=False):
                        dot_idx = entry.name.rfind(".")
                        if dot_idx != -1 and entry.name[dot_idx:].lower() in valid_exts:
                            cnt += 1
        except OSError:
            pass
        return cnt

    UNLABELED_DIR_NAMES = frozenset({
        "unlabeled",
        "unknown",
        "sin_etiquetar",
        "sin_clasificar",
        "sin_etiqueta",
        "sin-etiquetar",
        "sin-clasificar",
    })

    def detect_dataset_structure(p: Path) -> dict[str, Any]:
        """Detect dataset structure: Flat vs Pre-Split, and count unlabeled images."""
        from agrifed_khaos.data import IMAGE_EXTENSIONS

        candidates = [p]
        if p.name.lower() in ("train", "training"):
            candidates.append(p.parent)
        try:
            candidates.extend(
                [d for d in p.iterdir() if d.is_dir() and not d.name.startswith((".", "_"))]
            )
        except OSError:
            pass

        split_root: Path | None = None
        train_dir: Path | None = None
        val_dir: Path | None = None
        test_dir: Path | None = None

        for cand in candidates:
            if not cand.is_dir():
                continue
            try:
                entries = {e.name.lower(): e for e in cand.iterdir() if e.is_dir()}
            except OSError:
                continue

            c_train = entries.get("train") or entries.get("training")
            c_val = (
                entries.get("val")
                or entries.get("valid")
                or entries.get("validation")
            )
            c_test = entries.get("test") or entries.get("testing")

            if c_train and c_train.is_dir():
                train_subdirs = [
                    d
                    for d in c_train.iterdir()
                    if d.is_dir()
                    and not d.name.startswith((".", "_"))
                    and d.name.lower() not in UNLABELED_DIR_NAMES
                ]
                if train_subdirs:
                    split_root = cand
                    train_dir = c_train
                    val_dir = c_val if (c_val and c_val.is_dir()) else None
                    test_dir = c_test if (c_test and c_test.is_dir()) else None
                    break

        def count_images_in_tree(d: Path | None) -> int:
            if not d or not d.is_dir():
                return 0
            cnt = 0
            for _root, dirs, files in os.walk(d):
                dirs[:] = [
                    x
                    for x in dirs
                    if not x.startswith((".", "_"))
                    and x.lower() not in UNLABELED_DIR_NAMES
                ]
                for f in files:
                    dot_idx = f.rfind(".")
                    if dot_idx != -1 and f[dot_idx:].lower() in IMAGE_EXTENSIONS:
                        cnt += 1
            return cnt

        unlabeled_count = 0
        search_root = split_root or p

        # Root-level images directly outside any class folder
        try:
            with os.scandir(search_root) as it:
                for entry in it:
                    if entry.is_file(follow_symlinks=False):
                        dot = entry.name.rfind(".")
                        if dot != -1 and entry.name[dot:].lower() in IMAGE_EXTENSIONS:
                            unlabeled_count += 1
        except OSError:
            pass

        # Images inside explicit unlabeled folders
        for r, _dirs, files in os.walk(search_root):
            dname = Path(r).name.lower()
            if dname in UNLABELED_DIR_NAMES:
                for f in files:
                    dot = f.rfind(".")
                    if dot != -1 and f[dot:].lower() in IMAGE_EXTENSIONS:
                        unlabeled_count += 1

        if split_root and train_dir:
            train_cnt = count_images_in_tree(train_dir)
            val_cnt = count_images_in_tree(val_dir)
            test_cnt = count_images_in_tree(test_dir)
            total_split = train_cnt + val_cnt + test_cnt
            pct_train = (
                round((train_cnt / total_split * 100), 1) if total_split else 0.0
            )
            pct_val = (
                round((val_cnt / total_split * 100), 1) if total_split else 0.0
            )
            pct_test = (
                round((test_cnt / total_split * 100), 1) if total_split else 0.0
            )
            return {
                "is_pre_split": True,
                "split_distribution": {
                    "train": train_cnt,
                    "val": val_cnt,
                    "test": test_cnt,
                },
                "split_percentages": {
                    "train": pct_train,
                    "val": pct_val,
                    "test": pct_test,
                },
                "unlabeled_count": unlabeled_count,
                "train_dir": train_dir,
                "split_root": split_root,
            }

        return {
            "is_pre_split": False,
            "split_distribution": None,
            "split_percentages": None,
            "unlabeled_count": unlabeled_count,
            "train_dir": None,
            "split_root": None,
        }

    def profile_dataset_health_sync(dataset_dir: Path | str) -> dict[str, Any]:
        """Profile dataset health, detecting corruption, duplicate hashes, and size variance."""
        target_path = Path(dataset_dir)
        if not target_path.exists():
            return {
                "health_score": 0,
                "corrupted_count": 0,
                "duplicates_count": 0,
                "resolution_warning": False,
                "total_scanned": 0,
                "min_resolution": [0, 0],
                "max_resolution": [0, 0],
            }

        valid_exts = frozenset({
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp",
            ".tif",
            ".tiff",
            ".ppm",
            ".pgm",
        })

        image_paths: list[Path] = []
        try:
            for p_file in target_path.rglob("*"):
                if (
                    p_file.is_file()
                    and p_file.suffix.lower() in valid_exts
                    and not p_file.name.startswith((".", "_"))
                ):
                    if any(
                        part.startswith((".", "_")) or part.lower() == "__macosx"
                        for part in p_file.parts
                    ):
                        continue
                    image_paths.append(p_file)
        except OSError:
            pass

        if not image_paths:
            return {
                "health_score": 0,
                "corrupted_count": 0,
                "duplicates_count": 0,
                "resolution_warning": False,
                "total_scanned": 0,
                "min_resolution": [0, 0],
                "max_resolution": [0, 0],
            }

        corrupted_count = 0
        duplicates_count = 0
        seen_hashes: set[str] = set()

        min_w = float("inf")
        max_w = 0
        min_h = float("inf")
        max_h = 0
        min_area = float("inf")
        max_area = 0
        valid_count = 0

        for img_p in image_paths:
            # a) Duplicate detection via chunked binary MD5 (streaming, 64KB)
            try:
                hasher = hashlib.md5()
                with img_p.open("rb") as f_img:
                    while chunk := f_img.read(65536):
                        hasher.update(chunk)
                file_hash = hasher.hexdigest()
                if file_hash in seen_hashes:
                    duplicates_count += 1
                else:
                    seen_hashes.add(file_hash)
            except OSError:
                corrupted_count += 1
                continue

            # b) Header inspection & corruption check via PIL verify() (no full pixel decoding)
            try:
                with Image.open(img_p) as img:
                    w, h = img.size
                    img.verify()
            except Exception:
                corrupted_count += 1
                continue

            valid_count += 1
            area = w * h
            if w < min_w:
                min_w = w
            if w > max_w:
                max_w = w
            if h < min_h:
                min_h = h
            if h > max_h:
                max_h = h
            if area < min_area:
                min_area = area
            if area > max_area:
                max_area = area

        total_scanned = len(image_paths)
        resolution_warning = False
        if valid_count > 1 and min_area < float("inf") and max_area > 0:
            area_ratio = max_area / max(1, min_area)
            dim_ratio = max(max_w, max_h) / max(1, min(min_w, min_h))
            if (
                area_ratio >= 16.0
                or dim_ratio >= 6.0
                or (min(min_w, min_h) < 64 and max(max_w, max_h) >= 1024)
            ):
                resolution_warning = True

        score = 100
        if total_scanned > 0:
            if corrupted_count > 0:
                corrupt_ratio = corrupted_count / total_scanned
                score -= int(35 + 65 * corrupt_ratio)
            if duplicates_count > 0:
                dup_ratio = duplicates_count / total_scanned
                score -= min(25, int(10 + 40 * dup_ratio))
            if resolution_warning:
                score -= 15

        health_score = max(0, min(100, score))

        return {
            "health_score": health_score,
            "corrupted_count": corrupted_count,
            "duplicates_count": duplicates_count,
            "resolution_warning": resolution_warning,
            "total_scanned": total_scanned,
            "min_resolution": [int(min_w), int(min_h)] if valid_count > 0 else [0, 0],
            "max_resolution": [int(max_w), int(max_h)] if valid_count > 0 else [0, 0],
        }

    def deduplicate_dataset_sync(
        dataset_dir: Path | str,
        remove_duplicates: bool = True,
        remove_corrupted: bool = False,
    ) -> dict[str, Any]:
        """Scan dataset directory, delete exact MD5 duplicates (and optionally corrupted files)."""
        target_path = Path(dataset_dir)
        valid_exts = frozenset({
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp",
            ".tif",
            ".tiff",
            ".ppm",
            ".pgm",
        })

        image_paths: list[Path] = []
        try:
            for p_file in target_path.rglob("*"):
                if (
                    p_file.is_file()
                    and p_file.suffix.lower() in valid_exts
                    and not p_file.name.startswith((".", "_"))
                ):
                    if any(
                        part.startswith((".", "_")) or part.lower() == "__macosx"
                        for part in p_file.parts
                    ):
                        continue
                    image_paths.append(p_file)
        except OSError:
            pass

        image_paths.sort()
        seen_hashes: set[str] = set()
        deleted_count = 0
        corrupted_removed = 0

        for img_p in image_paths:
            if remove_corrupted:
                try:
                    with Image.open(img_p) as img:
                        img.verify()
                except Exception:
                    try:
                        os.remove(img_p)
                        corrupted_removed += 1
                    except OSError:
                        pass
                    continue

            try:
                hasher = hashlib.md5()
                with img_p.open("rb") as f_img:
                    while chunk := f_img.read(65536):
                        hasher.update(chunk)
                file_hash = hasher.hexdigest()
            except OSError:
                if remove_corrupted:
                    try:
                        os.remove(img_p)
                        corrupted_removed += 1
                    except OSError:
                        pass
                continue

            if file_hash in seen_hashes:
                if remove_duplicates:
                    try:
                        os.remove(img_p)
                        deleted_count += 1
                    except OSError:
                        pass
            else:
                seen_hashes.add(file_hash)

        # Recalculate per-class counts and total images
        subdirs = [
            d
            for d in target_path.iterdir()
            if d.is_dir()
            and not d.name.startswith((".", "_"))
            and d.name.lower() != "__macosx"
        ]
        image_counts: dict[str, int] = {}
        if subdirs:
            for d in sorted(subdirs, key=lambda x: x.name):
                image_counts[d.name] = _count_images_scandir(d, set(valid_exts))
            total_images = sum(image_counts.values())
        else:
            total_images = _count_images_scandir(target_path, set(valid_exts))

        new_health = profile_dataset_health_sync(target_path)
        if total_images == 0 and new_health.get("total_scanned", 0) > 0:
            total_images = new_health["total_scanned"]

        return {
            "status": "ok",
            "dataset_path": str(target_path),
            "deleted_count": deleted_count,
            "corrupted_removed": corrupted_removed,
            "total_images": total_images,
            "image_counts": image_counts,
            "health_profile": new_health,
        }

    @app.post("/api/master/datasets/deduplicate")
    async def deduplicate_dataset_endpoint(req: DatasetDeduplicateRequest) -> dict[str, Any]:
        """Deduplicate exact MD5 image files (and optionally purge corrupted headers) async."""
        raw_p = req.dataset_path.strip()
        if not raw_p:
            raise HTTPException(
                status_code=400, detail="La ruta del dataset ('dataset_path') es obligatoria."
            )

        p = Path(raw_p)
        if not p.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"El directorio '{raw_p}' no existe en el servidor o no es una carpeta.",
            )

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _profiler_executor,
            deduplicate_dataset_sync,
            p,
            req.remove_duplicates,
            req.remove_corrupted,
        )
        _inspect_cache.clear()
        return result

    @app.post("/api/master/datasets/inspect")
    def inspect_dataset_directory(req: DatasetInspectRequest) -> dict[str, Any]:
        """Inspect a dataset directory, discover classes, count image samples, and suggest arch."""
        raw_p = req.dataset_path.strip()
        if not raw_p:
            raise HTTPException(
                status_code=400, detail="La ruta del dataset ('dataset_path') es obligatoria."
            )

        p = Path(raw_p)
        if not p.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"El directorio '{raw_p}' no existe en el servidor o no es una carpeta.",
            )

        try:
            p_mtime = p.stat().st_mtime_ns
            cache_key = (raw_p, p_mtime)
            if cache_key in _inspect_cache:
                return dict(_inspect_cache[cache_key])
        except OSError:
            cache_key = None

        from agrifed_khaos.yolo import inspect_yolo_dataset, is_yolo_dataset

        # 1. Check if dataset is in Ultralytics YOLO Object Detection format
        if is_yolo_dataset(p):
            yolo_res = inspect_yolo_dataset(p, preferred_split="train")
            if yolo_res.get("valid"):
                yolo_res["original_path"] = raw_p
                yolo_res["auto_detected"] = (
                    Path(yolo_res["dataset_path"]).resolve() != p.resolve()
                )
                from agrifed_khaos.metadata import suggest_dataset_metadata

                suggested = suggest_dataset_metadata(
                    dataset_path=str(yolo_res["dataset_path"]),
                    classes=yolo_res.get("classes", []),
                    total_images=yolo_res.get("total_images", 0),
                    source_name=req.source_name or p.name,
                    is_yolo=True,
                    total_boxes=yolo_res.get("total_boxes", 0),
                )
                yolo_res.update(suggested)
                if "health_profile" not in yolo_res:
                    yolo_res["health_profile"] = profile_dataset_health_sync(
                        Path(yolo_res.get("dataset_path", p))
                    )
                yolo_res.setdefault("is_pre_split", True)
                yolo_res.setdefault("split_distribution", None)
                yolo_res.setdefault("split_percentages", None)
                yolo_res.setdefault("unlabeled_count", 0)
                if cache_key is not None:
                    _inspect_cache[cache_key] = dict(yolo_res)
                return yolo_res

        # 2. Otherwise process as standard ImageFolder classification dataset
        from agrifed_khaos.data import IMAGE_EXTENSIONS, find_dataset_class_dir

        struct_info = detect_dataset_structure(p)

        target_dir = find_dataset_class_dir(p, preferred_split="train")
        auto_detected = (target_dir.resolve() != p.resolve())

        subdirs = [
            d
            for d in target_dir.iterdir()
            if d.is_dir()
            and not d.name.startswith((".", "_"))
            and d.name.lower() != "__macosx"
            and d.name.lower() not in UNLABELED_DIR_NAMES
        ]

        if not subdirs:
            return {
                "valid": False,
                "dataset_path": str(target_dir),
                "original_path": raw_p,
                "auto_detected": auto_detected,
                "classes": [],
                "image_counts": {},
                "total_images": 0,
                "is_pre_split": struct_info["is_pre_split"],
                "split_distribution": struct_info["split_distribution"],
                "split_percentages": struct_info["split_percentages"],
                "unlabeled_count": struct_info["unlabeled_count"],
                "error": (
                    f"No se detectaron subcarpetas de clase en '{raw_p}'. "
                    f"Estructura esperada: {raw_p}/<clase_1>, {raw_p}/<clase_2>..."
                ),
            }

        image_counts: dict[str, int] = {}
        for d in sorted(subdirs, key=lambda x: x.name):
            image_counts[d.name] = _count_images_scandir(d, IMAGE_EXTENSIONS)

        total_images = sum(image_counts.values())
        classes = sorted(image_counts.keys())

        empty_classes = [cls for cls, cnt in image_counts.items() if cnt == 0]
        if total_images == 0 or empty_classes:
            empty_info = f" Clases vacías: {empty_classes}." if empty_classes else ""
            return {
                "valid": False,
                "dataset_path": str(target_dir),
                "original_path": raw_p,
                "auto_detected": auto_detected,
                "classes": classes,
                "image_counts": image_counts,
                "total_images": total_images,
                "is_pre_split": struct_info["is_pre_split"],
                "split_distribution": struct_info["split_distribution"],
                "split_percentages": struct_info["split_percentages"],
                "unlabeled_count": struct_info["unlabeled_count"],
                "error": (
                    f"Se encontraron subcarpetas ({classes}), pero no contienen "
                    "imágenes con extensiones soportadas (.jpg, .jpeg, .png, .webp, .bmp)."
                    f"{empty_info}"
                ),
            }

        if total_images > 5000:
            rec_arch = "efficientnet_v2_s"
            rec_reason = (
                "Volumen robusto (> 5.000 imgs): Alta capacidad representacional y "
                "convergencia SOTA."
            )
        elif total_images >= 1000:
            rec_arch = "efficientnet_b0"
            rec_reason = (
                "Volumen intermedio (1.000 - 5.000 imgs): Óptimo ratio precisión/parámetros."
            )
        else:
            rec_arch = "resnet18"
            rec_reason = (
                "Volumen reducido (< 1.000 imgs): Balance robusto y prevención de sobreajuste."
            )

        rec_result = {
            "recommended_arch": rec_arch,
            "reason": rec_reason,
        }

        from agrifed_khaos.metadata import suggest_dataset_metadata

        suggested = suggest_dataset_metadata(
            dataset_path=str(target_dir),
            classes=classes,
            total_images=total_images,
            image_counts=image_counts,
            source_name=req.source_name or p.name,
            is_yolo=False,
        )

        res_payload = {
            "valid": True,
            "dataset_path": str(target_dir),
            "original_path": raw_p,
            "auto_detected": auto_detected,
            "classes": classes,
            "image_counts": image_counts,
            "total_images": total_images,
            "recommended_arch": rec_arch,
            "recommendation_details": rec_result,
            "health_profile": profile_dataset_health_sync(target_dir),
            "is_pre_split": struct_info["is_pre_split"],
            "split_distribution": struct_info["split_distribution"],
            "split_percentages": struct_info["split_percentages"],
            "unlabeled_count": struct_info["unlabeled_count"],
            **suggested,
        }
        if cache_key is not None:
            _inspect_cache[cache_key] = dict(res_payload)
        return res_payload

    def _safe_extract_archive(
        archive_path: Path,
        dest_dir: Path,
    ) -> Path:
        """Extract a .zip or .tar archive safely guarding against zip-slip path traversal."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        resolved_dest = dest_dir.resolve()

        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, "r") as zf:
                for member in zf.infolist():
                    target_p = (dest_dir / member.filename).resolve()
                    if not str(target_p).startswith(str(resolved_dest)):
                        raise HTTPException(
                            status_code=400,
                            detail=f"Ruta maliciosa detectada en archivo: {member.filename}",
                        )
                zf.extractall(resolved_dest)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, "r:*") as tf:
                for member in tf.getmembers():
                    target_p = (dest_dir / member.name).resolve()
                    if not str(target_p).startswith(str(resolved_dest)):
                        raise HTTPException(
                            status_code=400,
                            detail=f"Ruta maliciosa detectada en archivo: {member.name}",
                        )
                tf.extractall(resolved_dest)
        else:
            raise HTTPException(
                status_code=400,
                detail="Formato de archivo comprimido no soportado (.zip, .tar, .tar.gz).",
            )

        # Drill down if archive contains YOLO dataset or nested dataset class directory
        from agrifed_khaos.yolo import is_yolo_dataset

        if is_yolo_dataset(dest_dir):
            return dest_dir

        from agrifed_khaos.data import find_dataset_class_dir

        return find_dataset_class_dir(dest_dir, preferred_split="train")

    @app.get("/api/master/datasets/available")
    def get_available_datasets_endpoint() -> dict[str, Any]:
        """List pre-mounted or discovered agricultural datasets on server and agriKhaos."""
        now_t = time.monotonic()
        cached_ds = _available_ds_cache["payload"]
        if cached_ds is not None and (now_t - _available_ds_cache["ts"]) < 30.0:
            return cached_ds

        from agrifed_khaos.data import IMAGE_EXTENSIONS, find_dataset_class_dir

        search_roots = [
            Path("/data/EnfermedadesHoja_v1.0"),
            Path("/data/agriKhaos/enfermedadeshoja_v1.0"),
            Path("/data/agriKhaos/mangofruit_v1.0"),
            Path("/data/agriKhaos/olivefruit_v1.1/classification"),
            Path("/data/agriKhaos/olivefruit_v1.1"),
            Path("/data"),
            Path("/data/agriKhaos"),
        ]
        skip_names = {"agrikhaos", "uploads", "agrifed-khaos", "agrivision"}
        seen_candidates: set[str] = set()
        found: dict[str, dict[str, Any]] = {}

        for root in search_roots:
            if not root.is_dir():
                continue
            candidates_to_check: list[Path] = []
            if root in (Path("/data"), Path("/data/agriKhaos")):
                try:
                    candidates_to_check.extend([
                        d
                        for d in root.iterdir()
                        if d.is_dir()
                        and not d.name.startswith((".", "_"))
                        and d.name.lower() not in skip_names
                    ])
                except OSError:
                    pass
            else:
                candidates_to_check.append(root)

            for cand in candidates_to_check:
                try:
                    cand_key = str(cand.resolve())
                except OSError:
                    cand_key = str(cand)
                if cand_key in seen_candidates:
                    continue
                seen_candidates.add(cand_key)

                try:
                    resolved = find_dataset_class_dir(cand, preferred_split="train")
                    if not resolved.is_dir():
                        continue
                    real_p = str(resolved.resolve())
                    if real_p in found:
                        continue
                    subdirs = [
                        d
                        for d in resolved.iterdir()
                        if d.is_dir() and not d.name.startswith((".", "_"))
                    ]
                    if not subdirs:
                        continue
                    cnt_map = {
                        d.name: _count_images_scandir(d, IMAGE_EXTENSIONS)
                        for d in subdirs
                    }
                    total = sum(cnt_map.values())
                    if total > 0 and all(c > 0 for c in cnt_map.values()):
                        title_name = cand.name.replace("_", " ").title()
                        if "Enfermedades" in title_name:
                            badge = "Olivo / Patologías"
                            title_name = "Enfermedades de Hoja de Olivo (v1.0)"
                        elif "Mango" in title_name:
                            badge = "Mango / Calidad y Defectos"
                            title_name = "Mango Fruit Quality (v1.0)"
                        elif "Olive" in title_name or "Aceituna" in title_name:
                            badge = "Aceituna / Calidad"
                            title_name = "Olive Fruit Classification (v1.1)"
                        else:
                            badge = "AgriVision Dataset"

                        found[real_p] = {
                            "title": title_name,
                            "badge": badge,
                            "root_path": str(cand),
                            "class_path": str(resolved),
                            "classes": sorted(cnt_map.keys()),
                            "image_counts": cnt_map,
                            "total_images": total,
                        }
                except Exception:
                    continue

        dataset_list = sorted(found.values(), key=lambda x: x["total_images"], reverse=True)
        payload = {"datasets": dataset_list, "total": len(dataset_list)}
        _available_ds_cache["ts"] = now_t
        _available_ds_cache["payload"] = payload
        return payload

    @app.post("/api/master/datasets/upload")
    async def upload_dataset_archive_or_folder(
        file: UploadFile | None = File(None),  # noqa: B008
        files: list[UploadFile] | None = File(None),  # noqa: B008
        paths: list[str] | None = Form(None),  # noqa: B008
    ) -> dict[str, Any]:
        """Upload a dataset as a .zip/.tar archive or as multi-file folder (no size limits)."""
        uploads_dir = Path("data/uploads")
        uploads_dir.mkdir(parents=True, exist_ok=True)
        session_id = f"upload_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        dest_dir = uploads_dir / session_id

        # 1. Single archive upload (.zip, .tar, .tar.gz)
        target_archive = None
        if file and file.filename:
            target_archive = file
        elif files and len(files) == 1 and files[0].filename:
            raw_ext_candidate = "".join(Path(files[0].filename).suffixes).lower()
            if any(
                raw_ext_candidate.endswith(ext)
                for ext in (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2")
            ):
                target_archive = files[0]

        if target_archive and target_archive.filename:
            raw_ext = "".join(Path(target_archive.filename).suffixes).lower()
            if not any(
                raw_ext.endswith(ext)
                for ext in (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2")
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Formato no soportado. Por favor suba un archivo .zip o .tar.gz.",
                )

            temp_archive = uploads_dir / f"{session_id}_archive{raw_ext}"
            try:
                with temp_archive.open("wb") as f_out:
                    while chunk := await target_archive.read(1024 * 1024):
                        f_out.write(chunk)

                resolved_path = await asyncio.to_thread(
                    _safe_extract_archive, temp_archive, dest_dir
                )
            finally:
                temp_archive.unlink(missing_ok=True)

        # 2. Multi-file folder upload with relative paths
        elif files and len(files) > 0:
            dest_dir.mkdir(parents=True, exist_ok=True)
            resolved_dest = dest_dir.resolve()
            for idx, f in enumerate(files):
                rel_str = (
                    paths[idx]
                    if paths and idx < len(paths)
                    else (f.filename or f"item_{idx}.jpg")
                )
                clean_rel = rel_str.strip().lstrip("/")
                target_p = (dest_dir / clean_rel).resolve()
                if not str(target_p).startswith(str(resolved_dest)):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Ruta relativa no válida o intento de path traversal: {rel_str}",
                    )
                target_p.parent.mkdir(parents=True, exist_ok=True)
                img_data = await f.read()
                target_p.write_bytes(img_data)

            from agrifed_khaos.yolo import is_yolo_dataset

            if is_yolo_dataset(dest_dir):
                resolved_path = dest_dir
            else:
                from agrifed_khaos.data import find_dataset_class_dir

                resolved_path = find_dataset_class_dir(dest_dir, preferred_split="train")
        else:
            raise HTTPException(
                status_code=400,
                detail="No se ha recibido ningún archivo comprimido ni archivos de carpeta.",
            )

        source_filename = None
        if target_archive and target_archive.filename:
            source_filename = target_archive.filename
        elif files and len(files) > 0 and files[0].filename:
            p_parts = Path(files[0].filename).parts
            if len(p_parts) > 1:
                source_filename = p_parts[0]

        inspect_req = DatasetInspectRequest(
            dataset_path=str(resolved_path),
            source_name=source_filename,
        )
        loop = asyncio.get_running_loop()
        health_profile = await loop.run_in_executor(
            _profiler_executor,
            profile_dataset_health_sync,
            resolved_path,
        )
        inspect_res = inspect_dataset_directory(inspect_req)
        inspect_res["health_profile"] = health_profile
        return inspect_res

    @app.post("/api/master/datasets/fetch-url")
    async def fetch_dataset_from_url(req: DatasetFetchURLRequest) -> dict[str, Any]:
        """Download and extract a dataset archive from a remote URL (no size limits)."""
        raw_url = req.url.strip()
        if not raw_url:
            raise HTTPException(status_code=400, detail="La URL es obligatoria.")

        parsed = urllib.parse.urlparse(raw_url)
        if parsed.scheme not in ("http", "https"):
            raise HTTPException(
                status_code=400, detail="Solo se admiten enlaces HTTP o HTTPS."
            )

        hostname = (parsed.hostname or "").lower()
        blocked_hosts = {
            "169.254.169.254",
            "metadata.google.internal",
            "metadata",
            "instance-data",
            "100.100.100.200",
        }
        if hostname in blocked_hosts or hostname.startswith("169.254."):
            raise HTTPException(
                status_code=400, detail="Host no permitido por política de seguridad."
            )

        downloads_dir = Path("data/downloads")
        downloads_dir.mkdir(parents=True, exist_ok=True)
        session_id = f"url_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        dest_dir = downloads_dir / session_id
        temp_archive = downloads_dir / f"{session_id}.archive"

        try:
            async with httpx.AsyncClient(timeout=600.0, follow_redirects=True) as client:
                async with client.stream("GET", raw_url) as response:
                    if response.status_code >= 400:
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                f"El servidor remoto devolvió código "
                                f"{response.status_code} ({response.reason_phrase})."
                            ),
                        )
                    with temp_archive.open("wb") as f_out:
                        async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                            f_out.write(chunk)

            resolved_path = _safe_extract_archive(temp_archive, dest_dir)
        except HTTPException:
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise
        except Exception as err:
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise HTTPException(
                status_code=400,
                detail=f"Error durante la descarga del dataset: {err}",
            ) from err
        finally:
            temp_archive.unlink(missing_ok=True)

        url_filename = Path(parsed.path).name or "archive"
        inspect_req = DatasetInspectRequest(
            dataset_path=str(resolved_path),
            source_name=url_filename,
        )
        loop = asyncio.get_running_loop()
        health_profile = await loop.run_in_executor(
            _profiler_executor,
            profile_dataset_health_sync,
            resolved_path,
        )
        inspect_res = inspect_dataset_directory(inspect_req)
        inspect_res["health_profile"] = health_profile
        return inspect_res

    @app.post("/api/master/models/auto-fl", status_code=202)
    async def auto_fl_train_model(req: AutoFLRequest) -> dict[str, Any]:
        """Zero-to-Model Pipeline: Train a new model from dataset and auto-register it."""
        if job_manager.is_running:
            raise HTTPException(
                status_code=409,
                detail=f"Trabajo en ejecución ('{job_manager.action}'). Espere a que finalice.",
            )

        if not req.title.strip():
            raise HTTPException(status_code=400, detail="El título del modelo es obligatorio.")

        if not req.dataset_path.strip():
            raise HTTPException(
                status_code=400, detail="La ruta del dataset ('dataset_path') es obligatoria."
            )

        ds_path = Path(req.dataset_path.strip())
        if not ds_path.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"El directorio del dataset '{req.dataset_path}' no existe en el servidor.",
            )

        from agrifed_khaos.data import find_dataset_class_dir

        resolved_ds_path = find_dataset_class_dir(ds_path, preferred_split="train")
        subdirs = [
            d
            for d in resolved_ds_path.iterdir()
            if d.is_dir()
            and not d.name.startswith((".", "_"))
            and d.name.lower() != "__macosx"
        ]

        if not subdirs:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"El directorio '{req.dataset_path}' no contiene subcarpetas de clases. "
                    f"Estructura esperada: {req.dataset_path}/<clase_1>..."
                ),
            )

        valid_exts = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
            ".tif",
            ".tiff",
            ".ppm",
            ".pgm",
        }
        image_counts: dict[str, int] = {}
        for d in subdirs:
            cnt = sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower() in valid_exts)
            image_counts[d.name] = cnt

        total_images = sum(image_counts.values())
        empty_classes = [cls for cls, cnt in image_counts.items() if cnt == 0]
        if total_images == 0 or empty_classes:
            empty_msg = f" (carpetas vacías: {empty_classes})" if empty_classes else ""
            raise HTTPException(
                status_code=400,
                detail=(
                    f"El directorio '{req.dataset_path}' no contiene imágenes válidas en todas "
                    f"sus clases{empty_msg}. Cada subcarpeta debe contener al menos 1 imagen "
                    "(.jpg, .jpeg, .png, .webp, .bmp)."
                ),
            )

        class_names = sorted([d.name for d in subdirs])
        slug = slugify(req.slug or req.title)

        cm = _get_catalog_manager()
        models_root = cm.models_dir
        models_root.mkdir(parents=True, exist_ok=True)
        out_weights = models_root / f"{slug}_base.pt"
        out_meta = models_root / f"{slug}.meta.json"

        cmd = [
            sys.executable,
            "scripts/train_base.py",
            "--data-dir",
            str(resolved_ds_path),
            "--output",
            str(out_weights),
            "--metadata",
            str(out_meta),
            "--epochs",
            str(max(1, req.epochs)),
            "--batch-size",
            str(max(1, req.batch_size)),
            "--lr",
            str(req.lr),
            "--arch",
            req.architecture,
            "--target-client-tier",
            req.target_client_tier,
            "--tl-strategy",
            req.transfer_learning,
            "--loss",
            req.loss_function,
            "--aug",
            req.augmentation,
        ]

        struct = detect_dataset_structure(ds_path)
        if not struct.get("is_pre_split", False) and req.train_split is not None:
            cmd.extend([
                "--train-split",
                str(req.train_split),
                "--val-split",
                str(req.val_split if req.val_split is not None else 0.1),
                "--test-split",
                str(req.test_split if req.test_split is not None else 0.1),
            ])

        async def on_training_success() -> None:
            nonlocal active_model_p, active_explicitly_cleared
            trained_arch = req.architecture if req.architecture != "auto" else "resnet18"
            trained_classes = class_names
            metrics: dict[str, Any] = {}
            if out_meta.is_file():
                try:
                    meta_json = json.loads(out_meta.read_text(encoding="utf-8"))
                    trained_arch = meta_json.get("arch", trained_arch)
                    trained_classes = meta_json.get("classes", trained_classes)
                    metrics = meta_json.get("metrics", {})
                except Exception:
                    pass

            now_ts = time.strftime("%Y-%m-%d %H:%M:%S")
            v1_rec = ModelVersionRecord(
                version_id="v1",
                checkpoint=out_weights.name,
                created_at=now_ts,
                parent_version=None,
                origin="train_base",
                status="active",
                metrics_summary=metrics,
            )

            new_entry = LogicalModelEntry(
                slug=slug,
                title=req.title.strip(),
                domain=req.domain.strip() or "General",
                crop=req.crop.strip(),
                target_element=req.target_element.strip(),
                purpose=req.purpose.strip(),
                thumbnail_url=req.thumbnail_url.strip() if req.thumbnail_url else None,
                description=req.description.strip()
                or f"Modelo entrenado con Asistente Auto-FL desde '{ds_path.name}'.",
                input_guide=req.input_guide.strip()
                or "Fotografía clara de la muestra con iluminación adecuada.",
                active_checkpoint=out_weights.name,
                versions=[out_weights.name],
                version_history=[v1_rec],
                architecture=trained_arch,
                num_classes=len(trained_classes),
                classes=trained_classes,
                updated_at=now_ts,
            )
            cm.upsert_entry(new_entry)

            # Record history
            append_model_history(
                models_dir=models_root,
                slug=slug,
                version="v1",
                checkpoint_name=out_weights.name,
                accuracy=metrics.get("accuracy"),
                macro_f1=metrics.get("macro_f1"),
                loss=metrics.get("loss"),
                metrics=metrics,
                metadata={
                    "origin": "auto_fl",
                    "dataset_path": str(ds_path),
                    "epochs": req.epochs,
                },
                timestamp=now_ts,
            )

            if active_model_p is None:
                active_model_p = out_weights
                active_explicitly_cleared = False
                _resolve_and_reload_model(force=True)

            job_manager.logs.append(
                f"[*] [AUTO-FL] Modelo '{slug}' ({new_entry.title}) registrado en el catálogo. "
                f"Clases detectadas ({len(trained_classes)}): {', '.join(trained_classes)}. "
                f"Checkpoint activo fijado en '{out_weights.name}'."
            )

        await job_manager.run_command(
            action=f"auto_fl_train:{slug}",
            cmd=cmd,
            on_success=on_training_success,
        )

        return {
            "status": "accepted",
            "action": f"auto_fl_train:{slug}",
            "slug": slug,
            "title": req.title,
            "dataset_path": str(ds_path),
            "classes_detected": class_names,
            "target_weights": out_weights.name,
            "message": f"Entrenamiento Auto-FL iniciado en segundo plano para '{slug}'.",
        }

    # =========================================================================
    # API: MASTER MLOPS STUDIO EXTENSIONS
    # =========================================================================

    @app.api_route("/api/master/recommend-arch", methods=["GET", "POST"])
    async def recommend_architecture(
        request: Request,
        task: str = Query("classification"),
        target_client_tier: str = Query("cpu-low"),
        dataset_sample_count: int | None = Query(None),
        num_classes: int | None = Query(None),
        network_speed_mbit: float = Query(50.0),
    ) -> dict[str, Any]:
        """Intelligently profile host hardware and recommend architectures for FL."""
        req_task = task
        req_tier = target_client_tier
        req_samples = dataset_sample_count
        req_classes = num_classes
        req_speed = network_speed_mbit

        is_json = "application/json" in request.headers.get("content-type", "")
        if request.method == "POST" and is_json:
            try:
                body = await request.json()
                if isinstance(body, dict):
                    req_task = body.get("task", req_task)
                    req_tier = body.get("target_client_tier", req_tier)
                    req_samples = body.get("dataset_sample_count", req_samples)
                    req_classes = body.get("num_classes", req_classes)
                    req_speed = float(body.get("network_speed_mbit", req_speed))
            except Exception:
                pass

        profile = HardwareProfile.detect()

        valid_tier: Any = "cpu-low"
        if req_tier in {"cpu-low", "cpu-high", "gpu"}:
            valid_tier = req_tier

        valid_task: Any = "classification"
        if req_task in {"classification", "detection", "segmentation"}:
            valid_task = req_task

        candidates = ModelSelector.get_recommendations(
            profile=profile,
            task=valid_task,
            network_speed_mbit=req_speed,
            target_client_tier=valid_tier,
            dataset_sample_count=req_samples,
            num_classes=req_classes,
        )

        return {
            "profile": {
                "device_type": profile.device_type,
                "device_name": profile.device_name,
                "cpu_count": profile.cpu_count,
                "memory_ram_gb": profile.memory_ram_gb,
                "gpu_count": profile.gpu_count,
                "gpu_memory_vram_gb": profile.gpu_memory_vram_gb,
                "cuda_compute_capability": profile.cuda_compute_capability,
                "has_mixed_precision_fp16": profile.has_mixed_precision_fp16,
            },
            "recommendations": [asdict(c) for c in candidates],
            "total_candidates": len(candidates),
        }

    @app.get("/api/master/models/{slug}/telemetry")
    def get_model_telemetry(slug: str) -> dict[str, Any]:
        """Retrieve confusion matrix, calibration thresholds, and canary telemetry for a model."""
        cm = _get_catalog_manager()
        entry = cm.get_by_slug(slug, include_deleted=True)
        if not entry:
            for m in cm.load_catalog(include_deleted=True):
                if m.active_checkpoint.startswith(slug) or m.slug == slug:
                    entry = m
                    break
        if not entry:
            raise HTTPException(status_code=404, detail=f"Modelo '{slug}' no encontrado.")

        models_root = (
            cm.models_dir if hasattr(cm, "models_dir") and cm.models_dir else Path("models")
        )
        chk_path = (
            Path(entry.active_checkpoint)
            if Path(entry.active_checkpoint).is_absolute()
            else models_root / entry.active_checkpoint
        )
        bundle_meta: dict[str, Any] = {}
        if chk_path.is_file():
            try:
                _, bundle_meta = load_checkpoint_bundle(chk_path, device="cpu")
            except Exception:
                pass

        global_meta: dict[str, Any] = {}
        meta_p = models_root / "metadata.json"
        if meta_p.is_file():
            try:
                global_meta = json.loads(meta_p.read_text(encoding="utf-8"))
            except Exception:
                pass

        # Look up evaluation report linked to the active version or matching candidate
        resolved_versions = entry.get_resolved_versions()
        eval_report: dict[str, Any] | None = None
        eval_source: str | None = None
        evaluated_checkpoint: str | None = None

        # 1. Check if active version has an explicit evaluation_report
        for v in resolved_versions:
            if v.checkpoint == entry.active_checkpoint and v.evaluation_report:
                cand = Path(v.evaluation_report)
                if not cand.is_absolute() and not cand.is_file():
                    cand = models_root / v.evaluation_report
                if cand.is_file():
                    try:
                        eval_report = json.loads(cand.read_text(encoding="utf-8"))
                        eval_source = str(cand)
                        evaluated_checkpoint = v.checkpoint
                        break
                    except Exception:
                        pass

        # 2. Check candidate paths matching slug or active checkpoint stem
        if not eval_report:
            report_candidates = [
                models_root / f"{slug}_evaluation_report.json",
                models_root / f"{Path(entry.active_checkpoint).stem}_evaluation_report.json",
                models_root / "evaluation_report.json",
                Path(f"docs/images/{slug}/evaluation_report.json"),
                Path("docs/images/evaluation_report.json"),
            ]
            for rep in report_candidates:
                if rep.is_file():
                    try:
                        candidate_data = json.loads(rep.read_text(encoding="utf-8"))
                        # Verify classes match to avoid attributing wrong report
                        cand_classes = list((candidate_data.get("per_class") or {}).keys())
                        chk_classes = (
                            bundle_meta.get("classes")
                            or global_meta.get("classes")
                            or entry.classes
                            or []
                        )
                        if (
                            not cand_classes
                            or not chk_classes
                            or set(cand_classes) == set(chk_classes)
                        ):
                            eval_report = candidate_data
                            eval_source = str(rep)
                            evaluated_checkpoint = entry.active_checkpoint
                            break
                    except Exception:
                        pass

        classes = bundle_meta.get("classes") or global_meta.get("classes") or entry.classes or []
        arch_name = (
            bundle_meta.get("architecture")
            or global_meta.get("architecture")
            or entry.architecture
        )
        cal_thresholds = (
            bundle_meta.get("calibrated_thresholds")
            or global_meta.get("calibrated_thresholds")
            or {}
        )
        temperature = float(bundle_meta.get("temperature") or global_meta.get("temperature") or 1.0)

        confusion_matrix_raw = None
        metrics: dict[str, Any] = {
            "overall_accuracy": None,
            "cross_entropy_loss": None,
            "macro_avg": None,
            "per_class": None,
        }
        if eval_report:
            confusion_matrix_raw = (
                eval_report.get("confusion_matrix_raw") or eval_report.get("confusion_matrix")
            )
            metrics["overall_accuracy"] = eval_report.get("overall_accuracy")
            metrics["cross_entropy_loss"] = eval_report.get("cross_entropy_loss")
            metrics["macro_avg"] = eval_report.get("macro_avg")
            metrics["per_class"] = eval_report.get("per_class")
            if eval_report.get("macro_avg"):
                metrics["macro_precision"] = eval_report["macro_avg"].get("precision")
                metrics["macro_recall"] = eval_report["macro_avg"].get("recall")
                metrics["macro_f1"] = eval_report["macro_avg"].get("f1_score")

        agg_history = global_meta.get("aggregation_history", [])

        matrix_classes = classes
        if eval_report and eval_report.get("per_class"):
            matrix_classes = list(eval_report["per_class"].keys())

        return {
            "slug": entry.slug,
            "title": entry.title,
            "architecture": arch_name,
            "checkpoint": entry.active_checkpoint,
            "size_bytes": chk_path.stat().st_size if chk_path.is_file() else 0,
            "classes": classes,
            "num_classes": len(classes),
            "temperature": temperature,
            "calibrated_thresholds": cal_thresholds,
            "metrics": metrics,
            "confusion_matrix": {
                "matrix": confusion_matrix_raw,
                "classes": matrix_classes,
            },
            "version_history": [v.model_dump() for v in resolved_versions],
            "aggregation_history": agg_history,
            "has_eval_report": eval_report is not None,
            "eval_source": eval_source,
            "evaluated_checkpoint": evaluated_checkpoint,
        }

    @app.post("/api/master/models/{slug}/evaluate")
    async def evaluate_model_performance(
        slug: str,
        val_dir: str | None = Query(None),
        batch_size: int = Query(32),
        calibrate_threshold: bool = Query(True),
    ) -> dict[str, Any]:
        """Trigger asynchronous performance evaluation of a model using scripts/evaluate.py."""
        if job_manager.is_running:
            raise HTTPException(
                status_code=409,
                detail=f"Trabajo en ejecución ('{job_manager.action}'). Espere a que finalice.",
            )

        cm = _get_catalog_manager()
        entry = cm.get_by_slug(slug, include_deleted=True)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Modelo '{slug}' no encontrado.")

        models_root = (
            cm.models_dir if hasattr(cm, "models_dir") and cm.models_dir else Path("models")
        )
        chk_path = (
            Path(entry.active_checkpoint)
            if Path(entry.active_checkpoint).is_absolute()
            else models_root / entry.active_checkpoint
        )
        if not chk_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"El checkpoint activo '{entry.active_checkpoint}' no existe en disco.",
            )

        # Resolve validation/test split directory compatible with the model's taxonomy
        resolved_val, _ = resolve_validation_directory_for_model(
            entry.classes, explicit_val_dir=val_dir
        )
        if not resolved_val or not resolved_val.is_dir():
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No se encontró un conjunto de test o validación compatible con las clases "
                    f"del modelo ({', '.join(entry.classes)}). "
                    f"Especifique un directorio válido en el parámetro 'val_dir'."
                ),
            )

        meta_candidates = [
            models_root / f"{slug}.meta.json",
            models_root / f"{chk_path.stem}.meta.json",
            models_root / "metadata.json",
        ]
        meta_arg: Path | None = None
        for cand in meta_candidates:
            if cand.is_file():
                meta_arg = cand
                break

        out_eval_dir = models_root / f"eval_{slug}"
        out_eval_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "scripts/evaluate.py",
            "--model",
            str(chk_path),
            "--data-dir",
            str(resolved_val),
            "--out-dir",
            str(out_eval_dir),
            "--batch-size",
            str(batch_size),
            "--arch",
            entry.architecture or "resnet18",
        ]
        if meta_arg:
            cmd.extend(["--metadata", str(meta_arg)])
        if calibrate_threshold:
            cmd.append("--calibrate-threshold")

        async def on_eval_success() -> None:
            generated_rep = out_eval_dir / "evaluation_report.json"
            target_rep = models_root / f"{slug}_evaluation_report.json"
            if generated_rep.is_file():
                try:
                    shutil.copyfile(generated_rep, target_rep)
                    rep_data = json.loads(target_rep.read_text(encoding="utf-8"))

                    # Update catalog version history
                    updated_entry = cm.get_by_slug(slug, include_deleted=True)
                    if updated_entry:
                        vh = updated_entry.get_resolved_versions()
                        for v in vh:
                            if v.checkpoint == updated_entry.active_checkpoint:
                                v.evaluation_report = target_rep.name
                                v.metrics_summary = {
                                    "accuracy": rep_data.get("overall_accuracy"),
                                    "loss": rep_data.get("cross_entropy_loss"),
                                    "macro_f1": (rep_data.get("macro_avg") or {}).get("f1_score"),
                                }
                                break
                        updated_entry.version_history = vh
                        cm.upsert_entry(updated_entry)
                    job_manager.logs.append(
                        f"[*] [EVAL] Reporte guardado en {target_rep.name} para modelo '{slug}'."
                    )
                except Exception as exc:
                    job_manager.logs.append(
                        f"[!] Error actualizando catálogo tras evaluación: {exc}"
                    )

        await job_manager.run_command(
            action=f"evaluate_model:{slug}",
            cmd=cmd,
            on_success=on_eval_success,
        )

        return {
            "status": "started",
            "action": f"evaluate_model:{slug}",
            "slug": slug,
            "checkpoint": entry.active_checkpoint,
            "validation_directory": str(resolved_val),
            "command": cmd,
        }

    @app.get("/api/master/federated/updates")
    def get_federated_updates(
        target_model: str | None = Query(None),
        updates_dir: str = Query("models/client_updates"),
    ) -> dict[str, Any]:
        """List client checkpoints with security and lineage audit against target model."""
        cm = _get_catalog_manager()
        models_root = (
            cm.models_dir if hasattr(cm, "models_dir") and cm.models_dir else Path("models")
        )
        root = Path(updates_dir)
        if not root.is_absolute():
            cand_root = models_root / updates_dir.replace("models/", "")
            if cand_root.is_dir():
                root = cand_root

        target_entry = None
        target_base_chk_path: Path | None = None
        target_base_sha: str | None = None
        target_arch = "resnet18"

        if target_model:
            target_entry = cm.get_by_slug(target_model, include_deleted=True)
            if not target_entry:
                for m in cm.load_catalog(include_deleted=True):
                    if m.active_checkpoint == target_model or m.slug == target_model:
                        target_entry = m
                        break
        elif active_model_p and active_model_p.is_file():
            target_base_chk_path = active_model_p

        if target_entry:
            target_base_chk_path = (
                Path(target_entry.active_checkpoint)
                if Path(target_entry.active_checkpoint).is_absolute()
                else models_root / target_entry.active_checkpoint
            )
            target_arch = target_entry.architecture

        if target_base_chk_path and target_base_chk_path.is_file():
            target_base_sha = compute_sha256(target_base_chk_path)

        target_slug = target_entry.slug if target_entry else (target_model or "")
        all_catalog_entries = cm.load_catalog(include_deleted=True)
        other_slugs = {
            m.slug for m in all_catalog_entries if target_slug and m.slug != target_slug
        }

        other_model_shas: set[str] = set()
        for other_m in all_catalog_entries:
            if target_slug and other_m.slug != target_slug:
                chk = (
                    Path(other_m.active_checkpoint)
                    if Path(other_m.active_checkpoint).is_absolute()
                    else models_root / other_m.active_checkpoint
                )
                if chk.is_file():
                    try:
                        other_model_shas.add(compute_sha256(chk))
                    except Exception:
                        pass

        target_model_shas: set[str] = set()
        if target_base_sha:
            target_model_shas.add(target_base_sha)
        if target_entry:
            for v in target_entry.get_resolved_versions():
                chk_v = (
                    Path(v.checkpoint)
                    if Path(v.checkpoint).is_absolute()
                    else models_root / v.checkpoint
                )
                if chk_v.is_file():
                    try:
                        target_model_shas.add(compute_sha256(chk_v))
                    except Exception:
                        pass

        candidates: list[Path] = []

        # 1. Model-specific partitioned directory: models/client_updates/<slug>/
        target_subdirs: list[Path] = []
        if target_slug:
            if (root / target_slug).is_dir():
                target_subdirs.append(root / target_slug)
            if (models_root / "client_updates" / target_slug).is_dir():
                target_subdirs.append(models_root / "client_updates" / target_slug)

        for sdir in target_subdirs:
            for p in sorted(sdir.glob("*.pt")):
                if not p.name.startswith(".") and not p.name.endswith("_candidate.pt"):
                    if p not in candidates:
                        candidates.append(p)

        # 2. Check root and shared models directories with strict isolation
        pool: list[Path] = []
        if root.is_dir():
            for p in sorted(root.glob("*.pt")):
                if not p.name.startswith(".") and not p.name.endswith("_candidate.pt"):
                    if p not in pool:
                        pool.append(p)

        for p in sorted(models_root.glob("*_parameters.pt")):
            if not p.name.startswith(".") and p not in pool:
                pool.append(p)

        for p in pool:
            if p in candidates:
                continue

            # Exclude if explicitly located inside another model's folder
            if p.parent.name in other_slugs:
                continue

            # Exclude if filename starts with another model's slug
            if any(
                p.name.startswith(f"{other}_") or p.name.startswith(f"{other}-")
                for other in other_slugs
            ):
                continue

            # Include if filename is explicitly prefixed with target_slug
            if target_slug and (
                p.name.startswith(f"{target_slug}_") or p.name.startswith(f"{target_slug}-")
            ):
                candidates.append(p)
                continue

            # Inspect companion metadata manifest
            meta_path = resolve_meta_path(p)
            meta_data: dict[str, Any] = {}
            if meta_path.is_file():
                try:
                    meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

            meta_target = (
                meta_data.get("target_model")
                or meta_data.get("model_slug")
                or meta_data.get("slug")
            )
            if meta_target:
                if target_slug and meta_target == target_slug:
                    candidates.append(p)
                    continue
                else:
                    # Belongs to a different model -> strictly isolate
                    continue

            meta_classes = meta_data.get("classes")
            if meta_classes and target_entry and target_entry.classes:
                if set(meta_classes) != set(target_entry.classes):
                    # Trained on different class taxonomy -> strictly isolate
                    continue

            meta_base_sha = meta_data.get("base_model_sha256")
            if meta_base_sha:
                if meta_base_sha in target_model_shas:
                    candidates.append(p)
                    continue
                if meta_base_sha in other_model_shas:
                    # Derived from a different model in catalog -> strictly isolate
                    continue

            # If unassigned and single model catalog (e.g. test environment), keep for testing
            if len(all_catalog_entries) <= 1:
                candidates.append(p)

        found_updates: list[dict[str, Any]] = []
        eligible_count = 0
        rejected_count = 0

        for p in candidates:
            audit = audit_client_update(
                p,
                target_base_checkpoint=target_base_chk_path,
                target_arch=target_arch,
            )
            if audit.is_eligible:
                eligible_count += 1
            else:
                rejected_count += 1

            found_updates.append(
                {
                    "filename": p.name,
                    "path": str(p),
                    "client_id": audit.client_id,
                    "num_samples": audit.num_samples,
                    "scope": audit.training_scope,
                    "sha256": audit.sha256,
                    "base_model_sha256": audit.base_model_sha256,
                    "target_base_sha256": target_base_sha,
                    "lineage_matched": audit.lineage_matched,
                    "has_hmac": audit.has_hmac,
                    "hmac_valid": audit.hmac_valid,
                    "size_bytes": audit.size_bytes,
                    "is_eligible": audit.is_eligible,
                    "rejection_reasons": audit.rejection_reasons,
                    "created_at": audit.created_at,
                }
            )

        return {
            "target_model": target_entry.slug if target_entry else None,
            "target_base_checkpoint": target_base_chk_path.name if target_base_chk_path else None,
            "target_base_sha256": target_base_sha,
            "updates": found_updates,
            "total": len(found_updates),
            "eligible_count": eligible_count,
            "rejected_count": rejected_count,
        }

    @app.post("/api/master/federated/aggregate")
    async def run_secure_federated_aggregation(
        req: FederatedAggregationRequest,
    ) -> dict[str, Any]:
        """Trigger FedAvgM aggregation producing an isolated candidate checkpoint."""
        if job_manager.is_running:
            raise HTTPException(
                status_code=409,
                detail=f"Trabajo en ejecución ('{job_manager.action}'). Espere a que finalice.",
            )

        if not req.selected_update_paths:
            raise HTTPException(
                status_code=400,
                detail="Debe seleccionar al menos una actualización federada válida para agregar.",
            )

        cm = _get_catalog_manager()
        entry = cm.get_by_slug(req.target_model_slug, include_deleted=True)
        if not entry:
            raise HTTPException(
                status_code=404,
                detail=f"Modelo objetivo '{req.target_model_slug}' no encontrado.",
            )

        models_root = (
            cm.models_dir if hasattr(cm, "models_dir") and cm.models_dir else Path("models")
        )
        base_chk_path = (
            Path(entry.active_checkpoint)
            if Path(entry.active_checkpoint).is_absolute()
            else models_root / entry.active_checkpoint
        )
        if not base_chk_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Checkpoint base '{entry.active_checkpoint}' no existe en disco.",
            )

        # REDUNDANT BACKEND PREFLIGHT: Validate EVERY selected artifact
        validated_paths: list[Path] = []
        for p_str in req.selected_update_paths:
            p = (
                Path(p_str)
                if Path(p_str).is_absolute()
                else models_root / p_str.replace("models/", "")
            )
            audit = audit_client_update(
                p,
                target_base_checkpoint=base_chk_path,
                target_arch=entry.architecture,
            )
            if not audit.is_eligible:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"RECHAZO DE SEGURIDAD: El artefacto '{audit.file_name}' no es apto "
                        f"para agregación: {'; '.join(audit.rejection_reasons)}"
                    ),
                )
            validated_paths.append(p)

        # Resolve validation dataset matching model classes
        val_dir, val_status = resolve_validation_directory_for_model(
            entry.classes, req.explicit_val_dir
        )

        candidate_name = f"{entry.slug}_candidate.pt"
        candidate_path = models_root / candidate_name
        op_id = f"fl-op-{int(time.time())}"

        op_record = FederatedOperationRecord(
            operation_id=op_id,
            target_model_slug=entry.slug,
            base_checkpoint=entry.active_checkpoint,
            base_checkpoint_sha256=compute_sha256(base_chk_path),
            candidate_checkpoint=candidate_name,
            selected_updates=[str(p) for p in validated_paths],
            aggregation_params={
                "server_momentum": req.server_momentum,
                "server_lr": req.server_lr,
                "allow_new_classes": req.allow_new_classes,
                "max_acc_drop": req.max_acc_drop,
            },
            val_dir_used=str(val_dir) if val_dir else None,
            guardrail_status="PASSED" if val_dir else "PENDING_EVALUATION",
            status="RUNNING",
        )
        _get_fl_ops_manager().record_operation(op_record)

        # Build CLI command
        cmd = [
            sys.executable,
            "scripts/aggregate.py",
            "--base-model",
            str(base_chk_path),
            "--output",
            str(candidate_path),
            "--server-momentum",
            str(req.server_momentum),
            "--server-lr",
            str(req.server_lr),
        ]
        if req.allow_new_classes:
            cmd.append("--allow-new-classes")
        if val_dir:
            cmd.extend(["--val-dir", str(val_dir), "--max-acc-drop", str(req.max_acc_drop)])

        cmd.append("--client-weights")
        cmd.extend(str(p) for p in validated_paths)

        await job_manager.run_command("federated_aggregate", cmd)

        return {
            "status": "started",
            "operation_id": op_id,
            "target_model": entry.slug,
            "base_checkpoint": entry.active_checkpoint,
            "candidate_checkpoint": candidate_name,
            "selected_updates_count": len(validated_paths),
            "val_dir_status": val_status,
            "val_dir": str(val_dir) if val_dir else None,
        }

    @app.get("/api/master/federated/operations")
    def list_federated_operations(
        slug: str | None = Query(None),
    ) -> dict[str, Any]:
        """List historical federated operations and candidate states."""
        ops = _get_fl_ops_manager().load_operations()
        if slug:
            ops = [op for op in ops if op.target_model_slug == slug]
        return {
            "operations": [op.model_dump() for op in reversed(ops)],
            "total": len(ops),
        }

    @app.post("/api/master/models/{slug}/promote-candidate")
    def promote_candidate_model(
        slug: str,
        force: bool = Query(False),
    ) -> dict[str, Any]:
        """Atomically promote a candidate checkpoint to active production version."""
        cm = _get_catalog_manager()
        entry = cm.get_by_slug(slug, include_deleted=True)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Modelo '{slug}' no encontrado.")

        models_root = (
            cm.models_dir if hasattr(cm, "models_dir") and cm.models_dir else Path("models")
        )
        candidate_p = models_root / f"{slug}_candidate.pt"
        if not candidate_p.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"No se ha encontrado el archivo candidato '{candidate_p.name}' en disco.",
            )

        # Check latest operation record for guardrail status
        latest_op = _get_fl_ops_manager().get_latest_candidate(slug)
        if latest_op and latest_op.guardrail_status == "REJECTED" and not force:
            raise HTTPException(
                status_code=400,
                detail="Candidato rechazado por guardarraíl de regresión. Promoción denegada.",
            )

        metrics_payload: dict[str, Any] = {}
        eval_report_str: str | None = None
        if latest_op:
            metrics_payload = {
                "pre_aggregation_acc": latest_op.pre_aggregation_acc,
                "pre_aggregation_loss": latest_op.pre_aggregation_loss,
                "post_aggregation_acc": latest_op.post_aggregation_acc,
                "post_aggregation_loss": latest_op.post_aggregation_loss,
                "accuracy": latest_op.post_aggregation_acc,
                "loss": latest_op.post_aggregation_loss,
                "accuracy_delta": latest_op.accuracy_delta,
                "loss_delta": latest_op.loss_delta,
                "val_dir_used": latest_op.val_dir_used,
            }
        eval_p = models_root / "evaluation_report.json"
        if eval_p.is_file():
            try:
                eval_data = json.loads(eval_p.read_text(encoding="utf-8"))
                if "macro_f1" in eval_data:
                    metrics_payload["macro_f1"] = eval_data["macro_f1"]
                if "accuracy" in eval_data and "accuracy" not in metrics_payload:
                    metrics_payload["accuracy"] = eval_data["accuracy"]
                if "loss" in eval_data and "loss" not in metrics_payload:
                    metrics_payload["loss"] = eval_data["loss"]
                eval_report_str = str(eval_p)
            except Exception:
                pass

        success, msg, new_rec = cm.promote_candidate(
            slug,
            candidate_p,
            metrics=metrics_payload,
            eval_report=eval_report_str,
        )
        if not success:
            raise HTTPException(status_code=400, detail=msg)

        if latest_op:
            latest_op.status = "PROMOTED"
            latest_op.promoted_at = time.strftime("%Y-%m-%d %H:%M:%S")
            _get_fl_ops_manager().record_operation(latest_op)

        _resolve_and_reload_model(force=True)
        updated_entry = cm.get_by_slug(slug, include_deleted=True)
        return {
            "success": True,
            "message": msg,
            "promoted_version": new_rec.model_dump() if new_rec else None,
            "active_checkpoint": updated_entry.active_checkpoint if updated_entry else None,
        }

    @app.post("/api/master/models/{slug}/rollback")
    def rollback_model_version(
        slug: str,
        target_version_id: str | None = Query(None),
    ) -> dict[str, Any]:
        """Rollback active model to a previous version or snapshot."""
        cm = _get_catalog_manager()
        success, msg = cm.rollback_to_version(slug, target_version_id)
        if not success:
            raise HTTPException(status_code=400, detail=msg)

        _resolve_and_reload_model(force=True)
        entry = cm.get_by_slug(slug, include_deleted=True)
        return {
            "success": True,
            "message": msg,
            "active_checkpoint": entry.active_checkpoint if entry else None,
        }

    @app.post("/api/master/models/{slug}/gradcam")
    async def generate_model_gradcam(
        slug: str,
        file: UploadFile = File(...),  # noqa: B008
        target_class: int | None = Query(None),
    ) -> dict[str, Any]:
        """Generate real-time in-memory Grad-CAM visual explanation overlay for a model."""
        import matplotlib
        import numpy as np

        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, detail="El archivo subido debe ser una imagen válida."
            )

        content = await file.read(MAX_UPLOAD_SIZE_BYTES + 1)
        if len(content) > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="La imagen excede el límite de 15MB.")

        cm = _get_catalog_manager()
        entry = cm.get_by_slug(slug, include_deleted=True)
        if not entry:
            for m in cm.load_catalog(include_deleted=True):
                if m.active_checkpoint.startswith(slug) or m.slug == slug:
                    entry = m
                    break
        if not entry:
            raise HTTPException(status_code=404, detail=f"Modelo '{slug}' no encontrado.")

        models_root = (
            cm.models_dir if hasattr(cm, "models_dir") and cm.models_dir else Path("models")
        )
        chk_path = (
            Path(entry.active_checkpoint)
            if Path(entry.active_checkpoint).is_absolute()
            else models_root / entry.active_checkpoint
        )
        if not chk_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"El archivo de pesos '{entry.active_checkpoint}' no existe en disco.",
            )

        try:
            pil_image = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception as e:
            raise HTTPException(
                status_code=400, detail="Formato de imagen corrupto o no reconocido."
            ) from e

        # Load model and state
        state_dict, b_meta = load_checkpoint_bundle(chk_path, device="cpu")
        model_arch = b_meta.get("architecture") or entry.architecture
        model_classes = b_meta.get("classes") or entry.classes or ["Clase 0", "Clase 1"]
        img_dim = b_meta.get("img_size", [224, 224])
        if isinstance(img_dim, int):
            img_dim = (img_dim, img_dim)
        else:
            img_dim = (int(img_dim[0]), int(img_dim[1]))

        eval_model = build_model(model_arch, num_classes=len(model_classes), pretrained=False)
        eval_model.load_state_dict(state_dict, strict=False)
        eval_model.eval()

        # Aligned inference preprocessing: Resize(256) -> CenterCrop(224)
        rs_dim = (int(round(img_dim[0] * 256 / 224)), int(round(img_dim[1] * 256 / 224)))
        pre_crop = transforms.Compose(
            [
                transforms.Resize(rs_dim),
                transforms.CenterCrop(img_dim),
            ]
        )
        cropped_pil = pre_crop(pil_image)
        to_tensor_norm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        x_tensor = to_tensor_norm(cropped_pil).unsqueeze(0)

        target_layer = get_cam_target_layer(eval_model, architecture=model_arch)
        temperature = float(b_meta.get("temperature", 1.0))
        cam_generator = GradCAM(eval_model, target_layer, temperature=temperature)

        try:
            cam_map, pred_class_idx, conf = cam_generator.generate(
                x_tensor, class_idx=target_class
            )
        finally:
            cam_generator.remove()

        cmap = matplotlib.colormaps.get_cmap("jet")
        cam_uint8 = (cmap(cam_map)[:, :, :3] * 255).astype(np.uint8)
        cam_pil = Image.fromarray(cam_uint8).resize(cropped_pil.size, Image.BILINEAR)
        blended = Image.blend(cropped_pil, cam_pil, alpha=0.5)

        buf = io.BytesIO()
        blended.save(buf, format="JPEG", quality=90)
        overlay_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        pred_name = (
            model_classes[pred_class_idx]
            if pred_class_idx < len(model_classes)
            else f"Clase {pred_class_idx}"
        )

        return {
            "slug": entry.slug,
            "architecture": model_arch,
            "predicted_class_index": pred_class_idx,
            "predicted_class_name": pred_name,
            "confidence": round(conf, 4),
            "cam_overlay_base64": f"data:image/jpeg;base64,{overlay_b64}",
        }

    @app.post("/api/master/models/{slug}/compare-ab")
    async def compare_model_versions(
        slug: str,
        file: UploadFile = File(...),  # noqa: B008
        version_a: str = Query(..., description="Version ID (e.g. v1) or checkpoint for A"),
        version_b: str = Query(..., description="Version ID (e.g. v2) or checkpoint for B"),
        ground_truth_class: str | None = Query(None, description="Optional ground truth class"),
        include_gradcam: bool = Query(True, description="Whether to compute Grad-CAM overlays"),
    ) -> dict[str, Any]:
        """A/B Comparator: Run live inference side-by-side across two checkpoints."""
        import matplotlib
        import numpy as np

        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, detail="El archivo subido debe ser una imagen válida."
            )

        content = await file.read(MAX_UPLOAD_SIZE_BYTES + 1)
        if len(content) > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="La imagen excede el límite de 15MB.")

        cm = _get_catalog_manager()
        entry = cm.get_by_slug(slug, include_deleted=True)
        if not entry:
            for m in cm.load_catalog(include_deleted=True):
                if m.active_checkpoint.startswith(slug) or m.slug == slug:
                    entry = m
                    break
        if not entry:
            raise HTTPException(status_code=404, detail=f"Modelo '{slug}' no encontrado.")

        models_root = (
            cm.models_dir if hasattr(cm, "models_dir") and cm.models_dir else Path("models")
        )

        def resolve_version_checkpoint(v_spec: str) -> tuple[ModelVersionRecord | None, Path]:
            matched_rec = None
            for v_rec in entry.get_resolved_versions():
                if v_rec.version_id.lower() == v_spec.lower() or v_rec.checkpoint == v_spec:
                    matched_rec = v_rec
                    break
            chk_name = matched_rec.checkpoint if matched_rec else v_spec
            chk_p = Path(chk_name) if Path(chk_name).is_absolute() else models_root / chk_name
            return matched_rec, chk_p

        rec_a, path_a = resolve_version_checkpoint(version_a)
        rec_b, path_b = resolve_version_checkpoint(version_b)

        if not path_a.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Checkpoint Versión A '{version_a}' ({path_a.name}) no encontrado.",
            )
        if not path_b.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Checkpoint Versión B '{version_b}' ({path_b.name}) no encontrado.",
            )

        try:
            pil_image = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception as e:
            raise HTTPException(
                status_code=400, detail="Formato de imagen corrupto o no reconocido."
            ) from e

        def eval_single_version(
            chk_path: Path,
            v_rec: ModelVersionRecord | None,
            v_label: str,
        ) -> dict[str, Any]:
            try:
                state_dict, b_meta = load_checkpoint_bundle(chk_path, device="cpu")
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error cargando checkpoint '{chk_path.name}' de {v_label}: {e}",
                ) from e

            v_arch = b_meta.get("architecture") or entry.architecture
            v_classes = (
                b_meta.get("classes")
                or entry.classes
                or [f"Clase {i}" for i in range(entry.num_classes or 2)]
            )
            v_temp = float(b_meta.get("temperature", 1.0))
            raw_dim = b_meta.get("img_size", [224, 224])
            img_dim = (
                (raw_dim, raw_dim)
                if isinstance(raw_dim, int)
                else (int(raw_dim[0]), int(raw_dim[1]))
            )

            v_model = build_model(v_arch, num_classes=len(v_classes), pretrained=False)
            if state_dict:
                v_model.load_state_dict(state_dict, strict=False)
            v_model.eval()

            rs_dim = (int(round(img_dim[0] * 256 / 224)), int(round(img_dim[1] * 256 / 224)))
            pre_crop = transforms.Compose(
                [
                    transforms.Resize(rs_dim),
                    transforms.CenterCrop(img_dim),
                ]
            )
            cropped_pil = pre_crop(pil_image)
            to_tensor_norm = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            x_tensor = to_tensor_norm(cropped_pil).unsqueeze(0)

            with torch.no_grad():
                logits = v_model(x_tensor)
                probs = F.softmax(logits / max(v_temp, 1e-3), dim=1)[0]
                pred_idx = int(probs.argmax().item())
                confidence = float(probs[pred_idx].item())
                pred_name = (
                    v_classes[pred_idx]
                    if pred_idx < len(v_classes)
                    else f"Clase {pred_idx}"
                )
                all_probs = {
                    v_classes[i]: round(float(probs[i].item()), 4)
                    for i in range(min(len(v_classes), len(probs)))
                }

            cam_overlay_b64: str | None = None
            cam_supported: bool = True
            cam_error: str | None = None
            if include_gradcam:
                try:
                    target_layer = get_cam_target_layer(v_model, architecture=v_arch)
                    cam_gen = GradCAM(v_model, target_layer, temperature=v_temp)
                    try:
                        cam_map, _, _ = cam_gen.generate(x_tensor, class_idx=pred_idx)
                    finally:
                        cam_gen.remove()

                    cmap = matplotlib.colormaps.get_cmap("jet")
                    cam_uint8 = (cmap(cam_map)[:, :, :3] * 255).astype(np.uint8)
                    cam_pil = Image.fromarray(cam_uint8).resize(cropped_pil.size, Image.BILINEAR)
                    blended = Image.blend(cropped_pil, cam_pil, alpha=0.5)
                    buf = io.BytesIO()
                    blended.save(buf, format="JPEG", quality=90)
                    cam_overlay_b64 = (
                        f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
                    )
                except Exception as cam_exc:
                    cam_supported = False
                    cam_error = str(cam_exc)

            agg_metrics: dict[str, Any] | None = None
            report_path = None
            if v_rec and v_rec.evaluation_report:
                cand_p = Path(v_rec.evaluation_report)
                if not cand_p.is_absolute() and not cand_p.is_file():
                    cand_p = models_root / v_rec.evaluation_report
                if cand_p.is_file():
                    report_path = cand_p
            else:
                stem_candidate = models_root / f"{chk_path.stem}_evaluation_report.json"
                if stem_candidate.is_file():
                    report_path = stem_candidate

            if report_path and report_path.is_file():
                try:
                    rep_json = json.loads(report_path.read_text(encoding="utf-8"))
                    agg_metrics = {
                        "source": str(report_path),
                        "overall_accuracy": rep_json.get("overall_accuracy"),
                        "cross_entropy_loss": rep_json.get("cross_entropy_loss"),
                        "macro_f1": (rep_json.get("macro_avg") or {}).get("f1_score"),
                    }
                except Exception:
                    pass

            matches_gt: bool | None = None
            if ground_truth_class:
                gt_clean = ground_truth_class.strip().lower()
                matches_gt = pred_name.lower() == gt_clean

            return {
                "version_id": v_rec.version_id if v_rec else v_label,
                "checkpoint": chk_path.name,
                "architecture": v_arch,
                "predicted_class_name": pred_name,
                "predicted_class_index": pred_idx,
                "confidence": round(confidence, 4),
                "probabilities": all_probs,
                "matches_ground_truth": matches_gt,
                "cam_overlay_base64": cam_overlay_b64,
                "gradcam_supported": cam_supported,
                "gradcam_error": cam_error,
                "aggregate_metrics": agg_metrics,
                "origin": v_rec.origin if v_rec else "unknown",
                "status": v_rec.status if v_rec else "unknown",
            }

        res_a = eval_single_version(path_a, rec_a, version_a)
        res_b = eval_single_version(path_b, rec_b, version_b)

        is_agreement = res_a["predicted_class_name"] == res_b["predicted_class_name"]

        summary_parts = []
        if is_agreement:
            summary_parts.append(
                f"Ambas versiones coinciden en predecir '{res_a['predicted_class_name']}'."
            )
        else:
            summary_parts.append(
                f"Discrepancia diagnóstica: Versión A predice '{res_a['predicted_class_name']}' "
                f"(conf: {res_a['confidence'] * 100:.1f}%) y Versión B predice "
                f"'{res_b['predicted_class_name']}' (conf: {res_b['confidence'] * 100:.1f}%)."
            )

        if ground_truth_class:
            gt_text = ground_truth_class.strip()
            if res_a["matches_ground_truth"] and res_b["matches_ground_truth"]:
                summary_parts.append(
                    f"Ambas versiones aciertan con la etiqueta de referencia ('{gt_text}')."
                )
            elif res_a["matches_ground_truth"] and not res_b["matches_ground_truth"]:
                summary_parts.append(
                    f"Versión A acierta con la etiqueta de referencia ('{gt_text}'), "
                    f"mientras Versión B difiere."
                )
            elif not res_a["matches_ground_truth"] and res_b["matches_ground_truth"]:
                summary_parts.append(
                    f"Versión B acierta con la etiqueta de referencia ('{gt_text}'), "
                    f"mientras Versión A difiere."
                )
            else:
                summary_parts.append(
                    f"Ninguna de las dos versiones coincide con la etiqueta de referencia "
                    f"('{gt_text}')."
                )

        summary_parts.append(
            "Nota MLOps: Una mayor confianza softmax no implica necesariamente mayor exactitud "
            "empírica."
        )

        return {
            "slug": entry.slug,
            "title": entry.title,
            "ground_truth_class": ground_truth_class,
            "agreement": is_agreement,
            "summary": " ".join(summary_parts),
            "version_a": res_a,
            "version_b": res_b,
        }

    # =========================================================================
    # CORE INFERENCE & CLIENT FEDERATION (Backward Compatible)
    # =========================================================================

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/health")
    def health() -> dict[str, Any]:
        _resolve_and_reload_model()
        return {
            "status": "online",
            "model_path": str(active_model_p) if active_model_p else str(model_p),
            "model_sha256": model_sha256,
            "architecture": arch,
            "num_classes": num_classes,
            "classes": classes,
            "img_size": list(img_size_tuple),
            "calibrated_thresholds": calibrated_thresholds,
        }

    @app.post("/predict")
    async def predict(
        file: UploadFile = File(...),  # noqa: B008
        model_name: str | None = Query(None),
        model_slug: str | None = Query(None),
        save_for_training: bool = Query(True),
        ood_threshold: float | None = Query(None),
    ) -> dict[str, Any]:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image")

        content = await file.read(MAX_UPLOAD_SIZE_BYTES + 1)
        if len(content) > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Uploaded image exceeds maximum allowed limit of "
                    f"{MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB"
                ),
            )

        target_model = model_slug or model_name
        res = _infer_single(
            content,
            file.filename,
            model_name=target_model,
            ood_threshold=ood_threshold,
        )

        # Enqueue for expert active learning review if opted in (default True)
        if save_for_training:
            try:
                active_hitl = getattr(app.state, "hitl", hitl_manager)
                chosen_slug = (
                    model_slug
                    or (
                        catalog_manager.get_by_slug(res.get("model_name", ""))
                        and catalog_manager.get_by_slug(res.get("model_name", "")).slug  # type: ignore[union-attr]
                    )
                    or res.get("model_name", "default")
                )
                await active_hitl.enqueue(
                    image_bytes=content,
                    filename=file.filename,
                    model_name=res.get("model_name", "default"),
                    predicted_class=res["predicted_class"],
                    confidence=res["confidence"],
                    probabilities=res.get("probabilities", {}),
                    threshold_applied=res.get("threshold_applied", "nominal_argmax"),
                    model_slug=chosen_slug,
                )
                res["saved_for_training"] = True
                res["model_slug"] = chosen_slug
            except Exception:
                res["saved_for_training"] = False

        return res

    @app.post("/predict/batch")
    async def predict_batch(
        files: list[UploadFile] = File(...),  # noqa: B008
        model_name: str | None = Query(None),
        model_slug: str | None = Query(None),
        ood_threshold: float | None = Query(None),
    ) -> dict[str, Any]:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded for batch inference")

        target_model = model_slug or model_name
        (
            infer_net,
            infer_classes,
            infer_num_classes,
            infer_cal,
            infer_temp,
            infer_tf,
            actual_model_name,
        ) = _get_model_for_inference(target_model)

        eff_thresh = ood_threshold
        if eff_thresh is None:
            for v in model_cache.values():
                if v.get("net") is infer_net:
                    eff_thresh = v.get("ood_threshold")
                    break

        predictions: list[dict[str, Any] | None] = [None] * len(files)
        valid_indices: list[int] = []
        valid_filenames: list[str | None] = []
        valid_tensors: list[torch.Tensor] = []

        for idx, file in enumerate(files):
            if not file.content_type or not file.content_type.startswith("image/"):
                predictions[idx] = {"filename": file.filename, "error": "Not an image"}
                continue
            content = await file.read(MAX_UPLOAD_SIZE_BYTES + 1)
            if len(content) > MAX_UPLOAD_SIZE_BYTES:
                predictions[idx] = {
                    "filename": file.filename,
                    "error": (
                        f"Image exceeds limit of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB"
                    ),
                }
                continue
            try:
                img = Image.open(io.BytesIO(content)).convert("RGB")
                valid_tensors.append(infer_tf(img))
                valid_indices.append(idx)
                valid_filenames.append(file.filename)
            except Image.DecompressionBombError as exc:
                predictions[idx] = {
                    "filename": file.filename,
                    "error": f"Decompression bomb detected: image exceeds pixel limit ({exc})",
                }
            except Exception as exc:
                predictions[idx] = {
                    "filename": file.filename,
                    "error": f"Invalid image format or corrupted payload: {exc}",
                }

        if valid_tensors:
            batch_tensor = torch.stack(valid_tensors, dim=0).to(device)
            with torch.inference_mode():
                batch_logits = infer_net(batch_tensor)
                batch_probs = F.softmax(batch_logits / max(infer_temp, 1e-3), dim=1).cpu()

            for b_i, orig_idx in enumerate(valid_indices):
                predictions[orig_idx] = _decode_probs_row(
                    batch_probs[b_i],
                    infer_classes,
                    infer_num_classes,
                    infer_cal,
                    valid_filenames[b_i],
                    actual_model_name,
                    logits=batch_logits[b_i].cpu(),
                    ood_threshold=eff_thresh,
                    ood_temperature=infer_temp if infer_temp > 0 else 1.0,
                )

        return {
            "batch_size": len(files),
            "predictions": [p for p in predictions if p is not None],
        }

    def _cleanup_dir(path: str) -> None:
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass

    def _sync_process_batch_zip(
        zip_in_path: str,
        temp_dir: str,
        target_model: str | None = None,
        ood_threshold: float | None = None,
    ) -> str:
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_in_path, "r") as zf:
                for member in zf.infolist():
                    if member.is_dir():
                        continue
                    clean_name = os.path.basename(member.filename)
                    if (
                        not clean_name
                        or clean_name.startswith(".")
                        or clean_name.startswith("__MACOSX")
                    ):
                        continue
                    dest = os.path.join(input_dir, clean_name)
                    count = 1
                    base, ext = os.path.splitext(clean_name)
                    while os.path.exists(dest):
                        dest = os.path.join(input_dir, f"{base}_{count}{ext}")
                        count += 1
                    with zf.open(member) as source, open(dest, "wb") as target:
                        shutil.copyfileobj(source, target)
        except zipfile.BadZipFile as e:
            raise HTTPException(
                status_code=400, detail="Uploaded file is not a valid ZIP file"
            ) from e

        supported_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
        image_files = [
            f
            for f in sorted(os.listdir(input_dir))
            if os.path.splitext(f)[1].lower() in supported_exts
        ]

        if not image_files:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No supported images found in the uploaded ZIP file "
                    "(supported: JPG, PNG, WEBP, BMP)"
                ),
            )

        records: list[dict[str, Any]] = []

        for filename in image_files:
            img_path = os.path.join(input_dir, filename)
            try:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()

                res = _infer_single(
                    img_bytes,
                    filename=filename,
                    model_name=target_model,
                    ood_threshold=ood_threshold,
                )
                pred_class = res.get("predicted_class", "Desconocido")
                confidence = float(res.get("confidence", 0.0))
                ood_flag = bool(res.get("ood_flag", False))

                safe_folder = re.sub(r"[^\w\-_.]", "_", str(pred_class))
                target_folder = os.path.join(output_dir, safe_folder)
                os.makedirs(target_folder, exist_ok=True)
                shutil.copy2(img_path, os.path.join(target_folder, filename))

                records.append(
                    {
                        "filename": filename,
                        "predicted_class": pred_class,
                        "confidence": confidence,
                        "ood_flag": ood_flag,
                    }
                )
            except Exception:
                err_folder = os.path.join(output_dir, "Error_Procesamiento")
                os.makedirs(err_folder, exist_ok=True)
                try:
                    shutil.copy2(img_path, os.path.join(err_folder, filename))
                except Exception:
                    pass
                records.append(
                    {
                        "filename": filename,
                        "predicted_class": "Error",
                        "confidence": 0.0,
                        "ood_flag": True,
                    }
                )

        csv_path = os.path.join(output_dir, "report.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["filename", "predicted_class", "confidence", "ood_flag"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(r)

        zip_out_path = os.path.join(temp_dir, "batch_inference_results.zip")
        with zipfile.ZipFile(zip_out_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(output_dir):
                for file in files:
                    abs_p = os.path.join(root, file)
                    rel_p = os.path.relpath(abs_p, output_dir)
                    zf.write(abs_p, rel_p)

        return zip_out_path

    @app.post("/api/predict/batch")
    async def api_predict_batch_zip(
        file: UploadFile = File(...),  # noqa: B008
        model_slug: str | None = Query(None),
        model_name: str | None = Query(None),
        ood_threshold: float | None = Query(None),
    ) -> FileResponse:
        """Batch inference on a ZIP of images, returns a categorized ZIP with report.csv."""
        if not file.filename or not (
            file.filename.lower().endswith(".zip")
            or (file.content_type and "zip" in file.content_type)
        ):
            raise HTTPException(status_code=400, detail="Uploaded file must be a .zip archive")

        target_model = model_slug or model_name

        temp_dir = tempfile.mkdtemp(prefix="agrifed_batch_")
        zip_in_path = os.path.join(temp_dir, "uploaded_batch.zip")

        try:
            with open(zip_in_path, "wb") as f_out:
                while chunk := await file.read(1024 * 1024):
                    f_out.write(chunk)
        except Exception as e:
            _cleanup_dir(temp_dir)
            raise HTTPException(
                status_code=500, detail=f"Failed to save uploaded ZIP: {e}"
            ) from e

        loop = asyncio.get_running_loop()
        try:
            zip_out_path = await loop.run_in_executor(
                None,
                _sync_process_batch_zip,
                zip_in_path,
                temp_dir,
                target_model,
                ood_threshold,
            )
        except HTTPException:
            _cleanup_dir(temp_dir)
            raise
        except Exception as e:
            _cleanup_dir(temp_dir)
            raise HTTPException(status_code=500, detail=f"Batch inference failed: {e}") from e

        download_name = f"lote_{target_model or 'modelo'}_clasificado.zip"
        return FileResponse(
            path=zip_out_path,
            filename=download_name,
            media_type="application/zip",
            background=BackgroundTask(_cleanup_dir, temp_dir),
        )

    @app.post("/api/models/{slug}/predict/batch")
    async def api_model_predict_batch_zip(
        slug: str,
        file: UploadFile = File(...),  # noqa: B008
        ood_threshold: float | None = Query(None),
    ) -> FileResponse:
        """Convenience route for batch inference targeting a specific model slug."""
        return await api_predict_batch_zip(
            file=file,
            model_slug=slug,
            ood_threshold=ood_threshold,
        )

    @app.get("/download-model")
    def download_model(model_slug: str | None = Query(None)) -> FileResponse:
        """Endpoint to allow collaborative edge clients to download the model checkpoint bundle."""
        _resolve_and_reload_model()
        target_file: Path | None = None
        if model_slug:
            entry = catalog_manager.get_by_slug(model_slug)
            if entry:
                cand = Path("models") / entry.active_checkpoint
                if cand.is_file():
                    target_file = cand
        if not target_file:
            target_file = active_model_p or (model_p if model_p.is_file() else None)
        if not target_file or not target_file.is_file():
            raise HTTPException(status_code=404, detail="Model file not found on server")
        return FileResponse(
            path=str(target_file),
            filename=target_file.name,
            media_type="application/octet-stream",
        )

    @app.get("/download-client-script")
    def download_client_script() -> FileResponse:
        """Endpoint to allow edge clients to download the official Python training script."""
        script_path = Path("scripts/client_trainer.py")
        if not script_path.is_file():
            raise HTTPException(status_code=404, detail="Client trainer script not found on server")
        return FileResponse(
            path=str(script_path),
            filename="client_trainer.py",
            media_type="text/x-python",
        )

    @app.post("/reload-model")
    def reload_model() -> dict[str, Any]:
        """Manually trigger checkpoint reloading from disk."""
        _resolve_and_reload_model(force=True)
        return {
            "status": "reloaded",
            "model_path": str(active_model_p) if active_model_p else None,
            "model_sha256": model_sha256,
            "calibrated": calibrated_thresholds is not None,
        }

    @app.post("/submit-update")
    async def submit_update(
        file: UploadFile = File(...),  # noqa: B008
        manifest: UploadFile | None = File(None),  # noqa: B008
        client_id: str = "client",
        target_model: str | None = Query(None),
        authorization: str | None = Header(None),  # noqa: B008
        client_hmac: str | None = Header(None, alias="X-Client-HMAC"),  # noqa: B008
    ) -> dict[str, Any]:
        """Automated ingestion endpoint for collaborative edge clients to upload
        trained parameter bundles and companion cryptographic manifests.
        """
        server_token = os.getenv("AGRIFED_AUTH_TOKEN")
        server_hmac_key = os.getenv("AGRIFED_HMAC_KEY")
        if not server_token and not server_hmac_key:
            raise HTTPException(
                status_code=503,
                detail="Update ingestion is disabled until an auth token or HMAC key is configured",
            )

        # 1. Optional client authentication via pre-shared Bearer token
        if server_token:
            expected_auth = f"Bearer {server_token}"
            if not authorization or authorization.strip() != expected_auth:
                raise HTTPException(
                    status_code=401,
                    detail="Unauthorized: invalid or missing Bearer token in Authorization header",
                )

        # 2. Client ID validation and normalization
        trimmed_id = client_id.strip()
        if not re.match(r"^[a-zA-Z0-9_\-]{1,64}$", trimmed_id):
            raise HTTPException(
                status_code=400,
                detail="Invalid client_id: must match regex ^[a-zA-Z0-9_-]{1,64}$",
            )
        safe_client_id = trimmed_id

        # 3. Read uploaded parameter file
        content = await file.read(MAX_UPDATE_SIZE_BYTES + 1)
        if len(content) > MAX_UPDATE_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="Parameter file exceeds 50 MB limit")
        if not content:
            raise HTTPException(status_code=400, detail="Empty parameter file uploaded")
        file_sha256 = hashlib.sha256(content).hexdigest()

        # 4. Manifest processing & integrity validation
        if manifest is None:
            raise HTTPException(status_code=400, detail="A client manifest is required")
        manifest_bytes = await manifest.read(MAX_MANIFEST_SIZE_BYTES + 1)
        if len(manifest_bytes) > MAX_MANIFEST_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="Manifest exceeds 1 MB limit")
        try:
            manifest_data = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail="Malformed manifest JSON") from exc
        if not isinstance(manifest_data, dict):
            raise HTTPException(status_code=400, detail="Manifest must be a JSON object")

        declared_sha = manifest_data.get("sha256")
        if not isinstance(declared_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", declared_sha):
            raise HTTPException(status_code=400, detail="Manifest must include a valid sha256")
        if declared_sha.lower() != file_sha256:
            raise HTTPException(
                status_code=400,
                detail="Integrity check failed: uploaded payload SHA-256 does not match manifest",
            )
        if manifest_data.get("client_id") != safe_client_id:
            raise HTTPException(status_code=400, detail="Manifest client_id does not match request")
        base_sha = manifest_data.get("base_model_sha256")
        if not isinstance(base_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", base_sha):
            raise HTTPException(status_code=400, detail="Manifest has an invalid base_model_sha256")
        if manifest_data.get("training_scope") not in ("head_only", "full"):
            raise HTTPException(status_code=400, detail="Manifest has an invalid training_scope")
        sample_count = manifest_data.get("num_samples")
        if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 1:
            raise HTTPException(
                status_code=400, detail="Manifest num_samples must be a positive integer"
            )

        manifest_verified = True
        hmac_verified = False
        signature = manifest_data.get("hmac_signature")
        if client_hmac and client_hmac != signature:
            raise HTTPException(status_code=403, detail="HMAC header does not match manifest")
        if server_hmac_key:
            from agrifed_khaos.crypto import verify_manifest_hmac

            if not verify_manifest_hmac(server_hmac_key, manifest_data, file_sha256):
                raise HTTPException(status_code=403, detail="HMAC signature verification failed")
            hmac_verified = True

        # 5. Persist weights and companion manifest atomically under per-client lock
        client_lock = await _get_client_lock(safe_client_id)
        async with client_lock:
            target_dir = Path(os.getenv("AGRIFED_UPDATES_DIR", "models/client_updates"))
            target_dir.mkdir(parents=True, exist_ok=True)
            archive_dir = target_dir / "archive"
            archive_dir.mkdir(parents=True, exist_ok=True)

            updates_dir = target_dir / "updates" / safe_client_id
            updates_dir.mkdir(parents=True, exist_ok=True)

            stamp = f"{time.time_ns()}_{uuid.uuid4().hex[:8]}"
            final_update_dir = updates_dir / stamp
            staging_dir = updates_dir / f".staging_{stamp}"
            staging_dir.mkdir(parents=True, exist_ok=True)

            try:
                (staging_dir / "parameters.meta.json").write_bytes(manifest_bytes)
                (staging_dir / "parameters.pt").write_bytes(content)
                os.replace(staging_dir, final_update_dir)
            except Exception:
                shutil.rmtree(staging_dir, ignore_errors=True)
                raise

            dest_path = target_dir / f"{safe_client_id}_parameters.pt"
            dest_meta = target_dir / f"{safe_client_id}_parameters.meta.json"

            _atomic_write(dest_meta, manifest_bytes)
            _atomic_write(dest_path, content)

            target_slug = (
                target_model
                or manifest_data.get("target_model")
                or manifest_data.get("model_slug")
            )
            if target_slug:
                model_dir = target_dir / target_slug
                model_dir.mkdir(parents=True, exist_ok=True)
                _atomic_write(model_dir / f"{safe_client_id}_parameters.meta.json", manifest_bytes)
                _atomic_write(model_dir / f"{safe_client_id}_parameters.pt", content)

            _atomic_write(
                archive_dir / f"{safe_client_id}_{stamp}_parameters.meta.json", manifest_bytes
            )
            _atomic_write(archive_dir / f"{safe_client_id}_{stamp}_parameters.pt", content)

            return {
                "status": "ready_for_aggregation",
                "client_id": safe_client_id,
                "bytes_received": len(content),
                "sha256": file_sha256,
                "has_manifest": True,
                "manifest_verified": manifest_verified,
                "hmac_verified": hmac_verified,
                "saved_to": str(dest_path),
                "meta_saved_to": str(dest_meta),
                "update_dir": str(final_update_dir),
            }

    return app


def main() -> None:
    args = parse_args()
    app = create_app(args.model, args.metadata, architecture=args.arch, device_name=args.device)

    if args.ssl_keyfile and args.ssl_certfile:
        print(
            f"[*] Starting AgriFed REST Inference Server on https://{args.host}:{args.port} "
            "(TLS enabled)"
        )
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            log_level="info",
        )
    else:
        auth_configured = bool(os.getenv("AGRIFED_AUTH_TOKEN") or os.getenv("AGRIFED_HMAC_KEY"))
        if args.host not in ("127.0.0.1", "localhost") and auth_configured:
            print(
                "[!] SECURITY NOTICE: Server is listening on unencrypted HTTP with "
                "authentication/HMAC enabled.\n"
                "    Bearer tokens and HMAC signatures travel in plaintext unless encrypted.\n"
                "    Outside a trusted network, provide --ssl-certfile and --ssl-keyfile or run\n"
                "    behind an HTTPS reverse proxy (e.g., Nginx, Caddy) terminating TLS."
            )
        print(f"[*] Starting AgriFed REST Inference Server on http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
