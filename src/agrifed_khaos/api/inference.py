"""Inference, Batch ZIP Evaluation, Edge Downloads & Federated Client Ingestion API."""

from __future__ import annotations

import asyncio
import hashlib
import hmac as hmac_mod
import io
import json
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import torch
from fastapi import APIRouter, File, Header, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from starlette.background import BackgroundTask

from agrifed_khaos.catalog import ModelCatalogManager
from agrifed_khaos.crypto import compute_manifest_hmac, compute_sha256
from agrifed_khaos.hitl import HITLQueueManager
from agrifed_khaos.inference_service import InferenceEngine
from agrifed_khaos.models import build_model, load_checkpoint_bundle, save_checkpoint_bundle
from agrifed_khaos.onnx_export import get_or_export_edge_bundle, isolate_backbone_and_head
from agrifed_khaos.schemas import LightClientHeadUpdateRequest, ModelBundleResponse
from agrifed_khaos.wrappers import ParameterValidator

logger = logging.getLogger("agrifed.api.inference")
router = APIRouter(tags=["Inference & Edge"])

MAX_UPLOAD_SIZE_BYTES = 15 * 1024 * 1024
MAX_UPDATE_SIZE_BYTES = 50 * 1024 * 1024
MAX_MANIFEST_SIZE_BYTES = 1 * 1024 * 1024

_client_locks: dict[str, asyncio.Lock] = {}
_client_locks_guard = asyncio.Lock()


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


def _get_engine(request: Request) -> InferenceEngine:
    engine = getattr(request.app.state, "inference_engine", None)
    if not engine:
        raise HTTPException(
            status_code=503,
            detail="InferenceEngine is not initialized on application.",
        )
    return engine


def _get_cm(request: Request) -> ModelCatalogManager:
    return getattr(
        request.app.state,
        "catalog",
        ModelCatalogManager(catalog_path=Path("models/catalog.json"), models_dir=Path("models")),
    )


def _get_hitl(request: Request) -> HITLQueueManager:
    return getattr(request.app.state, "hitl", HITLQueueManager())


def _cleanup_dir(path: str) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except OSError as exc:
        logger.debug("Failed cleaning up directory %s: %s", path, exc)


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    """Health check reporting active model status, architecture and calibration metadata."""
    engine = _get_engine(request)
    engine.resolve_and_reload_model()
    return {
        "status": "online",
        "task": getattr(engine, "task", "classification"),
        "model_path": str(engine.active_model_p) if engine.active_model_p else str(engine.model_p),
        "model_sha256": engine.model_sha256,
        "architecture": engine.arch,
        "num_classes": engine.num_classes,
        "classes": engine.classes,
        "img_size": list(engine.img_size_tuple),
        "calibrated_thresholds": engine.calibrated_thresholds,
    }


@router.post("/predict")
@router.post("/api/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),  # noqa: B008
    model_name: str | None = Query(None),
    model_slug: str | None = Query(None),
    save_for_training: bool = Query(True),
    ood_threshold: float | None = Query(None),
) -> dict[str, Any]:
    """Single-image vision inference with OOD detection and optional HITL feedback queueing."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    content = await file.read(MAX_UPLOAD_SIZE_BYTES + 1)
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        max_mb = MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded image exceeds maximum allowed limit of {max_mb} MB",
        )

    target_model = model_slug or model_name
    engine = _get_engine(request)
    cm = _get_cm(request)
    res = engine.infer_single(
        content,
        file.filename,
        model_name=target_model,
        ood_threshold=ood_threshold,
    )

    if save_for_training:
        try:
            active_hitl = _get_hitl(request)
            matching_entry = cm.get_by_slug(res.get("model_name", ""))
            chosen_slug = (
                model_slug
                or (matching_entry.slug if matching_entry else None)
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
                detections=res.get("detections"),
            )
            res["saved_for_training"] = True
            res["model_slug"] = chosen_slug
        except (OSError, RuntimeError) as exc:
            logger.warning("HITL enqueue failed for prediction: %s", exc)
            res["saved_for_training"] = False

    return res


@router.post("/predict/batch")
async def predict_batch(
    request: Request,
    files: list[UploadFile] = File(...),  # noqa: B008
    model_name: str | None = Query(None),
    model_slug: str | None = Query(None),
    ood_threshold: float | None = Query(None),
) -> dict[str, Any]:
    """Multi-file image batch inference with vectorization."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded for batch inference")

    target_model = model_slug or model_name
    engine = _get_engine(request)

    if engine.is_detection_model(target_model):
        (
            yolo_model,
            yolo_classes,
            actual_model_name,
        ) = engine.get_yolo_model_for_inference(target_model)
        predictions: list[dict[str, Any] | None] = [None] * len(files)
        for idx, file in enumerate(files):
            if not file.content_type or not file.content_type.startswith("image/"):
                predictions[idx] = {"filename": file.filename, "error": "Not an image"}
                continue
            content = await file.read(MAX_UPLOAD_SIZE_BYTES + 1)
            if len(content) > MAX_UPLOAD_SIZE_BYTES:
                predictions[idx] = {
                    "filename": file.filename,
                    "error": f"Image exceeds limit of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB",
                }
                continue
            try:
                img = Image.open(io.BytesIO(content)).convert("RGB")
                predictions[idx] = engine.infer_detection(
                    img, yolo_model, yolo_classes, file.filename, actual_model_name
                )
            except Exception as exc:
                predictions[idx] = {"filename": file.filename, "error": str(exc)}
        return {
            "model_name": actual_model_name,
            "total_samples": len(files),
            "predictions": predictions,
            "task": "detection",
        }

    (
        infer_net,
        infer_classes,
        infer_num_classes,
        infer_cal,
        infer_temp,
        infer_tf,
        actual_model_name,
    ) = engine.get_model_for_inference(target_model)

    eff_thresh = ood_threshold
    if eff_thresh is None:
        for v in engine.model_cache.values():
            if v.get("net") is infer_net:
                eff_thresh = v.get("ood_threshold")
                break

    predictions: list[dict[str, Any] | None] = [None] * len(files)

    for idx, file in enumerate(files):
        if not file.content_type or not file.content_type.startswith("image/"):
            predictions[idx] = {"filename": file.filename, "error": "Not an image"}
            continue
        content = await file.read(MAX_UPLOAD_SIZE_BYTES + 1)
        if len(content) > MAX_UPLOAD_SIZE_BYTES:
            predictions[idx] = {
                "filename": file.filename,
                "error": f"Image exceeds limit of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB",
            }
            continue
        try:
            img = Image.open(io.BytesIO(content)).convert("RGB")
            probs, fused_logits = engine.infer_pil_multiview(
                img, infer_net, infer_tf, infer_temp
            )
            predictions[idx] = engine.decode_probs_row(
                probs,
                infer_classes,
                infer_num_classes,
                infer_cal,
                file.filename,
                actual_model_name,
                logits=fused_logits,
                ood_threshold=eff_thresh,
                ood_temperature=infer_temp if infer_temp > 0 else 1.0,
            )
        except Image.DecompressionBombError as exc:
            predictions[idx] = {
                "filename": file.filename,
                "error": f"Decompression bomb detected: image exceeds pixel limit ({exc})",
            }
        except (OSError, ValueError) as exc:
            predictions[idx] = {
                "filename": file.filename,
                "error": f"Invalid image format or corrupted payload: {exc}",
            }

    return {
        "batch_size": len(files),
        "predictions": [p for p in predictions if p is not None],
    }


@router.post("/api/predict/batch")
async def api_predict_batch_zip(
    request: Request,
    file: UploadFile = File(...),  # noqa: B008
    model_slug: str | None = Query(None),
    model_name: str | None = Query(None),
    ood_threshold: float | None = Query(None),
    save_for_training: bool = Query(True),
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
    except OSError as e:
        _cleanup_dir(temp_dir)
        raise HTTPException(
            status_code=500, detail=f"Failed to save uploaded ZIP: {e}"
        ) from e

    engine = _get_engine(request)
    hitl_mgr = _get_hitl(request)
    loop = asyncio.get_running_loop()
    try:
        zip_out_path = await loop.run_in_executor(
            None,
            engine.sync_process_batch_zip,
            zip_in_path,
            temp_dir,
            target_model,
            ood_threshold,
            save_for_training,
            hitl_mgr,
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


@router.post("/api/models/{slug}/predict/batch")
async def api_model_predict_batch_zip(
    slug: str,
    request: Request,
    file: UploadFile = File(...),  # noqa: B008
    ood_threshold: float | None = Query(None),
    save_for_training: bool = Query(True),
) -> FileResponse:
    """Convenience route for batch inference targeting a specific model slug."""
    return await api_predict_batch_zip(
        request=request,
        file=file,
        model_slug=slug,
        ood_threshold=ood_threshold,
        save_for_training=save_for_training,
    )


@router.get("/download-model")
def download_model(request: Request, model_slug: str | None = Query(None)) -> FileResponse:
    """Endpoint to allow collaborative edge clients to download the model checkpoint bundle."""
    engine = _get_engine(request)
    cm = _get_cm(request)
    engine.resolve_and_reload_model()
    target_file: Path | None = None
    if model_slug:
        entry = cm.get_by_slug(model_slug)
        if entry and entry.active_checkpoint:
            cand = Path("models") / entry.active_checkpoint
            if cand.is_file():
                target_file = cand
    if not target_file:
        target_file = engine.active_model_p or (
            engine.model_p if engine.model_p.is_file() else None
        )
    if not target_file or not target_file.is_file():
        raise HTTPException(status_code=404, detail="Model file not found on server")
    return FileResponse(
        path=str(target_file),
        filename=target_file.name,
        media_type="application/octet-stream",
    )


@router.get("/download-client-script")
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


@router.post("/reload-model")
def reload_model(request: Request) -> dict[str, Any]:
    """Manually trigger checkpoint reloading from disk."""
    engine = _get_engine(request)
    engine.resolve_and_reload_model(force=True)
    return {
        "status": "reloaded",
        "model_path": str(engine.active_model_p) if engine.active_model_p else None,
        "model_sha256": engine.model_sha256,
        "calibrated": engine.calibrated_thresholds is not None,
    }


@router.post("/submit-update")
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

    if server_token:
        expected_auth = f"Bearer {server_token}"
        if not authorization or authorization.strip() != expected_auth:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: invalid or missing Bearer token in Authorization header",
            )

    trimmed_id = client_id.strip()
    if not re.match(r"^[a-zA-Z0-9_\-]{1,64}$", trimmed_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid client_id: must match regex ^[a-zA-Z0-9_-]{1,64}$",
        )
    safe_client_id = trimmed_id

    content = await file.read(MAX_UPDATE_SIZE_BYTES + 1)
    if len(content) > MAX_UPDATE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Parameter file exceeds 50 MB limit")
    if not content:
        raise HTTPException(status_code=400, detail="Empty parameter file uploaded")
    file_sha256 = hashlib.sha256(content).hexdigest()

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
        except OSError:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

        dest_path = target_dir / f"{safe_client_id}_parameters.pt"
        dest_meta = target_dir / f"{safe_client_id}_parameters.meta.json"

        _atomic_write(dest_meta, manifest_bytes)
        _atomic_write(dest_path, content)

        target_slug = (
            target_model or manifest_data.get("target_model") or manifest_data.get("model_slug")
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


def _resolve_target_checkpoint_and_meta(
    request: Request, model_slug: str | None
) -> tuple[Path, Path | None, str]:
    """Resolve active model weights, companion metadata, and canonical slug."""
    cm = _get_cm(request)
    engine = _get_engine(request)
    entry = None
    if model_slug:
        entry = cm.get_by_slug(model_slug)
        if entry:
            chk = Path("models") / (entry.active_checkpoint or f"{entry.slug}_base.pt")
            if chk.is_file():
                meta = Path("models") / f"{entry.slug}.meta.json"
                return chk, meta if meta.is_file() else None, entry.slug
    chk = engine.active_model_p or engine.model_p
    slug = (entry.slug if entry else None) or getattr(engine, "model_name", "global_model")
    meta = Path("models") / "metadata.json"
    return chk, meta if meta.is_file() else None, slug


def _reject_if_detection_model(request: Request, model_slug: str | None) -> None:
    """Edge PWA bundles/updates only exist for classification heads; reject YOLO cleanly (400)."""
    if not model_slug:
        return
    entry = _get_cm(request).get_by_slug(model_slug)
    if entry and (
        entry.task == "detection" or (entry.architecture and "yolo" in entry.architecture.lower())
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Edge PWA on-device federation is only supported for vision classification "
                "architectures, not object detection models."
            ),
        )


MAX_EDGE_NUM_SAMPLES = 100_000


@router.api_route(
    "/federate/model-bundle", methods=["GET", "HEAD"], response_model=ModelBundleResponse
)
async def get_federate_model_bundle(
    request: Request,
    response: Response,
    model_slug: str | None = Query(None),
) -> ModelBundleResponse:
    """Deliver synchronized ONNX backbone and lightweight head to edge PWA clients."""
    _reject_if_detection_model(request, model_slug)

    chk_p, meta_p, slug = _resolve_target_checkpoint_and_meta(request, model_slug)
    if not chk_p or not chk_p.is_file():
        raise HTTPException(status_code=404, detail="Active model checkpoint not found on server")

    loop = asyncio.get_running_loop()
    try:
        bundle = await loop.run_in_executor(
            None,
            get_or_export_edge_bundle,
            chk_p,
            slug,
            Path("models"),
            meta_p,
        )
    except Exception as exc:
        logger.error("Error creating edge model bundle for %s: %s", slug, exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to generate edge bundle: {exc}"
        ) from exc

    response.headers["ETag"] = f'"{bundle.base_model_sha256}"'
    response.headers["Cache-Control"] = "no-cache"

    head_json_bytes = bundle.head_weights_path.read_bytes()
    head_data = json.loads(head_json_bytes.decode("utf-8"))

    return ModelBundleResponse(
        schema_version="1.0",
        model_slug=bundle.model_slug,
        base_model_sha256=bundle.base_model_sha256,
        architecture=bundle.architecture,
        num_classes=bundle.num_classes,
        classes=bundle.classes,
        embedding_dim=bundle.embedding_dim,
        head_tensor_keys=head_data.get("head_tensor_keys", {}),
        preprocessing=bundle.preprocessing,
        head_weights=head_data.get("head_weights", {}),
        onnx_download_url=f"/federate/backbone.onnx?model_slug={quote(bundle.model_slug)}&sha256={bundle.base_model_sha256[:16]}",
        cached=bundle.cached,
    )


@router.api_route("/federate/backbone.onnx", methods=["GET", "HEAD"])
def get_federate_backbone_onnx(
    request: Request,
    model_slug: str = Query(...),
    sha256: str | None = Query(None),
) -> FileResponse:
    """Serve immutable ONNX backbone feature extractor for client-side local inference."""
    if not re.match(r"^[a-zA-Z0-9_\-]+$", model_slug):
        raise HTTPException(status_code=400, detail="Invalid model_slug parameter")
    _reject_if_detection_model(request, model_slug)
    chk_p, meta_p, slug = _resolve_target_checkpoint_and_meta(request, model_slug)
    if not chk_p or not chk_p.is_file():
        raise HTTPException(status_code=404, detail=f"Model '{model_slug}' not found")

    bundle = get_or_export_edge_bundle(chk_p, slug, Path("models"), meta_p)
    if not bundle.onnx_path.is_file():
        raise HTTPException(status_code=404, detail="ONNX backbone file missing")

    return FileResponse(
        path=str(bundle.onnx_path),
        filename=f"{model_slug}_backbone.onnx",
        media_type="application/octet-stream",
        headers={"ETag": f'"{bundle.base_model_sha256}"', "Cache-Control": "public, max-age=86400"},
    )


@router.post("/federate/submit-update")
async def submit_federate_update(
    req: LightClientHeadUpdateRequest,
    request: Request,
    authorization: str | None = Header(None),
    client_hmac: str | None = Header(None, alias="X-Client-HMAC"),
) -> dict[str, Any]:
    """Ingest lightweight head-only updates (~10-45 KB) trained on-device via PWA."""
    server_token = os.getenv("AGRIFED_AUTH_TOKEN")
    server_hmac_key = os.getenv("AGRIFED_HMAC_KEY")

    if server_token:
        expected_auth = f"Bearer {server_token}"
        if not authorization or authorization.strip() != expected_auth:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: invalid or missing Bearer token in Authorization header",
            )

    trimmed_id = req.client_id.strip()
    if not re.match(r"^[a-zA-Z0-9_\-]{1,64}$", trimmed_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid client_id: must match regex ^[a-zA-Z0-9_-]{1,64}$",
        )
    safe_client_id = trimmed_id

    if req.training_scope != "head_only":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid training_scope '{req.training_scope}': light clients must use 'head_only'"
            ),
        )

    if req.num_samples < 1:
        raise HTTPException(status_code=400, detail="num_samples must be positive integer >= 1")
    if req.num_samples > MAX_EDGE_NUM_SAMPLES:
        # num_samples is client-declared and drives FedAvg weighting: cap it to bound inflation.
        raise HTTPException(
            status_code=400,
            detail=f"num_samples exceeds the accepted maximum ({MAX_EDGE_NUM_SAMPLES})",
        )

    _reject_if_detection_model(request, req.model_slug)
    chk_p, meta_p, slug = _resolve_target_checkpoint_and_meta(request, req.model_slug)
    if not chk_p or not chk_p.is_file():
        raise HTTPException(status_code=404, detail=f"Target model '{req.model_slug}' not found")

    actual_base_sha = compute_sha256(chk_p)
    if req.base_model_sha256 != actual_base_sha:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Lineage conflict: base model on server has been updated "
                f"({actual_base_sha[:16]}...), but client update was trained against "
                f"({req.base_model_sha256[:16]}...). "
                "Please sync the new model bundle before submitting."
            ),
        )

    # Validate head tensors
    raw_w = req.head_tensors.get("weight")
    raw_b = req.head_tensors.get("bias")
    if raw_w is None or raw_b is None:
        raise HTTPException(
            status_code=422, detail="head_tensors must contain both 'weight' and 'bias'"
        )

    try:
        w_arr = np.asarray(raw_w, dtype=np.float32)
        b_arr = np.asarray(raw_b, dtype=np.float32)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to parse head tensors: {exc}") from exc

    if not np.all(np.isfinite(w_arr)) or not np.all(np.isfinite(b_arr)):
        raise HTTPException(status_code=422, detail="Head tensors contain NaN or Infinite values")

    # Integrity: when the client declares payload_sha256 it must match the received tensors
    # (float32 weight bytes followed by bias bytes), independently of HMAC being enabled.
    if req.payload_sha256:
        received_sha = hashlib.sha256(
            np.ascontiguousarray(w_arr).tobytes() + np.ascontiguousarray(b_arr).tobytes()
        ).hexdigest()
        if not hmac_mod.compare_digest(received_sha, req.payload_sha256.lower()):
            raise HTTPException(
                status_code=422,
                detail="payload_sha256 does not match the received head tensors",
            )

    # Load base model state to get target keys and reference parameters
    base_state, bundle_meta = load_checkpoint_bundle(chk_p, device="cpu")
    arch = bundle_meta.get("architecture") or "resnet18"
    base_num_classes = bundle_meta.get("num_classes", len(bundle_meta.get("classes", [])) or 9)
    base_classes = list(bundle_meta.get("classes", [])) or [
        f"class_{i}" for i in range(base_num_classes)
    ]

    base_model = build_model(architecture=arch, num_classes=base_num_classes)
    base_model.load_state_dict(base_state)
    _, base_w_tensor, base_b_tensor, w_key, b_key = isolate_backbone_and_head(base_model, arch)

    base_w_arr = base_w_tensor.cpu().numpy()
    base_b_arr = base_b_tensor.cpu().numpy()

    # Canonical Class Projection (Architectural Rule 1: Anti-Ontology Pollution)
    # If client created extra classes locally, project only the canonical classes
    if w_arr.shape[0] != base_w_arr.shape[0]:
        if req.classes and set(base_classes).issubset(set(req.classes)):
            indices = [req.classes.index(c) for c in base_classes]
            w_arr = w_arr[indices, :]
            b_arr = b_arr[indices]
            logger.info(
                "Client %s submitted %d classes; projected to %d canonical classes",
                safe_client_id,
                len(req.classes),
                len(base_classes),
            )

    if w_arr.shape != base_w_arr.shape or b_arr.shape != base_b_arr.shape:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Head tensor shape mismatch: expected weight {base_w_arr.shape}, "
                f"bias {base_b_arr.shape}; got weight {w_arr.shape}, bias {b_arr.shape}"
            ),
        )

    # HMAC verification if required
    if server_hmac_key:
        sig = req.hmac_signature or client_hmac
        if not sig:
            raise HTTPException(
                status_code=403,
                detail="Server requires HMAC authentication, but no signature was provided",
            )
        canonical_bytes = w_arr.tobytes() + b_arr.tobytes()
        tensors_sha256 = hashlib.sha256(canonical_bytes).hexdigest()
        expected_sig = compute_manifest_hmac(
            key=server_hmac_key,
            client_sha256=tensors_sha256,
            base_model_sha256=actual_base_sha,
            training_scope="head_only",
            client_id=safe_client_id,
            num_samples=req.num_samples,
        )
        import hmac
        if not hmac.compare_digest(sig, expected_sig):
            raise HTTPException(status_code=403, detail="Invalid HMAC signature for head update")

    # ParameterValidator audit (norm limits & divergence sanity checks)
    validator = ParameterValidator(max_update_norm=150.0, max_relative_norm=0.50)
    audit_res = validator.validate([w_arr, b_arr], [base_w_arr, base_b_arr])
    if not audit_res.valid:
        raise HTTPException(
            status_code=422,
            detail=f"Parameter audit rejected client update: {'; '.join(audit_res.issues)}",
        )

    # Package into native .pt + .meta.json checkpoint bundle
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

        saved_state = {
            w_key: torch.from_numpy(w_arr),
            b_key: torch.from_numpy(b_arr),
        }
        dest_pt_path = staging_dir / "parameters.pt"
        dest_meta_path = staging_dir / "parameters.meta.json"

        meta_payload = {
            "client_weights_file": str(dest_pt_path),
            "client_id": safe_client_id,
            "base_model_sha256": actual_base_sha,
            "num_samples": req.num_samples,
            "delta_norm": round(audit_res.delta_norm or 0.0, 4),
            "relative_norm": round(audit_res.relative_norm or 0.0, 4),
            "training_scope": "head_only",
            "architecture": arch,
            "num_classes": base_num_classes,
            "classes": base_classes,
            "missing_classes": req.missing_classes,
            "trainable_tensors": [w_key, b_key],
            "training_stats": req.training_stats,
            "client_tier": "pwa-edge-light",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        save_checkpoint_bundle(dest_pt_path, saved_state, meta_payload)
        out_sha = compute_sha256(dest_pt_path)
        meta_payload["sha256"] = out_sha

        if server_hmac_key:
            meta_payload["hmac_signature"] = compute_manifest_hmac(
                key=server_hmac_key,
                client_sha256=out_sha,
                base_model_sha256=actual_base_sha,
                training_scope="head_only",
                client_id=safe_client_id,
                num_samples=req.num_samples,
            )

        manifest_bytes = json.dumps(meta_payload, indent=2).encode("utf-8")
        dest_meta_path.write_bytes(manifest_bytes)
        pt_content = dest_pt_path.read_bytes()

        os.replace(staging_dir, final_update_dir)

        # Write top-level updates files
        root_dest_path = target_dir / f"{safe_client_id}_parameters.pt"
        root_dest_meta = target_dir / f"{safe_client_id}_parameters.meta.json"
        _atomic_write(root_dest_meta, manifest_bytes)
        _atomic_write(root_dest_path, pt_content)

        # Write canonical {slug}__{safe_client_id}.pt for quorum aggregation
        slug_dest_path = target_dir / f"{slug}__{safe_client_id}.pt"
        slug_dest_meta = target_dir / f"{slug}__{safe_client_id}.meta.json"
        _atomic_write(slug_dest_meta, manifest_bytes)
        _atomic_write(slug_dest_path, pt_content)

        # Model slug scoped directory
        model_scoped_dir = target_dir / req.model_slug
        model_scoped_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(model_scoped_dir / f"{safe_client_id}_parameters.meta.json", manifest_bytes)
        _atomic_write(model_scoped_dir / f"{safe_client_id}_parameters.pt", pt_content)

        # Archive
        _atomic_write(
            archive_dir / f"{safe_client_id}_{stamp}_parameters.meta.json", manifest_bytes
        )
        _atomic_write(archive_dir / f"{safe_client_id}_{stamp}_parameters.pt", pt_content)

        return {
            "status": "ready_for_aggregation",
            "client_id": safe_client_id,
            "training_scope": "head_only",
            "delta_norm": meta_payload["delta_norm"],
            "relative_norm": meta_payload["relative_norm"],
            "sha256": out_sha,
            "bytes_received": len(pt_content),
            "manifest_verified": True,
            "hmac_verified": server_hmac_key is not None,
            "saved_to": str(root_dest_path),
            "update_dir": str(final_update_dir),
        }

