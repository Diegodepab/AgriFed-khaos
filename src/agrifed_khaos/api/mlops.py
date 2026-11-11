"""MLOps Studio, Model Telemetry, Federated Aggregation & Evaluation Router."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel

from agrifed_khaos.catalog import (
    LogicalModelEntry,
    ModelCatalogManager,
    ModelVersionRecord,
    append_model_history,
    slugify,
)
from agrifed_khaos.dataset_service import detect_dataset_structure
from agrifed_khaos.federated_ops import (
    FederatedAggregationRequest,
    FederatedOperationRecord,
    FederatedOperationsManager,
    audit_client_update,
    compute_sha256,
    resolve_validation_directory_for_model,
)
from agrifed_khaos.hardware import HardwareProfile, ModelSelector
from agrifed_khaos.jobs import JobManager
from agrifed_khaos.mlops_service import (
    compare_model_versions as run_compare_ab,
)
from agrifed_khaos.mlops_service import (
    generate_model_gradcam as run_gradcam,
)
from agrifed_khaos.mlops_service import (
    get_federated_updates as run_federated_updates,
)
from agrifed_khaos.mlops_service import (
    get_model_telemetry as run_model_telemetry,
)
from agrifed_khaos.schemas import AutoFLRequest

logger = logging.getLogger("agrifed.api.mlops")
router = APIRouter(tags=["MLOps & Evaluation"])


def _get_cm(request: Request) -> ModelCatalogManager:
    return getattr(
        request.app.state,
        "catalog",
        ModelCatalogManager(catalog_path=Path("models/catalog.json"), models_dir=Path("models")),
    )


def _get_jm(request: Request) -> JobManager:
    jm = getattr(request.app.state, "job_manager", None)
    if not jm:
        raise HTTPException(status_code=500, detail="JobManager is not initialized on application.")
    return jm


def _get_fl_ops(request: Request) -> FederatedOperationsManager:
    fl = getattr(request.app.state, "fl_ops", None)
    if fl:
        return fl
    cm = _get_cm(request)
    models_root = cm.models_dir if hasattr(cm, "models_dir") and cm.models_dir else Path("models")
    return FederatedOperationsManager(storage_path=models_root / "federated_operations.json")


def _get_engine(request: Request) -> Any:
    return getattr(request.app.state, "inference_engine", None)


class GPUWorkerConfigRequest(BaseModel):
    gpu_worker_url: str = ""


def _get_gpu_worker_url(request: Request) -> str:
    runtime_url = getattr(request.app.state, "gpu_worker_url", None)
    if runtime_url is not None:
        return str(runtime_url).strip().rstrip("/")
    url = (
        os.environ.get("AGRIFED_GPU_WORKER_URL")
        or os.environ.get("AGRIFED_GPU_ENDPOINT")
        or ""
    )
    return url.strip().rstrip("/")


@router.get("/api/master/gpu-worker")
async def get_gpu_worker_status(request: Request) -> dict[str, Any]:
    """Check if a remote GPU worker (agrifed-gpu) is configured and reachable."""
    gpu_url = _get_gpu_worker_url(request)
    if not gpu_url:
        return {
            "configured": False,
            "reachable": False,
            "health": None,
        }
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{gpu_url}/health")
            resp.raise_for_status()
            health_data = resp.json()
        return {
            "configured": True,
            "reachable": True,
            "health": health_data,
        }
    except Exception as exc:
        logger.debug("GPU worker probe failed: %s", exc)
        return {
            "configured": True,
            "reachable": False,
            "error": "Nodo GPU no disponible",
            "health": None,
        }


@router.post("/api/master/gpu-worker")
async def configure_gpu_worker(cfg: GPUWorkerConfigRequest, request: Request) -> dict[str, Any]:
    """Configure or clear the remote GPU worker URL (e.g. http://<gpu-ip>:8088) at runtime."""
    clean_url = cfg.gpu_worker_url.strip().rstrip("/")
    request.app.state.gpu_worker_url = clean_url
    os.environ["AGRIFED_GPU_WORKER_URL"] = clean_url
    return await get_gpu_worker_status(request)


@router.post("/api/master/models/auto-fl", status_code=202)
async def auto_fl_train_model(req: AutoFLRequest, request: Request) -> dict[str, Any]:
    """Zero-to-Model Pipeline: Train a new model from dataset and auto-register it."""
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

    from agrifed_khaos.coco import (
        coco_to_classification_crops,
        discover_coco_splits,
        inspect_coco_dataset,
        is_coco_dataset,
    )
    from agrifed_khaos.data import find_dataset_class_dir
    from agrifed_khaos.yolo import inspect_yolo_dataset, is_yolo_dataset

    coco_root = None
    if is_coco_dataset(ds_path):
        coco_root = ds_path
    elif ds_path.name.lower() in ("train", "val", "test") and is_coco_dataset(ds_path.parent):
        coco_root = ds_path.parent

    yolo_root = None
    if is_yolo_dataset(ds_path):
        yolo_root = ds_path
    elif ds_path.name.lower() in ("train", "val", "test", "images", "labels") and is_yolo_dataset(
        ds_path.parent
    ):
        yolo_root = ds_path.parent

    jm = _get_jm(request)
    cm = _get_cm(request)

    # Determine dataset format & task
    is_detection_source = (coco_root is not None) or (yolo_root is not None)
    det_classes: list[str] = []
    if coco_root is not None:
        try:
            insp = inspect_coco_dataset(coco_root)
            det_classes = insp.get("classes", [])
        except Exception:
            det_classes = []
    elif yolo_root is not None:
        try:
            insp = inspect_yolo_dataset(yolo_root)
            det_classes = insp.get("classes", [])
        except Exception:
            det_classes = []

    # Decide if we route to detection:
    # 1. Dataset is detection source AND (req.task == "detection" or arch is "yolo" or auto)
    # 2. Single-class detection dataset (e.g. wheat_head): classification impossible
    is_detection_run = False
    if is_detection_source:
        req_task = getattr(req, "task", "auto").lower()
        arch_lower = req.architecture.lower()
        if len(det_classes) == 1:
            # Single class detection dataset (e.g. wheathead)
            is_detection_run = True
        elif req_task == "detection" or arch_lower.startswith("yolo"):
            is_detection_run = True
        elif req_task == "auto":
            is_detection_run = True
        elif req_task == "classification":
            is_detection_run = False

    slug = slugify(req.slug or req.title)
    models_root = cm.models_dir
    models_root.mkdir(parents=True, exist_ok=True)
    out_weights = models_root / f"{slug}_base.pt"
    out_meta = models_root / f"{slug}.meta.json"

    if is_detection_run:
        det_data_dir = coco_root or yolo_root or ds_path
        chosen_yolo_arch = (
            req.architecture if req.architecture.lower().startswith("yolo") else "yolov8n"
        )
        class_names = det_classes if det_classes else ["object"]

        cmd = [
            sys.executable,
            "scripts/train_detection.py",
            "--data-dir",
            str(det_data_dir),
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
            chosen_yolo_arch,
        ]

        jm.logs.append(
            f"[*] [AUTO-FL] Pipeline Inteligente: Dataset de Detección de Objetos detectado "
            f"({len(class_names)} clase(s): {', '.join(class_names)}). "
            f"Enrutado automáticamente al motor Ultralytics YOLO ({chosen_yolo_arch})."
        )
    else:
        # Classification path: if COCO dataset was selected for crop classification
        if coco_root is not None:
            splits_dict = discover_coco_splits(coco_root)
            if splits_dict:
                uploads_root = Path("data/uploads")
                uploads_root.mkdir(parents=True, exist_ok=True)
                out_crops_dir = uploads_root / f"crops_{int(time.time())}_{slug}"
                total_crops: dict[str, int] = {}
                for s_name, (s_json, s_img_d) in splits_dict.items():
                    if "unannotated" in s_name:
                        continue
                    max_c = 2000 if s_name == "train" else 500
                    crops_per_cat = coco_to_classification_crops(
                        coco_json_path=s_json,
                        images_dir=s_img_d,
                        output_dir=out_crops_dir,
                        split_name=s_name,
                        max_crops_per_class=max_c,
                    )
                    for c_name, count in crops_per_cat.items():
                        total_crops[c_name] = total_crops.get(c_name, 0) + count

                if any(total_crops.values()):
                    ds_path = out_crops_dir
                    jm.logs.append(
                        f"[*] [AUTO-FL] Extracción de recortes para clasificación completada: "
                        f"{sum(total_crops.values())} recortes en {len(total_crops)} clases."
                    )

        resolved_ds_path = find_dataset_class_dir(ds_path, preferred_split="train")
        subdirs = [
            d
            for d in resolved_ds_path.iterdir()
            if d.is_dir() and not d.name.startswith((".", "_")) and d.name.lower() != "__macosx"
        ]

        if not subdirs:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"El directorio '{req.dataset_path}' no contiene subcarpetas de clases. "
                    f"Estructura esperada: {req.dataset_path}/<clase_1>..."
                ),
            )

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".ppm", ".pgm"}
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
        if len(class_names) < 2:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"El dataset en '{req.dataset_path}' solo contiene {len(class_names)} "
                    f"clase(s) ({class_names}). Un modelo de clasificación supervisada "
                    "requiere al menos 2 clases distintas. Para datasets monoclase de detección, "
                    "el sistema entrena un detector de objetos YOLO."
                ),
            )

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
            "--max-samples",
            "0",
        ]

        struct = detect_dataset_structure(ds_path)
        if not struct.get("is_pre_split", False) and req.train_split is not None:
            cmd.extend(
                [
                    "--train-split",
                    str(req.train_split),
                    "--val-split",
                    str(req.val_split if req.val_split is not None else 0.1),
                    "--test-split",
                    str(req.test_split if req.test_split is not None else 0.1),
                ]
            )

    gpu_url = _get_gpu_worker_url(request)
    if gpu_url:
        dispatch_data_dir = (
            str(coco_root or yolo_root or ds_path) if is_detection_run else str(resolved_ds_path)
        )
        dispatch_arch = (
            (req.architecture if req.architecture.lower().startswith("yolo") else "yolov8n")
            if is_detection_run
            else req.architecture
        )
        cmd = [
            sys.executable,
            "scripts/dispatch_to_gpu.py",
            "--gpu-url",
            gpu_url,
            "--slug",
            slug,
            "--task",
            "detection" if is_detection_run else "classification",
            "--data-dir",
            dispatch_data_dir,
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
            dispatch_arch,
            "--target-client-tier",
            req.target_client_tier,
            "--tl-strategy",
            req.transfer_learning,
            "--loss",
            req.loss_function,
            "--aug",
            req.augmentation,
        ]
        if not is_detection_run and req.train_split is not None:
            cmd.extend(
                [
                    "--train-split",
                    str(req.train_split),
                    "--val-split",
                    str(req.val_split if req.val_split is not None else 0.1),
                    "--test-split",
                    str(req.test_split if req.test_split is not None else 0.1),
                ]
            )
        jm.logs.append(
            "[*] [AUTO-FL] Acelerador GPU dedicado detectado. "
            "Delegando entrenamiento a 'agrifed-gpu' con cola FIFO y fallback seguro."
        )

    async def on_training_success() -> None:
        task_type = "detection" if is_detection_run else "classification"
        trained_arch = (
            req.architecture
            if req.architecture != "auto"
            else ("yolov8n" if is_detection_run else "resnet18")
        )
        trained_classes = class_names
        metrics: dict[str, Any] = {}
        if out_meta.is_file():
            try:
                meta_json = json.loads(out_meta.read_text(encoding="utf-8"))
                trained_arch = meta_json.get("arch", meta_json.get("architecture", trained_arch))
                trained_classes = meta_json.get("classes", trained_classes)
                task_type = meta_json.get("task", task_type)
                metrics = meta_json.get("metrics") or {}
                if not metrics:
                    val_m = meta_json.get("hyperparameters", {}).get("val_metrics_last_epoch") or {}
                    if val_m:
                        raw_acc = val_m.get("accuracy")
                        metrics = {
                            "accuracy": (raw_acc * 100.0)
                            if (raw_acc is not None and raw_acc <= 1.0)
                            else raw_acc,
                            "loss": val_m.get("loss"),
                            "macro_f1": val_m.get("macro_f1"),
                        }
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Error parsing training metadata JSON for %s: %s", slug, exc)

        target_eval_rep = out_weights.parent / f"{slug}_evaluation_report.json"
        eval_rep_name = target_eval_rep.name if target_eval_rep.is_file() else None
        now_ts = time.strftime("%Y-%m-%d %H:%M:%S")

        v1_rec = ModelVersionRecord(
            version_id="v1",
            checkpoint=out_weights.name,
            created_at=now_ts,
            parent_version=None,
            origin="train_detection" if is_detection_run else "train_base",
            status="active",
            evaluation_report=eval_rep_name,
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
            task=task_type,
            num_classes=len(trained_classes),
            classes=trained_classes,
            updated_at=now_ts,
        )
        cm.upsert_entry(new_entry)

        append_model_history(
            models_dir=models_root,
            slug=slug,
            version="v1",
            checkpoint_name=out_weights.name,
            accuracy=metrics.get("accuracy"),
            macro_f1=metrics.get("macro_f1"),
            loss=metrics.get("loss"),
            metrics=metrics,
            metadata={"origin": "auto_fl", "dataset_path": str(ds_path), "epochs": req.epochs},
            timestamp=now_ts,
        )

        engine = _get_engine(request)
        if engine and engine.active_model_p is None:
            engine.active_model_p = out_weights
            engine.active_explicitly_cleared = False
            engine.resolve_and_reload_model(force=True)

        jm.logs.append(
            f"[*] [AUTO-FL] Modelo '{slug}' ({new_entry.title}) registrado en el catálogo. "
            f"Clases detectadas ({len(trained_classes)}): {', '.join(trained_classes)}. "
            f"Checkpoint activo fijado en '{out_weights.name}'."
        )

    job_info = await jm.enqueue_or_run(
        action=f"auto_fl_train:{slug}",
        cmd=cmd,
        on_success=on_training_success,
    )

    resp = dict(job_info)
    orig_job_status = resp.pop("status", None)
    return {
        "status": "accepted",
        "job_status": orig_job_status,
        **resp,
        "slug": slug,
        "title": req.title,
        "dataset_path": str(ds_path),
        "classes_detected": class_names,
        "target_weights": out_weights.name,
    }


@router.api_route("/api/master/recommend-arch", methods=["GET", "POST"])
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
        except (json.JSONDecodeError, ValueError) as exc:
            logger.debug("Failed parsing JSON body in recommend-arch: %s", exc)

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


@router.get("/api/master/models/{slug}/telemetry")
def get_model_telemetry(slug: str, request: Request) -> dict[str, Any]:
    """Retrieve confusion matrix, calibration thresholds, and canary telemetry for a model."""
    cm = _get_cm(request)
    jm = _get_jm(request)
    return run_model_telemetry(cm, jm, slug)


@router.post("/api/master/models/{slug}/evaluate")
async def evaluate_model_performance(
    slug: str,
    request: Request,
    val_dir: str | None = Query(None),
    batch_size: int = Query(32),
    calibrate_threshold: bool = Query(True),
) -> dict[str, Any]:
    """Trigger asynchronous performance evaluation of a model using scripts/evaluate.py."""
    cm = _get_cm(request)
    entry = cm.get_by_slug(slug, include_deleted=True)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Modelo '{slug}' no encontrado.")

    if not entry.active_checkpoint:
        raise HTTPException(
            status_code=400,
            detail=f"El modelo '{slug}' no tiene un checkpoint activo para evaluar.",
        )

    models_root = cm.models_dir if hasattr(cm, "models_dir") and cm.models_dir else Path("models")
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

    meta_candidates = [
        models_root / f"{slug}.meta.json",
        models_root / f"{chk_path.stem}.meta.json",
        models_root / "metadata.json",
    ]
    meta_arg: Path | None = None
    dataset_source_cand: str | None = None
    for cand in meta_candidates:
        if cand.is_file():
            meta_arg = cand
            try:
                meta_data = json.loads(cand.read_text(encoding="utf-8"))
                dataset_source_cand = meta_data.get("dataset_source")
            except (json.JSONDecodeError, OSError) as exc:
                logger.debug("Failed reading metadata candidate %s: %s", cand, exc)
            break

    is_det = (getattr(entry, "task", None) == "detection") or str(getattr(entry, "architecture", "") or "").lower().startswith("yolo")
    if is_det:
        cmd = [
            sys.executable,
            "scripts/backfill_detection_eval.py",
            "--slug",
            slug,
        ]
        jm = _get_jm(request)
        job_info = await jm.enqueue_or_run(
            action=f"evaluate_model:{slug}",
            cmd=cmd,
        )
        return {
            **job_info,
            "slug": slug,
            "checkpoint": entry.active_checkpoint,
            "command": cmd,
        }

    effective_val_dir = val_dir or dataset_source_cand
    resolved_val, _ = resolve_validation_directory_for_model(
        entry.classes, explicit_val_dir=effective_val_dir
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

    jm = _get_jm(request)

    async def on_eval_success() -> None:
        generated_rep = out_eval_dir / "evaluation_report.json"
        target_rep = models_root / f"{slug}_evaluation_report.json"
        if generated_rep.is_file():
            try:
                shutil.copyfile(generated_rep, target_rep)
                rep_data = json.loads(target_rep.read_text(encoding="utf-8"))
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
                jm.logs.append(
                    f"[*] [EVAL] Reporte guardado en {target_rep.name} para modelo '{slug}'."
                )
            except (OSError, json.JSONDecodeError) as exc:
                jm.logs.append(f"[!] Error actualizando catálogo tras evaluación: {exc}")

    job_info = await jm.enqueue_or_run(
        action=f"evaluate_model:{slug}",
        cmd=cmd,
        on_success=on_eval_success,
    )

    return {
        **job_info,
        "slug": slug,
        "checkpoint": entry.active_checkpoint,
        "validation_directory": str(resolved_val),
        "command": cmd,
    }


@router.get("/api/master/federated/updates")
def get_federated_updates(
    request: Request,
    target_model: str | None = Query(None),
    updates_dir: str = Query("models/client_updates"),
) -> dict[str, Any]:
    """List client checkpoints with security and lineage audit against target model."""
    cm = _get_cm(request)
    engine = _get_engine(request)
    active_model_p = engine.active_model_p if engine else None
    return run_federated_updates(cm, target_model, updates_dir, active_model_p)


@router.post("/api/master/federated/aggregate")
async def run_secure_federated_aggregation(
    req: FederatedAggregationRequest,
    request: Request,
) -> dict[str, Any]:
    """Trigger FedAvgM aggregation producing an isolated candidate checkpoint."""
    if not req.selected_update_paths:
        raise HTTPException(
            status_code=400,
            detail="Debe seleccionar al menos una actualización federada válida para agregar.",
        )

    cm = _get_cm(request)
    entry = cm.get_by_slug(req.target_model_slug, include_deleted=True)
    if not entry:
        raise HTTPException(
            status_code=404,
            detail=f"Modelo objetivo '{req.target_model_slug}' no encontrado.",
        )

    if not entry.active_checkpoint:
        raise HTTPException(
            status_code=400,
            detail=(
                f"El modelo objetivo '{req.target_model_slug}' no tiene "
                "checkpoint activo para agregar."
            ),
        )

    models_root = cm.models_dir if hasattr(cm, "models_dir") and cm.models_dir else Path("models")
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

    validated_paths: list[Path] = []
    for p_str in req.selected_update_paths:
        p = Path(p_str) if Path(p_str).is_absolute() else models_root / p_str.replace("models/", "")
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
    _get_fl_ops(request).record_operation(op_record)

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

    jm = _get_jm(request)
    job_info = await jm.enqueue_or_run("federated_aggregate", cmd)

    return {
        **job_info,
        "operation_id": op_id,
        "target_model": entry.slug,
        "base_checkpoint": entry.active_checkpoint,
        "candidate_checkpoint": candidate_name,
        "selected_updates_count": len(validated_paths),
        "val_dir_status": val_status,
        "val_dir": str(val_dir) if val_dir else None,
    }


@router.get("/api/master/federated/operations")
def list_federated_operations(
    request: Request,
    slug: str | None = Query(None),
) -> dict[str, Any]:
    """List historical federated operations and candidate states."""
    ops = _get_fl_ops(request).load_operations()
    if slug:
        ops = [op for op in ops if op.target_model_slug == slug]
    return {
        "operations": [op.model_dump() for op in reversed(ops)],
        "total": len(ops),
    }


@router.post("/api/master/models/{slug}/promote-candidate")
def promote_candidate_model(
    slug: str,
    request: Request,
    force: bool = Query(False),
) -> dict[str, Any]:
    """Atomically promote a candidate checkpoint to active production version."""
    cm = _get_cm(request)
    entry = cm.get_by_slug(slug, include_deleted=True)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Modelo '{slug}' no encontrado.")

    models_root = cm.models_dir if hasattr(cm, "models_dir") and cm.models_dir else Path("models")
    candidate_p = models_root / f"{slug}_candidate.pt"
    if not candidate_p.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"No se ha encontrado el archivo candidato '{candidate_p.name}' en disco.",
        )

    fl_ops = _get_fl_ops(request)
    latest_op = fl_ops.get_latest_candidate(slug)
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
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Failed reading evaluation report %s: %s", eval_p, exc)

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
        fl_ops.record_operation(latest_op)

    engine = _get_engine(request)
    if engine:
        engine.resolve_and_reload_model(force=True)

    updated_entry = cm.get_by_slug(slug, include_deleted=True)
    return {
        "success": True,
        "message": msg,
        "promoted_version": new_rec.model_dump() if new_rec else None,
        "active_checkpoint": updated_entry.active_checkpoint if updated_entry else None,
    }


@router.post("/api/master/models/{slug}/rollback")
def rollback_model_version(
    slug: str,
    request: Request,
    target_version_id: str | None = Query(None),
) -> dict[str, Any]:
    """Rollback active model to a previous version or snapshot."""
    cm = _get_cm(request)
    success, msg = cm.rollback_to_version(slug, target_version_id)
    if not success:
        raise HTTPException(status_code=400, detail=msg)

    engine = _get_engine(request)
    if engine:
        engine.resolve_and_reload_model(force=True)

    entry = cm.get_by_slug(slug, include_deleted=True)
    return {
        "success": True,
        "message": msg,
        "active_checkpoint": entry.active_checkpoint if entry else None,
    }


@router.post("/api/master/models/{slug}/gradcam")
async def generate_model_gradcam(
    slug: str,
    request: Request,
    file: UploadFile = File(...),  # noqa: B008
    target_class: int | None = Query(None),
) -> dict[str, Any]:
    """Generate real-time in-memory Grad-CAM visual explanation overlay for a model."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo subido debe ser una imagen válida.")

    content = await file.read(15 * 1024 * 1024 + 1)
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="La imagen excede el límite de 15MB.")

    cm = _get_cm(request)
    return run_gradcam(cm, slug, content, target_class=target_class)


@router.post("/api/master/models/{slug}/compare-ab")
async def compare_model_versions(
    slug: str,
    request: Request,
    file: UploadFile = File(...),  # noqa: B008
    version_a: str = Query(..., description="Version ID (e.g. v1) or checkpoint for A"),
    version_b: str = Query(..., description="Version ID (e.g. v2) or checkpoint for B"),
    ground_truth_class: str | None = Query(None, description="Optional ground truth class"),
    include_gradcam: bool = Query(True, description="Whether to compute Grad-CAM overlays"),
) -> dict[str, Any]:
    """A/B Comparator: Run live inference side-by-side across two checkpoints."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo subido debe ser una imagen válida.")

    content = await file.read(15 * 1024 * 1024 + 1)
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="La imagen excede el límite de 15MB.")

    cm = _get_cm(request)
    return run_compare_ab(
        cm,
        slug,
        content,
        version_a=version_a,
        version_b=version_b,
        ground_truth_class=ground_truth_class,
        include_gradcam=include_gradcam,
    )
