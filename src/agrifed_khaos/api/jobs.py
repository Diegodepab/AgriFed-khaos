"""Master Background Job Runner and Asynchronous Queue API Endpoints."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from agrifed_khaos.catalog import ModelCatalogManager
from agrifed_khaos.federated_ops import audit_client_update, resolve_validation_directory_for_model
from agrifed_khaos.hitl import HITLQueueManager, _clean_slug
from agrifed_khaos.jobs import JobManager

logger = logging.getLogger("agrifed.api.jobs")
router = APIRouter(tags=["Jobs & Queue"])


def _get_jm(request: Request) -> JobManager:
    jm = getattr(request.app.state, "job_manager", None)
    if not jm:
        raise HTTPException(status_code=500, detail="JobManager is not initialized on application.")
    return jm


def _get_cm(request: Request) -> ModelCatalogManager:
    return getattr(request.app.state, "catalog", ModelCatalogManager(catalog_path=Path("models/catalog.json"), models_dir=Path("models")))


@router.get("/api/master/status")
def get_master_job_status(request: Request) -> dict[str, Any]:
    """Get active job status, FIFO execution queue, and recent terminal logs."""
    return _get_jm(request).get_status()


@router.post("/api/master/jobs/clear")
def clear_master_job_logs(request: Request) -> dict[str, Any]:
    """Clear execution logs from job manager terminal."""
    jm = _get_jm(request)
    jm.logs.clear()
    return {"status": "cleared", "logs": []}


@router.post("/api/master/jobs/cancel")
async def cancel_master_job(
    request: Request,
    job_id: str | None = Query(None),
    action: str | None = Query(None),
) -> dict[str, Any]:
    """Cancel an active running job or remove a pending job from the FIFO queue."""
    jm = _get_jm(request)
    target = job_id or action
    if not target:
        if jm.is_running and jm.active_job_id:
            target = jm.active_job_id
        elif jm.action:
            target = jm.action
        else:
            raise HTTPException(
                status_code=400,
                detail="Debe indicar 'job_id' o 'action' para cancelar.",
            )
    res = await jm.cancel_job(target)
    if not res.get("cancelled"):
        raise HTTPException(status_code=404, detail=res.get("detail"))
    return res


@router.post("/api/master/run")
async def run_master_job(request: Request, action: str = Query(...)) -> dict[str, Any]:
    """Enqueue and execute an asynchronous administrative job."""
    jm = _get_jm(request)
    engine = getattr(request.app.state, "inference_engine", None)
    active_model_p = engine.active_model_p if engine else None
    classes = engine.classes if engine else []

    cmd: list[str] = [sys.executable]
    if action == "train_hitl":
        active_hitl = getattr(request.app.state, "hitl", HITLQueueManager())
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
            if cand.is_dir():
                v_images = cand / "images"
                if v_images.is_dir() and any(f.is_file() for f in v_images.iterdir()):
                    chosen_verified = cand
                    break
                if any(
                    f.is_file()
                    for class_dir in cand.iterdir()
                    if class_dir.is_dir() and class_dir.name not in ("images", "labels", "metadata")
                    for f in class_dir.iterdir()
                ):
                    chosen_verified = cand
                    break

        if not chosen_verified:
            raise HTTPException(
                status_code=400,
                detail="No verified images found in hitl_queue to train on.",
            )

        cm = _get_cm(request)
        is_detection = False
        v_images = chosen_verified / "images"
        if v_images.is_dir() and ((chosen_verified / "dataset.yaml").is_file() or (chosen_verified / "data.yaml").is_file()):
            is_detection = True

        resolved_slug = _clean_slug(chosen_verified.parent.name)
        entry = cm.get_by_slug(resolved_slug) if cm else None
        if entry and entry.task == "detection":
            is_detection = True

        if is_detection:
            base_weights = (
                str(entry.checkpoint_path)
                if (entry and entry.checkpoint_path and Path(entry.checkpoint_path).is_file())
                else f"models/{resolved_slug}_base.pt"
            )
            if not Path(base_weights).is_file():
                base_weights = "yolov8n.pt"

            cmd.extend(
                [
                    "scripts/train_detection.py",
                    "--data-dir",
                    str(chosen_verified),
                    "--output",
                    f"models/{resolved_slug}_base.pt",
                    "--metadata",
                    f"models/{resolved_slug}.meta.json",
                    "--arch",
                    base_weights,
                    "--epochs",
                    "5",
                    "--workers",
                    "2",
                ]
            )
        else:
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

        cm = _get_cm(request)
        target_slug = "patologias-foliares-olivo"
        entry = cm.get_by_slug(target_slug)

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

    job_info = await jm.enqueue_or_run(action, cmd)
    return {**job_info, "command": cmd}

