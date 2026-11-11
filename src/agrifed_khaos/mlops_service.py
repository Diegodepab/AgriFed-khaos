"""MLOps Services: Model Telemetry, Grad-CAM Visualization, Federated Lineage, and A/B Comparison."""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from fastapi import HTTPException
from PIL import Image
from torchvision import transforms

from agrifed_khaos.catalog import ModelCatalogManager, ModelVersionRecord
from agrifed_khaos.crypto import compute_sha256
from agrifed_khaos.federated_ops import audit_client_update, resolve_meta_path
from agrifed_khaos.models import (
    GradCAM,
    build_model,
    get_cam_target_layer,
    load_checkpoint_bundle,
)

logger = logging.getLogger("agrifed.mlops_service")


def get_model_telemetry(
    cm: ModelCatalogManager,
    job_manager: Any,
    slug: str,
) -> dict[str, Any]:
    """Retrieve confusion matrix, calibration thresholds, and canary telemetry for a model."""
    entry = cm.get_by_slug(slug, include_deleted=True)
    if not entry:
        for m in cm.load_catalog(include_deleted=True):
            if (m.active_checkpoint and m.active_checkpoint.startswith(slug)) or m.slug == slug:
                entry = m
                break
    if not entry:
        raise HTTPException(status_code=404, detail=f"Modelo '{slug}' no encontrado.")

    models_root = cm.models_dir if hasattr(cm, "models_dir") and cm.models_dir else Path("models")
    chk_path = (
        Path(entry.active_checkpoint)
        if entry.active_checkpoint and Path(entry.active_checkpoint).is_absolute()
        else (models_root / entry.active_checkpoint if entry.active_checkpoint else None)
    )
    chk_stem = Path(entry.active_checkpoint).stem if entry.active_checkpoint else slug
    is_detection = (getattr(entry, "task", None) == "detection") or str(getattr(entry, "architecture", "") or "").lower().startswith("yolo")

    # Load model-specific metadata (.meta.json) if present
    model_meta: dict[str, Any] = {}
    meta_candidates = [
        models_root / f"{slug}.meta.json",
        models_root / f"{chk_stem}.meta.json",
    ]
    for mc in meta_candidates:
        if mc and mc.is_file():
            try:
                model_meta = json.loads(mc.read_text(encoding="utf-8"))
                break
            except Exception as exc:
                logger.warning("Could not load model meta '%s': %s", mc, exc)

    bundle_meta: dict[str, Any] = {}
    if not is_detection and chk_path and chk_path.is_file():
        try:
            _, bundle_meta = load_checkpoint_bundle(chk_path, device="cpu")
        except Exception as exc:
            logger.warning("Could not load checkpoint bundle for telemetry '%s': %s", chk_path, exc)

    global_meta: dict[str, Any] = {}
    meta_p = models_root / "metadata.json"
    if meta_p.is_file():
        try:
            global_meta = json.loads(meta_p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load global metadata: %s", exc)

    resolved_versions = entry.get_resolved_versions()
    eval_report: dict[str, Any] | None = None
    eval_source: str | None = None
    evaluated_checkpoint: str | None = None

    for v in resolved_versions:
        if (
            entry.active_checkpoint
            and v.checkpoint == entry.active_checkpoint
            and v.evaluation_report
        ):
            cand = Path(v.evaluation_report)
            if not cand.is_absolute() and not cand.is_file():
                cand = models_root / v.evaluation_report
            if cand.is_file():
                try:
                    eval_report = json.loads(cand.read_text(encoding="utf-8"))
                    eval_source = str(cand)
                    evaluated_checkpoint = v.checkpoint
                    break
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning("Could not read evaluation report '%s': %s", cand, exc)

    if not eval_report:
        report_candidates = [
            models_root / f"{slug}_evaluation_report.json",
            models_root / f"{Path(entry.active_checkpoint).stem}_evaluation_report.json"
            if entry.active_checkpoint
            else None,
            models_root / "evaluation_report.json",
            Path(f"docs/images/{slug}/evaluation_report.json"),
            Path("docs/images/evaluation_report.json"),
        ]
        for rep in report_candidates:
            if rep and rep.is_file():
                try:
                    candidate_data = json.loads(rep.read_text(encoding="utf-8"))
                    raw_pc = candidate_data.get("per_class") or {}
                    if isinstance(raw_pc, list):
                        cand_classes = [c.get("class_name") for c in raw_pc if isinstance(c, dict) and c.get("class_name")]
                    elif isinstance(raw_pc, dict):
                        cand_classes = list(raw_pc.keys())
                    else:
                        cand_classes = []
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
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning("Could not read candidate evaluation report '%s': %s", rep, exc)

    classes = (
        model_meta.get("classes")
        or bundle_meta.get("classes")
        or global_meta.get("classes")
        or entry.classes
        or []
    )
    arch_name = (
        model_meta.get("architecture")
        or bundle_meta.get("architecture")
        or global_meta.get("architecture")
        or entry.architecture
        or ("yolov8s" if is_detection else "resnet18")
    )
    cal_thresholds = (
        model_meta.get("calibrated_thresholds")
        or bundle_meta.get("calibrated_thresholds")
        or global_meta.get("calibrated_thresholds")
        or entry.calibrated_thresholds
        or {}
    )
    temperature = float(
        model_meta.get("temperature")
        or bundle_meta.get("temperature")
        or global_meta.get("temperature")
        or entry.temperature
        or 1.0
    )

    confusion_matrix_raw = None
    metrics: dict[str, Any] = {
        "overall_accuracy": None,
        "cross_entropy_loss": None,
        "macro_avg": None,
        "per_class": None,
    }
    if is_detection:
        det_metrics = model_meta.get("metrics") or {}
        if not det_metrics and eval_report:
            det_metrics = eval_report.get("metrics") or {}
        if not det_metrics:
            for v in resolved_versions:
                if getattr(v, "metrics_summary", None):
                    det_metrics = v.metrics_summary
                    break
        metrics = {
            "mAP50": det_metrics.get("mAP50") or det_metrics.get("accuracy"),
            "mAP50_95": det_metrics.get("mAP50_95"),
            "precision": det_metrics.get("precision"),
            "recall": det_metrics.get("recall"),
            "accuracy": det_metrics.get("accuracy") or det_metrics.get("mAP50"),
            "mean_iou": det_metrics.get("mean_iou"),
            "matched_detections": det_metrics.get("matched_detections"),
            "overall_accuracy": det_metrics.get("mAP50") or det_metrics.get("accuracy"),
        }
        if not eval_source and (det_metrics or (models_root / f"{slug}_eval_preview.jpg").is_file()):
            eval_source = f"Métricas de validación YOLO ({entry.active_checkpoint or slug})"
            evaluated_checkpoint = entry.active_checkpoint
    elif eval_report:
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
    else:
        fb_metrics = (
            model_meta.get("metrics")
            or global_meta.get("metrics")
            or bundle_meta.get("metrics")
            or global_meta.get("hyperparameters", {}).get("val_metrics_last_epoch")
            or bundle_meta.get("hyperparameters", {}).get("val_metrics_last_epoch")
        )
        if fb_metrics:
            raw_acc = fb_metrics.get("overall_accuracy", fb_metrics.get("accuracy"))
            if raw_acc is not None and raw_acc <= 1.0:
                raw_acc = raw_acc * 100.0
            metrics["overall_accuracy"] = raw_acc
            metrics["cross_entropy_loss"] = fb_metrics.get(
                "cross_entropy_loss", fb_metrics.get("loss")
            )
            metrics["macro_f1"] = fb_metrics.get("macro_f1")
            metrics["macro_precision"] = fb_metrics.get(
                "macro_precision", fb_metrics.get("macro_f1")
            )
            metrics["macro_recall"] = fb_metrics.get(
                "macro_recall", fb_metrics.get("macro_f1")
            )
            eval_source = "Validación de Entrenamiento (Mejor Época)"
            evaluated_checkpoint = entry.active_checkpoint

        if not confusion_matrix_raw:
            confusion_matrix_raw = (
                global_meta.get("confusion_matrix")
                or bundle_meta.get("confusion_matrix")
            )

    eval_job_status = "idle"
    eval_job_pos = None
    if job_manager and job_manager.is_running and job_manager.action:
        if job_manager.action in (f"evaluate_model:{slug}", f"evaluate:{slug}"):
            eval_job_status = "in_progress"
        elif job_manager.action in (f"auto_fl_train:{slug}", f"train:{slug}"):
            eval_job_status = "training"

    if eval_job_status == "idle" and job_manager:
        for q_idx, item in enumerate(job_manager.queue):
            if item.action in (f"evaluate_model:{slug}", f"evaluate:{slug}"):
                eval_job_status = "queued"
                eval_job_pos = q_idx + 1
                break

    history_candidates = [
        models_root / f"{slug}_history.json",
        models_root / f"{chk_stem}_history.json",
    ]
    agg_history = []
    for hc in history_candidates:
        if hc and hc.is_file():
            try:
                agg_history = json.loads(hc.read_text(encoding="utf-8"))
                break
            except Exception as exc:
                logger.warning("Could not read history '%s': %s", hc, exc)

    if not agg_history:
        agg_history = global_meta.get("aggregation_history", [])

    if not agg_history and resolved_versions:
        for v in resolved_versions:
            m_sum = getattr(v, "metrics_summary", {}) or {}
            agg_history.append({
                "version": v.version_id,
                "timestamp": v.created_at,
                "checkpoint_name": v.checkpoint,
                "accuracy": m_sum.get("accuracy") or m_sum.get("mAP50") or (m_sum.get("macro_f1") * 100 if m_sum.get("macro_f1") else None),
                "metrics": m_sum,
            })

    matrix_classes = classes
    if eval_report and eval_report.get("per_class"):
        raw_pc = eval_report["per_class"]
        if isinstance(raw_pc, list):
            matrix_classes = [c.get("class_name") for c in raw_pc if isinstance(c, dict) and c.get("class_name")]
        elif isinstance(raw_pc, dict):
            matrix_classes = list(raw_pc.keys())

    eval_preview_exists = (
        (models_root / f"{slug}_eval_preview.jpg").is_file()
        or (models_root / f"{chk_stem}_eval_preview.jpg").is_file()
    )

    return {
        "slug": entry.slug,
        "title": entry.title,
        "task": "detection" if is_detection else getattr(entry, "task", "classification"),
        "architecture": arch_name,
        "checkpoint": entry.active_checkpoint,
        "size_bytes": chk_path.stat().st_size if (chk_path and chk_path.is_file()) else 0,
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
        "has_eval_report": (eval_report is not None)
        or (metrics.get("overall_accuracy") is not None)
        or (metrics.get("mAP50") is not None)
        or eval_preview_exists,
        "eval_preview_available": eval_preview_exists,
        "eval_source": eval_source,
        "evaluated_checkpoint": evaluated_checkpoint,
        "evaluation_status": eval_job_status,
        "queue_position": eval_job_pos,
    }


def get_federated_updates(
    cm: ModelCatalogManager,
    target_model: str | None = None,
    updates_dir: str = "models/client_updates",
    active_model_p: Path | None = None,
) -> dict[str, Any]:
    """List client checkpoints with security and lineage audit against target model."""
    models_root = cm.models_dir if hasattr(cm, "models_dir") and cm.models_dir else Path("models")
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
        target_arch = target_entry.architecture or "resnet18"

    if target_base_chk_path and target_base_chk_path.is_file():
        try:
            target_base_sha = compute_sha256(target_base_chk_path)
        except (OSError, ValueError) as exc:
            logger.warning("Could not compute sha256 for base: %s", exc)

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
                except (OSError, ValueError) as exc:
                    logger.warning("Could not compute sha256: %s", exc)

    target_model_shas: set[str] = set()
    if target_base_sha:
        target_model_shas.add(target_base_sha)
    if target_entry:
        for v in target_entry.get_resolved_versions():
            if not v.checkpoint:
                continue
            chk_v = (
                Path(v.checkpoint)
                if Path(v.checkpoint).is_absolute()
                else models_root / v.checkpoint
            )
            if chk_v.is_file():
                try:
                    target_model_shas.add(compute_sha256(chk_v))
                except (OSError, ValueError) as exc:
                    logger.warning("Could not compute sha256: %s", exc)

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

        if p.parent.name in other_slugs:
            continue

        if any(
            p.name.startswith(f"{other}_") or p.name.startswith(f"{other}-")
            for other in other_slugs
        ):
            continue

        if target_slug and (
            p.name.startswith(f"{target_slug}_") or p.name.startswith(f"{target_slug}-")
        ):
            candidates.append(p)
            continue

        meta_path = resolve_meta_path(p)
        meta_data: dict[str, Any] = {}
        if meta_path.is_file():
            try:
                meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.debug("Failed reading meta path %s: %s", meta_path, exc)

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
                continue

        meta_classes = meta_data.get("classes")
        if meta_classes and target_entry and target_entry.classes:
            if set(meta_classes) != set(target_entry.classes):
                continue

        meta_base_sha = meta_data.get("base_model_sha256")
        if meta_base_sha:
            if meta_base_sha in target_model_shas:
                candidates.append(p)
                continue
            if meta_base_sha in other_model_shas:
                continue

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
        "target_architecture": target_arch,
        "updates": found_updates,
        "total": len(found_updates),
        "eligible_count": eligible_count,
        "rejected_count": rejected_count,
    }


def generate_model_gradcam(
    cm: ModelCatalogManager,
    slug: str,
    content: bytes,
    target_class: int | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Generate real-time in-memory Grad-CAM visual explanation overlay for a model."""
    import matplotlib
    import numpy as np

    entry = cm.get_by_slug(slug, include_deleted=True)
    if not entry:
        for m in cm.load_catalog(include_deleted=True):
            if (m.active_checkpoint and m.active_checkpoint.startswith(slug)) or m.slug == slug:
                entry = m
                break
    if not entry:
        raise HTTPException(status_code=404, detail=f"Modelo '{slug}' no encontrado.")

    if not entry.active_checkpoint:
        raise HTTPException(
            status_code=404, detail=f"El modelo '{slug}' no tiene checkpoint activo en disco."
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
            detail=f"El archivo de pesos '{entry.active_checkpoint}' no existe en disco.",
        )

    try:
        pil_image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400, detail="Formato de imagen corrupto o no reconocido."
        ) from e

    dev = device or torch.device("cpu")
    is_yolo = (getattr(entry, "task", None) == "detection") or str(getattr(entry, "architecture", "") or "").lower().startswith("yolo")

    if is_yolo:
        from ultralytics import YOLO
        y_model = YOLO(str(chk_path))
        preds = y_model.predict(source=pil_image, conf=0.25, imgsz=640, verbose=False)
        annotated_bgr = preds[0].plot() if preds and len(preds) > 0 else np.array(pil_image)
        annotated_rgb = annotated_bgr[..., ::-1] if (preds and len(preds) > 0) else annotated_bgr
        blended = Image.fromarray(annotated_rgb)
        buf = io.BytesIO()
        blended.save(buf, format="JPEG", quality=90)
        b64_str = base64.b64encode(buf.getvalue()).decode("ascii")

        boxes = getattr(preds[0], "boxes", None) if preds else None
        top_cls_name = "Sin detecciones"
        conf = 0.0
        if boxes and len(boxes) > 0:
            top_box = boxes[0]
            cls_id = int(top_box.cls[0].item())
            conf = float(top_box.conf[0].item())
            names_dict = getattr(y_model, "names", {})
            top_cls_name = names_dict.get(cls_id, f"Clase {cls_id}")

        return {
            "slug": entry.slug,
            "architecture": entry.architecture or "yolo",
            "checkpoint": entry.active_checkpoint,
            "target_class_index": 0,
            "target_class_name": top_cls_name,
            "predicted_class_index": 0,
            "predicted_class_name": top_cls_name,
            "confidence": round(conf, 4),
            "cam_overlay_base64": f"data:image/jpeg;base64,{b64_str}",
            "overlay_base64": f"data:image/jpeg;base64,{b64_str}",
        }

    state_dict, b_meta = load_checkpoint_bundle(chk_path, device=dev)
    arch = b_meta.get("architecture") or entry.architecture
    classes = b_meta.get("classes") or entry.classes or [f"Clase {i}" for i in range(entry.num_classes or 2)]
    temperature = float(b_meta.get("temperature", 1.0))

    model = build_model(arch, num_classes=len(classes), pretrained=False).to(dev)
    if state_dict:
        model.load_state_dict(state_dict, strict=False)
    model.eval()

    raw_dim = b_meta.get("img_size", [224, 224])
    img_dim = (raw_dim, raw_dim) if isinstance(raw_dim, int) else (int(raw_dim[0]), int(raw_dim[1]))
    rs_dim = (int(round(img_dim[0] * 256 / 224)), int(round(img_dim[1] * 256 / 224)))

    pre_crop = transforms.Compose([transforms.Resize(rs_dim), transforms.CenterCrop(img_dim)])
    cropped_pil = pre_crop(pil_image)
    to_tensor_norm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    x_tensor = to_tensor_norm(cropped_pil).unsqueeze(0).to(dev)

    with torch.no_grad():
        logits = model(x_tensor)
        probs = F.softmax(logits / max(temperature, 1e-3), dim=1)[0]
        top_idx = int(probs.argmax().item())
        confidence = float(probs[top_idx].item())

    cls_idx = target_class if (target_class is not None and 0 <= target_class < len(classes)) else top_idx

    target_layer = get_cam_target_layer(model, architecture=arch)
    cam_gen = GradCAM(model, target_layer, temperature=temperature)
    try:
        cam_map, _, _ = cam_gen.generate(x_tensor, class_idx=cls_idx)
    finally:
        cam_gen.remove()

    cmap = matplotlib.colormaps.get_cmap("jet")
    cam_uint8 = (cmap(cam_map)[:, :, :3] * 255).astype(np.uint8)
    cam_pil = Image.fromarray(cam_uint8).resize(cropped_pil.size, Image.BILINEAR)
    blended = Image.blend(cropped_pil, cam_pil, alpha=0.5)

    buf = io.BytesIO()
    blended.save(buf, format="JPEG", quality=90)
    b64_str = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "slug": entry.slug,
        "architecture": arch,
        "checkpoint": entry.active_checkpoint,
        "target_class_index": cls_idx,
        "target_class_name": classes[cls_idx] if cls_idx < len(classes) else f"Clase {cls_idx}",
        "predicted_class_index": top_idx,
        "predicted_class_name": classes[top_idx] if top_idx < len(classes) else f"Clase {top_idx}",
        "confidence": round(confidence, 4),
        "cam_overlay_base64": f"data:image/jpeg;base64,{b64_str}",
        "overlay_base64": f"data:image/jpeg;base64,{b64_str}",
    }


def compare_model_versions(
    cm: ModelCatalogManager,
    slug: str,
    content: bytes,
    version_a: str,
    version_b: str,
    ground_truth_class: str | None = None,
    include_gradcam: bool = True,
) -> dict[str, Any]:
    """A/B Comparator: Run live inference side-by-side across two checkpoints."""
    import matplotlib
    import numpy as np

    entry = cm.get_by_slug(slug, include_deleted=True)
    if not entry:
        for m in cm.load_catalog(include_deleted=True):
            if (m.active_checkpoint and m.active_checkpoint.startswith(slug)) or m.slug == slug:
                entry = m
                break
    if not entry:
        raise HTTPException(status_code=404, detail=f"Modelo '{slug}' no encontrado.")

    models_root = cm.models_dir if hasattr(cm, "models_dir") and cm.models_dir else Path("models")

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
        is_yolo = (getattr(entry, "task", None) == "detection") or str(getattr(entry, "architecture", "") or "").lower().startswith("yolo")
        if is_yolo:
            from ultralytics import YOLO
            try:
                y_model = YOLO(str(chk_path))
                preds = y_model.predict(source=pil_image, conf=0.25, imgsz=640, verbose=False)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error cargando modelo YOLO '{chk_path.name}' de {v_label}: {e}",
                ) from e

            annotated_bgr = preds[0].plot() if preds and len(preds) > 0 else np.array(pil_image)
            annotated_rgb = annotated_bgr[..., ::-1] if (preds and len(preds) > 0) else annotated_bgr
            blended = Image.fromarray(annotated_rgb)
            buf = io.BytesIO()
            blended.save(buf, format="JPEG", quality=90)
            b64_str = base64.b64encode(buf.getvalue()).decode("ascii")

            boxes = getattr(preds[0], "boxes", None) if preds else None
            pred_name = "Sin detecciones"
            confidence = 0.0
            all_probs: dict[str, float] = {}
            if boxes and len(boxes) > 0:
                top_box = boxes[0]
                cls_id = int(top_box.cls[0].item())
                confidence = float(top_box.conf[0].item())
                names_dict = getattr(y_model, "names", {})
                pred_name = names_dict.get(cls_id, f"Clase {cls_id}")
                for b in boxes:
                    c_id = int(b.cls[0].item())
                    c_name = names_dict.get(c_id, f"Clase {c_id}")
                    all_probs[c_name] = max(all_probs.get(c_name, 0.0), round(float(b.conf[0].item()), 4))

            return {
                "version_id": v_rec.version_id if v_rec else v_label,
                "checkpoint": chk_path.name,
                "predicted_class": pred_name,
                "confidence": round(confidence, 4),
                "is_correct": (ground_truth_class is not None and pred_name.lower() == ground_truth_class.lower()) if ground_truth_class else None,
                "probabilities": all_probs,
                "cam_overlay_base64": f"data:image/jpeg;base64,{b64_str}" if include_gradcam else None,
                "cam_supported": True,
                "cam_error": None,
                "metrics_summary": v_rec.metrics_summary if v_rec else {},
            }

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
        img_dim = (raw_dim, raw_dim) if isinstance(raw_dim, int) else (int(raw_dim[0]), int(raw_dim[1]))

        v_model = build_model(v_arch, num_classes=len(v_classes), pretrained=False)
        if state_dict:
            v_model.load_state_dict(state_dict, strict=False)
        v_model.eval()

        rs_dim = (int(round(img_dim[0] * 256 / 224)), int(round(img_dim[1] * 256 / 224)))
        pre_crop = transforms.Compose([transforms.Resize(rs_dim), transforms.CenterCrop(img_dim)])
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
            pred_name = v_classes[pred_idx] if pred_idx < len(v_classes) else f"Clase {pred_idx}"
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
                cam_overlay_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
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
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read AB report from '%s': %s", report_path, exc)

        matches_gt: bool | None = None
        if ground_truth_class:
            matches_gt = pred_name.lower() == ground_truth_class.strip().lower()

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
        summary_parts.append(f"Ambas versiones coinciden en predecir '{res_a['predicted_class_name']}'.")
    else:
        summary_parts.append(
            f"Discrepancia diagnóstica: Versión A predice '{res_a['predicted_class_name']}' "
            f"(conf: {res_a['confidence'] * 100:.1f}%) y Versión B predice "
            f"'{res_b['predicted_class_name']}' (conf: {res_b['confidence'] * 100:.1f}%)."
        )

    if ground_truth_class:
        gt_text = ground_truth_class.strip()
        if res_a["matches_ground_truth"] and res_b["matches_ground_truth"]:
            summary_parts.append(f"Ambas versiones aciertan con la etiqueta de referencia ('{gt_text}').")
        elif res_a["matches_ground_truth"] and not res_b["matches_ground_truth"]:
            summary_parts.append(f"Versión A acierta con la etiqueta de referencia ('{gt_text}'), mientras Versión B difiere.")
        elif not res_a["matches_ground_truth"] and res_b["matches_ground_truth"]:
            summary_parts.append(f"Versión B acierta con la etiqueta de referencia ('{gt_text}'), mientras Versión A difiere.")
        else:
            summary_parts.append(f"Ninguna de las dos versiones coincide con la etiqueta de referencia ('{gt_text}').")

    summary_parts.append("Nota MLOps: Una mayor confianza softmax no implica necesariamente mayor exactitud empírica.")

    return {
        "slug": entry.slug,
        "title": entry.title,
        "ground_truth_class": ground_truth_class,
        "agreement": is_agreement,
        "summary": " ".join(summary_parts),
        "version_a": res_a,
        "version_b": res_b,
    }

