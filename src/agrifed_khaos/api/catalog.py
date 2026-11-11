"""Model Catalog, Lifecycle, Quarantine & Model Management API Endpoints."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from agrifed_khaos.catalog import LogicalModelEntry, ModelCatalogManager, get_model_history, slugify

logger = logging.getLogger("agrifed.api.catalog")
router = APIRouter(tags=["Catalog & Models"])


def _get_cm(request: Request) -> ModelCatalogManager:
    if hasattr(request.app.state, "catalog"):
        return request.app.state.catalog
    return ModelCatalogManager(catalog_path=Path("models/catalog.json"), models_dir=Path("models"))


def _extract_dataset_info(slug: str, m_dict: dict[str, Any], models_dir: Path = Path("models")) -> dict[str, Any]:
    """Extract dataset, training, and evaluation metadata from sidecar JSON files."""
    meta: dict[str, Any] = {}
    meta_path = models_dir / f"{slug}.meta.json"
    if not meta_path.is_file() and m_dict.get("active_checkpoint"):
        base_cp = str(m_dict["active_checkpoint"]).replace("_base.pt", "").replace(".pt", "")
        cand = models_dir / f"{base_cp}.meta.json"
        if cand.is_file():
            meta_path = cand

    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            meta = {}

    eval_report: dict[str, Any] = {}
    eval_path = models_dir / f"{slug}_evaluation_report.json"
    if eval_path.is_file():
        try:
            eval_report = json.loads(eval_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            eval_report = {}

    hp = meta.get("hyperparameters") or {}
    val_last = hp.get("val_metrics_last_epoch") or {}
    early_stop = hp.get("early_stopping") or {}
    meta_metrics = meta.get("metrics") or {}

    # 1. Image resolution
    raw_img_size = meta.get("img_size", 224)
    if isinstance(raw_img_size, (list, tuple)) and len(raw_img_size) >= 2:
        img_size_num = raw_img_size[0]
        img_size_str = f"{raw_img_size[0]} × {raw_img_size[1]} px"
    elif isinstance(raw_img_size, int):
        img_size_num = raw_img_size
        img_size_str = f"{raw_img_size} × {raw_img_size} px"
    elif isinstance(raw_img_size, str) and "×" in raw_img_size:
        img_size_str = raw_img_size
        try:
            img_size_num = int(raw_img_size.split("×")[0].strip())
        except (ValueError, TypeError):
            img_size_num = 224
    else:
        img_size_num = 224
        img_size_str = "224 × 224 px"

    # 2. Clean dataset source name
    ds_source = str(meta.get("dataset_source") or eval_report.get("dataset_evaluated") or "")
    dataset_name = ""
    if ds_source:
        parts = [
            p for p in Path(ds_source).parts
            if p.lower() not in ("data", "uploads", "dataset", "datasets", "train", "val", "test", "/")
            and not p.lower().startswith("upload_")
        ]
        if parts:
            dataset_name = parts[-1]

    # 3. Total dataset images
    total_images: int | None = None
    desc = str(m_dict.get("description") or "")
    m_desc = re.search(r"\b(\d{2,7})\s+(?:imágenes|capturas|muestras|ejemplares)", desc, re.IGNORECASE)
    if m_desc:
        total_images = int(m_desc.group(1))
    if not total_images and eval_report.get("total_samples"):
        total_images = int(eval_report["total_samples"])
    if not total_images and hp.get("samples_trained"):
        total_images = int(hp["samples_trained"])

    # 4. Accuracy & Macro F1 (normalized to 0-100 percentage)
    acc_pct: float | None = None
    if eval_report.get("overall_accuracy") is not None:
        acc_val = float(eval_report["overall_accuracy"])
        acc_pct = round(acc_val if acc_val > 1.0 else acc_val * 100.0, 2)
    elif meta_metrics.get("overall_accuracy") is not None:
        acc_val = float(meta_metrics["overall_accuracy"])
        acc_pct = round(acc_val if acc_val > 1.0 else acc_val * 100.0, 2)
    elif val_last.get("accuracy") is not None:
        acc_val = float(val_last["accuracy"])
        acc_pct = round(acc_val if acc_val > 1.0 else acc_val * 100.0, 2)
    else:
        vh = m_dict.get("version_history") or []
        if vh and isinstance(vh[-1], dict):
            ms = vh[-1].get("metrics_summary") or {}
            if ms.get("accuracy") is not None:
                acc_val = float(ms["accuracy"])
                acc_pct = round(acc_val if acc_val > 1.0 else acc_val * 100.0, 2)

    f1_pct: float | None = None
    macro_avg = eval_report.get("macro_avg") or {}
    if macro_avg.get("f1_score") is not None:
        f1_val = float(macro_avg["f1_score"])
        f1_pct = round(f1_val if f1_val > 1.0 else f1_val * 100.0, 2)
    elif meta_metrics.get("macro_f1") is not None:
        f1_val = float(meta_metrics["macro_f1"])
        f1_pct = round(f1_val if f1_val > 1.0 else f1_val * 100.0, 2)
    elif early_stop.get("monitor") == "val_macro_f1" and early_stop.get("best_metric") is not None:
        f1_val = float(early_stop["best_metric"])
        f1_pct = round(f1_val if f1_val > 1.0 else f1_val * 100.0, 2)
    elif val_last.get("macro_f1") is not None:
        f1_val = float(val_last["macro_f1"])
        f1_pct = round(f1_val if f1_val > 1.0 else f1_val * 100.0, 2)

    loss_val: float | None = None
    if eval_report.get("cross_entropy_loss") is not None:
        loss_val = round(float(eval_report["cross_entropy_loss"]), 4)
    elif val_last.get("loss") is not None:
        loss_val = round(float(val_last["loss"]), 4)

    return {
        "dataset_name": dataset_name or m_dict.get("title") or slug,
        "total_images": total_images,
        "eval_samples": eval_report.get("total_samples"),
        "samples_trained": hp.get("samples_trained"),
        "img_size": img_size_num,
        "img_size_str": img_size_str,
        "accuracy_pct": acc_pct,
        "macro_f1_pct": f1_pct,
        "loss": loss_val,
        "epochs_run": hp.get("epochs_run") or hp.get("epochs"),
        "best_epoch": early_stop.get("best_epoch"),
        "batch_size": hp.get("batch_size"),
        "learning_rate": hp.get("learning_rate"),
        "loss_function": hp.get("loss_function"),
        "augmentation": hp.get("augmentation"),
        "transfer_learning": hp.get("transfer_learning"),
        "temperature": meta.get("temperature") or m_dict.get("temperature"),
        "created_at": meta.get("created_at") or m_dict.get("updated_at"),
        "sha256": meta.get("sha256"),
        "per_class_metrics": eval_report.get("per_class") or {},
        "eval_split": eval_report.get("eval_split") or meta.get("eval_split") or "val",
        "detection_metrics": eval_report.get("metrics") or meta.get("metrics") or {},
        "eval_preview_available": (models_dir / f"{slug}_eval_preview.jpg").is_file() or eval_report.get("eval_preview_available", False),
        "eval_report": eval_report,
    }


@router.get("/api/models")
def list_models(
    request: Request,
    q: str | None = None,
    domain: str | None = None,
    page: int = 1,
    page_size: int = 12,
) -> dict[str, Any]:
    """Search, filter, and paginate logical models in the Data Space catalog."""
    engine = getattr(request.app.state, "inference_engine", None)
    if engine:
        engine.resolve_and_reload_model()
        classes = engine.classes
        arch = engine.arch
        num_classes = engine.num_classes
        calibrated_thresholds = engine.calibrated_thresholds
        active_model_p = engine.active_model_p
    else:
        classes = []
        arch = "resnet18"
        num_classes = 9
        calibrated_thresholds = None
        active_model_p = None

    cm = _get_cm(request)
    cm.sync_with_filesystem(default_classes=classes)
    catalog_res = cm.search_and_paginate(q=q, domain=domain, page=page, page_size=page_size)

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
        item["dataset_info"] = _extract_dataset_info(item.get("slug", ""), item, cm.models_dir)
        enriched_items.append(item)

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


@router.get("/api/models/{slug}")
def get_model_detail(request: Request, slug: str) -> dict[str, Any]:
    """Fetch full details for a single logical model by slug or filename."""
    engine = getattr(request.app.state, "inference_engine", None)
    if engine:
        engine.resolve_and_reload_model()
        classes = engine.classes
        arch = engine.arch
        num_classes = engine.num_classes
        active_model_p = engine.active_model_p
    else:
        classes = []
        arch = "resnet18"
        num_classes = 9
        active_model_p = None

    cm = _get_cm(request)
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

    model_dict = entry.model_dump()
    dataset_info = _extract_dataset_info(entry.slug, model_dict, cm.models_dir)
    model_dict["dataset_info"] = dataset_info

    return {
        "model": model_dict,
        "dataset_info": dataset_info,
        "size_bytes": sz,
        "is_active": is_active,
        "has_calibration": bool(entry.calibrated_thresholds),
        "download_url": (
            f"/download-model?model_slug={entry.slug}" if entry.active_checkpoint else None
        ),
    }


@router.get("/api/models/{slug}/thumbnail")
def get_model_thumbnail(request: Request, slug: str) -> FileResponse:
    """Serve representative thumbnail image for a logical model if available."""
    cm = _get_cm(request)
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
            Path("models") / f"{slug}_eval_preview.jpg",
            Path("models") / f"{slug}_eval_preview.png",
        ]
    )
    if entry.active_checkpoint:
        base_name = entry.active_checkpoint.replace("_base.pt", "").replace(".pt", "")
        candidates.append(Path("models") / f"{base_name}_eval_preview.jpg")
        candidates.append(Path("models") / f"{base_name}_eval_preview.png")

    allowed_roots = [
        Path("models").resolve(),
        Path("data").resolve(),
        Path("docs").resolve(),
        Path("src").resolve(),
    ]
    for cand in candidates:
        if cand.is_file():
            try:
                resolved_cand = cand.resolve()
                if not any(resolved_cand.is_relative_to(r) for r in allowed_roots):
                    continue
            except (OSError, ValueError):
                continue
            suffix = cand.suffix.lower()
            media_type = (
                "image/png"
                if suffix == ".png"
                else ("image/webp" if suffix == ".webp" else "image/jpeg")
            )
            return FileResponse(str(cand), media_type=media_type)

    raise HTTPException(status_code=404, detail=f"No thumbnail image available for model '{slug}'")


@router.get("/api/models/{slug}/eval-preview")
def get_model_eval_preview(request: Request, slug: str) -> FileResponse:
    """Serve detection evaluation prediction mosaic (val_batch0_pred.jpg) if available."""
    cm = _get_cm(request)
    entry = cm.get_by_slug(slug, include_deleted=False)
    models_dir = Path("models").resolve()
    candidates = [
        models_dir / f"{slug}_eval_preview.jpg",
        models_dir / f"{slug}_eval_preview.jpeg",
        models_dir / f"{slug}_eval_preview.png",
    ]
    if entry and entry.active_checkpoint:
        base_name = entry.active_checkpoint.replace("_base.pt", "").replace(".pt", "")
        candidates.append(models_dir / f"{base_name}_eval_preview.jpg")
        candidates.append(models_dir / f"{base_name}_eval_preview.png")

    for cand in candidates:
        if cand.is_file():
            try:
                resolved = cand.resolve()
                if resolved.is_relative_to(models_dir):
                    media_type = "image/png" if cand.suffix.lower() == ".png" else "image/jpeg"
                    return FileResponse(str(cand), media_type=media_type)
            except (OSError, ValueError):
                continue

    raise HTTPException(status_code=404, detail=f"No evaluation preview available for model '{slug}'")


@router.get("/api/models/{slug}/sample-raw")
def get_model_sample_raw(request: Request, slug: str) -> FileResponse:
    """Serve sample raw photograph (unannotated) for visual comparison."""
    cm = _get_cm(request)
    entry = cm.get_by_slug(slug, include_deleted=False)
    models_dir = Path("models").resolve()
    candidates = [
        models_dir / f"{slug}_sample_raw.jpg",
        models_dir / f"{slug}_sample_raw.jpeg",
        models_dir / f"{slug}_sample_raw.png",
    ]
    if entry and entry.active_checkpoint:
        base_name = entry.active_checkpoint.replace("_base.pt", "").replace(".pt", "")
        candidates.append(models_dir / f"{base_name}_sample_raw.jpg")
        candidates.append(models_dir / f"{base_name}_sample_raw.png")

    for cand in candidates:
        if cand.is_file():
            try:
                resolved = cand.resolve()
                if resolved.is_relative_to(models_dir):
                    media_type = "image/png" if cand.suffix.lower() == ".png" else "image/jpeg"
                    return FileResponse(str(cand), media_type=media_type)
            except (OSError, ValueError):
                continue

    # Fallback to thumbnail if raw sample is not explicitly available
    if entry and entry.thumbnail_url:
        cand_thumb = Path(entry.thumbnail_url.lstrip("/"))
        if cand_thumb.is_file():
            suffix = cand_thumb.suffix.lower()
            media_type = "image/png" if suffix == ".png" else ("image/webp" if suffix == ".webp" else "image/jpeg")
            return FileResponse(str(cand_thumb), media_type=media_type)

    raise HTTPException(status_code=404, detail=f"No raw sample image available for model '{slug}'")


@router.get("/api/models/{slug}/sample-annotated")
def get_model_sample_annotated(request: Request, slug: str) -> FileResponse:
    """Serve sample annotated photograph (with bounding boxes) for visual comparison."""
    cm = _get_cm(request)
    entry = cm.get_by_slug(slug, include_deleted=False)
    models_dir = Path("models").resolve()
    candidates = [
        models_dir / f"{slug}_sample_annotated.jpg",
        models_dir / f"{slug}_sample_annotated.jpeg",
        models_dir / f"{slug}_sample_annotated.png",
    ]
    if entry and entry.active_checkpoint:
        base_name = entry.active_checkpoint.replace("_base.pt", "").replace(".pt", "")
        candidates.append(models_dir / f"{base_name}_sample_annotated.jpg")
        candidates.append(models_dir / f"{base_name}_sample_annotated.png")

    for cand in candidates:
        if cand.is_file():
            try:
                resolved = cand.resolve()
                if resolved.is_relative_to(models_dir):
                    media_type = "image/png" if cand.suffix.lower() == ".png" else "image/jpeg"
                    return FileResponse(str(cand), media_type=media_type)
            except (OSError, ValueError):
                continue

    # Fallback to eval preview mosaic if single annotated sample is not present
    cand_prev = models_dir / f"{slug}_eval_preview.jpg"
    if cand_prev.is_file():
        return FileResponse(str(cand_prev), media_type="image/jpeg")

    raise HTTPException(status_code=404, detail=f"No annotated sample image available for model '{slug}'")


@router.post("/api/models/save")
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
        _get_cm(request).upsert_entry(entry)
        return {"status": "saved", "model": entry.model_dump()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to save model card: {e}") from e


@router.post("/api/models/activate")
def activate_model(request: Request, model_name: str = Query(...)) -> dict[str, Any]:
    """Set the default active global model for the server."""
    safe_name = Path(model_name).name
    engine = getattr(request.app.state, "inference_engine", None)
    resolved_models_dir = Path("models")

    candidate = Path("models") / safe_name
    if not candidate.is_file():
        if (resolved_models_dir / safe_name).is_file():
            candidate = resolved_models_dir / safe_name
        elif engine and Path(engine.model_path).name == safe_name and Path(engine.model_path).is_file():
            candidate = Path(engine.model_path)
        elif Path(model_name).is_file():
            candidate = Path(model_name)
        else:
            raise HTTPException(status_code=404, detail=f"Model '{safe_name}' not found")

    if engine:
        engine.model_path = str(candidate)
        engine.resolve_and_reload_model(force=True)
        arch = engine.arch
        num_classes = engine.num_classes
    else:
        arch = "resnet18"
        num_classes = 9

    return {
        "status": "activated",
        "active_model": safe_name,
        "architecture": arch,
        "num_classes": num_classes,
    }


@router.get("/api/master/models")
@router.get("/api/master/checkpoints")
def get_master_models(request: Request) -> dict[str, Any]:
    """List valid inference models and physical checkpoints in models/ directory."""
    res = list_models(request)
    models_list = res.get("models", [])
    checkpoints: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for m in models_list:
        cp_name = m.get("active_checkpoint") or m.get("name")
        if cp_name and cp_name not in seen_names:
            seen_names.add(cp_name)
            checkpoints.append(
                {
                    "name": cp_name,
                    "size_mb": round(m.get("size_bytes", 0) / (1024 * 1024), 2),
                    "size_bytes": m.get("size_bytes", 0),
                    "architecture": m.get("architecture", "resnet18"),
                    "classes": m.get("classes", []),
                    "is_active": m.get("is_active", False),
                }
            )

    engine = getattr(request.app.state, "inference_engine", None)
    active_model_p = engine.active_model_p if engine else None
    arch = engine.arch if engine else "resnet18"
    classes = engine.classes if engine else []

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
                checkpoints.append(
                    {
                        "name": name,
                        "size_mb": round(sz / (1024 * 1024), 2),
                        "size_bytes": sz,
                        "architecture": arch,
                        "classes": classes,
                        "is_active": active_model_p is not None and p.resolve() == active_model_p.resolve(),
                    }
                )

    res["checkpoints"] = checkpoints
    return res


@router.post("/api/master/set-active")
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
        except (json.JSONDecodeError, ValueError) as exc:
            logger.debug("Failed parsing JSON body in set_active_model: %s", exc)
    if not target:
        raise HTTPException(status_code=400, detail="Missing required parameter 'model_name'")
    return activate_model(request, model_name=target)


@router.get("/api/master/trash")
def get_trash_models(request: Request) -> dict[str, Any]:
    """List models currently held in the soft-delete trash quarantine."""
    cm = _get_cm(request)
    cm.auto_purge_expired()
    trash_items = cm.list_trash()
    total_bytes = sum(t.get("size_bytes", 0) for t in trash_items)
    return {
        "trash": trash_items,
        "total": len(trash_items),
        "total_bytes": total_bytes,
    }


@router.post("/api/master/models/{slug}/delete")
async def soft_delete_model_endpoint(
    request: Request,
    slug: str,
    retention_days: int = Query(7, ge=1, le=90),
) -> dict[str, Any]:
    """Move a model to trash quarantine (soft delete) with automatic retention period."""
    cm = _get_cm(request)
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

    engine = getattr(request.app.state, "inference_engine", None)
    if engine and engine.active_model_p and entry.active_checkpoint and (
        entry.active_checkpoint == engine.active_model_p.name
        or (Path("models") / entry.active_checkpoint).resolve() == engine.active_model_p.resolve()
    ):
        engine.active_model_p = None
        engine.active_explicitly_cleared = True

    updated_entry = cm.soft_delete_model(slug, retention_days=retention_days)
    jm = getattr(request.app.state, "job_manager", None)
    if jm:
        jm.logs.append(
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


@router.post("/api/master/models/{slug}/archive")
async def archive_model_endpoint(request: Request, slug: str) -> dict[str, Any]:
    """Archive a model's active version, transitioning its active state to None."""
    cm = _get_cm(request)
    entry = cm.get_by_slug(slug, include_deleted=True)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Modelo '{slug}' no encontrado.")

    engine = getattr(request.app.state, "inference_engine", None)
    if engine and engine.active_model_p and entry.active_checkpoint and (
        entry.active_checkpoint == engine.active_model_p.name
        or (Path("models") / entry.active_checkpoint).resolve() == engine.active_model_p.resolve()
    ):
        engine.active_model_p = None
        engine.active_explicitly_cleared = True

    updated_entry = cm.archive_model(slug)
    jm = getattr(request.app.state, "job_manager", None)
    if jm:
        jm.logs.append(
            f"[*] [AUDIT] Modelo '{slug}' ({updated_entry.title}) archivado. "
            f"Estado activo transicionado a None."
        )
    return {
        "status": "archived",
        "slug": slug,
        "active_checkpoint": None,
        "message": f"Modelo '{updated_entry.title}' archivado correctamente.",
    }


@router.post("/api/master/models/{slug}/restore")
async def restore_model_endpoint(request: Request, slug: str) -> dict[str, Any]:
    """Restore a soft-deleted model back to active catalog."""
    cm = _get_cm(request)
    entry = cm.get_by_slug(slug, include_deleted=True)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Modelo '{slug}' no encontrado.")
    if not entry.is_deleted:
        return {"status": "already_active", "slug": slug}

    restored_entry = cm.restore_model(slug)
    engine = getattr(request.app.state, "inference_engine", None)
    if engine and engine.active_model_p is None and restored_entry.active_checkpoint:
        engine.active_explicitly_cleared = False
        engine.resolve_and_reload_model(force=True)

    jm = getattr(request.app.state, "job_manager", None)
    if jm:
        jm.logs.append(
            f"[*] [AUDIT] Modelo '{slug}' ({restored_entry.title}) "
            "restaurado con éxito al catálogo activo."
        )

    return {
        "status": "restored",
        "slug": slug,
        "active_checkpoint": restored_entry.active_checkpoint,
        "model": restored_entry.model_dump(),
    }


@router.post("/api/master/models/{slug}/purge")
async def purge_model_endpoint(
    request: Request,
    slug: str,
    confirm_slug: str = Query(..., description="Confirmation matching the slug to be permanently destroyed"),
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

    cm = _get_cm(request)
    entry = cm.get_by_slug(slug, include_deleted=True)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Modelo '{slug}' no encontrado.")

    engine = getattr(request.app.state, "inference_engine", None)
    if engine and engine.active_model_p and entry.active_checkpoint and (
        entry.active_checkpoint == engine.active_model_p.name
        or (Path("models") / entry.active_checkpoint).resolve() == engine.active_model_p.resolve()
    ):
        engine.active_model_p = None
        engine.active_explicitly_cleared = True

    result = cm.purge_model(slug, remove_files=True)
    jm = getattr(request.app.state, "job_manager", None)
    if jm:
        jm.logs.append(
            f"[*] [AUDIT] PURGA PERMANENTE: Modelo '{slug}' destruido. "
            f"Archivos físicos eliminados: {result['deleted_files']}, "
            f"Espacio liberado: {result['freed_bytes'] // 1024} KB"
        )

    return {
        "status": "purged",
        "slug": slug,
        "details": result,
    }


@router.get("/api/master/models/{slug}/history")
def get_model_history_endpoint(request: Request, slug: str) -> dict[str, Any]:
    """Fetch history iterations (accuracy, loss, F1) for a model."""
    cm = _get_cm(request)
    history = get_model_history(cm.models_dir, slug)
    return {"slug": slug, "history": history, "total": len(history)}

