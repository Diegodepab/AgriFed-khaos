"""Dataset Inspection, Ingestion, Profiling, and Deduplication API Endpoints."""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import time
import urllib.parse
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from agrifed_khaos.dataset_service import (
    UNLABELED_DIR_NAMES,
    count_images_scandir,
    deduplicate_dataset_sync,
    detect_dataset_structure,
    profile_dataset_health_sync,
    safe_extract_archive,
)
from agrifed_khaos.schemas import (
    DatasetConvertCropsRequest,
    DatasetDeduplicateRequest,
    DatasetFetchURLRequest,
    DatasetInspectRequest,
)

logger = logging.getLogger("agrifed.api.datasets")
router = APIRouter(tags=["Datasets"])

_profiler_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ds_profiler")
_inspect_cache: dict[tuple[str, int], dict[str, Any]] = {}
_available_ds_cache: dict[str, Any] = {"ts": 0.0, "payload": None}


@router.post("/api/master/datasets/deduplicate")
async def deduplicate_dataset_endpoint(req: DatasetDeduplicateRequest) -> dict[str, Any]:
    """Deduplicate exact MD5 image files (and optionally purge corrupted headers) async."""
    raw_p = req.dataset_path.strip()
    if not raw_p:
        raise HTTPException(status_code=400, detail="La ruta del dataset ('dataset_path') es obligatoria.")

    p = Path(raw_p)
    if not p.is_dir():
        raise HTTPException(status_code=400, detail=f"El directorio '{raw_p}' no existe en el servidor o no es una carpeta.")

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


@router.get("/api/master/datasets/preview-image")
def get_dataset_preview_image(image_path: str = Query(...)) -> FileResponse:
    """Serve a sample annotated image file securely for dataset inspection preview."""
    safe_path = Path(image_path).resolve()
    repo_root = Path.cwd().resolve()
    try:
        safe_path.relative_to(repo_root)
    except ValueError:
        raise HTTPException(
            status_code=403,
            detail="Acceso denegado: ruta fuera del directorio de trabajo.",
        )

    if not safe_path.is_file():
        raise HTTPException(status_code=404, detail="Imagen de previsualización no encontrada.")

    return FileResponse(str(safe_path))


@router.post("/api/master/datasets/convert-crops")
async def api_convert_dataset_crops(req: DatasetConvertCropsRequest) -> dict[str, Any]:
    """Convert a COCO dataset into ImageFolder classification crops."""
    raw_p = req.dataset_path.strip()
    p = Path(raw_p)
    if not p.is_dir():
        raise HTTPException(status_code=400, detail=f"El directorio '{raw_p}' no existe en el servidor o no es una carpeta.")

    from agrifed_khaos.coco import (
        coco_to_classification_crops,
        discover_coco_splits,
        is_coco_dataset,
    )

    if not is_coco_dataset(p):
        raise HTTPException(
            status_code=400,
            detail=f"El directorio '{raw_p}' no contiene anotaciones válidas en formato COCO JSON.",
        )

    splits_dict = discover_coco_splits(p)
    if not splits_dict:
        raise HTTPException(
            status_code=400,
            detail="No se encontraron divisiones de datos (splits) en el dataset COCO.",
        )

    base_name = req.output_name or f"{p.name}_crops"
    safe_name = re.sub(r"[^\w\-_.]", "_", base_name)
    uploads_root = Path("data/uploads")
    uploads_root.mkdir(parents=True, exist_ok=True)
    out_crops_dir = uploads_root / f"crops_{int(time.time())}_{safe_name}"

    loop = asyncio.get_running_loop()

    def _do_crops_sync() -> dict[str, int]:
        total_crops: dict[str, int] = {}
        for s_name, (s_json, s_img_d) in splits_dict.items():
            if "unannotated" in s_name:
                continue
            max_c = req.max_crops_per_class if req.max_crops_per_class is not None else 2000
            if s_name != "train":
                max_c = min(500, max_c)
            crops_per_cat = coco_to_classification_crops(
                coco_json_path=s_json,
                images_dir=s_img_d,
                output_dir=out_crops_dir,
                split_name=s_name,
                min_box_size=req.min_box_size,
                padding_pct=req.padding_pct,
                max_crops_per_class=max_c,
                target_categories=req.target_categories,
            )
            for c_name, count in crops_per_cat.items():
                total_crops[c_name] = total_crops.get(c_name, 0) + count
        return total_crops

    total_crops = await loop.run_in_executor(_profiler_executor, _do_crops_sync)

    if not any(total_crops.values()):
        raise HTTPException(
            status_code=400,
            detail="No se generaron recortes válidos. Comprueba el tamaño mínimo de caja o las categorías seleccionadas.",
        )

    _inspect_cache.clear()
    inspect_res = inspect_dataset_directory(DatasetInspectRequest(dataset_path=str(out_crops_dir)))
    inspect_res["converted_crops"] = total_crops
    inspect_res["message"] = (
        f"Se han extraído con éxito {sum(total_crops.values()):,} recortes de fruto "
        f"distribuidos en {len(total_crops)} clases ({', '.join(total_crops.keys())})."
    )
    return inspect_res


@router.post("/api/master/datasets/inspect")
def inspect_dataset_directory(req: DatasetInspectRequest) -> dict[str, Any]:
    """Inspect a dataset directory, discover classes, count image samples, and suggest arch."""
    raw_p = req.dataset_path.strip()
    if not raw_p:
        raise HTTPException(status_code=400, detail="La ruta del dataset ('dataset_path') es obligatoria.")

    p = Path(raw_p)
    if not p.is_dir():
        raise HTTPException(status_code=400, detail=f"El directorio '{raw_p}' no existe en el servidor o no es una carpeta.")

    try:
        p_mtime = p.stat().st_mtime_ns
        cache_key = (raw_p, p_mtime)
        if cache_key in _inspect_cache:
            return dict(_inspect_cache[cache_key])
    except OSError:
        cache_key = None

    from agrifed_khaos.yolo import inspect_yolo_dataset, is_yolo_dataset

    if is_yolo_dataset(p):
        yolo_res = inspect_yolo_dataset(p, preferred_split="train")
        if yolo_res.get("valid"):
            yolo_res["original_path"] = raw_p
            yolo_res["auto_detected"] = Path(yolo_res["dataset_path"]).resolve() != p.resolve()
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
                yolo_res["health_profile"] = profile_dataset_health_sync(Path(yolo_res.get("dataset_path", p)))
            yolo_res.setdefault("is_pre_split", True)
            yolo_res.setdefault("split_distribution", None)
            yolo_res.setdefault("split_percentages", None)
            yolo_res.setdefault("unlabeled_count", 0)
            if cache_key is not None:
                _inspect_cache[cache_key] = dict(yolo_res)
            return yolo_res

    from agrifed_khaos.coco import inspect_coco_dataset, is_coco_dataset

    if is_coco_dataset(p):
        coco_res = inspect_coco_dataset(p, preferred_split="train")
        if coco_res.get("valid"):
            coco_res["original_path"] = raw_p
            coco_res["auto_detected"] = Path(coco_res["dataset_path"]).resolve() != p.resolve()
            if cache_key is not None:
                _inspect_cache[cache_key] = dict(coco_res)
            return coco_res

    from agrifed_khaos.data import IMAGE_EXTENSIONS, find_dataset_class_dir

    struct_info = detect_dataset_structure(p)
    target_dir = find_dataset_class_dir(p, preferred_split="train")
    auto_detected = target_dir.resolve() != p.resolve()

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
            "error": f"No se detectaron subcarpetas de clase en '{raw_p}'. Estructura esperada: {raw_p}/<clase_1>, {raw_p}/<clase_2>...",
        }

    image_counts: dict[str, int] = {}
    for d in sorted(subdirs, key=lambda x: x.name):
        image_counts[d.name] = count_images_scandir(d, IMAGE_EXTENSIONS)

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
            "error": f"Se encontraron subcarpetas ({classes}), pero no contienen imágenes con extensiones soportadas (.jpg, .jpeg, .png, .webp, .bmp).{empty_info}",
        }

    if total_images > 5000:
        rec_arch = "efficientnet_v2_s"
        rec_reason = "Volumen robusto (> 5.000 imgs): Alta capacidad representacional y convergencia SOTA."
    elif total_images >= 1000:
        rec_arch = "efficientnet_b0"
        rec_reason = "Volumen intermedio (1.000 - 5.000 imgs): Óptimo ratio precisión/parámetros."
    else:
        rec_arch = "resnet18"
        rec_reason = "Volumen reducido (< 1.000 imgs): Balance robusto y prevención de sobreajuste."

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
        "recommendation_details": {"recommended_arch": rec_arch, "reason": rec_reason},
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


@router.get("/api/master/datasets/available")
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
                candidates_to_check.extend(
                    [
                        d
                        for d in root.iterdir()
                        if d.is_dir()
                        and not d.name.startswith((".", "_"))
                        and d.name.lower() not in skip_names
                    ]
                )
            except OSError as exc:
                logger.debug("Cannot scan root '%s': %s", root, exc)
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
                subdirs = [d for d in resolved.iterdir() if d.is_dir() and not d.name.startswith((".", "_"))]
                if not subdirs:
                    continue
                cnt_map = {d.name: count_images_scandir(d, IMAGE_EXTENSIONS) for d in subdirs}
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
            except Exception as exc:
                logger.debug("Failed scanning candidate dataset '%s': %s", cand, exc)
                continue

    dataset_list = sorted(found.values(), key=lambda x: x["total_images"], reverse=True)
    payload = {"datasets": dataset_list, "total": len(dataset_list)}
    _available_ds_cache["ts"] = now_t
    _available_ds_cache["payload"] = payload
    return payload


@router.post("/api/master/datasets/upload")
async def upload_dataset_archive_or_folder(
    file: UploadFile | None = File(None),
    files: list[UploadFile] | None = File(None),
    paths: list[str] | None = Form(None),
) -> dict[str, Any]:
    """Upload a dataset as a .zip/.tar archive or as multi-file folder (no size limits)."""
    uploads_dir = Path("data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    session_id = f"upload_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    dest_dir = uploads_dir / session_id

    target_archive = None
    if file and file.filename:
        target_archive = file
    elif files and len(files) == 1 and files[0].filename:
        raw_ext_candidate = "".join(Path(files[0].filename).suffixes).lower()
        if any(raw_ext_candidate.endswith(ext) for ext in (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2")):
            target_archive = files[0]

    if target_archive and target_archive.filename:
        raw_ext = "".join(Path(target_archive.filename).suffixes).lower()
        if not any(raw_ext.endswith(ext) for ext in (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2")):
            raise HTTPException(
                status_code=400,
                detail="Formato no soportado. Por favor suba un archivo .zip o .tar.gz.",
            )

        temp_archive = uploads_dir / f"{session_id}_archive{raw_ext}"
        try:
            with temp_archive.open("wb") as f_out:
                while chunk := await target_archive.read(1024 * 1024):
                    f_out.write(chunk)

            resolved_path = await asyncio.to_thread(safe_extract_archive, temp_archive, dest_dir)
        except ValueError as val_err:
            raise HTTPException(status_code=400, detail=str(val_err)) from val_err
        finally:
            temp_archive.unlink(missing_ok=True)
    elif files and len(files) > 0:
        dest_dir.mkdir(parents=True, exist_ok=True)
        resolved_dest = dest_dir.resolve()
        for idx, f in enumerate(files):
            rel_str = paths[idx] if paths and idx < len(paths) else (f.filename or f"item_{idx}.jpg")
            clean_rel = rel_str.strip().lstrip("/")
            target_p = (dest_dir / clean_rel).resolve()
            if not str(target_p).startswith(str(resolved_dest)):
                raise HTTPException(status_code=400, detail=f"Ruta relativa no válida o intento de path traversal: {rel_str}")
            target_p.parent.mkdir(parents=True, exist_ok=True)
            img_data = await f.read()
            target_p.write_bytes(img_data)

        from agrifed_khaos.coco import is_coco_dataset
        from agrifed_khaos.yolo import is_yolo_dataset

        if is_yolo_dataset(dest_dir):
            resolved_path = dest_dir
        elif is_coco_dataset(dest_dir):
            resolved_path = dest_dir
        else:
            from agrifed_khaos.data import find_dataset_class_dir

            resolved_path = find_dataset_class_dir(dest_dir, preferred_split="train")
    else:
        raise HTTPException(status_code=400, detail="No se ha recibido ningún archivo comprimido ni archivos de carpeta.")

    source_filename = None
    if target_archive and target_archive.filename:
        source_filename = target_archive.filename
    elif files and len(files) > 0 and files[0].filename:
        p_parts = Path(files[0].filename).parts
        if len(p_parts) > 1:
            source_filename = p_parts[0]

    inspect_req = DatasetInspectRequest(dataset_path=str(resolved_path), source_name=source_filename)
    loop = asyncio.get_running_loop()
    health_profile = await loop.run_in_executor(_profiler_executor, profile_dataset_health_sync, resolved_path)
    inspect_res = inspect_dataset_directory(inspect_req)
    inspect_res["health_profile"] = health_profile
    return inspect_res


@router.post("/api/master/datasets/fetch-url")
async def fetch_dataset_from_url(req: DatasetFetchURLRequest) -> dict[str, Any]:
    """Download and extract a dataset archive from a remote URL (no size limits)."""
    raw_url = req.url.strip()
    if not raw_url:
        raise HTTPException(status_code=400, detail="La URL es obligatoria.")

    parsed = urllib.parse.urlparse(raw_url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="Solo se admiten enlaces HTTP o HTTPS.")

    hostname = (parsed.hostname or "").lower()
    blocked_hosts = {
        "169.254.169.254",
        "metadata.google.internal",
        "metadata",
        "instance-data",
        "100.100.100.200",
    }
    if hostname in blocked_hosts or hostname.startswith("169.254."):
        raise HTTPException(status_code=400, detail="Host no permitido por política de seguridad.")

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
                        detail=f"El servidor remoto devolvió código {response.status_code} ({response.reason_phrase}).",
                    )
                with temp_archive.open("wb") as f_out:
                    async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                        f_out.write(chunk)

        resolved_path = safe_extract_archive(temp_archive, dest_dir)
    except HTTPException:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise
    except ValueError as val_err:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(val_err)) from val_err
    except Exception as err:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Error durante la descarga del dataset: {err}") from err
    finally:
        temp_archive.unlink(missing_ok=True)

    url_filename = Path(parsed.path).name or "archive"
    inspect_req = DatasetInspectRequest(dataset_path=str(resolved_path), source_name=url_filename)
    loop = asyncio.get_running_loop()
    health_profile = await loop.run_in_executor(_profiler_executor, profile_dataset_health_sync, resolved_path)
    inspect_res = inspect_dataset_directory(inspect_req)
    inspect_res["health_profile"] = health_profile
    return inspect_res

