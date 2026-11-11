"""Dataset Inspection, Health Profiling, Deduplication, and Archive Ingestion Service."""

from __future__ import annotations

import hashlib
import logging
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image

from agrifed_khaos.data import IMAGE_EXTENSIONS, find_dataset_class_dir
from agrifed_khaos.yolo import is_yolo_dataset

logger = logging.getLogger("agrifed.dataset_service")

UNLABELED_DIR_NAMES = frozenset({
    "unlabeled",
    "unknown",
    "sin_etiquetar",
    "sin_clasificar",
    "sin_etiqueta",
    "sin-etiquetar",
    "sin-clasificar",
})

VALID_IMAGE_EXTS = frozenset({
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


def count_images_scandir(dir_path: Path, valid_exts: set[str] | frozenset[str] = VALID_IMAGE_EXTS) -> int:
    """Fast non-recursive image counter for a single directory."""
    cnt = 0
    try:
        with os.scandir(dir_path) as it:
            for entry in it:
                if entry.is_file(follow_symlinks=False):
                    dot_idx = entry.name.rfind(".")
                    if dot_idx != -1 and entry.name[dot_idx:].lower() in valid_exts:
                        cnt += 1
    except OSError as exc:
        logger.debug("Cannot scan directory '%s': %s", dir_path, exc)
    return cnt


def detect_dataset_structure(p: Path) -> dict[str, Any]:
    """Detect dataset structure: Flat vs Pre-Split, and count unlabeled images."""
    candidates = [p]
    if p.name.lower() in ("train", "training"):
        candidates.append(p.parent)
    try:
        candidates.extend(
            [d for d in p.iterdir() if d.is_dir() and not d.name.startswith((".", "_"))]
        )
    except OSError as exc:
        logger.debug("Cannot read candidate subdirectories in '%s': %s", p, exc)

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
            try:
                train_subdirs = [
                    d
                    for d in c_train.iterdir()
                    if d.is_dir()
                    and not d.name.startswith((".", "_"))
                    and d.name.lower() not in UNLABELED_DIR_NAMES
                ]
            except OSError:
                train_subdirs = []
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

    try:
        with os.scandir(search_root) as it:
            for entry in it:
                if entry.is_file(follow_symlinks=False):
                    dot = entry.name.rfind(".")
                    if dot != -1 and entry.name[dot:].lower() in IMAGE_EXTENSIONS:
                        unlabeled_count += 1
    except OSError as exc:
        logger.debug("Cannot scan search root '%s': %s", search_root, exc)

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
    """Profile dataset health, detecting corruption, duplicate hashes, size variance, and background bias."""
    import numpy as np

    target_path = Path(dataset_dir)
    empty_result = {
        "health_score": 0,
        "corrupted_count": 0,
        "duplicates_count": 0,
        "resolution_warning": False,
        "total_scanned": 0,
        "min_resolution": [0, 0],
        "max_resolution": [0, 0],
        "background_uniformity_pct": 0.0,
        "dominant_bg_type": "natural_diverse",
        "domain_shift_risk": "low",
        "recommended_augmentation": "basic",
    }
    if not target_path.exists():
        return empty_result

    image_paths: list[Path] = []
    try:
        for p_file in target_path.rglob("*"):
            if (
                p_file.is_file()
                and p_file.suffix.lower() in VALID_IMAGE_EXTS
                and not p_file.name.startswith((".", "_"))
            ):
                if any(
                    part.startswith((".", "_")) or part.lower() == "__macosx"
                    for part in p_file.parts
                ):
                    continue
                image_paths.append(p_file)
    except OSError as exc:
        logger.warning("Error traversing dataset directory '%s': %s", target_path, exc)

    if not image_paths:
        return empty_result

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

    sample_stride = max(1, len(image_paths) // 100)
    bg_uniform_count = 0
    bg_dark_count = 0
    bg_light_count = 0
    bg_sampled = 0
    border_means: list[np.ndarray] = []
    border_stds: list[float] = []

    for idx, img_p in enumerate(image_paths):
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
        except OSError as exc:
            logger.debug("Unreadable file '%s': %s", img_p, exc)
            corrupted_count += 1
            continue

        try:
            with Image.open(img_p) as img:
                w, h = img.size
                if idx % sample_stride == 0 and bg_sampled < 100:
                    rgb_small = img.convert("RGB").resize((64, 64), Image.Resampling.NEAREST)
                    arr = np.asarray(rgb_small, dtype=np.float32)
                    b = 5
                    border_px = np.concatenate(
                        [
                            arr[:b, :, :].reshape(-1, 3),
                            arr[-b:, :, :].reshape(-1, 3),
                            arr[:, :b, :].reshape(-1, 3),
                            arr[:, -b:, :].reshape(-1, 3),
                        ],
                        axis=0,
                    )
                    b_med = np.median(border_px, axis=0)
                    dists = np.linalg.norm(border_px - b_med[None, :], axis=1)
                    inlier_mask = dists < 32.0
                    inlier_ratio = float(np.mean(inlier_mask))
                    inlier_std = (
                        float(border_px[inlier_mask].std(axis=0).mean())
                        if np.any(inlier_mask)
                        else 50.0
                    )
                    b_std = float(border_px.std(axis=0).mean())
                    b_lum = (
                        float(border_px[inlier_mask].mean())
                        if np.any(inlier_mask)
                        else float(b_med.mean())
                    )
                    border_means.append(b_med)
                    border_stds.append(b_std)
                    bg_sampled += 1
                    if b_std < 18.0 or (inlier_ratio >= 0.60 and inlier_std < 16.0):
                        bg_uniform_count += 1
                        if b_lum < 55.0:
                            bg_dark_count += 1
                        elif b_lum > 200.0:
                            bg_light_count += 1
                else:
                    img.verify()
        except Exception as exc:
            logger.debug("Corrupted image '%s': %s", img_p, exc)
            corrupted_count += 1
            continue

        valid_count += 1
        area = w * h
        min_w = min(min_w, w)
        max_w = max(max_w, w)
        min_h = min(min_h, h)
        max_h = max(max_h, h)
        min_area = min(min_area, area)
        max_area = max(max_area, area)

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

    bg_uniformity_pct = 0.0
    dominant_bg_type = "natural_diverse"
    domain_shift_risk = "low"
    recommended_aug = "basic"

    if bg_sampled > 0:
        bg_uniformity_pct = round((bg_uniform_count / float(bg_sampled)) * 100.0, 1)
        avg_intra_std = float(np.mean(border_stds)) if border_stds else 50.0
        inter_img_std = (
            float(np.stack(border_means, axis=0).std(axis=0).mean())
            if len(border_means) > 1
            else 50.0
        )
        if inter_img_std < 16.0 and avg_intra_std < 20.0 and bg_uniformity_pct < 65.0:
            bg_uniformity_pct = max(
                bg_uniformity_pct, round((1.0 - inter_img_std / 32.0) * 100.0, 1)
            )

        if bg_uniformity_pct >= 60.0:
            domain_shift_risk = "high"
            recommended_aug = "in_the_wild"
            if bg_dark_count >= max(1.0, bg_uniform_count * 0.5):
                dominant_bg_type = "dark_studio"
            elif bg_light_count >= max(1.0, bg_uniform_count * 0.5):
                dominant_bg_type = "light_studio"
            else:
                dominant_bg_type = "uniform_surface"
        elif bg_uniformity_pct >= 35.0:
            domain_shift_risk = "medium"
            recommended_aug = "in_the_wild"
            dominant_bg_type = "uniform_surface"
        else:
            domain_shift_risk = "low"
            recommended_aug = "randaugment"
            dominant_bg_type = "natural_diverse"

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
        "background_uniformity_pct": bg_uniformity_pct,
        "dominant_bg_type": dominant_bg_type,
        "domain_shift_risk": domain_shift_risk,
        "recommended_augmentation": recommended_aug,
    }


def deduplicate_dataset_sync(
    dataset_dir: Path | str,
    remove_duplicates: bool = True,
    remove_corrupted: bool = False,
) -> dict[str, Any]:
    """Scan dataset directory, delete exact MD5 duplicates (and optionally corrupted files)."""
    target_path = Path(dataset_dir)
    image_paths: list[Path] = []
    try:
        for p_file in target_path.rglob("*"):
            if (
                p_file.is_file()
                and p_file.suffix.lower() in VALID_IMAGE_EXTS
                and not p_file.name.startswith((".", "_"))
            ):
                if any(
                    part.startswith((".", "_")) or part.lower() == "__macosx"
                    for part in p_file.parts
                ):
                    continue
                image_paths.append(p_file)
    except OSError as exc:
        logger.warning("Error traversing dataset directory '%s': %s", target_path, exc)

    image_paths.sort()
    seen_hashes: set[str] = set()
    deleted_count = 0
    corrupted_removed = 0

    for img_p in image_paths:
        if remove_corrupted:
            try:
                with Image.open(img_p) as img:
                    img.verify()
            except Exception as exc:
                logger.info("Purging corrupted file '%s': %s", img_p, exc)
                try:
                    os.remove(img_p)
                    corrupted_removed += 1
                except OSError as rm_exc:
                    logger.warning("Could not delete corrupted file '%s': %s", img_p, rm_exc)
                continue

        try:
            hasher = hashlib.md5()
            with img_p.open("rb") as f_img:
                while chunk := f_img.read(65536):
                    hasher.update(chunk)
            file_hash = hasher.hexdigest()
        except OSError as exc:
            if remove_corrupted:
                logger.info("Purging unreadable file '%s': %s", img_p, exc)
                try:
                    os.remove(img_p)
                    corrupted_removed += 1
                except OSError as rm_exc:
                    logger.warning("Could not delete unreadable file '%s': %s", img_p, rm_exc)
            continue

        if file_hash in seen_hashes:
            if remove_duplicates:
                try:
                    os.remove(img_p)
                    deleted_count += 1
                except OSError as exc:
                    logger.warning("Could not delete duplicate file '%s': %s", img_p, exc)
        else:
            seen_hashes.add(file_hash)

    try:
        subdirs = [
            d
            for d in target_path.iterdir()
            if d.is_dir()
            and not d.name.startswith((".", "_"))
            and d.name.lower() != "__macosx"
        ]
    except OSError:
        subdirs = []

    image_counts: dict[str, int] = {}
    if subdirs:
        for d in sorted(subdirs, key=lambda x: x.name):
            image_counts[d.name] = count_images_scandir(d, VALID_IMAGE_EXTS)
        total_images = sum(image_counts.values())
    else:
        total_images = count_images_scandir(target_path, VALID_IMAGE_EXTS)

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


def safe_extract_archive(
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
                    raise ValueError(f"Ruta maliciosa detectada en archivo: {member.filename}")
            zf.extractall(resolved_dest)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tf:
            for member in tf.getmembers():
                target_p = (dest_dir / member.name).resolve()
                if not str(target_p).startswith(str(resolved_dest)):
                    raise ValueError(f"Ruta maliciosa detectada en archivo: {member.name}")
            tf.extractall(resolved_dest)
    else:
        raise ValueError("Formato de archivo comprimido no soportado (.zip, .tar, .tar.gz).")

    if is_yolo_dataset(dest_dir):
        return dest_dir

    from agrifed_khaos.coco import is_coco_dataset

    if is_coco_dataset(dest_dir):
        return dest_dir

    return find_dataset_class_dir(dest_dir, preferred_split="train")

