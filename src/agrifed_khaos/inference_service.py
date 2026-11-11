"""PyTorch Vision Inference Service, OOD Guardrails, and Batch Processing."""

from __future__ import annotations

import collections
import csv
import hashlib
import io
import json
import logging
import math
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import HTTPException
from PIL import Image
from torchvision import transforms

from agrifed_khaos.catalog import ModelCatalogManager
from agrifed_khaos.models import build_model, load_checkpoint_bundle

logger = logging.getLogger("agrifed.inference")

MAX_LRU_MODELS = 2


def extract_smart_roi_crop(img: Image.Image, padding_pct: float = 0.15) -> Image.Image:
    """Detect and crop the primary foreground object (ROI) to mitigate background domain shift.

    Uses perimeter color/luminance contrast combined with spatial edge energy to locate
    subjects surrounded by uniform or distracting backgrounds, returning a padded square crop.
    If the subject already fills the frame or no distinct ROI is separable, returns the
    original image unchanged (safe no-op).
    """
    w, h = img.size
    if w < 48 or h < 48:
        return img

    try:
        work_size = 160
        small = img.resize((work_size, work_size), Image.Resampling.BILINEAR)
        arr = np.asarray(small, dtype=np.float32)

        # Sample outer 6% border ring to estimate background color distribution
        border = max(4, int(work_size * 0.06))
        top = arr[:border, :, :].reshape(-1, 3)
        bottom = arr[-border:, :, :].reshape(-1, 3)
        left = arr[:, :border, :].reshape(-1, 3)
        right = arr[:, -border:, :].reshape(-1, 3)
        border_pixels = np.concatenate([top, bottom, left, right], axis=0)
        bg_median = np.median(border_pixels, axis=0)

        # Color distance from perimeter background
        color_dist = np.linalg.norm(arr - bg_median, axis=-1)

        # Spatial gradient energy (Sobel-like finite differences)
        gray = arr.mean(axis=-1)
        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        grad_mag = np.hypot(gx, gy)

        # Normalize and fuse color contrast + edge salience
        c_max = float(np.percentile(color_dist, 98))
        g_max = float(np.percentile(grad_mag, 98))
        if c_max < 8.0 and g_max < 5.0:
            # Uniform/flat image -> no distinct ROI
            return img

        c_norm = color_dist / max(c_max, 1.0)
        g_norm = grad_mag / max(g_max, 1.0)
        saliency = 0.65 * c_norm + 0.35 * g_norm

        # Suppress extreme 2px image border artifacts
        saliency[:2, :] = 0.0
        saliency[-2:, :] = 0.0
        saliency[:, :2] = 0.0
        saliency[:, -2:] = 0.0

        thresh = max(0.32, float(np.percentile(saliency, 72)))
        mask = saliency >= thresh
        ys, xs = np.where(mask)
        if len(xs) < 25:
            return img

        # Use robust percentiles (5% to 95%) to ignore isolated background specks
        x0_s = float(np.percentile(xs, 4))
        x1_s = float(np.percentile(xs, 96))
        y0_s = float(np.percentile(ys, 4))
        y1_s = float(np.percentile(ys, 96))

        box_w = x1_s - x0_s
        box_h = y1_s - y0_s
        area_ratio = (box_w * box_h) / float(work_size * work_size)

        # Only crop if ROI is a meaningful sub-region (between 6% and 82% of frame)
        if area_ratio < 0.06 or area_ratio > 0.82:
            return img

        cx = (x0_s + x1_s) / 2.0
        cy = (y0_s + y1_s) / 2.0
        side = max(box_w, box_h) * (1.0 + 2.0 * padding_pct)
        side = min(float(work_size), max(side, work_size * 0.30))

        half = side / 2.0
        nx0 = max(0.0, min(work_size - side, cx - half))
        ny0 = max(0.0, min(work_size - side, cy - half))
        nx1 = min(float(work_size), nx0 + side)
        ny1 = min(float(work_size), ny0 + side)

        scale_x = w / float(work_size)
        scale_y = h / float(work_size)
        crop_box = (
            int(round(nx0 * scale_x)),
            int(round(ny0 * scale_y)),
            int(round(nx1 * scale_x)),
            int(round(ny1 * scale_y)),
        )
        if crop_box[2] - crop_box[0] < 32 or crop_box[3] - crop_box[1] < 32:
            return img
        return img.crop(crop_box)
    except Exception as exc:
        logger.debug("Smart ROI extraction skipped: %s", exc)
        return img


def build_aspect_letterbox(
    img: Image.Image, target_size: tuple[int, int] = (224, 224)
) -> Image.Image:
    """Resize image preserving aspect ratio and pad with perimeter median color.

    Prevents CenterCrop from amputating extremities of elongated or wide objects
    captured in 16:9 or 4:3 mobile aspect ratios.
    """
    w, h = img.size
    tw, th = target_size
    if w <= 0 or h <= 0:
        return img.resize(target_size, Image.Resampling.BILINEAR)

    aspect_diff = abs(w - h) / float(max(w, h))
    if aspect_diff < 0.08:
        return img.resize(target_size, Image.Resampling.BILINEAR)

    try:
        arr = np.asarray(img.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.uint8)
        border_px = np.concatenate(
            [arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]], axis=0
        )
        fill_rgb = tuple(int(v) for v in np.median(border_px, axis=0))
    except Exception:
        fill_rgb = (128, 128, 128)

    scale = min(tw / float(w), th / float(h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = img.resize((nw, nh), Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (tw, th), fill_rgb)
    paste_x = (tw - nw) // 2
    paste_y = (th - nh) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def fuse_multiview_logits(
    logits_batch: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """Fuse multi-view logits (CenterCrop, SmartROI, Letterbox) weighted by inverse entropy.

    Views that isolate the target object cleanly produce lower Shannon entropy (sharper
    in-domain activation) and receive higher weight, while views polluted by unfamiliar
    backgrounds produce flatter distributions and are down-weighted automatically.
    """
    if logits_batch.dim() == 1:
        return logits_batch
    if logits_batch.shape[0] == 1:
        return logits_batch[0]

    t_val = max(float(temperature), 1e-3)
    probs = F.softmax(logits_batch / t_val, dim=-1)
    num_classes = probs.shape[-1]
    max_ent = math.log(max(num_classes, 2))
    entropies = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
    norm_ent = entropies / max(max_ent, 1e-6)

    # Sharpness score in [0, 1]: higher when entropy is low
    sharpness = torch.clamp(1.0 - norm_ent, min=1e-3)
    # Give slight prior preference (1.05x) to standard view (idx 0) so ties preserve legacy
    prior = torch.ones_like(sharpness)
    prior[0] = 1.05

    weights = F.softmax(sharpness * 5.0, dim=0) * prior
    weights = weights / weights.sum().clamp(min=1e-6)
    fused_logits = (logits_batch * weights.unsqueeze(-1)).sum(dim=0)
    return fused_logits


def compute_ood_guardrail(
    logits: torch.Tensor | None,
    probs: torch.Tensor,
    num_classes: int,
    temperature: float = 1.0,
    threshold: float | None = None,
    entropy_threshold: float | None = None,
) -> dict[str, Any]:
    """Compute Energy-based Out-of-Distribution (OoD) score and Shannon Entropy guardrails."""
    t_val = max(float(temperature), 1e-3)

    if logits is not None:
        logits_1d = logits.squeeze(0) if logits.dim() > 1 else logits
        free_energy = float((-t_val * torch.logsumexp(logits_1d / t_val, dim=-1)).item())
    else:
        free_energy = float(
            (-t_val * torch.logsumexp(torch.log(probs + 1e-9) / t_val, dim=-1)).item()
        )

    energy_score = -free_energy
    probs_1d = probs.squeeze(0) if probs.dim() > 1 else probs
    entropy = float(-(probs_1d * torch.log(probs_1d + 1e-9)).sum().item())
    max_entropy = math.log(max(num_classes, 2))
    norm_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    explicit_thresh = threshold is not None
    if explicit_thresh:
        eff_thresh = float(threshold)  # type: ignore[arg-type]
    else:
        env_thresh = os.getenv("AGRIFED_OOD_ENERGY_THRESHOLD")
        if env_thresh:
            eff_thresh = float(env_thresh)
            explicit_thresh = True
        else:
            eff_thresh = max(0.5, t_val * (max_entropy - 1.5))

    explicit_entropy = entropy_threshold is not None
    if explicit_entropy:
        eff_entropy_thresh = float(entropy_threshold)  # type: ignore[arg-type]
    else:
        env_entropy = os.getenv("AGRIFED_OOD_ENTROPY_THRESHOLD")
        if env_entropy:
            eff_entropy_thresh = float(env_entropy)
            explicit_entropy = True
        else:
            eff_entropy_thresh = 0.88 if num_classes >= 10 else 0.78

    low_energy = energy_score < eff_thresh
    high_entropy = norm_entropy >= eff_entropy_thresh

    if explicit_thresh or explicit_entropy:
        is_ood = bool(
            (low_energy if explicit_thresh else False)
            or (high_entropy if explicit_entropy else False)
        )
    else:
        is_ood = bool(
            (low_energy and high_entropy)
            or (energy_score < 0.0)
            or (norm_entropy >= 0.90)
        )

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


class InferenceEngine:
    """Manages active PyTorch model loading, LRU caching, and vision inference."""

    def __init__(
        self,
        model_path: str = "models/model.pt",
        metadata_path: str = "models/metadata.json",
        architecture: str | None = None,
        device: torch.device | None = None,
        device_name: str = "auto",
        app: Any = None,
    ) -> None:
        self.app = app
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.architecture = architecture
        if device is not None:
            self.device = device
        elif device_name and device_name != "auto":
            self.device = torch.device(device_name)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_p = Path(model_path)
        self.meta_p = Path(metadata_path)

        self.active_model_p: Path | None = None
        self.active_explicitly_cleared: bool = False
        self.last_mtime: float = -1.0
        self.bundle_meta: dict[str, Any] = {}
        self.classes: list[str] = []
        self.arch: str = architecture or "resnet18"
        self.num_classes: int = 9
        self.calibrated_thresholds: Any = None
        self.temperature: float = 1.0
        self.model_sha256: str = ""
        self.img_size_tuple: tuple[int, int] = (224, 224)
        self.resize_tuple: tuple[int, int] = (256, 256)
        self.transform: transforms.Compose = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.net: torch.nn.Module = build_model(architecture=self.arch, num_classes=self.num_classes).to(self.device)
        self.model_cache: collections.OrderedDict[str, dict[str, Any]] = collections.OrderedDict()
        self.yolo_models: collections.OrderedDict[str, dict[str, Any]] = collections.OrderedDict()
        self.task: str = "classification"

        self.resolve_and_reload_model()

    def _get_catalog_manager(self) -> ModelCatalogManager:
        if self.app and hasattr(self.app.state, "catalog"):
            return self.app.state.catalog
        models_dir = self.model_p.parent if self.model_p.parent.is_dir() else Path("models")
        return ModelCatalogManager(catalog_path=models_dir / "catalog.json", models_dir=models_dir)

    def _load_metadata_for_path(self, target_path: Path) -> dict[str, Any]:
        """Load companion .meta.json for a checkpoint if available."""
        stem_clean = target_path.stem.replace("_base", "")
        candidates = [
            target_path.parent / f"{stem_clean}.meta.json",
            target_path.with_suffix(".meta.json"),
            target_path.parent / "metadata.json",
        ]
        for c in candidates:
            if c.is_file():
                try:
                    data = json.loads(c.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        return data
                except Exception:
                    pass
        return {}

    def is_detection_model(self, model_name_or_path: Path | str | None) -> bool:
        """Check if target model is an Ultralytics YOLO object detection model."""
        if not model_name_or_path:
            if self.active_model_p:
                return self.is_detection_model(self.active_model_p)
            return False

        # 1. Check catalog entry if string slug
        cm = self._get_catalog_manager()
        name_str = str(model_name_or_path).strip()
        slug_clean = Path(name_str).stem.replace("_base", "")
        entry = cm.get_by_slug(slug_clean, include_deleted=True) or cm.get_by_slug(name_str, include_deleted=True)
        if entry:
            if getattr(entry, "task", "") == "detection" or entry.architecture.lower().startswith("yolo"):
                return True

        # 2. Check metadata on disk
        target_p = Path(model_name_or_path)
        if not target_p.is_file():
            models_root = self.model_p.parent if self.model_p.parent.is_dir() else Path("models")
            target_p = models_root / target_p.name
        if target_p.is_file():
            meta = self._load_metadata_for_path(target_p)
            if meta.get("task") == "detection" or str(meta.get("architecture", "")).lower().startswith("yolo"):
                return True

        # 3. Check filename heuristics
        fname = target_p.name.lower()
        if fname.startswith("yolo") or "yolov8" in fname or "yolov11" in fname or "yolo11" in fname:
            return True

        return False

    def _resolve_target_checkpoint_path(self, target_model_name: str | None = None) -> Path:
        """Resolve on-disk path for a target checkpoint or active model."""
        if not target_model_name:
            if self.active_model_p and self.active_model_p.is_file():
                return self.active_model_p
            raise HTTPException(status_code=503, detail="No hay ningún modelo activo en el sistema.")

        target_checkpoint = target_model_name
        cm = self._get_catalog_manager()
        catalog_entry = cm.get_by_slug(target_model_name, include_deleted=True)
        if catalog_entry and catalog_entry.active_checkpoint:
            target_checkpoint = catalog_entry.active_checkpoint

        cand = Path(target_checkpoint)
        if cand.is_file():
            return cand

        safe_name = cand.name
        base_dir = self.active_model_p.parent if self.active_model_p else Path("models")
        target_path = base_dir / safe_name
        if not target_path.is_file():
            target_path = Path("models") / safe_name
        if not target_path.is_file() and Path(safe_name).is_file():
            target_path = Path(safe_name)
        if not target_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Checkpoint '{target_checkpoint}' no encontrado en disco.",
            )
        return target_path

    def get_yolo_model_for_inference(
        self, target_model_name: str | None = None
    ) -> tuple[Any, list[str], str]:
        """Fetch or load Ultralytics YOLO model from cache."""
        target_path = self._resolve_target_checkpoint_path(target_model_name)
        resolved_key = str(target_path.resolve())
        mtime = target_path.stat().st_mtime

        if resolved_key in self.yolo_models:
            entry = self.yolo_models[resolved_key]
            if entry["mtime"] == mtime:
                self.yolo_models.move_to_end(resolved_key)
                return entry["model"], entry["classes"], target_path.name

        from ultralytics import YOLO

        yolo_net = YOLO(str(target_path))
        meta = self._load_metadata_for_path(target_path)
        classes = meta.get("classes") or []
        if not classes and hasattr(yolo_net, "names") and yolo_net.names:
            classes = [str(yolo_net.names[k]) for k in sorted(yolo_net.names.keys())]

        while len(self.yolo_models) >= MAX_LRU_MODELS:
            self.yolo_models.popitem(last=False)

        self.yolo_models[resolved_key] = {
            "model": yolo_net,
            "classes": classes,
            "mtime": mtime,
        }
        return yolo_net, classes, target_path.name

    def infer_detection(
        self,
        img: Image.Image,
        yolo_model: Any,
        classes: list[str],
        filename: str | None,
        actual_model_name: str,
        conf_threshold: float = 0.25,
    ) -> dict[str, Any]:
        """Run Ultralytics YOLO object detection on PIL image and produce structured detections."""
        results = yolo_model(img, conf=conf_threshold, verbose=False)
        boxes_out: list[dict[str, Any]] = []
        counts: dict[str, int] = {c: 0 for c in classes}

        orig_w, orig_h = img.size
        top_conf = 0.0
        primary_class = classes[0] if classes else "object"

        if results and len(results) > 0:
            res0 = results[0]
            if hasattr(res0, "boxes") and res0.boxes is not None:
                for b in res0.boxes:
                    conf = float(b.conf[0].item()) if hasattr(b, "conf") and len(b.conf) > 0 else 1.0
                    cls_id = int(b.cls[0].item()) if hasattr(b, "cls") and len(b.cls) > 0 else 0
                    if cls_id < len(classes):
                        cls_name = classes[cls_id]
                    elif hasattr(res0, "names") and isinstance(res0.names, dict) and cls_id in res0.names:
                        cls_name = str(res0.names[cls_id])
                    else:
                        cls_name = f"class_{cls_id}"

                    counts[cls_name] = counts.get(cls_name, 0) + 1
                    if conf > top_conf:
                        top_conf = conf
                        primary_class = cls_name

                    xyxy = b.xyxy[0].tolist() if hasattr(b, "xyxy") and len(b.xyxy) > 0 else [0, 0, 0, 0]
                    xywhn = b.xywhn[0].tolist() if hasattr(b, "xywhn") and len(b.xywhn) > 0 else [0, 0, 0, 0]

                    boxes_out.append({
                        "box_xyxy": [round(float(v), 1) for v in xyxy],
                        "box_xywhn": [round(float(v), 4) for v in xywhn],
                        "confidence": round(conf, 4),
                        "class_id": cls_id,
                        "class_name": cls_name,
                    })

        total_dets = len(boxes_out)
        if total_dets == 0:
            primary_class = "Sin detecciones"
            top_conf = 0.0

        prob_dict: dict[str, float] = {}
        if total_dets > 0:
            for c in classes:
                prob_dict[c] = round(counts.get(c, 0) / float(total_dets), 4)
        else:
            for c in classes:
                prob_dict[c] = 0.0

        return {
            "task": "detection",
            "predicted_class": primary_class,
            "confidence": round(top_conf, 4),
            "probabilities": prob_dict,
            "filename": filename,
            "model_name": actual_model_name,
            "threshold_applied": f"yolo_conf_{conf_threshold}",
            "total_detections": total_dets,
            "detection_counts": counts,
            "detections": boxes_out,
            "image_width": orig_w,
            "image_height": orig_h,
            "ood_flag": False,
            "ood_score": 0.0,
            "energy": 0.0,
            "free_energy": 0.0,
            "energy_score": 0.0,
            "entropy": 0.0,
            "normalized_entropy": 0.0,
            "ood_threshold": 0.0,
        }

    def resolve_and_reload_model(self, force: bool = False) -> None:
        """Resolve active model weights from disk or catalog."""
        if self.active_explicitly_cleared and not force:
            return
        if force:
            self.active_explicitly_cleared = False

        cm = self._get_catalog_manager()
        active_entries = cm.load_catalog(include_deleted=False)
        operational = [e for e in active_entries if e.active_checkpoint]
        if active_entries and not operational and not force:
            self.active_model_p = None
            self.active_explicitly_cleared = True
            return

        candidate_p = Path(self.model_path)
        if not candidate_p.is_file():
            base_cand = Path("models/base_model.pt")
            if base_cand.is_file():
                candidate_p = base_cand
            else:
                return
        else:
            if candidate_p == Path("models/base_model.pt") and Path("models/global_model.pt").is_file():
                candidate_p = Path("models/global_model.pt")

        current_mtime = candidate_p.stat().st_mtime
        if not force and self.active_model_p == candidate_p and current_mtime == self.last_mtime:
            return

        if self.is_detection_model(candidate_p):
            meta = self._load_metadata_for_path(candidate_p)
            self.classes = meta.get("classes", [])
            self.arch = meta.get("architecture", "yolov8n")
            self.num_classes = len(self.classes)
            self.active_model_p = candidate_p
            self.last_mtime = current_mtime
            self.task = "detection"
            sha = hashlib.sha256()
            try:
                with candidate_p.open("rb") as handle:
                    while chunk := handle.read(65536):
                        sha.update(chunk)
                self.model_sha256 = sha.hexdigest()
            except OSError:
                self.model_sha256 = ""
            return

        state, b_meta = load_checkpoint_bundle(candidate_p, device=self.device)
        sha = hashlib.sha256()
        try:
            with candidate_p.open("rb") as handle:
                while chunk := handle.read(65536):
                    sha.update(chunk)
            self.model_sha256 = sha.hexdigest()
        except OSError as exc:
            logger.warning("Could not compute hash for checkpoint '%s': %s", candidate_p, exc)
            self.model_sha256 = ""

        meta: dict[str, Any] = {}
        if self.meta_p.is_file():
            try:
                meta = json.loads(self.meta_p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read metadata '%s': %s", self.meta_p, exc)

        self.classes = b_meta.get("classes") or meta.get("classes") or []
        self.arch = (
            self.architecture
            or b_meta.get("architecture")
            or meta.get("architecture")
            or "resnet18"
        )
        self.num_classes = (
            b_meta.get("num_classes")
            or (len(self.classes) if self.classes else None)
            or meta.get("num_classes")
            or 9
        )
        self.calibrated_thresholds = b_meta.get("calibrated_thresholds") or meta.get("calibrated_thresholds")
        self.temperature = float(b_meta.get("temperature") or meta.get("temperature") or 1.0)

        self.net = build_model(architecture=self.arch, num_classes=self.num_classes).to(self.device)
        if state:
            self.net.load_state_dict(state)
            self.net.eval()

        if self.app:
            self.app.state.net = self.net

        raw_img_size = b_meta.get("img_size") or meta.get("img_size") or 224
        if isinstance(raw_img_size, int):
            self.img_size_tuple = (raw_img_size, raw_img_size)
        elif isinstance(raw_img_size, (list, tuple)) and len(raw_img_size) == 2:
            self.img_size_tuple = (int(raw_img_size[0]), int(raw_img_size[1]))
        else:
            self.img_size_tuple = (224, 224)

        resize_dim = (
            int(round(self.img_size_tuple[0] * 256 / 224)),
            int(round(self.img_size_tuple[1] * 256 / 224)),
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize(resize_dim),
                transforms.CenterCrop(self.img_size_tuple),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.active_model_p = candidate_p
        self.last_mtime = current_mtime

    def get_model_for_inference(
        self, target_model_name: str | None = None
    ) -> tuple[torch.nn.Module, list[str], int, Any, float, transforms.Compose, str]:
        """Fetch model, metadata and transforms via LRU cache without mutating global active."""
        self.resolve_and_reload_model()
        override_net = getattr(self.app.state, "net", None) if self.app else None
        active_net = override_net if override_net is not None else self.net

        if not target_model_name:
            if not self.active_model_p or active_net is None:
                raise HTTPException(
                    status_code=503,
                    detail="Modelo temporalmente fuera de servicio: No hay ningún modelo activo en el sistema.",
                )
            return (
                active_net,
                self.classes,
                self.num_classes,
                self.calibrated_thresholds,
                self.temperature,
                self.transform,
                self.active_model_p.name,
            )

        target_checkpoint = target_model_name
        catalog_entry = self._get_catalog_manager().get_by_slug(target_model_name, include_deleted=True)
        if catalog_entry:
            if catalog_entry.is_deleted:
                raise HTTPException(
                    status_code=400,
                    detail=f"El modelo '{target_model_name}' está en la papelera y no admite inferencias.",
                )
            if not catalog_entry.active_checkpoint:
                raise HTTPException(
                    status_code=503,
                    detail=f"Modelo temporalmente fuera de servicio: El modelo '{catalog_entry.title}' no tiene ningún checkpoint activo asignado.",
                )
            target_checkpoint = catalog_entry.active_checkpoint

        safe_name = Path(target_checkpoint).name
        base_dir = self.active_model_p.parent if self.active_model_p else Path("models")
        target_path = base_dir / safe_name
        if not target_path.is_file():
            target_path = Path("models") / safe_name
        if not target_path.is_file():
            if not self.active_model_p:
                raise HTTPException(
                    status_code=503,
                    detail=f"Modelo temporalmente fuera de servicio: Checkpoint '{target_checkpoint}' no disponible en disco y no hay modelo activo por defecto.",
                )
            return (
                active_net,
                self.classes,
                self.num_classes,
                self.calibrated_thresholds,
                self.temperature,
                self.transform,
                self.active_model_p.name,
            )

        resolved_key = str(target_path.resolve())
        mtime = target_path.stat().st_mtime

        if resolved_key in self.model_cache:
            entry = self.model_cache[resolved_key]
            if entry["mtime"] == mtime:
                self.model_cache.move_to_end(resolved_key)
                return (
                    entry["net"],
                    entry["classes"],
                    entry["num_classes"],
                    entry["calibrated_thresholds"],
                    entry["temperature"],
                    entry["transform"],
                    safe_name,
                )

        try:
            state, b_meta = load_checkpoint_bundle(target_path, device=self.device)
            stem_clean = target_path.stem.replace("_base", "")
            for meta_cand in (
                target_path.parent / f"{stem_clean}.meta.json",
                target_path.with_suffix(".meta.json"),
            ):
                if meta_cand.is_file():
                    try:
                        disk_meta = json.loads(meta_cand.read_text(encoding="utf-8"))
                        if isinstance(disk_meta, dict):
                            for k, v in disk_meta.items():
                                if v is not None and b_meta.get(k) is None:
                                    b_meta[k] = v
                    except Exception:
                        pass
                    break

            m_classes = b_meta.get("classes") or self.classes
            m_arch = b_meta.get("architecture") or "resnet18"
            m_num = b_meta.get("num_classes") or len(m_classes) or 9
            m_cal = b_meta.get("calibrated_thresholds")
            m_temp = float(b_meta.get("temperature") or 1.0)

            m_net = build_model(architecture=m_arch, num_classes=m_num).to(self.device)
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

            while len(self.model_cache) >= MAX_LRU_MODELS:
                self.model_cache.popitem(last=False)

            self.model_cache[resolved_key] = {
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
        except Exception as exc:
            logger.warning("Could not load target model '%s', falling back to default: %s", target_path, exc)
            return (
                active_net,
                self.classes,
                self.num_classes,
                self.calibrated_thresholds,
                self.temperature,
                self.transform,
                Path(str(self.active_model_p)).name if self.active_model_p else "default",
            )

    def decode_probs_row(
        self,
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
        """Decode output probabilities into structured class predictions and OoD scores."""
        probs_list: list[float] = probs.tolist()
        pred_idx = int(probs.argmax().item())
        confidence = float(probs_list[pred_idx])

        if infer_cal:
            if infer_num_classes == 2 and any(
                k in infer_cal for k in ("calibrated_youden", "calibrated_f1", "optimal_threshold")
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
            infer_classes[pred_idx] if infer_classes and pred_idx < len(infer_classes) else f"class_{pred_idx}"
        )
        prob_dict = {
            infer_classes[i] if i < len(infer_classes) else f"class_{i}": round(float(probs_list[i]), 4)
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

    def infer_pil_multiview(
        self,
        img: Image.Image,
        infer_net: torch.nn.Module,
        infer_tf: transforms.Compose,
        infer_temp: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run multi-view inference (CenterCrop + SmartROI + Letterbox) on a PIL image."""
        t_std = infer_tf(img)
        views = [t_std]

        roi_img = extract_smart_roi_crop(img)
        if roi_img is not img and roi_img.size != img.size:
            views.append(infer_tf(roi_img))

        w, h = img.size
        if max(w, h) > 0 and (abs(w - h) / float(max(w, h))) >= 0.08:
            target_hw = (int(t_std.shape[2]), int(t_std.shape[1]))
            lb_img = build_aspect_letterbox(img, target_size=target_hw)
            views.append(infer_tf(lb_img))

        batch_tensor = torch.stack(views, dim=0).to(self.device)
        t_val = max(infer_temp, 1e-3)

        with torch.inference_mode():
            logits_batch = infer_net(batch_tensor)
            fused_logits = fuse_multiview_logits(logits_batch, temperature=t_val).cpu()
            probs = F.softmax(fused_logits / t_val, dim=-1).cpu()
        return probs, fused_logits

    def infer_single(
        self,
        img_bytes: bytes,
        filename: str | None,
        model_name: str | None = None,
        ood_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Perform vision inference on raw image bytes."""
        target_check = model_name or (self.active_model_p.name if self.active_model_p else None)
        if self.is_detection_model(target_check):
            yolo_model, yolo_classes, actual_name = self.get_yolo_model_for_inference(target_check)
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
            return self.infer_detection(img, yolo_model, yolo_classes, filename, actual_name)

        (
            infer_net,
            infer_classes,
            infer_num_classes,
            infer_cal,
            infer_temp,
            infer_tf,
            actual_model_name,
        ) = self.get_model_for_inference(model_name)

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

        probs, fused_logits = self.infer_pil_multiview(img, infer_net, infer_tf, infer_temp)

        eff_thresh = ood_threshold
        if eff_thresh is None:
            for v in self.model_cache.values():
                if v.get("net") is infer_net:
                    eff_thresh = v.get("ood_threshold")
                    break

        return self.decode_probs_row(
            probs,
            infer_classes,
            infer_num_classes,
            infer_cal,
            filename,
            actual_model_name,
            logits=fused_logits,
            ood_threshold=eff_thresh,
            ood_temperature=infer_temp if infer_temp > 0 else 1.0,
        )

    def sync_process_batch_zip(
        self,
        zip_in_path: str,
        temp_dir: str,
        target_model: str | None = None,
        ood_threshold: float | None = None,
        save_for_training: bool = False,
        hitl_manager: Any = None,
    ) -> str:
        """Process images from ZIP file in batch, classifying them into categorized folders."""
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
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid ZIP file") from e

        supported_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
        image_files = [
            f
            for f in sorted(os.listdir(input_dir))
            if os.path.splitext(f)[1].lower() in supported_exts
        ]

        if not image_files:
            raise HTTPException(
                status_code=400,
                detail="No supported images found in the uploaded ZIP file (supported: JPG, PNG, WEBP, BMP)",
            )

        records: list[dict[str, Any]] = []

        for filename in image_files:
            img_path = os.path.join(input_dir, filename)
            try:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()

                res = self.infer_single(
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

                if save_for_training and self.app and hasattr(self.app.state, "hitl"):
                    try:
                        active_hitl = self.app.state.hitl
                        chosen_slug = target_model or res.get("model_name", "default")
                        active_hitl.enqueue_sync(
                            image_bytes=img_bytes,
                            filename=filename,
                            model_name=res.get("model_name", "default"),
                            predicted_class=pred_class,
                            confidence=confidence,
                            probabilities=res.get("probabilities", {}),
                            threshold_applied=res.get("threshold_applied", "nominal_argmax"),
                            model_slug=chosen_slug,
                            detections=res.get("detections"),
                        )
                    except Exception as hitl_err:
                        logger.warning("HITL enqueue_sync failed for batch image %s: %s", filename, hitl_err)

                records.append(
                    {
                        "filename": filename,
                        "predicted_class": pred_class,
                        "confidence": confidence,
                        "ood_flag": ood_flag,
                    }
                )
            except Exception as proc_err:
                logger.warning("Error processing batch image '%s': %s", filename, proc_err)
                err_folder = os.path.join(output_dir, "Error_Procesamiento")
                os.makedirs(err_folder, exist_ok=True)
                try:
                    shutil.copy2(img_path, os.path.join(err_folder, filename))
                except OSError as copy_err:
                    logger.warning("Could not copy failed file '%s': %s", filename, copy_err)
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
