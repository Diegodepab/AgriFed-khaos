"""Human-In-The-Loop (HITL) active learning and queue management.

Provides persistence, locking, and workflow queues for inference feedback,
partitioned by logical model_slug to prevent dataset contamination across models:
- pending: photos submitted by users with model predictions and metadata
- verified: photos audited, corrected or approved by experts, grouped by class
- rejected: photos discarded as invalid/non-representative
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, Field


def _clean_slug(name: str) -> str:
    s = name.replace(".pt", "").lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_-]+", "-", s)
    return s.strip("-") or "default"


class DetectionBoxItem(BaseModel):
    """Normalized bounding box detection item for HITL persistence."""

    class_name: str
    box_xywhn: list[float]  # [x_center, y_center, width, height] normalized in [0.0, 1.0]
    confidence: float | None = None


class HITLItemMetadata(BaseModel):
    """Metadata recorded for each inference submitted to the HITL queue."""

    item_id: str
    timestamp: float = Field(default_factory=time.time)
    created_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    filename: str | None = None
    image_relpath: str
    model_name: str
    model_slug: str = "default"
    predicted_class: str
    confidence: float
    probabilities: dict[str, float] = Field(default_factory=dict)
    threshold_applied: str = "nominal_argmax"
    status: str = "pending"  # "pending", "verified", "rejected"
    verified_class: str | None = None
    detections: list[DetectionBoxItem] | None = None
    notes: str | None = None
    reviewer: str | None = None
    votes: list[dict[str, Any]] = Field(default_factory=list)


class HITLQueueManager:
    """Manages atomic file persistence and concurrency for partitioned HITL queues."""

    def __init__(self, root_dir: Path | str = "models/hitl_queue") -> None:
        self.root_dir = Path(root_dir)
        self._lock = asyncio.Lock()
        # Default directories for backward compatibility with tests
        self.pending_dir = self.root_dir / "default" / "pending"
        self.verified_dir = self.root_dir / "default" / "verified"
        self.rejected_dir = self.root_dir / "default" / "rejected"
        self._dirs_for("default")

    def _dirs_for(self, model_slug: str = "default") -> tuple[Path, Path, Path]:
        slug = _clean_slug(model_slug)
        p = self.root_dir / slug / "pending"
        v = self.root_dir / slug / "verified"
        r = self.root_dir / slug / "rejected"
        p.mkdir(parents=True, exist_ok=True)
        v.mkdir(parents=True, exist_ok=True)
        r.mkdir(parents=True, exist_ok=True)
        return p, v, r

    def get_verified_dir(self, model_slug: str = "default") -> Path:
        """Return the verified dataset directory partitioned for the specific model."""
        slug = _clean_slug(model_slug)
        v = self.root_dir / slug / "verified"
        v.mkdir(parents=True, exist_ok=True)
        return v

    def enqueue_sync(
        self,
        image_bytes: bytes,
        filename: str | None,
        model_name: str,
        predicted_class: str,
        confidence: float,
        probabilities: dict[str, float] | None = None,
        threshold_applied: str = "nominal_argmax",
        model_slug: str | None = None,
        detections: list[dict[str, Any] | DetectionBoxItem] | None = None,
    ) -> HITLItemMetadata:
        """Synchronously persist an inference result and image into the model's pending queue."""
        resolved_slug = _clean_slug(model_slug or model_name or "default")
        p_dir, _, _ = self._dirs_for(resolved_slug)

        item_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        ext = ".jpg"
        if filename:
            orig_ext = Path(filename).suffix.lower()
            if orig_ext in (".jpg", ".jpeg", ".png", ".webp"):
                ext = orig_ext

        img_filename = f"{item_id}{ext}"
        img_path = p_dir / img_filename
        meta_path = p_dir / f"{item_id}.meta.json"

        parsed_detections: list[DetectionBoxItem] | None = None
        if detections is not None:
            parsed_detections = []
            for d in detections:
                if isinstance(d, DetectionBoxItem):
                    parsed_detections.append(d)
                elif isinstance(d, dict):
                    c_name = str(d.get("class_name") or d.get("name") or "object").strip()
                    xywhn = d.get("box_xywhn") or []
                    if len(xywhn) == 4:
                        norm_box = [round(max(0.0, min(1.0, float(v))), 6) for v in xywhn]
                    else:
                        norm_box = [0.0, 0.0, 0.0, 0.0]
                    conf = float(d["confidence"]) if d.get("confidence") is not None else None
                    parsed_detections.append(
                        DetectionBoxItem(
                            class_name=c_name,
                            box_xywhn=norm_box,
                            confidence=round(conf, 4) if conf is not None else None,
                        )
                    )

        meta = HITLItemMetadata(
            item_id=item_id,
            timestamp=time.time(),
            filename=filename,
            image_relpath=str(img_path.relative_to(self.root_dir)),
            model_name=model_name,
            model_slug=resolved_slug,
            predicted_class=predicted_class,
            confidence=round(confidence, 4),
            probabilities=probabilities or {},
            threshold_applied=threshold_applied,
            status="pending",
            detections=parsed_detections,
        )

        tmp_img = img_path.with_name(f".{img_filename}.{uuid.uuid4().hex}.tmp")
        tmp_meta = meta_path.with_name(f".{item_id}.meta.json.{uuid.uuid4().hex}.tmp")
        try:
            tmp_img.write_bytes(image_bytes)
            tmp_meta.write_text(json.dumps(meta.model_dump(), indent=2), encoding="utf-8")
            os.replace(tmp_img, img_path)
            os.replace(tmp_meta, meta_path)
        finally:
            tmp_img.unlink(missing_ok=True)
            tmp_meta.unlink(missing_ok=True)

        return meta

    async def enqueue(
        self,
        image_bytes: bytes,
        filename: str | None,
        model_name: str,
        predicted_class: str,
        confidence: float,
        probabilities: dict[str, float],
        threshold_applied: str = "nominal_argmax",
        model_slug: str | None = None,
        detections: list[dict[str, Any] | DetectionBoxItem] | None = None,
    ) -> HITLItemMetadata:
        """Atomically persist an inference result and image into the model's pending queue."""
        async with self._lock:
            return self.enqueue_sync(
                image_bytes=image_bytes,
                filename=filename,
                model_name=model_name,
                predicted_class=predicted_class,
                confidence=confidence,
                probabilities=probabilities,
                threshold_applied=threshold_applied,
                model_slug=model_slug,
                detections=detections,
            )

    async def list_pending(
        self,
        sort_by_uncertainty: bool = True,
        model_slug: str | None = None,
        allowed_slugs: set[str] | list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """List pending items, optionally scoped to a model_slug or active allowed_slugs."""
        async with self._lock:
            items: list[dict[str, Any]] = []
            if model_slug:
                resolved = _clean_slug(model_slug)
                if allowed_slugs is not None and resolved not in allowed_slugs:
                    return []
                dirs_to_check = [self.root_dir / resolved / "pending"]
            elif allowed_slugs is not None:
                dirs_to_check = [self.root_dir / _clean_slug(s) / "pending" for s in allowed_slugs]
            else:
                dirs_to_check = list(self.root_dir.glob("*/pending"))
                legacy = self.root_dir / "pending"
                if legacy.is_dir():
                    dirs_to_check.append(legacy)

            for p_dir in dirs_to_check:
                if not p_dir.is_dir():
                    continue
                for meta_path in p_dir.glob("*.meta.json"):
                    try:
                        data = dict(json.loads(meta_path.read_text(encoding="utf-8")))
                        votes = list(data.get("votes", []))

                        # Multi-expert consensus computation
                        verified_votes = [v for v in votes if v.get("decision") == "verified"]
                        rejected_votes = [v for v in votes if v.get("decision") == "rejected"]
                        classes_voted = sorted(
                            list(set(v.get("class_name") for v in verified_votes if v.get("class_name")))
                        )

                        if not votes:
                            consensus_status = "none"
                            consensus_label = "Sin votos"
                        elif rejected_votes and not verified_votes:
                            consensus_status = "reject_consensus"
                            n_rej = len(rejected_votes)
                            consensus_label = f"Consenso Descarte ({n_rej} voto{'s' if n_rej > 1 else ''})"
                        elif len(classes_voted) == 1 and not rejected_votes:
                            consensus_status = "consensus"
                            n_ver = len(verified_votes)
                            consensus_label = f"Consenso: {classes_voted[0]} ({n_ver} voto{'s' if n_ver > 1 else ''})"
                        elif len(classes_voted) > 1 or (verified_votes and rejected_votes):
                            consensus_status = "disputed"
                            desc_parts = [f"{c}" for c in classes_voted]
                            if rejected_votes:
                                desc_parts.append(f"{len(rejected_votes)} descarte(s)")
                            consensus_label = f"En Discusión ({', '.join(desc_parts)})"
                        else:
                            consensus_status = "pending"
                            consensus_label = f"{len(votes)} voto(s)"

                        data["consensus_status"] = consensus_status
                        data["consensus_label"] = consensus_label
                        data["votes_count"] = len(votes)
                        data["votes"] = votes

                        items.append(data)
                    except Exception:
                        continue

            if sort_by_uncertainty:
                items.sort(key=lambda x: (x.get("confidence", 1.0), -x.get("timestamp", 0.0)))
            else:
                items.sort(key=lambda x: -x.get("timestamp", 0.0))

            return items

    def _find_pending_meta(self, item_id: str) -> Path | None:
        """Fast lookup of pending .meta.json without full tree rglob."""
        target_name = f"{item_id}.meta.json"
        legacy_cand = self.root_dir / "pending" / target_name
        if legacy_cand.is_file():
            return legacy_cand
        try:
            with os.scandir(self.root_dir) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        cand = Path(entry.path) / "pending" / target_name
                        if cand.is_file():
                            return cand
        except OSError:
            pass
        for p in self.root_dir.rglob(target_name):
            if "pending" in p.parts:
                return p
        return None

    @staticmethod
    def _count_meta_files(directory: Path) -> int:
        """Count *.meta.json files in a directory using os.scandir without Path allocations."""
        cnt = 0
        try:
            with os.scandir(directory) as it:
                for entry in it:
                    if entry.name.endswith(".meta.json") and entry.is_file(follow_symlinks=False):
                        cnt += 1
        except OSError:
            pass
        return cnt

    def get_image_path(self, item_id: str) -> Path | None:
        """Find the image path for a given item_id across all partitioned queues."""
        img_exts = (".jpg", ".jpeg", ".png", ".webp")
        # Fast-path 1: check pending metadata for exact image_relpath
        pending_meta = self._find_pending_meta(item_id)
        if pending_meta is not None:
            for ext in img_exts:
                cand = pending_meta.parent / f"{item_id}{ext}"
                if cand.is_file():
                    return cand
        # Fast-path 2: check slug pending/rejected/verified directories directly
        try:
            with os.scandir(self.root_dir) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        slug_p = Path(entry.path)
                        for sub in ("pending", "rejected", "verified/images"):
                            sub_dir = slug_p / sub
                            for ext in img_exts:
                                cand = sub_dir / f"{item_id}{ext}"
                                if cand.is_file():
                                    return cand
        except OSError:
            pass
        # Fallback: full rglob (covers verified/<class>/ subdirectories)
        for m in self.root_dir.rglob(f"{item_id}.*"):
            if m.suffix.lower() in img_exts and not m.name.endswith(".meta.json"):
                return m
        return None

    async def verify(
        self,
        item_id: str,
        verified_class: str,
        reviewer: str = "expert",
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Move a pending image to verified/<class> partitioned by model_slug."""
        async with self._lock:
            meta_path = self._find_pending_meta(item_id)

            if not meta_path or not meta_path.is_file():
                raise FileNotFoundError(f"Pending HITL item '{item_id}' not found")

            meta_data = dict(json.loads(meta_path.read_text(encoding="utf-8")))
            slug = _clean_slug(
                meta_data.get("model_slug") or meta_data.get("model_name") or "default"
            )
            _, verified_dir, _ = self._dirs_for(slug)

            img_path = self.get_image_path(item_id)
            if not img_path or not img_path.is_file():
                raise FileNotFoundError(f"Pending image file for '{item_id}' not found")

            safe_class = "".join(
                c for c in verified_class if c.isalnum() or c in ("_", "-")
            ).strip()
            if not safe_class:
                safe_class = "unclassified"

            target_class_dir = verified_dir / safe_class
            target_class_dir.mkdir(parents=True, exist_ok=True)

            target_img = target_class_dir / img_path.name
            target_meta = target_class_dir / f"{item_id}.meta.json"

            meta_data["status"] = "verified"
            meta_data["verified_class"] = safe_class
            meta_data["reviewer"] = reviewer
            meta_data["notes"] = notes
            meta_data["verified_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            meta_data["image_relpath"] = str(target_img.relative_to(self.root_dir))

            shutil.move(str(img_path), str(target_img))
            target_meta.write_text(json.dumps(meta_data, indent=2), encoding="utf-8")
            meta_path.unlink(missing_ok=True)

            return cast(dict[str, Any], meta_data)

    async def verify_detection(
        self,
        item_id: str,
        boxes: list[dict[str, Any] | DetectionBoxItem],
        class_names: list[str] | None = None,
        reviewer: str = "expert",
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Verify an object detection item and write standard YOLO images/, labels/ and dataset.yaml."""
        async with self._lock:
            meta_path = self._find_pending_meta(item_id)

            if not meta_path or not meta_path.is_file():
                raise FileNotFoundError(f"Pending HITL item '{item_id}' not found")

            meta_data = dict(json.loads(meta_path.read_text(encoding="utf-8")))
            slug = _clean_slug(
                meta_data.get("model_slug") or meta_data.get("model_name") or "default"
            )
            _, verified_dir, _ = self._dirs_for(slug)

            img_path = self.get_image_path(item_id)
            if not img_path or not img_path.is_file():
                raise FileNotFoundError(f"Pending image file for '{item_id}' not found")

            v_images = verified_dir / "images"
            v_labels = verified_dir / "labels"
            v_meta = verified_dir / "metadata"
            v_images.mkdir(parents=True, exist_ok=True)
            v_labels.mkdir(parents=True, exist_ok=True)
            v_meta.mkdir(parents=True, exist_ok=True)

            canonical_classes: list[str] = [c for c in class_names] if class_names else []
            if not canonical_classes and meta_data.get("probabilities"):
                canonical_classes = list(meta_data["probabilities"].keys())

            verified_boxes: list[dict[str, Any]] = []
            label_lines: list[str] = []

            for b in boxes:
                if isinstance(b, BaseModel):
                    b_dict = b.model_dump()
                else:
                    b_dict = dict(b)

                c_name = str(b_dict.get("class_name") or b_dict.get("name") or "object").strip()
                xywhn = b_dict.get("box_xywhn") or []
                if len(xywhn) != 4:
                    continue
                try:
                    xc, yc, w, h = [float(v) for v in xywhn]
                except (ValueError, TypeError):
                    continue

                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and w > 0 and h > 0):
                    continue

                xc = max(0.0, min(1.0, round(xc, 6)))
                yc = max(0.0, min(1.0, round(yc, 6)))
                w = max(0.0001, min(1.0, round(w, 6)))
                h = max(0.0001, min(1.0, round(h, 6)))

                if canonical_classes and c_name in canonical_classes:
                    cls_id = canonical_classes.index(c_name)
                elif canonical_classes:
                    canonical_classes.append(c_name)
                    cls_id = len(canonical_classes) - 1
                else:
                    if c_name not in canonical_classes:
                        canonical_classes.append(c_name)
                    cls_id = canonical_classes.index(c_name)

                conf = float(b_dict["confidence"]) if b_dict.get("confidence") is not None else None
                verified_boxes.append({
                    "class_name": c_name,
                    "class_id": cls_id,
                    "box_xywhn": [xc, yc, w, h],
                    "confidence": round(conf, 4) if conf is not None else None,
                })
                label_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

            label_file = v_labels / f"{item_id}.txt"
            label_file.write_text("".join(label_lines), encoding="utf-8")

            target_img = v_images / img_path.name
            shutil.move(str(img_path), str(target_img))

            yaml_lines = [
                f"path: {verified_dir.resolve()}",
                "train: images",
                "val: images",
                "names:",
            ]
            for idx, name in enumerate(canonical_classes):
                yaml_lines.append(f"  {idx}: {name}")
            yaml_content = "\n".join(yaml_lines) + "\n"
            (verified_dir / "dataset.yaml").write_text(yaml_content, encoding="utf-8")
            (verified_dir / "data.yaml").write_text(yaml_content, encoding="utf-8")

            meta_data["status"] = "verified"
            meta_data["task"] = "detection"
            meta_data["verified_boxes"] = verified_boxes
            meta_data["reviewer"] = reviewer
            meta_data["notes"] = notes
            meta_data["verified_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            meta_data["image_relpath"] = str(target_img.relative_to(self.root_dir))
            (v_meta / f"{item_id}.meta.json").write_text(
                json.dumps(meta_data, indent=2), encoding="utf-8"
            )

            meta_path.unlink(missing_ok=True)
            if img_path.exists() and img_path != target_img:
                img_path.unlink(missing_ok=True)

            return cast(dict[str, Any], meta_data)

    async def reject(
        self,
        item_id: str,
        reviewer: str = "expert",
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Move a pending image to rejected/ partitioned by model_slug."""
        async with self._lock:
            meta_path = self._find_pending_meta(item_id)

            if not meta_path or not meta_path.is_file():
                raise FileNotFoundError(f"Pending HITL item '{item_id}' not found")

            meta_data = dict(json.loads(meta_path.read_text(encoding="utf-8")))
            slug = _clean_slug(
                meta_data.get("model_slug") or meta_data.get("model_name") or "default"
            )
            _, _, rejected_dir = self._dirs_for(slug)

            img_path = self.get_image_path(item_id)
            if not img_path or not img_path.is_file():
                raise FileNotFoundError(f"Pending image file for '{item_id}' not found")

            target_img = rejected_dir / img_path.name
            target_meta = rejected_dir / f"{item_id}.meta.json"

            meta_data["status"] = "rejected"
            meta_data["reviewer"] = reviewer
            meta_data["notes"] = reason
            meta_data["rejected_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            meta_data["image_relpath"] = str(target_img.relative_to(self.root_dir))

            shutil.move(str(img_path), str(target_img))
            target_meta.write_text(json.dumps(meta_data, indent=2), encoding="utf-8")
            meta_path.unlink(missing_ok=True)

            return cast(dict[str, Any], meta_data)

    async def record_vote(
        self,
        item_id: str,
        reviewer: str = "expert",
        decision: str = "verified",
        class_name: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Record or update an expert's review vote on a pending item with atomic persistence."""
        async with self._lock:
            meta_path = self._find_pending_meta(item_id)
            if not meta_path or not meta_path.is_file():
                raise FileNotFoundError(f"Pending HITL item '{item_id}' not found")

            meta_data = dict(json.loads(meta_path.read_text(encoding="utf-8")))
            votes = list(meta_data.get("votes", []))

            # Upsert vote for this reviewer
            existing_idx = next(
                (i for i, v in enumerate(votes) if v.get("reviewer") == reviewer), None
            )
            new_vote = {
                "reviewer": reviewer,
                "decision": decision,
                "class_name": class_name or meta_data.get("predicted_class", "unclassified"),
                "notes": notes,
                "voted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            if existing_idx is not None:
                votes[existing_idx] = new_vote
            else:
                votes.append(new_vote)

            meta_data["votes"] = votes

            # Write atomically
            tmp_meta = meta_path.with_name(f".{item_id}.meta.json.{uuid.uuid4().hex}.tmp")
            try:
                tmp_meta.write_text(json.dumps(meta_data, indent=2), encoding="utf-8")
                os.replace(tmp_meta, meta_path)
            finally:
                tmp_meta.unlink(missing_ok=True)

            return cast(dict[str, Any], meta_data)

    async def stats(
        self,
        model_slug: str | None = None,
        allowed_slugs: set[str] | list[str] | None = None,
    ) -> dict[str, Any]:
        """Return counts of items in each queue and per verified class, optionally scoped."""
        async with self._lock:
            if model_slug:
                clean_m = _clean_slug(model_slug)
                slugs = [clean_m] if (allowed_slugs is None or clean_m in allowed_slugs) else []
            elif allowed_slugs is not None:
                slugs = [_clean_slug(s) for s in allowed_slugs]
            else:
                slugs = [
                    p.name
                    for p in self.root_dir.iterdir()
                    if p.is_dir() and (p / "pending").is_dir()
                ]
                if not slugs:
                    slugs = ["default"]

            pending_count = 0
            rejected_count = 0
            total_verified = 0
            verified_negative_count = 0
            verified_by_class: dict[str, int] = {}

            for slug in slugs:
                p_dir, v_dir, r_dir = self._dirs_for(slug)
                pending_count += self._count_meta_files(p_dir)
                rejected_count += self._count_meta_files(r_dir)

                # Check if detection structure exists: v_dir / images
                v_images = v_dir / "images"
                v_labels = v_dir / "labels"
                if v_images.is_dir():
                    det_img_cnt = 0
                    try:
                        with os.scandir(v_images) as img_it:
                            for img_e in img_it:
                                if img_e.is_file(follow_symlinks=False) and Path(img_e.name).suffix.lower() in (
                                    ".jpg", ".jpeg", ".png", ".webp"
                                ):
                                    det_img_cnt += 1
                    except OSError:
                        pass
                    total_verified += det_img_cnt

                    if v_labels.is_dir():
                        yolo_names: dict[int, str] = {}
                        ds_yaml = v_dir / "dataset.yaml"
                        if not ds_yaml.is_file():
                            ds_yaml = v_dir / "data.yaml"
                        if ds_yaml.is_file():
                            try:
                                for line in ds_yaml.read_text(encoding="utf-8").splitlines():
                                    line_s = line.strip()
                                    if ":" in line_s and not line_s.startswith(("path", "train", "val", "names")):
                                        k, v = line_s.split(":", 1)
                                        if k.strip().isdigit():
                                            yolo_names[int(k.strip())] = v.strip().strip("'\"")
                            except Exception:
                                pass

                        try:
                            with os.scandir(v_labels) as lbl_it:
                                for lbl_e in lbl_it:
                                    if lbl_e.is_file(follow_symlinks=False) and lbl_e.name.endswith(".txt"):
                                        st_size = lbl_e.stat().st_size
                                        if st_size == 0:
                                            verified_negative_count += 1
                                        else:
                                            try:
                                                txt_content = Path(lbl_e.path).read_text(encoding="utf-8")
                                                for line in txt_content.strip().splitlines():
                                                    parts = line.strip().split()
                                                    if parts:
                                                        cid = int(parts[0])
                                                        cname = yolo_names.get(cid, f"class_{cid}")
                                                        verified_by_class[cname] = verified_by_class.get(cname, 0) + 1
                                            except Exception:
                                                pass
                        except OSError:
                            pass

                try:
                    with os.scandir(v_dir) as v_it:
                        for class_entry in v_it:
                            if class_entry.is_dir(follow_symlinks=False):
                                if class_entry.name in ("images", "labels", "metadata"):
                                    continue
                                cnt = self._count_meta_files(Path(class_entry.path))
                                verified_by_class[class_entry.name] = (
                                    verified_by_class.get(class_entry.name, 0) + cnt
                                )
                                total_verified += cnt
                except OSError:
                    pass

            # Also check legacy root dirs if they exist and no slug filter applied
            if not model_slug:
                legacy_p = self.root_dir / "pending"
                if legacy_p.is_dir():
                    pending_count += self._count_meta_files(legacy_p)
                legacy_r = self.root_dir / "rejected"
                if legacy_r.is_dir():
                    rejected_count += self._count_meta_files(legacy_r)
                legacy_v = self.root_dir / "verified"
                if legacy_v.is_dir():
                    try:
                        with os.scandir(legacy_v) as lv_it:
                            for c_entry in lv_it:
                                if c_entry.is_dir(follow_symlinks=False):
                                    if c_entry.name in ("images", "labels", "metadata"):
                                        continue
                                    cnt = self._count_meta_files(Path(c_entry.path))
                                    verified_by_class[c_entry.name] = (
                                        verified_by_class.get(c_entry.name, 0) + cnt
                                    )
                                    total_verified += cnt
                    except OSError:
                        pass

            return {
                "pending": pending_count,
                "verified_total": total_verified,
                "verified_negative_count": verified_negative_count,
                "rejected": rejected_count,
                "verified_by_class": verified_by_class,
            }
