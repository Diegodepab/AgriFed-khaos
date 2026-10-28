"""Domain-Agnostic Model Registry & Catalog Manager for KHAOS Data Space Hub."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def slugify(text: str) -> str:
    """Generate a clean URL-safe slug from title or filename."""
    s = text.replace(".pt", "").lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_-]+", "-", s)
    return s.strip("-") or "model"


class ModelVersionRecord(BaseModel):
    """Traceable version record linking checkpoint, lineage, origin, and evaluation."""

    version_id: str
    checkpoint: str
    created_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    parent_version: str | None = None
    # "train_base" | "hitl_curation" | "federated_aggregation" | "imported"
    origin: str = "imported"
    evaluation_report: str | None = None
    status: str = "candidate"  # "candidate" | "active" | "archived" | "validated"
    metrics_summary: dict[str, Any] = Field(default_factory=dict)


class LogicalModelEntry(BaseModel):
    """Metadata entity for a logical service model registered in the Data Space."""

    slug: str
    title: str
    domain: str = "General"
    crop: str = ""
    target_element: str = ""
    purpose: str = ""
    description: str = ""
    input_guide: str = "Imagen en formato JPG, PNG o WEBP con iluminación adecuada."
    active_checkpoint: str | None = None
    versions: list[str] = Field(default_factory=list)
    version_history: list[ModelVersionRecord] = Field(default_factory=list)
    architecture: str = "resnet18"
    num_classes: int = 0
    classes: list[str] = Field(default_factory=list)
    thumbnail_url: str | None = None
    example_images: list[str] = Field(default_factory=list)
    calibrated_thresholds: dict[str, float] | None = None
    temperature: float = 1.0
    updated_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    is_deleted: bool = False
    deleted_at: str | None = None
    purge_after: str | None = None

    def get_resolved_versions(self) -> list[ModelVersionRecord]:
        """Return structured version records, normalizing legacy string versions if needed."""
        if self.version_history:
            return self.version_history
        resolved: list[ModelVersionRecord] = []
        for idx, v_str in enumerate(self.versions):
            is_active = (self.active_checkpoint is not None) and (v_str == self.active_checkpoint)
            resolved.append(
                ModelVersionRecord(
                    version_id=f"v{idx + 1}",
                    checkpoint=v_str,
                    created_at=self.updated_at,
                    parent_version=f"v{idx}" if idx > 0 else None,
                    origin="train_base" if idx == 0 else "federated_aggregation",
                    status="active" if is_active else "archived",
                )
            )
        return resolved


class ModelCatalogManager:
    """Manages catalog.json registry, soft-delete quarantine, and auto-sync with models/."""

    def __init__(
        self,
        catalog_path: Path | str = "models/catalog.json",
        models_dir: Path | str = "models",
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.models_dir = Path(models_dir)
        self._cache_mtime_ns: int | None = None
        self._cache_size: int | None = None
        self._cache_entries: list[LogicalModelEntry] | None = None
        self._last_sync_dir_mtime_ns: int | None = None
        self._last_sync_cat_mtime_ns: int | None = None

    def _invalidate_cache(self) -> None:
        self._cache_mtime_ns = None
        self._cache_size = None
        self._cache_entries = None
        self._last_sync_dir_mtime_ns = None
        self._last_sync_cat_mtime_ns = None

    def load_catalog(self, include_deleted: bool = True) -> list[LogicalModelEntry]:
        """Load registered models from catalog_path with mtime-based in-memory caching."""
        try:
            st = self.catalog_path.stat()
        except OSError:
            self._invalidate_cache()
            return []

        if (
            self._cache_entries is not None
            and self._cache_mtime_ns == st.st_mtime_ns
            and self._cache_size == st.st_size
        ):
            entries = [e.model_copy(deep=True) for e in self._cache_entries]
            if not include_deleted:
                return [e for e in entries if not e.is_deleted]
            return entries

        try:
            data = json.loads(self.catalog_path.read_text(encoding="utf-8"))
            raw_entries = data.get("models", [])
            parsed = [LogicalModelEntry(**item) for item in raw_entries]
            self._cache_mtime_ns = st.st_mtime_ns
            self._cache_size = st.st_size
            self._cache_entries = parsed
            entries = [e.model_copy(deep=True) for e in parsed]
            if not include_deleted:
                entries = [e for e in entries if not e.is_deleted]
            return entries
        except Exception:
            self._invalidate_cache()
            return []

    def save_catalog(self, entries: list[LogicalModelEntry]) -> None:
        """Atomically persist catalog entries to catalog_path."""
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"models": [e.model_dump() for e in entries]}
        content = json.dumps(payload, indent=2, ensure_ascii=False)
        temp_path = self.catalog_path.with_name(f".{self.catalog_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temp_path.write_text(content, encoding="utf-8")
            os.replace(temp_path, self.catalog_path)
            try:
                st = self.catalog_path.stat()
                self._cache_mtime_ns = st.st_mtime_ns
                self._cache_size = st.st_size
                self._cache_entries = [e.model_copy(deep=True) for e in entries]
            except OSError:
                self._invalidate_cache()
        finally:
            temp_path.unlink(missing_ok=True)

    def upsert_entry(self, entry: LogicalModelEntry) -> None:
        """Insert or update a logical model entry in the catalog."""
        entries = self.load_catalog(include_deleted=True)
        updated = False
        for idx, e in enumerate(entries):
            if e.slug == entry.slug:
                entries[idx] = entry
                updated = True
                break
        if not updated:
            entries.append(entry)
        self.save_catalog(entries)

    def sync_with_filesystem(
        self, default_classes: list[str] | None = None
    ) -> list[LogicalModelEntry]:
        """Auto-sync catalog with valid inference .pt bundles in models/ directory."""
        try:
            dir_mtime_ns = self.models_dir.stat().st_mtime_ns if self.models_dir.is_dir() else 0
            cat_mtime_ns = (
                self.catalog_path.stat().st_mtime_ns if self.catalog_path.is_file() else 0
            )
        except OSError:
            dir_mtime_ns = -1
            cat_mtime_ns = -1

        if (
            dir_mtime_ns > 0
            and cat_mtime_ns > 0
            and self._last_sync_dir_mtime_ns == dir_mtime_ns
            and self._last_sync_cat_mtime_ns == cat_mtime_ns
        ):
            cached = self.load_catalog(include_deleted=True)
            if not any(e.is_deleted and e.purge_after for e in cached):
                return cached

        # Clean up any expired trash models
        self.auto_purge_expired()

        existing = self.load_catalog(include_deleted=True)
        by_slug = {e.slug: e for e in existing}
        by_checkpoint = {v: e for e in existing for v in e.versions}
        for e in existing:
            if e.active_checkpoint:
                by_checkpoint[e.active_checkpoint] = e

        # Scan models directory
        pt_files: list[str] = []
        if self.models_dir.is_dir():
            for p in sorted(self.models_dir.glob("*.pt")):
                name = p.name
                if name.endswith("_momentum.pt") or name == "server_momentum.pt":
                    continue
                if name.startswith("client") or "parameters" in name:
                    continue
                if name.endswith("_candidate.pt") or name.endswith(".backup"):
                    continue
                pt_files.append(name)

        changed = False

        # If catalog is empty but global_model.pt or base_model.pt exists, bootstrap initial entry
        if not existing and pt_files:
            preferred = "global_model.pt" if "global_model.pt" in pt_files else pt_files[0]
            initial_entry = LogicalModelEntry(
                slug="patologias-foliares-olivo",
                title="Patologías Foliares del Olivo",
                domain="Agricultura",
                crop="Olivo",
                target_element="Hoja",
                description=(
                    "Clasificación federada de enfermedades y anomalías nutricionales en hojas de "
                    "olivar (Repilo, Antracnosis, Fumagina, Tuberculosis, entre otras)."
                ),
                input_guide=(
                    "Fotografía clara del haz y envés de la hoja afectada con "
                    "iluminación natural difusa."
                ),
                active_checkpoint=preferred,
                versions=[
                    f
                    for f in pt_files
                    if f in ("global_model.pt", "base_model.pt", "global_model_t1.pt")
                ],
                architecture="resnet18",
                num_classes=len(default_classes) if default_classes else 9,
                classes=default_classes
                or [
                    "aculus_olearius",
                    "antracnosis",
                    "cochinilla_tizne",
                    "deficiencia_nutricional",
                    "fumagina",
                    "repilo",
                    "sano",
                    "tuberculosis",
                    "virosis",
                ],
            )
            existing.append(initial_entry)
            by_slug[initial_entry.slug] = initial_entry
            for v in initial_entry.versions:
                by_checkpoint[v] = initial_entry
            by_checkpoint[preferred] = initial_entry
            changed = True

        # Auto-register any orphaned full model .pt files
        for pt_name in pt_files:
            if pt_name not in by_checkpoint:
                new_slug = slugify(pt_name)
                # Deduplicate slug if necessary
                base_slug = new_slug
                counter = 1
                while new_slug in by_slug:
                    new_slug = f"{base_slug}-{counter}"
                    counter += 1

                formatted_title = pt_name.replace(".pt", "").replace("_", " ").title()
                entry = LogicalModelEntry(
                    slug=new_slug,
                    title=formatted_title,
                    domain="General",
                    description=(
                        "Modelo de visión artificial para clasificación o inspección especializada."
                    ),
                    active_checkpoint=pt_name,
                    versions=[pt_name],
                    architecture="resnet18",
                    num_classes=len(default_classes) if default_classes else 9,
                    classes=default_classes or [],
                )
                existing.append(entry)
                by_slug[new_slug] = entry
                by_checkpoint[pt_name] = entry
                changed = True

        if changed:
            self.save_catalog(existing)

        try:
            if self.models_dir.is_dir():
                self._last_sync_dir_mtime_ns = self.models_dir.stat().st_mtime_ns
            if self.catalog_path.is_file():
                self._last_sync_cat_mtime_ns = self.catalog_path.stat().st_mtime_ns
        except OSError:
            pass

        return existing

    def get_by_slug(self, slug: str, include_deleted: bool = False) -> LogicalModelEntry | None:
        """Find model entry by slug or checkpoint name."""
        entries = self.load_catalog(include_deleted=include_deleted)
        for e in entries:
            if e.slug == slug or (e.active_checkpoint and e.active_checkpoint == slug):
                return e
        return None

    def search_and_paginate(
        self,
        q: str | None = None,
        domain: str | None = None,
        page: int = 1,
        page_size: int = 12,
        include_deleted: bool = False,
    ) -> dict[str, Any]:
        """Search, filter by domain, and paginate catalog models."""
        self.sync_with_filesystem()
        entries = self.load_catalog(include_deleted=include_deleted)

        # Collect all unique domains from active models
        domains = sorted({e.domain for e in entries if e.domain})

        filtered = entries
        if domain and domain.lower() not in ("todos", "all", ""):
            filtered = [e for e in filtered if e.domain.lower() == domain.lower()]

        if q and q.strip():
            query = q.strip().lower()
            filtered = [
                e
                for e in filtered
                if query in e.title.lower()
                or query in e.slug.lower()
                or query in e.description.lower()
                or query in e.domain.lower()
                or (e.crop and query in e.crop.lower())
                or (e.target_element and query in e.target_element.lower())
                or (e.purpose and query in e.purpose.lower())
                or any(query in c.lower() for c in e.classes)
            ]

        total = len(filtered)
        try:
            p = max(1, int(page))
        except (TypeError, ValueError):
            p = 1
        try:
            ps = max(1, min(int(page_size), 100))
        except (TypeError, ValueError):
            ps = 12
        total_pages = max(1, (total + ps - 1) // ps)
        start_idx = (p - 1) * ps
        end_idx = start_idx + ps
        page_items = filtered[start_idx:end_idx]

        return {
            "models": [e.model_dump() for e in page_items],
            "total": total,
            "page": p,
            "page_size": ps,
            "total_pages": total_pages,
            "domains": domains,
        }

    def soft_delete_model(self, slug: str, retention_days: int = 7) -> LogicalModelEntry:
        """Soft-delete a model: hide it from public views and place it in the trash quarantine."""
        entries = self.load_catalog(include_deleted=True)
        target: LogicalModelEntry | None = None
        for idx, e in enumerate(entries):
            if e.slug == slug:
                now_ts = time.time()
                purge_ts = now_ts + (retention_days * 86400)
                e.is_deleted = True
                e.deleted_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_ts))
                e.purge_after = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(purge_ts))
                e.updated_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_ts))
                # Gracefully transition active state to None
                e.active_checkpoint = None
                for v in e.version_history:
                    if v.status == "active":
                        v.status = "archived"
                entries[idx] = e
                target = e
                break

        if not target:
            raise KeyError(f"Modelo '{slug}' no encontrado en el registro.")

        self.save_catalog(entries)
        return target

    def archive_model(self, slug: str) -> LogicalModelEntry:
        """Archive a model's active checkpoint, transitioning active state to None."""
        entries = self.load_catalog(include_deleted=True)
        target: LogicalModelEntry | None = None
        for idx, e in enumerate(entries):
            if e.slug == slug:
                now_ts = time.strftime("%Y-%m-%d %H:%M:%S")
                e.active_checkpoint = None
                for v in e.version_history:
                    if v.status == "active":
                        v.status = "archived"
                e.updated_at = now_ts
                entries[idx] = e
                target = e
                break

        if not target:
            raise KeyError(f"Modelo '{slug}' no encontrado en el registro.")

        self.save_catalog(entries)
        return target

    def restore_model(self, slug: str) -> LogicalModelEntry:
        """Restore a soft-deleted model back to active status in the catalog."""
        entries = self.load_catalog(include_deleted=True)
        target: LogicalModelEntry | None = None
        for idx, e in enumerate(entries):
            if e.slug == slug:
                now_ts = time.time()
                e.is_deleted = False
                e.deleted_at = None
                e.purge_after = None
                e.updated_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_ts))
                # Restore active checkpoint if available
                if not e.active_checkpoint and e.versions:
                    e.active_checkpoint = e.versions[-1]
                    for v in e.version_history:
                        if v.checkpoint == e.active_checkpoint:
                            v.status = "active"
                entries[idx] = e
                target = e
                break

        if not target:
            raise KeyError(f"Modelo '{slug}' no encontrado en la papelera.")

        self.save_catalog(entries)
        return target

    def list_trash(self) -> list[dict[str, Any]]:
        """List all models currently in trash quarantine with time remaining and disk size."""
        entries = self.load_catalog(include_deleted=True)
        trash: list[dict[str, Any]] = []
        now_ts = time.time()

        for e in entries:
            if not e.is_deleted:
                continue

            # Compute disk size
            size_bytes = 0
            checkpoints = set(e.versions)
            if e.active_checkpoint:
                checkpoints.add(e.active_checkpoint)
            for v in checkpoints:
                p = self.models_dir / v
                if p.is_file():
                    size_bytes += p.stat().st_size

            # Compute remaining time
            days_left = 0.0
            if e.purge_after:
                try:
                    purge_time = time.mktime(time.strptime(e.purge_after, "%Y-%m-%d %H:%M:%S"))
                    diff_sec = purge_time - now_ts
                    days_left = max(0.0, round(diff_sec / 86400, 1))
                except Exception:
                    days_left = 0.0

            item = e.model_dump()
            item["size_bytes"] = size_bytes
            item["days_remaining"] = days_left
            trash.append(item)

        return trash

    def purge_model(self, slug: str, remove_files: bool = True) -> dict[str, Any]:
        """Permanently remove model from catalog and delete unshared .pt checkpoints."""
        entries = self.load_catalog(include_deleted=True)
        target: LogicalModelEntry | None = None
        survivors: list[LogicalModelEntry] = []

        for e in entries:
            if e.slug == slug:
                target = e
            else:
                survivors.append(e)

        if not target:
            raise KeyError(f"Modelo '{slug}' no encontrado.")

        # Persist catalog without the target
        self.save_catalog(survivors)

        deleted_files: list[str] = []
        freed_bytes = 0

        if remove_files:
            # Checkpoints needed by remaining models (both active and in trash)
            used_checkpoints: set[str] = set()
            for s in survivors:
                if s.active_checkpoint:
                    used_checkpoints.add(s.active_checkpoint)
                used_checkpoints.update(s.versions)

            # Checkpoints belonging to target
            target_checkpoints = set(target.versions)
            if target.active_checkpoint:
                target_checkpoints.add(target.active_checkpoint)

            for cp in target_checkpoints:
                # ONLY delete if no remaining model references this file
                if cp not in used_checkpoints:
                    cp_path = self.models_dir / cp
                    if cp_path.is_file():
                        sz = cp_path.stat().st_size
                        try:
                            cp_path.unlink()
                            deleted_files.append(cp)
                            freed_bytes += sz
                        except OSError:
                            pass

            # Also remove scoped HITL queue if present
            hitl_queue_dir = self.models_dir / "hitl_queue" / slug
            if hitl_queue_dir.is_dir():
                shutil.rmtree(hitl_queue_dir, ignore_errors=True)

        return {
            "slug": slug,
            "title": target.title,
            "deleted_files": deleted_files,
            "freed_bytes": freed_bytes,
        }

    def auto_purge_expired(self) -> list[str]:
        """Automatically purge models whose retention period in trash has expired."""
        entries = self.load_catalog(include_deleted=True)
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        purged: list[str] = []

        for e in entries:
            if e.is_deleted and e.purge_after and e.purge_after <= now_str:
                try:
                    self.purge_model(e.slug, remove_files=True)
                    purged.append(e.slug)
                except Exception:
                    pass

        return purged

    def promote_candidate(
        self,
        slug: str,
        candidate_path: Path | str,
        metrics: dict[str, Any] | None = None,
        eval_report: str | None = None,
    ) -> tuple[bool, str, ModelVersionRecord | None]:
        """Promote candidate checkpoint to production with snapshot backup and version tracking."""
        entries = self.load_catalog(include_deleted=True)
        target: LogicalModelEntry | None = None
        for e in entries:
            if e.slug == slug:
                target = e
                break

        if not target:
            return False, f"Modelo '{slug}' no encontrado en el catálogo.", None

        c_path = Path(candidate_path)
        if not c_path.is_file():
            return False, f"Archivo candidato '{candidate_path}' no encontrado en disco.", None

        # 1. Snapshot backup of current active checkpoint
        current_active = target.active_checkpoint
        if current_active:
            active_p = self.models_dir / current_active
            if active_p.is_file():
                backup_p = self.models_dir / f"{current_active}.backup"
                try:
                    shutil.copy2(active_p, backup_p)
                except Exception as exc:
                    return (
                        False,
                        f"Error creando copia de respaldo del modelo activo: {exc}",
                        None,
                    )

        # 2. Assign traceable version ID
        existing_versions = target.get_resolved_versions()
        new_version_num = len(existing_versions) + 1
        new_v_id = f"v{new_version_num}"
        prefix = Path(current_active).stem if current_active else target.slug
        promoted_filename = f"{prefix}_{new_v_id}.pt"
        promoted_target = self.models_dir / promoted_filename

        try:
            shutil.copy2(c_path, promoted_target)
        except Exception as exc:
            return (
                False,
                f"Error copiando artefacto candidato a checkpoint de versión: {exc}",
                None,
            )

        # If current active was global_model.pt or base_model.pt, also refresh canonical active file
        if current_active and current_active in ("global_model.pt", "base_model.pt"):
            try:
                shutil.copy2(c_path, self.models_dir / current_active)
            except Exception:
                pass

        # 3. Create version record and archive previous active
        now_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        parent_id = existing_versions[-1].version_id if existing_versions else None

        updated_history: list[ModelVersionRecord] = []
        for v in existing_versions:
            v_copy = v.model_copy()
            if v_copy.status == "active":
                v_copy.status = "archived"
            updated_history.append(v_copy)

        new_rec = ModelVersionRecord(
            version_id=new_v_id,
            checkpoint=promoted_filename,
            created_at=now_ts,
            parent_version=parent_id,
            origin="federated_aggregation",
            evaluation_report=eval_report,
            status="active",
            metrics_summary=metrics or {},
        )
        updated_history.append(new_rec)

        target.version_history = updated_history
        target.active_checkpoint = promoted_filename
        if promoted_filename not in target.versions:
            target.versions.append(promoted_filename)
        target.updated_at = now_ts

        self.save_catalog(entries)

        # 4. Observability: Append to models/<slug>_history.json
        acc = None
        f1 = None
        loss_val = None
        if metrics:
            acc = metrics.get("accuracy") or metrics.get("post_aggregation_acc")
            f1 = metrics.get("macro_f1")
            loss_val = metrics.get("loss") or metrics.get("post_aggregation_loss")

        append_model_history(
            models_dir=self.models_dir,
            slug=slug,
            version=new_v_id,
            checkpoint_name=promoted_filename,
            accuracy=acc,
            macro_f1=f1,
            loss=loss_val,
            metrics=metrics,
            eval_report=eval_report,
            metadata={"origin": "federated_aggregation"},
            timestamp=now_ts,
        )

        return True, f"Candidato promovido exitosamente como versión {new_v_id}.", new_rec

    def rollback_to_version(
        self,
        slug: str,
        target_version_id: str | None = None,
    ) -> tuple[bool, str]:
        """Rollback the active checkpoint to a previous version or snapshot."""
        entries = self.load_catalog(include_deleted=True)
        target: LogicalModelEntry | None = None
        for e in entries:
            if e.slug == slug:
                target = e
                break

        if not target:
            return False, f"Modelo '{slug}' no encontrado en el catálogo."

        versions = target.get_resolved_versions()
        if len(versions) < 2:
            # Check if .backup exists
            if target.active_checkpoint:
                backup_p = self.models_dir / f"{target.active_checkpoint}.backup"
                if backup_p.is_file():
                    active_p = self.models_dir / target.active_checkpoint
                    shutil.copy2(backup_p, active_p)
                    return True, f"Modelo restaurado desde respaldo '{backup_p.name}'."
            return False, "No existen versiones previas disponibles para rollback."

        # Find version to rollback to
        target_v: ModelVersionRecord | None = None
        if target_version_id:
            for v in versions:
                if v.version_id == target_version_id:
                    target_v = v
                    break
        else:
            # Find previous active or second to last
            for v in reversed(versions[:-1]):
                if v.status in ("archived", "candidate"):
                    target_v = v
                    break
            if not target_v and len(versions) >= 2:
                target_v = versions[-2]

        if not target_v:
            return False, f"Versión objetivo '{target_version_id}' no encontrada."

        # Verify target checkpoint file exists
        target_chk = self.models_dir / target_v.checkpoint
        if not target_chk.is_file():
            if target.active_checkpoint:
                backup_p = self.models_dir / f"{target.active_checkpoint}.backup"
                if backup_p.is_file():
                    shutil.copy2(backup_p, target_chk)
                else:
                    return (
                        False,
                        f"El archivo de pesos '{target_v.checkpoint}' no existe en disco.",
                    )
            else:
                return (
                    False,
                    f"El archivo de pesos '{target_v.checkpoint}' no existe en disco.",
                )

        # Update statuses
        now_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        updated_history: list[ModelVersionRecord] = []
        for v in versions:
            v_copy = v.model_copy()
            if v_copy.version_id == target_v.version_id:
                v_copy.status = "active"
            elif v_copy.checkpoint == target.active_checkpoint:
                v_copy.status = "archived"
            updated_history.append(v_copy)

        target.version_history = updated_history
        target.active_checkpoint = target_v.checkpoint
        target.updated_at = now_ts
        self.save_catalog(entries)
        return (
            True,
            f"Rollback exitoso a versión {target_v.version_id} ({target_v.checkpoint}).",
        )


def append_model_history(
    models_dir: Path | str,
    slug: str,
    version: str,
    checkpoint_name: str,
    accuracy: float | None = None,
    macro_f1: float | None = None,
    loss: float | None = None,
    metrics: dict[str, Any] | None = None,
    eval_report: str | None = None,
    metadata: dict[str, Any] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Append a traceable version record to models/<slug>_history.json."""
    models_p = Path(models_dir)
    models_p.mkdir(parents=True, exist_ok=True)
    history_file = models_p / f"{slug}_history.json"

    history: list[dict[str, Any]] = []
    if history_file.is_file():
        try:
            loaded = json.loads(history_file.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                history = loaded
        except Exception:
            history = []

    m = metrics or {}
    final_acc = (
        accuracy if accuracy is not None else m.get("accuracy", m.get("post_aggregation_acc"))
    )
    final_f1 = macro_f1 if macro_f1 is not None else m.get("macro_f1")
    final_loss = loss if loss is not None else m.get("loss", m.get("post_aggregation_loss"))

    record = {
        "version": version,
        "timestamp": timestamp or time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_name": checkpoint_name,
        "accuracy": final_acc,
        "macro_f1": final_f1,
        "loss": final_loss,
        "metrics": m,
        "eval_report": eval_report,
        "metadata": metadata or {},
    }

    history.append(record)

    temp_path = history_file.with_name(f".{history_file.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temp_path, history_file)
    finally:
        temp_path.unlink(missing_ok=True)

    return record


def get_model_history(models_dir: Path | str, slug: str) -> list[dict[str, Any]]:
    """Retrieve history records from models/<slug>_history.json."""
    history_file = Path(models_dir) / f"{slug}_history.json"
    if not history_file.is_file():
        return []
    try:
        data = json.loads(history_file.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []

