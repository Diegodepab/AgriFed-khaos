"""Web user interface template loader for KHAOS Federated Model Hub."""

from __future__ import annotations

import json
from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent / "templates"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_model_metadata(slug: str) -> dict:
    """Retrieve catalog metadata for pre-rendering model detail view without layout shifts."""
    catalog_path = PROJECT_ROOT / "models" / "catalog.json"
    if catalog_path.exists():
        try:
            cat = json.loads(catalog_path.read_text(encoding="utf-8"))
            for m in cat.get("models", []):
                if m.get("slug") == slug:
                    return m
        except (json.JSONDecodeError, OSError):
            pass
    meta_path = PROJECT_ROOT / "models" / f"{slug}.meta.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def render_agnostic_client_page() -> str:
    """Return HTML string for GET / (Data Space Model Hub Catalog)."""
    return (TEMPLATES_DIR / "client.html").read_text(encoding="utf-8")


def render_model_consumer_page(slug: str) -> str:
    """Return HTML string for GET /models/{slug} (Dedicated Consumer Model Page)."""
    from agrifed_khaos.api.catalog import _extract_dataset_info

    content = (TEMPLATES_DIR / "model_detail.html").read_text(encoding="utf-8")
    meta = get_model_metadata(slug)
    ds_info = _extract_dataset_info(slug, meta)
    title = meta.get("title") or slug.replace("-", " ").title()
    desc = meta.get("description") or "Catálogo de modelos de visión artificial e inferencia federada KHAOS Hub."
    domain = meta.get("domain") or "General"
    arch = meta.get("architecture") or "resnet18"
    classes_cnt = str(len(meta.get("classes", [])))
    ckpt = meta.get("active_checkpoint")
    ckpt_text = f"pesos: {ckpt}" if ckpt else "fuera de servicio"

    thumb_url = meta.get("thumbnail_url") or ""
    thumb_path = PROJECT_ROOT / "models" / "thumbnails" / f"{slug}.jpg"
    if not thumb_url and thumb_path.exists():
        thumb_url = f"/api/models/{slug}/thumbnail"

    if thumb_url:
        thumb_hero_html = (
            f'<button type="button" onclick="openDatasetThumbModal()" '
            f'class="kh-hero-thumb group" '
            f'aria-label="Ampliar muestra representativa del dataset {title}" '
            f'title="Clic para ampliar muestra representativa">'
            f'<img id="modelHeroThumbnail" src="{thumb_url}" width="190" height="135" alt="Muestra representativa de {title}" '
            f'fetchpriority="high">'
            f'<span class="absolute bottom-2 right-2 px-2 py-0.5 rounded text-[10px] font-medium bg-slate-900/80 text-white leading-none">Muestra</span>'
            f'</button>'
        )
    else:
        thumb_hero_html = (
            '<div id="modelHeroThumbnailWrap" class="hidden"></div>'
        )

    size_str = "-"
    if ckpt:
        ckpt_path = PROJECT_ROOT / "models" / ckpt
        if ckpt_path.exists():
            size_str = f"{(ckpt_path.stat().st_size / (1024 * 1024)):.1f} MB"

    total_imgs = ds_info.get("total_images")
    total_imgs_str = f"{total_imgs:,}".replace(",", ".") + " imágenes" if total_imgs else "Corpus validado"
    acc_pct = ds_info.get("accuracy_pct")
    acc_str = f"{acc_pct:.1f}%" if isinstance(acc_pct, (int, float)) else "-"
    f1_pct = ds_info.get("macro_f1_pct")
    f1_str = f"{f1_pct:.1f}%" if isinstance(f1_pct, (int, float)) else "-"
    res_str = ds_info.get("img_size_str") or f"{ds_info.get('img_size') or 224} × {ds_info.get('img_size') or 224} px"

    is_cal = bool(meta.get("calibrated_thresholds"))
    cal_html = (
        '<svg class="w-3.5 h-3.5 text-emerald-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M20 6 9 17l-5-5"/></svg><span class="text-emerald-800 font-semibold">Calibrado OvR</span>'
        if is_cal
        else 'Nominal Argmax'
    )
    cal_cls = (
        'inline-flex items-center gap-1 px-2.5 py-0.5 rounded-md text-xs bg-emerald-50 border border-emerald-200'
        if is_cal
        else 'inline-flex items-center gap-1 px-2.5 py-0.5 rounded-md text-xs bg-slate-100 text-slate-700 border border-slate-200'
    )

    task = meta.get("task") or ("detection" if arch.lower().startswith("yolo") else "classification")
    is_detection = task == "detection" or arch.lower().startswith("yolo")
    if is_detection:
        task_badge_html = (
            '<span class="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md text-xs font-semibold bg-emerald-50 text-emerald-800 border border-emerald-200/80 uppercase tracking-wider" title="Detección de Objetos">'
            '<svg class="w-3.5 h-3.5 text-emerald-600 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 8V4h4M20 8V4h-4M4 16v4h4M20 16v4h-4"/></svg>'
            'Detección de Objetos'
            '</span>'
        )
        task_notice_html = ""
        task_kind_str = "detection"
    else:
        task_badge_html = (
            '<span class="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md text-xs font-semibold bg-slate-100 text-slate-700 border border-slate-200/80 uppercase tracking-wider" title="Clasificación de Imagen">'
            '<svg class="w-3.5 h-3.5 text-slate-500 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="m21 15-5-5L5 21"/></svg>'
            'Clasificación de Imagen'
            '</span>'
        )
        task_notice_html = ""
        task_kind_str = "classification"

    return (
        content.replace("{{MODEL_SLUG}}", slug)
        .replace("{{MODEL_TITLE}}", title)
        .replace("{{MODEL_DESCRIPTION}}", desc)
        .replace("{{MODEL_DOMAIN}}", domain)
        .replace("{{MODEL_ARCH}}", arch)
        .replace("{{CLASSES_COUNT}}", classes_cnt)
        .replace("{{MODEL_CHECKPOINT}}", ckpt_text)
        .replace("{{MODEL_CAL_BADGE}}", cal_html)
        .replace("{{MODEL_CAL_CLASS}}", cal_cls)
        .replace("{{MODEL_TASK_BADGE}}", task_badge_html)
        .replace("{{MODEL_TASK_NOTICE}}", task_notice_html)
        .replace("{{MODEL_TASK_KIND}}", task_kind_str)
        .replace("{{MODEL_SIZE}}", size_str)
        .replace("{{MODEL_THUMB_HERO}}", thumb_hero_html)
        .replace("{{MODEL_THUMB_URL}}", thumb_url)
        .replace("{{DATASET_TOTAL_IMAGES}}", total_imgs_str)
        .replace("{{DATASET_ACCURACY}}", acc_str)
        .replace("{{DATASET_MACRO_F1}}", f1_str)
        .replace("{{DATASET_RESOLUTION}}", res_str)
    )


def render_expert_page(slug: str | None = None) -> str:
    """Return HTML string for GET /expert or GET /models/{slug}/expert."""
    content = (TEMPLATES_DIR / "expert.html").read_text(encoding="utf-8")
    return content.replace("{{MODEL_SLUG}}", slug or "")


def render_master_admin_page() -> str:
    """Return HTML string for GET /master (Data Space Governance Console)."""
    return (TEMPLATES_DIR / "master.html").read_text(encoding="utf-8")


def render_model_create_page() -> str:
    """Return HTML string for GET /master/models/new (Model Creation Studio)."""
    return (TEMPLATES_DIR / "model_create.html").read_text(encoding="utf-8")


def render_login_page() -> str:
    """Return HTML string for GET /login (Dedicated User Authentication View)."""
    return (TEMPLATES_DIR / "login.html").read_text(encoding="utf-8")


def render_accounts_page() -> str:
    """Return HTML string for GET /accounts (Dedicated Accounts & Policies Management Console)."""
    return (TEMPLATES_DIR / "accounts.html").read_text(encoding="utf-8")


def render_access_denied_page(
    user: dict | None,
    required_roles: list[str],
    next_url: str = "/",
    reason_message: str | None = None,
) -> str:
    """Return customized HTML string for 403 Forbidden Access Denied state."""
    raw = (TEMPLATES_DIR / "access_denied.html").read_text(encoding="utf-8")
    u_label = f"@{user['username']}" if user and user.get("username") else "Invitado / No autenticado"
    role_val = (user.get("role") if user else "anónimo") or "anónimo"

    if role_val == "admin":
        role_label = "Administrador"
        role_cls = "bg-slate-900 text-white"
    elif role_val == "expert":
        role_label = "Experto"
        role_cls = "bg-emerald-100 text-emerald-800 border border-emerald-200"
    elif role_val == "user":
        role_label = "Usuario Agnóstico"
        role_cls = "bg-blue-100 text-blue-800 border border-blue-200"
    else:
        role_label = "Sin Iniciar Sesión"
        role_cls = "bg-slate-100 text-slate-700 border border-slate-200"

    req_labels = []
    for r in required_roles:
        if r == "admin":
            req_labels.append("Administrador")
        elif r == "expert":
            req_labels.append("Experto")
        elif r == "user":
            req_labels.append("Usuario")
        else:
            req_labels.append(r)
    req_str = " o ".join(req_labels)

    if not reason_message:
        if role_val == "user":
            reason_message = (
                "Tu cuenta tiene rol de Usuario Agnóstico (orientado a explorar modelos y ejecutar inferencia). "
                f"Para acceder a esta sección se requiere rol de {req_str}."
            )
        elif role_val == "expert":
            reason_message = (
                f"Tu cuenta tiene rol de Experto. Esta sección requiere permisos de {req_str}."
            )
        else:
            reason_message = (
                f"Debes iniciar sesión con una cuenta que tenga permisos de {req_str}."
            )

    return (
        raw.replace("{{REASON_MESSAGE}}", reason_message)
        .replace("{{CURRENT_USER_LABEL}}", u_label)
        .replace("{{CURRENT_ROLE_LABEL}}", role_label)
        .replace("{{ROLE_BADGE_CLASS}}", role_cls)
        .replace("{{REQUIRED_ROLES_LABEL}}", req_str)
        .replace("{{NEXT_URL}}", next_url)
    )

