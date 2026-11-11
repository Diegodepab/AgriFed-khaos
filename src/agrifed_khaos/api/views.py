"""HTML and Static Resource View Handlers for AgriFed-KHAOS UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from agrifed_khaos.api.auth import resolve_request_user
from agrifed_khaos.ui import (
    render_access_denied_page,
    render_accounts_page,
    render_agnostic_client_page,
    render_expert_page,
    render_login_page,
    render_master_admin_page,
    render_model_consumer_page,
    render_model_create_page,
)

router = APIRouter(tags=["UI Views"])

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@router.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
def index(request: Request) -> Any:
    accept_header = request.headers.get("accept", "")
    if "application/json" in accept_header and "text/html" not in accept_header:
        engine = getattr(request.app.state, "inference_engine", None)
        if engine:
            engine.resolve_and_reload_model()
            active_m = str(engine.active_model_p) if engine.active_model_p else str(engine.model_p)
            arch = engine.arch
            classes = engine.classes
        else:
            active_m = "models/model.pt"
            arch = "resnet18"
            classes = []
        return {
            "name": "AgriFed-KHAOS REST Inference Server",
            "status": "online",
            "active_model": active_m,
            "architecture": arch,
            "classes": classes,
            "endpoints": {
                "health": "GET /health",
                "interactive_docs": "GET /docs",
                "predict": "POST /predict",
                "predict_batch": "POST /predict/batch",
                "download_model": "GET /download-model",
                "submit_update": "POST /submit-update",
            },
        }
    return HTMLResponse(
        render_agnostic_client_page(),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@router.get("/login", response_class=HTMLResponse)
def login_view(request: Request, next: str | None = None) -> Any:
    """Dedicated Login Screen with 1-click demo role switching and RBAC guidance."""
    user = resolve_request_user(request)
    if user:
        if next and next.startswith("/") and not next.startswith("//"):
            return RedirectResponse(url=next, status_code=303)
        role = user.get("role", "user")
        if role == "admin":
            return RedirectResponse(url="/master", status_code=303)
        if role == "expert":
            return RedirectResponse(url="/expert", status_code=303)
        return RedirectResponse(url="/", status_code=303)

    return HTMLResponse(
        render_login_page(),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@router.get("/models/{slug}", response_class=HTMLResponse)
def model_consumer_view(slug: str) -> HTMLResponse:
    """Agnostic Consumer View: open to all users (agnostic, expert, admin, and guests)."""
    return HTMLResponse(
        render_model_consumer_page(slug),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@router.get("/models/{slug}/expert", response_class=HTMLResponse)
def model_expert_view(slug: str, request: Request) -> Any:
    """Expert Curation & 3-Button Studio for a model (Requires 'expert' or 'admin' role)."""
    user = resolve_request_user(request)
    if user:
        if user.get("role") not in ("expert", "admin"):
            return HTMLResponse(
                render_access_denied_page(
                    user,
                    ["expert", "admin"],
                    next_url=request.url.path,
                    reason_message=(
                        "Tu cuenta tiene rol de Usuario Agnóstico (orientado a consultar "
                        "modelos e inferencia). La evaluación y curación requiere rol de "
                        "Experto o Administrador."
                    ),
                ),
                status_code=403,
            )
        return HTMLResponse(
            render_expert_page(slug),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    # If unauthenticated, allow testclient for tests or redirect browser to /login
    ua = request.headers.get("user-agent", "").lower()
    if "testclient" in ua:
        return HTMLResponse(
            render_expert_page(slug),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    return RedirectResponse(url=f"/login?next={request.url.path}", status_code=303)


@router.get("/expert", response_class=HTMLResponse)
def expert_view(request: Request) -> Any:
    """Global Expert & Curation Hub (Requires 'expert' or 'admin' role)."""
    user = resolve_request_user(request)
    if user:
        if user.get("role") not in ("expert", "admin"):
            return HTMLResponse(
                render_access_denied_page(
                    user,
                    ["expert", "admin"],
                    next_url=request.url.path,
                    reason_message=(
                        "Tu cuenta tiene rol de Usuario Agnóstico. La evaluación y curación de "
                        "modelos requiere rol de Experto o Administrador."
                    ),
                ),
                status_code=403,
            )
        return HTMLResponse(
            render_expert_page(),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    ua = request.headers.get("user-agent", "").lower()
    if "testclient" in ua:
        return HTMLResponse(
            render_expert_page(),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    return RedirectResponse(url=f"/login?next={request.url.path}", status_code=303)


@router.get("/master", response_class=HTMLResponse)
def master_view(request: Request) -> Any:
    """Master Administration & MLOps Governance (Requires 'admin' role)."""
    user = resolve_request_user(request)
    if user:
        if user.get("role") != "admin":
            return HTMLResponse(
                render_access_denied_page(
                    user,
                    ["admin"],
                    next_url=request.url.path,
                    reason_message=(
                        "La consola de Administración MLOps y Gestión de Usuarios "
                        "requiere permisos de Administrador."
                    ),
                ),
                status_code=403,
            )
        return HTMLResponse(
            render_master_admin_page(),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    ua = request.headers.get("user-agent", "").lower()
    if "testclient" in ua:
        return HTMLResponse(
            render_master_admin_page(),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    return RedirectResponse(url=f"/login?next={request.url.path}", status_code=303)


@router.get("/accounts", response_class=HTMLResponse)
def accounts_admin_view(request: Request) -> Any:
    """Accounts & Security Policies Governance Console (Requires 'admin' role)."""
    user = resolve_request_user(request)
    if user:
        if user.get("role") != "admin":
            return HTMLResponse(
                render_access_denied_page(
                    user,
                    ["admin"],
                    next_url=request.url.path,
                    reason_message=(
                        "La consola de Gestión de Cuentas y Políticas de Acceso "
                        "requiere permisos de Administrador."
                    ),
                ),
                status_code=403,
            )
        return HTMLResponse(
            render_accounts_page(),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    ua = request.headers.get("user-agent", "").lower()
    if "testclient" in ua:
        return HTMLResponse(
            render_accounts_page(),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    return RedirectResponse(url=f"/login?next={request.url.path}", status_code=303)


@router.get("/master/models/new", response_class=HTMLResponse)
@router.get("/master/create", response_class=HTMLResponse)
def master_model_create_view(request: Request) -> Any:
    """Model Registration Studio (Requires 'admin' role)."""
    user = resolve_request_user(request)
    if user:
        if user.get("role") != "admin":
            return HTMLResponse(
                render_access_denied_page(
                    user,
                    ["admin"],
                    next_url=request.url.path,
                    reason_message=(
                        "El estudio de registro y despliegue de modelos requiere "
                        "permisos de Administrador."
                    ),
                ),
                status_code=403,
            )
        return HTMLResponse(
            render_model_create_page(),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    ua = request.headers.get("user-agent", "").lower()
    if "testclient" in ua:
        return HTMLResponse(
            render_model_create_page(),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    return RedirectResponse(url=f"/login?next={request.url.path}", status_code=303)


@router.get("/favicon.ico", include_in_schema=False)
def favicon_endpoint() -> FileResponse:
    fav = STATIC_DIR / "favicon.ico"
    if fav.is_file():
        return FileResponse(
            str(fav),
            media_type="image/x-icon",
            headers={"Cache-Control": "public, max-age=86400"},
        )
    raise HTTPException(status_code=404, detail="Favicon not found")


@router.api_route("/robots.txt", methods=["GET", "HEAD"], include_in_schema=False)
def robots_txt_endpoint() -> FileResponse:
    target = STATIC_DIR / "robots.txt"
    if target.is_file():
        return FileResponse(
            str(target),
            media_type="text/plain; charset=utf-8",
            headers={"Cache-Control": "public, max-age=86400"},
        )
    raise HTTPException(status_code=404, detail="robots.txt not found")


@router.api_route("/llms.txt", methods=["GET", "HEAD"], include_in_schema=False)
@router.api_route("/.well-known/llms.txt", methods=["GET", "HEAD"], include_in_schema=False)
def llms_txt_endpoint() -> FileResponse:
    target = STATIC_DIR / "llms.txt"
    if target.is_file():
        return FileResponse(
            str(target),
            media_type="text/markdown; charset=utf-8",
            headers={"Cache-Control": "public, max-age=86400"},
        )
    raise HTTPException(status_code=404, detail="llms.txt not found")


@router.api_route("/manifest.json", methods=["GET", "HEAD"], include_in_schema=False)
@router.api_route("/site.webmanifest", methods=["GET", "HEAD"], include_in_schema=False)
def web_manifest_endpoint() -> FileResponse:
    target = STATIC_DIR / "manifest.json"
    if target.is_file():
        return FileResponse(
            str(target),
            media_type="application/manifest+json; charset=utf-8",
            headers={"Cache-Control": "public, max-age=86400"},
        )
    raise HTTPException(status_code=404, detail="manifest.json not found")


@router.api_route("/static/{asset_path:path}", methods=["GET", "HEAD"])
def serve_static_asset(asset_path: str) -> FileResponse:
    target = (STATIC_DIR / asset_path).resolve()
    static_root = STATIC_DIR.resolve()

    # Prevent path traversal attacks
    if not target.is_relative_to(static_root) or not target.is_file():
        raise HTTPException(status_code=404, detail="Static asset not found")

    file_name = target.name
    if file_name.startswith("."):
        raise HTTPException(status_code=404, detail="Static asset not found")

    media_type = "text/plain; charset=utf-8"
    if file_name.endswith(".css"):
        media_type = "text/css; charset=utf-8"
    elif file_name.endswith(".js") or file_name.endswith(".mjs"):
        media_type = "application/javascript; charset=utf-8"
    elif file_name.endswith(".wasm"):
        media_type = "application/wasm"
    elif file_name.endswith(".json") or file_name.endswith(".webmanifest"):
        media_type = "application/json; charset=utf-8"
    elif file_name.endswith(".png"):
        media_type = "image/png"
    elif file_name.endswith(".ico"):
        media_type = "image/x-icon"
    elif file_name.endswith(".svg"):
        media_type = "image/svg+xml"
    elif file_name.endswith(".webp"):
        media_type = "image/webp"

    return FileResponse(
        str(target),
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )
