"""Authentication, User Profile & Privacy-Preserving Admin User CRUD Endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from agrifed_khaos.users_db import SESSION_TTL_SECONDS, UserDatabaseManager

router = APIRouter(tags=["Auth & Users"])

SESSION_COOKIE_NAME = "khaos_session"


def get_users_db(request: Request) -> UserDatabaseManager:
    """Retrieve or lazily initialize the UserDatabaseManager attached to app.state."""
    db: UserDatabaseManager | None = getattr(request.app.state, "users_db", None)
    if db is None:
        cm = getattr(request.app.state, "catalog", None)
        models_dir = getattr(cm, "models_dir", None) or Path("models")
        db = UserDatabaseManager(db_path=Path(models_dir) / "users.db")
        request.app.state.users_db = db
    return db


def resolve_request_user(request: Request) -> dict[str, Any] | None:
    """Resolve the currently authenticated user from cookie or Authorization Bearer token."""
    db = get_users_db(request)
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer "):
            token = auth_header.split(" ", 1)[1].strip()
    if not token:
        return None
    return db.get_user_by_session_token(token)


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=254)
    password: str = Field(..., min_length=1, max_length=128)


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    full_name: str = Field(..., min_length=2, max_length=120)
    email: str = Field(..., min_length=5, max_length=254)
    password: str = Field(..., min_length=6, max_length=128)


class ProfileUpdateRequest(BaseModel):
    username: str | None = Field(default=None, max_length=32)
    current_password: str | None = Field(default=None, max_length=128)
    new_password: str | None = Field(default=None, max_length=128)


class AdminCreateUserRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    full_name: str = Field(..., min_length=2, max_length=120)
    email: str = Field(..., min_length=5, max_length=254)
    role: str = Field(default="user")
    password: str | None = Field(default=None, max_length=128)


class AdminUpdateUserRequest(BaseModel):
    full_name: str | None = Field(default=None, max_length=120)
    role: str | None = Field(default=None)
    is_active: bool | None = Field(default=None)


class AdminResetPasswordRequest(BaseModel):
    custom_password: str | None = Field(default=None, max_length=128)


class ApproveUserRequest(BaseModel):
    role: str = Field(default="user")


class UpdatePoliciesRequest(BaseModel):
    require_admin_approval: bool | None = None
    allow_public_registration: bool | None = None


class AddBannedPatternRequest(BaseModel):
    pattern: str = Field(..., min_length=3, max_length=254)
    reason: str | None = Field(default=None, max_length=254)


def _require_authenticated_user(request: Request) -> dict[str, Any]:
    user = resolve_request_user(request)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Debes iniciar sesión para realizar esta operación.",
        )
    return user


def _require_admin_user(request: Request) -> dict[str, Any]:
    """Verify the caller is an authenticated admin (or fallback when no session in automated tests if allowed)."""
    user = resolve_request_user(request)
    if user:
        if user.get("role") != "admin":
            raise HTTPException(
                status_code=403,
                detail="Acceso denegado: Se requieren permisos de Administrador.",
            )
        return user
    raise HTTPException(
        status_code=401,
        detail="Debes iniciar sesión como Administrador para gestionar usuarios.",
    )


@router.post("/api/auth/login")
def login(payload: LoginRequest, request: Request, response: Response) -> dict[str, Any]:
    """Authenticate a user and set an HttpOnly session cookie."""
    db = get_users_db(request)
    user, err_msg = db.authenticate(payload.username, payload.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail=err_msg or "Usuario o contraseña incorrectos.",
        )
    token = db.create_session(int(user["id"]))
    is_secure = (
        request.url.scheme == "https"
        or request.headers.get("x-forwarded-proto", "").lower() == "https"
    )
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=is_secure,
        path="/",
    )
    return {
        "authenticated": True,
        "token": token,
        "user": user,
    }


@router.post("/api/auth/register", status_code=201)
def register(payload: RegisterRequest, request: Request) -> dict[str, Any]:
    """Public self-registration for new users, subject to approval policies."""
    db = get_users_db(request)
    try:
        user, requires_approval, message = db.register_user(
            username=payload.username,
            full_name=payload.full_name,
            email=payload.email,
            password=payload.password,
        )
        return {
            "success": True,
            "requires_approval": requires_approval,
            "message": message,
            "user": user,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/auth/logout")
def logout(request: Request, response: Response) -> dict[str, Any]:
    """Revoke current session and clear session cookie."""
    db = get_users_db(request)
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer "):
            token = auth_header.split(" ", 1)[1].strip()
    db.revoke_session(token)
    response.delete_cookie(key=SESSION_COOKIE_NAME, path="/")
    return {"authenticated": False, "message": "Sesión cerrada correctamente."}


@router.get("/api/auth/me")
def get_me(request: Request) -> dict[str, Any]:
    """Return the currently logged-in user's profile (including their own email)."""
    user = resolve_request_user(request)
    if not user:
        return {"authenticated": False, "user": None}
    return {"authenticated": True, "user": user}


@router.patch("/api/auth/profile")
def update_profile(payload: ProfileUpdateRequest, request: Request) -> dict[str, Any]:
    """Allow the logged-in user to change their own username and/or password."""
    user = _require_authenticated_user(request)
    db = get_users_db(request)
    token = request.cookies.get(SESSION_COOKIE_NAME)
    try:
        updated = db.update_own_profile(
            int(user["id"]),
            new_username=payload.username,
            current_password=payload.current_password,
            new_password=payload.new_password,
            current_token=token,
        )
        return {
            "success": True,
            "message": "Perfil actualizado correctamente.",
            "user": updated,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ==============================================================================
# ADMINISTRATOR USER MANAGEMENT (CRUD without Email Exposure)
# ==============================================================================
@router.get("/api/admin/users")
def admin_list_users(request: Request) -> dict[str, Any]:
    """List all users for the Administrator panel. Emails are strictly omitted."""
    _require_admin_user(request)
    db = get_users_db(request)
    users = db.list_users_for_admin()
    return {"users": users, "total": len(users)}


@router.post("/api/admin/users", status_code=201)
def admin_create_user(payload: AdminCreateUserRequest, request: Request) -> dict[str, Any]:
    """Create a new user (Admin, Expert, or Agnostic User) with email conflict validation."""
    _require_admin_user(request)
    db = get_users_db(request)
    try:
        created_user, generated_pwd = db.create_user(
            username=payload.username,
            full_name=payload.full_name,
            email=payload.email,
            password=payload.password,
            role=payload.role,
        )
        return {
            "success": True,
            "message": f"Usuario '{created_user['username']}' creado con rol '{created_user['role']}'.",
            "user": created_user,
            "generated_password": generated_pwd,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch("/api/admin/users/{user_id}")
def admin_update_user_endpoint(
    user_id: int, payload: AdminUpdateUserRequest, request: Request
) -> dict[str, Any]:
    """Update a user's full name, role, or active status."""
    _require_admin_user(request)
    db = get_users_db(request)
    try:
        updated = db.admin_update_user(
            user_id,
            full_name=payload.full_name,
            role=payload.role,
            is_active=payload.is_active,
        )
        return {
            "success": True,
            "message": f"Usuario '{updated['username']}' actualizado.",
            "user": updated,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/api/admin/users/{user_id}/reset-password")
def admin_reset_user_password(
    user_id: int,
    request: Request,
    payload: AdminResetPasswordRequest | None = None,
) -> dict[str, Any]:
    """Generate a new password for a user without exposing their email address."""
    _require_admin_user(request)
    db = get_users_db(request)
    try:
        custom_pwd = payload.custom_password if payload else None
        updated_user, new_password = db.admin_reset_password(user_id, custom_password=custom_pwd)
        return {
            "success": True,
            "message": f"Nueva contraseña generada para '{updated_user['username']}'.",
            "user": updated_user,
            "generated_password": new_password,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/api/admin/users/{user_id}")
def admin_delete_user_endpoint(user_id: int, request: Request) -> dict[str, Any]:
    """Delete a user account."""
    caller = _require_admin_user(request)
    if int(caller["id"]) == int(user_id):
        raise HTTPException(
            status_code=400,
            detail="No puedes eliminar tu propia cuenta de administrador mientras tienes la sesión iniciada.",
        )
    db = get_users_db(request)
    try:
        deleted = db.admin_delete_user(user_id)
        return {
            "success": True,
            "message": f"Usuario '{deleted['username']}' eliminado.",
            "user": deleted,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ==============================================================================
# ADMINISTRATOR ACCOUNTS & POLICIES MANAGEMENT (/accounts)
# ==============================================================================
@router.get("/api/admin/accounts/summary")
def admin_accounts_summary(request: Request) -> dict[str, Any]:
    """Get high-level statistics for the Accounts & Policies dashboard."""
    _require_admin_user(request)
    db = get_users_db(request)
    return db.get_accounts_summary()


@router.get("/api/admin/accounts/pending")
def admin_list_pending_accounts(request: Request) -> dict[str, Any]:
    """List all account registrations waiting for administrator approval."""
    _require_admin_user(request)
    db = get_users_db(request)
    pending = db.list_pending_users()
    return {"pending_users": pending, "count": len(pending)}


@router.post("/api/admin/accounts/pending/{user_id}/approve")
def admin_approve_account(user_id: int, request: Request, payload: ApproveUserRequest | None = None) -> dict[str, Any]:
    """Approve a pending user registration and assign their platform role."""
    _require_admin_user(request)
    db = get_users_db(request)
    assigned_role = payload.role if payload else "user"
    try:
        user = db.approve_user(user_id, assigned_role=assigned_role)
        return {
            "success": True,
            "message": f"Cuenta @{user['username']} aprobada con rol '{user['role']}'.",
            "user": user,
        }
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/admin/accounts/pending/{user_id}/reject")
def admin_reject_account(user_id: int, request: Request) -> dict[str, Any]:
    """Reject a pending user registration."""
    _require_admin_user(request)
    db = get_users_db(request)
    try:
        user = db.reject_user(user_id)
        return {
            "success": True,
            "message": f"Solicitud de cuenta @{user['username']} rechazada.",
            "user": user,
        }
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/admin/accounts/{user_id}/ban")
def admin_ban_account(user_id: int, request: Request) -> dict[str, Any]:
    """Ban and deactivate a user account."""
    _require_admin_user(request)
    db = get_users_db(request)
    try:
        user = db.ban_user(user_id, ban_email=True)
        return {
            "success": True,
            "message": f"Cuenta @{user['username']} suspendida y bloqueada.",
            "user": user,
        }
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/admin/accounts/{user_id}/unban")
def admin_unban_account(user_id: int, request: Request) -> dict[str, Any]:
    """Unban a previously suspended user account."""
    _require_admin_user(request)
    db = get_users_db(request)
    try:
        user = db.unban_user(user_id)
        return {
            "success": True,
            "message": f"Cuenta @{user['username']} reactivada correctamente.",
            "user": user,
        }
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/admin/policies")
def admin_get_policies(request: Request) -> dict[str, Any]:
    """Get system registration and validation policies."""
    _require_admin_user(request)
    db = get_users_db(request)
    return db.get_policies()


@router.post("/api/admin/policies")
def admin_update_policies(payload: UpdatePoliciesRequest, request: Request) -> dict[str, Any]:
    """Update system registration and validation policies."""
    _require_admin_user(request)
    db = get_users_db(request)
    if payload.require_admin_approval is not None:
        db.set_policy("require_admin_approval", payload.require_admin_approval)
    if payload.allow_public_registration is not None:
        db.set_policy("allow_public_registration", payload.allow_public_registration)
    return {
        "success": True,
        "message": "Políticas actualizadas correctamente.",
        "policies": db.get_policies(),
    }


@router.get("/api/admin/banned-entries")
def admin_list_banned_entries(request: Request) -> dict[str, Any]:
    """List banned email patterns and domains."""
    _require_admin_user(request)
    db = get_users_db(request)
    entries = db.list_banned_entries()
    return {"banned_entries": entries, "count": len(entries)}


@router.post("/api/admin/banned-entries", status_code=201)
def admin_add_banned_entry(payload: AddBannedPatternRequest, request: Request) -> dict[str, Any]:
    """Add a banned email or wildcard domain."""
    _require_admin_user(request)
    db = get_users_db(request)
    try:
        entry = db.add_banned_entry(payload.pattern, payload.reason or "")
        return {
            "success": True,
            "message": f"Patrón '{payload.pattern}' bloqueado.",
            "entry": entry,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/api/admin/banned-entries/{entry_id}")
def admin_delete_banned_entry(entry_id: int, request: Request) -> dict[str, Any]:
    """Remove a pattern from the blacklist."""
    _require_admin_user(request)
    db = get_users_db(request)
    db.remove_banned_entry(entry_id)
    return {"success": True, "message": "Patrón desbloqueado correctamente."}


