"""User Authentication, RBAC & Privacy-Preserving User Management Database.

Provides a lightweight, high-concurrency SQLite (WAL) database (with optional
MariaDB/MySQL compatibility) to manage platform access across three roles:
- 'admin': Full platform & MLOps access (/master, /expert, /). Cannot read user emails,
           but can create users, change roles, and generate new passwords.
- 'expert': Agronomist / Federated participant (/expert, /). Has an isolated per-user
            workspace for class folders, local model retraining, and federation.
- 'user': Agnostic consumer (/). Can browse the catalog and run model inference.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import string
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

VALID_ROLES = ("admin", "expert", "user")

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,63}$")
USERNAME_REGEX = re.compile(r"^[a-zA-Z0-9_.\-]{3,32}$")

PBKDF2_ITERATIONS = 260_000
SESSION_TTL_SECONDS = 7 * 24 * 3600  # 7 days


def validate_email_format(email: str) -> str:
    """Validate and normalize an email address."""
    cleaned = (email or "").strip().lower()
    if not cleaned or len(cleaned) > 254 or not EMAIL_REGEX.match(cleaned):
        raise ValueError("El formato del correo electrónico no es válido.")
    return cleaned


def validate_username_format(username: str) -> str:
    """Validate and normalize a username."""
    cleaned = (username or "").strip().lower()
    if not cleaned or not USERNAME_REGEX.match(cleaned):
        raise ValueError(
            "El nombre de usuario debe tener entre 3 y 32 caracteres (letras, números, guiones o puntos)."
        )
    return cleaned


def validate_full_name(full_name: str) -> str:
    """Sanitize and validate full name."""
    cleaned = re.sub(r"\s+", " ", (full_name or "").strip())
    if len(cleaned) < 2 or len(cleaned) > 120:
        raise ValueError("El nombre completo debe tener entre 2 y 120 caracteres.")
    return cleaned


def validate_password_strength(password: str) -> None:
    """Ensure password meets basic security requirements."""
    if not password or len(password) < 6 or len(password) > 128:
        raise ValueError("La contraseña debe tener al menos 6 caracteres.")


def hash_password(password: str, salt_hex: str | None = None) -> str:
    """Hash a password using PBKDF2-HMAC-SHA256 with a cryptographic random salt."""
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a plaintext password against a stored PBKDF2-HMAC-SHA256 hash in constant time."""
    try:
        parts = stored_hash.split("$")
        if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
            return False
        iterations = int(parts[1])
        salt = bytes.fromhex(parts[2])
        expected_hex = parts[3]
        dk = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            iterations,
        )
        return hmac.compare_digest(dk.hex(), expected_hex)
    except Exception:
        return False


def generate_secure_password(length: int = 12) -> str:
    """Generate a strong human-typeable random password."""
    alphabet = string.ascii_letters + string.digits + "!@#$%&*"
    while True:
        pwd = "".join(secrets.choice(alphabet) for _ in range(max(length, 8)))
        if (
            any(c.islower() for c in pwd)
            and any(c.isupper() for c in pwd)
            and any(c.isdigit() for c in pwd)
        ):
            return pwd


class UserDatabaseManager:
    """Thread-safe SQLite (WAL) user & session repository with privacy controls."""

    def __init__(self, db_path: str | Path = "models/users.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema_and_seed()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_schema_and_seed(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    full_name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('admin', 'expert', 'user')),
                    is_active INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS system_policies (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS banned_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern TEXT NOT NULL UNIQUE,
                    reason TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS login_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identifier TEXT NOT NULL,
                    attempt_time REAL NOT NULL,
                    success INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS expert_saved_versions (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    model_slug TEXT NOT NULL,
                    version_tag TEXT NOT NULL,
                    custom_name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    num_samples INTEGER DEFAULT 0,
                    accuracy REAL DEFAULT 0.0,
                    final_loss REAL DEFAULT 0.0,
                    delta_norm REAL DEFAULT 0.0,
                    base_model_sha256 TEXT NOT NULL,
                    is_pinned INTEGER DEFAULT 0,
                    is_active INTEGER DEFAULT 0,
                    weights_json TEXT NOT NULL,
                    classes_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
                CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
                CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_banned_pattern ON banned_entries(pattern);
                CREATE INDEX IF NOT EXISTS idx_login_attempts_ident_time ON login_attempts(identifier, attempt_time);
                CREATE INDEX IF NOT EXISTS idx_saved_versions_user_slug ON expert_saved_versions(user_id, model_slug);
                """
            )
            # Schema migrations for existing databases
            cols = [r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
            if "status" not in cols:
                conn.execute("ALTER TABLE users ADD COLUMN status TEXT NOT NULL DEFAULT 'active';")

            now_str = time.strftime("%Y-%m-%d %H:%M:%S")
            default_policies = [
                ("require_admin_approval", "true"),
                ("allow_public_registration", "true"),
            ]
            for pkey, pval in default_policies:
                conn.execute(
                    "INSERT OR IGNORE INTO system_policies (key, value, updated_at) VALUES (?, ?, ?)",
                    (pkey, pval, now_str),
                )

            row = conn.execute("SELECT COUNT(*) AS cnt FROM users").fetchone()
            if row and row["cnt"] == 0:
                default_accounts = [
                    (
                        (os.getenv("AGRIFED_ADMIN_USER") or "admin").strip().lower(),
                        "Administrador Principal",
                        (os.getenv("AGRIFED_ADMIN_EMAIL") or "admin@khaos.local").strip().lower(),
                        hash_password(os.getenv("AGRIFED_ADMIN_PASS") or "admin123"),
                        "admin",
                    ),
                    (
                        (os.getenv("AGRIFED_EXPERT_USER") or "experto").strip().lower(),
                        "Experto Agrónomo",
                        (os.getenv("AGRIFED_EXPERT_EMAIL") or "experto@khaos.local").strip().lower(),
                        hash_password(os.getenv("AGRIFED_EXPERT_PASS") or "experto123"),
                        "expert",
                    ),
                    (
                        (os.getenv("AGRIFED_DEFAULT_USER") or "usuario").strip().lower(),
                        "Usuario Agnóstico",
                        (os.getenv("AGRIFED_DEFAULT_EMAIL") or "usuario@khaos.local").strip().lower(),
                        hash_password(os.getenv("AGRIFED_DEFAULT_PASS") or "usuario123"),
                        "user",
                    ),
                ]
                for uname, fname, mail, phash, role in default_accounts:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO users
                        (username, full_name, email, password_hash, role, is_active, status, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, 1, 'active', ?, ?)
                        """,
                        (uname, fname, mail, phash, role, now_str, now_str),
                    )
                conn.commit()

    # ------------------------------------------------------------------
    # Serialization helpers (Strict Email Privacy for Admin vs Self)
    # ------------------------------------------------------------------
    @staticmethod
    def _to_self_dict(row: sqlite3.Row) -> dict[str, Any]:
        """Serialize user for their OWN profile header/modal (includes email)."""
        keys = row.keys()
        status_val = (
            str(row["status"]) if "status" in keys else ("active" if row["is_active"] else "banned")
        )
        return {
            "id": int(row["id"]),
            "username": str(row["username"]),
            "full_name": str(row["full_name"]),
            "email": str(row["email"]),
            "role": str(row["role"]),
            "is_active": bool(row["is_active"]),
            "status": status_val,
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    @staticmethod
    def _to_admin_dict(row: sqlite3.Row) -> dict[str, Any]:
        """Serialize user for Administrator view (NEVER exposes email address)."""
        keys = row.keys()
        status_val = (
            str(row["status"]) if "status" in keys else ("active" if row["is_active"] else "banned")
        )
        return {
            "id": int(row["id"]),
            "username": str(row["username"]),
            "full_name": str(row["full_name"]),
            "role": str(row["role"]),
            "is_active": bool(row["is_active"]),
            "status": status_val,
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    # ------------------------------------------------------------------
    # CRUD Operations
    # ------------------------------------------------------------------
    def create_user(
        self,
        *,
        username: str,
        full_name: str,
        email: str,
        password: str | None = None,
        role: str = "user",
    ) -> tuple[dict[str, Any], str]:
        """Create a new user after validating inputs and checking uniqueness conflicts.

        Returns (admin_safe_user_dict, plaintext_password_used).
        """
        clean_username = validate_username_format(username)
        clean_name = validate_full_name(full_name)
        clean_email = validate_email_format(email)
        clean_role = (role or "user").strip().lower()
        if clean_role not in VALID_ROLES:
            raise ValueError(f"Rol inválido '{role}'. Roles permitidos: {', '.join(VALID_ROLES)}.")

        plain_pwd = password if (password and password.strip()) else generate_secure_password(10)
        validate_password_strength(plain_pwd)
        pwd_hash = hash_password(plain_pwd)
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")

        with self._connect() as conn:
            existing_u = conn.execute(
                "SELECT id FROM users WHERE username = ?", (clean_username,)
            ).fetchone()
            if existing_u:
                raise ValueError(f"El nombre de usuario '{clean_username}' ya está en uso.")

            existing_e = conn.execute(
                "SELECT id FROM users WHERE email = ?", (clean_email,)
            ).fetchone()
            if existing_e:
                raise ValueError("El correo electrónico indicado ya está registrado en el sistema.")

            cur = conn.execute(
                """
                INSERT INTO users (username, full_name, email, password_hash, role, is_active, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 1, 'active', ?, ?)
                """,
                (clean_username, clean_name, clean_email, pwd_hash, clean_role, now_str, now_str),
            )
            user_id = cur.lastrowid
            conn.commit()
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            return self._to_admin_dict(row), plain_pwd

    def list_users_for_admin(self) -> list[dict[str, Any]]:
        """List all users for the Admin panel WITHOUT exposing any email addresses."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, username, full_name, role, is_active, status, created_at, updated_at FROM users ORDER BY id ASC"
            ).fetchall()
            return [self._to_admin_dict(r) for r in rows]

    def get_user_by_id(self, user_id: int, include_email: bool = False) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            if not row:
                return None
            return self._to_self_dict(row) if include_email else self._to_admin_dict(row)

    def get_user_by_username(
        self, username: str, include_email: bool = False
    ) -> dict[str, Any] | None:
        clean_u = (username or "").strip().lower()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE username = ?", (clean_u,)).fetchone()
            if not row:
                return None
            return self._to_self_dict(row) if include_email else self._to_admin_dict(row)

    def is_email_banned(self, email: str) -> bool:
        """Check if an email matches any banned address or wildcard domain pattern (e.g. @spammer.org)."""
        clean_e = (email or "").strip().lower()
        with self._connect() as conn:
            rows = conn.execute("SELECT pattern FROM banned_entries").fetchall()
            for r in rows:
                pat = str(r["pattern"]).strip().lower()
                if pat.startswith("@") and clean_e.endswith(pat):
                    return True
                if pat == clean_e:
                    return True
        return False

    def register_user(
        self,
        *,
        username: str,
        full_name: str,
        email: str,
        password: str,
    ) -> tuple[dict[str, Any], bool, str]:
        """Self-registration of a new user account, governed by active system policies."""
        policies = self.get_policies()
        if not policies.get("allow_public_registration", True):
            raise ValueError(
                "El registro público de nuevas cuentas está deshabilitado por el administrador."
            )

        clean_username = validate_username_format(username)
        clean_name = validate_full_name(full_name)
        clean_email = validate_email_format(email)
        validate_password_strength(password)

        if self.is_email_banned(clean_email):
            raise ValueError(
                "El registro con esta dirección o dominio de correo no está permitido."
            )

        require_approval = bool(policies.get("require_admin_approval", True))
        status = "pending" if require_approval else "active"
        is_active = 0 if require_approval else 1
        role = "user"

        pwd_hash = hash_password(password)
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")

        with self._connect() as conn:
            existing_u = conn.execute(
                "SELECT id FROM users WHERE username = ?", (clean_username,)
            ).fetchone()
            if existing_u:
                raise ValueError(f"El nombre de usuario '{clean_username}' ya está en uso.")

            existing_e = conn.execute(
                "SELECT id FROM users WHERE email = ?", (clean_email,)
            ).fetchone()
            if existing_e:
                raise ValueError(
                    "El correo electrónico indicado ya está registrado en la plataforma."
                )

            cur = conn.execute(
                """
                INSERT INTO users (username, full_name, email, password_hash, role, is_active, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    clean_username,
                    clean_name,
                    clean_email,
                    pwd_hash,
                    role,
                    is_active,
                    status,
                    now_str,
                    now_str,
                ),
            )
            user_id = cur.lastrowid
            conn.commit()
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            user_dict = self._to_self_dict(row)

        if require_approval:
            msg = "Tu solicitud de cuenta ha sido registrada y está pendiente de aprobación por el administrador."
        else:
            msg = "Cuenta creada exitosamente. Ya puedes iniciar sesión."
        return user_dict, require_approval, msg

    def authenticate(
        self, username_or_email: str, password: str
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Verify credentials and account status with brute-force rate-limiting.

        Returns (user_dict, None) on success, or (None, informative_error_message).
        """
        ident = (username_or_email or "").strip().lower()
        if not ident or not password:
            return None, "Introduce tu usuario o correo y contraseña."

        now_ts = time.time()
        with self._connect() as conn:
            # Check recent failed attempts (max 5 failed attempts in 60 seconds)
            cutoff = now_ts - 60.0
            failed_row = conn.execute(
                "SELECT COUNT(*) AS c FROM login_attempts WHERE identifier = ? AND attempt_time > ? AND success = 0",
                (ident, cutoff),
            ).fetchone()
            if failed_row and int(failed_row["c"]) >= 5:
                return (
                    None,
                    "Demasiados intentos fallidos. Por seguridad, espera 1 minuto antes de reintentar.",
                )

            row = conn.execute(
                "SELECT * FROM users WHERE username = ? OR email = ?",
                (ident, ident),
            ).fetchone()
            if not row or not verify_password(password, str(row["password_hash"])):
                conn.execute(
                    "INSERT INTO login_attempts (identifier, attempt_time, success) VALUES (?, ?, 0)",
                    (ident, now_ts),
                )
                conn.commit()
                return None, "Usuario o contraseña incorrectos."

            keys = row.keys()
            status_val = (
                str(row["status"])
                if "status" in keys
                else ("active" if row["is_active"] else "banned")
            )
            if status_val == "pending":
                return (
                    None,
                    "Tu cuenta está pendiente de validación y aprobación por un administrador.",
                )
            if status_val == "banned" or not bool(row["is_active"]):
                return None, "Esta cuenta ha sido suspendida o bloqueada por la administración."
            if status_val == "rejected":
                return None, "Tu solicitud de registro no fue aprobada por la administración."

            # Record success and prune old attempts (> 1 hour)
            conn.execute(
                "INSERT INTO login_attempts (identifier, attempt_time, success) VALUES (?, ?, 1)",
                (ident, now_ts),
            )
            conn.execute("DELETE FROM login_attempts WHERE attempt_time < ?", (now_ts - 3600.0,))
            conn.commit()
            return self._to_self_dict(row), None

    def create_session(self, user_id: int) -> str:
        """Create a signed opaque session token and store its SHA-256 digest."""
        raw_token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        expires_at = time.time() + SESSION_TTL_SECONDS
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE expires_at < ?", (time.time(),))
            conn.execute(
                "INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (token_hash, user_id, now_str, expires_at),
            )
            conn.commit()
        return raw_token

    def get_user_by_session_token(self, raw_token: str | None) -> dict[str, Any] | None:
        """Resolve an active user from a session token."""
        if not raw_token:
            return None
        token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
        now_ts = time.time()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT u.* FROM sessions s
                JOIN users u ON u.id = s.user_id
                WHERE s.token_hash = ? AND s.expires_at > ? AND u.is_active = 1
                """,
                (token_hash, now_ts),
            ).fetchone()
            if not row:
                return None
            return self._to_self_dict(row)

    def revoke_session(self, raw_token: str | None) -> None:
        if not raw_token:
            return
        token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
            conn.commit()

    def update_own_profile(
        self,
        user_id: int,
        *,
        new_username: str | None = None,
        current_password: str | None = None,
        new_password: str | None = None,
        current_token: str | None = None,
    ) -> dict[str, Any]:
        """Allow a logged-in user to change their own username and/or password from the header profile."""
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            if not row:
                raise KeyError("Usuario no encontrado.")

            final_username = str(row["username"])
            if new_username is not None and new_username.strip().lower() != final_username:
                clean_u = validate_username_format(new_username)
                conflict = conn.execute(
                    "SELECT id FROM users WHERE username = ? AND id != ?",
                    (clean_u, user_id),
                ).fetchone()
                if conflict:
                    raise ValueError(f"El nombre de usuario '{clean_u}' ya está ocupado.")
                final_username = clean_u

            final_pwd_hash = str(row["password_hash"])
            pwd_changed = False
            if new_password is not None and new_password.strip() != "":
                if not current_password or not verify_password(current_password, final_pwd_hash):
                    raise ValueError("La contraseña actual introducida es incorrecta.")
                validate_password_strength(new_password)
                final_pwd_hash = hash_password(new_password)
                pwd_changed = True

            conn.execute(
                """
                UPDATE users
                SET username = ?, password_hash = ?, updated_at = ?
                WHERE id = ?
                """,
                (final_username, final_pwd_hash, now_str, user_id),
            )
            # When password is changed, revoke all other active sessions for security
            if pwd_changed:
                if current_token:
                    t_hash = hashlib.sha256(current_token.encode("utf-8")).hexdigest()
                    conn.execute(
                        "DELETE FROM sessions WHERE user_id = ? AND token_hash != ?",
                        (user_id, t_hash),
                    )
                else:
                    conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            conn.commit()
            updated = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            return self._to_self_dict(updated)

    def admin_update_user(
        self,
        user_id: int,
        *,
        full_name: str | None = None,
        role: str | None = None,
        is_active: bool | None = None,
    ) -> dict[str, Any]:
        """Allow an administrator to update a user's full_name, role, or active status (no email access)."""
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            if not row:
                raise KeyError("Usuario no encontrado.")

            final_name = (
                validate_full_name(full_name) if full_name is not None else str(row["full_name"])
            )
            final_role = str(row["role"])
            if role is not None:
                clean_r = role.strip().lower()
                if clean_r not in VALID_ROLES:
                    raise ValueError(f"Rol inválido '{role}'.")
                final_role = clean_r

            final_active = int(bool(is_active)) if is_active is not None else int(row["is_active"])

            # Prevent deactivating or demoting the last active admin
            if str(row["role"]) == "admin" and (final_role != "admin" or final_active == 0):
                admin_cnt = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM users WHERE role = 'admin' AND is_active = 1 AND id != ?",
                    (user_id,),
                ).fetchone()
                if not admin_cnt or int(admin_cnt["cnt"]) == 0:
                    raise ValueError(
                        "No puedes desactivar ni quitar el rol al único administrador activo."
                    )

            conn.execute(
                """
                UPDATE users
                SET full_name = ?, role = ?, is_active = ?, updated_at = ?
                WHERE id = ?
                """,
                (final_name, final_role, final_active, now_str, user_id),
            )
            if final_active == 0:
                conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            conn.commit()
            updated = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            return self._to_admin_dict(updated)

    def admin_reset_password(
        self, user_id: int, custom_password: str | None = None
    ) -> tuple[dict[str, Any], str]:
        """Generate a new password for a user as Administrator without exposing their email."""
        new_pwd = (
            custom_password.strip()
            if (custom_password and custom_password.strip())
            else generate_secure_password(12)
        )
        validate_password_strength(new_pwd)
        new_hash = hash_password(new_pwd)
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")

        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            if not row:
                raise KeyError("Usuario no encontrado.")

            conn.execute(
                "UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?",
                (new_hash, now_str, user_id),
            )
            # Invalidate old sessions so compromised or old sessions are logged out
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            conn.commit()
            updated = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            return self._to_admin_dict(updated), new_pwd

    def admin_delete_user(self, user_id: int) -> dict[str, Any]:
        """Delete a user account (preventing deletion of the last active admin)."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            if not row:
                raise KeyError("Usuario no encontrado.")

            if str(row["role"]) == "admin":
                admin_cnt = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM users WHERE role = 'admin' AND is_active = 1 AND id != ?",
                    (user_id,),
                ).fetchone()
                if not admin_cnt or int(admin_cnt["cnt"]) == 0:
                    raise ValueError(
                        "No se puede eliminar al único administrador activo del sistema."
                    )

            user_info = self._to_admin_dict(row)
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
            return user_info

    # ------------------------------------------------------------------
    # Account Approvals, Banning & Policies Management (/accounts)
    # ------------------------------------------------------------------
    def list_pending_users(self) -> list[dict[str, Any]]:
        """List users whose account registrations are pending approval."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, username, full_name, email, role, status, created_at FROM users WHERE status = 'pending' ORDER BY id ASC"
            ).fetchall()
            res = []
            for r in rows:
                em = str(r["email"])
                domain = em.split("@")[-1] if "@" in em else ""
                res.append(
                    {
                        "id": int(r["id"]),
                        "username": str(r["username"]),
                        "full_name": str(r["full_name"]),
                        "email_domain": domain,
                        "role": str(r["role"]),
                        "status": str(r["status"]),
                        "created_at": str(r["created_at"]),
                    }
                )
            return res

    def approve_user(self, user_id: int, assigned_role: str = "user") -> dict[str, Any]:
        """Approve a pending account registration and assign a role."""
        clean_role = (assigned_role or "user").strip().lower()
        if clean_role not in VALID_ROLES:
            raise ValueError(f"Rol inválido '{assigned_role}'.")
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            if not row:
                raise KeyError("Usuario no encontrado.")
            conn.execute(
                "UPDATE users SET status = 'active', is_active = 1, role = ?, updated_at = ? WHERE id = ?",
                (clean_role, now_str, user_id),
            )
            conn.commit()
            updated = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            return self._to_admin_dict(updated)

    def reject_user(self, user_id: int) -> dict[str, Any]:
        """Reject a pending account registration."""
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            if not row:
                raise KeyError("Usuario no encontrado.")
            conn.execute(
                "UPDATE users SET status = 'rejected', is_active = 0, updated_at = ? WHERE id = ?",
                (now_str, user_id),
            )
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            conn.commit()
            updated = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            return self._to_admin_dict(updated)

    def ban_user(self, user_id: int, ban_email: bool = True) -> dict[str, Any]:
        """Ban and deactivate a user account, terminating active sessions."""
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            if not row:
                raise KeyError("Usuario no encontrado.")
            if str(row["role"]) == "admin":
                raise ValueError("No se puede banear a un usuario con rol de Administrador.")

            email = str(row["email"]).strip().lower()
            conn.execute(
                "UPDATE users SET status = 'banned', is_active = 0, updated_at = ? WHERE id = ?",
                (now_str, user_id),
            )
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            if ban_email and email:
                conn.execute(
                    "INSERT OR IGNORE INTO banned_entries (pattern, reason, created_at) VALUES (?, ?, ?)",
                    (email, f"Baneado junto con cuenta @{row['username']}", now_str),
                )
            conn.commit()
            updated = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            return self._to_admin_dict(updated)

    def unban_user(self, user_id: int) -> dict[str, Any]:
        """Unban a previously suspended user account."""
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            if not row:
                raise KeyError("Usuario no encontrado.")
            conn.execute(
                "UPDATE users SET status = 'active', is_active = 1, updated_at = ? WHERE id = ?",
                (now_str, user_id),
            )
            email = str(row["email"]).strip().lower()
            if email:
                conn.execute("DELETE FROM banned_entries WHERE pattern = ?", (email,))
            conn.commit()
            updated = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            return self._to_admin_dict(updated)

    def get_policies(self) -> dict[str, Any]:
        """Fetch registration and account validation system policies."""
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM system_policies").fetchall()
            result = {
                "require_admin_approval": True,
                "allow_public_registration": True,
            }
            for r in rows:
                k = str(r["key"])
                v = str(r["value"]).strip().lower()
                result[k] = v in ("1", "true", "yes", "on")
            return result

    def set_policy(self, key: str, value: bool) -> None:
        """Update a system policy flag."""
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        val_str = "true" if value else "false"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO system_policies (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, val_str, now_str),
            )
            conn.commit()

    def list_banned_entries(self) -> list[dict[str, Any]]:
        """List all explicitly banned email addresses and domain wildcard patterns."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, pattern, reason, created_at FROM banned_entries ORDER BY id DESC"
            ).fetchall()
            return [
                {
                    "id": int(r["id"]),
                    "pattern": str(r["pattern"]),
                    "reason": str(r["reason"] or ""),
                    "created_at": str(r["created_at"]),
                }
                for r in rows
            ]

    def add_banned_entry(self, pattern: str, reason: str = "") -> dict[str, Any]:
        """Add a new banned email or wildcard domain (e.g. '@tempmail.com')."""
        clean_p = (pattern or "").strip().lower()
        if not clean_p or len(clean_p) < 3:
            raise ValueError(
                "El patrón debe tener al menos 3 caracteres (ej. '@tempmail.com' o 'spammer@empresa.es')."
            )
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM banned_entries WHERE pattern = ?", (clean_p,)
            ).fetchone()
            if existing:
                raise ValueError(f"El patrón '{clean_p}' ya se encuentra bloqueado.")
            cur = conn.execute(
                "INSERT INTO banned_entries (pattern, reason, created_at) VALUES (?, ?, ?)",
                (clean_p, reason.strip() or None, now_str),
            )
            conn.commit()
            return {
                "id": cur.lastrowid,
                "pattern": clean_p,
                "reason": reason.strip(),
                "created_at": now_str,
            }

    def remove_banned_entry(self, entry_id: int) -> None:
        """Unban a pattern from the blacklist."""
        with self._connect() as conn:
            conn.execute("DELETE FROM banned_entries WHERE id = ?", (entry_id,))
            conn.commit()

    def save_expert_version(
        self,
        user_id: int,
        model_slug: str,
        custom_name: str,
        base_model_sha256: str,
        weights_dict: dict[str, Any],
        classes_list: list[str],
        version_tag: str | None = None,
        description: str = "",
        num_samples: int = 0,
        accuracy: float = 0.0,
        final_loss: float = 0.0,
        delta_norm: float = 0.0,
        is_pinned: bool = False,
    ) -> dict[str, Any]:
        """Save a frozen expert model snapshot for personal versioning and pinning."""
        now_str = (
            time.strftime("%Y-%m-%d %H:%M:%S")
            + f".{int(time.time() * 1000) % 1000:03d}"
        )
        with self._connect() as conn:
            # Generate next version tag if omitted
            if not version_tag:
                existing_count = conn.execute(
                    "SELECT COUNT(*) AS c FROM expert_saved_versions "
                    "WHERE user_id = ? AND model_slug = ?",
                    (user_id, model_slug),
                ).fetchone()["c"]
                version_tag = f"v1.{existing_count + 1}-local"

            v_id = f"{model_slug}::{version_tag}::{uuid.uuid4().hex[:8]}"

            # Set as active by default, marking others as inactive for this model
            conn.execute(
                "UPDATE expert_saved_versions SET is_active = 0 "
                "WHERE user_id = ? AND model_slug = ?",
                (user_id, model_slug),
            )

            conn.execute(
                """
                INSERT INTO expert_saved_versions (
                    id, user_id, model_slug, version_tag, custom_name, description,
                    num_samples, accuracy, final_loss, delta_norm, base_model_sha256,
                    is_pinned, is_active, weights_json, classes_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                """,
                (
                    v_id,
                    user_id,
                    model_slug,
                    version_tag,
                    custom_name.strip() or f"Versión {version_tag}",
                    description.strip(),
                    int(num_samples),
                    float(accuracy),
                    float(final_loss),
                    float(delta_norm),
                    base_model_sha256,
                    1 if is_pinned else 0,
                    json.dumps(weights_dict),
                    json.dumps(classes_list),
                    now_str,
                ),
            )
            conn.commit()

            return {
                "id": v_id,
                "version_id": v_id,
                "version_tag": version_tag,
                "custom_name": custom_name.strip() or f"Versión {version_tag}",
                "description": description.strip(),
                "num_samples": int(num_samples),
                "accuracy": float(accuracy),
                "final_loss": float(final_loss),
                "delta_norm": float(delta_norm),
                "base_model_sha256": base_model_sha256,
                "is_pinned": bool(is_pinned),
                "is_active": True,
                "classes": classes_list,
                "created_at": now_str,
            }

    def list_expert_versions(self, user_id: int, model_slug: str) -> list[dict[str, Any]]:
        """List all saved versions of a model for the given expert."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, version_tag, custom_name, description, num_samples, accuracy,
                       final_loss, delta_norm, base_model_sha256, is_pinned, is_active,
                       classes_json, created_at
                FROM expert_saved_versions
                WHERE user_id = ? AND model_slug = ?
                ORDER BY created_at DESC, rowid DESC
                """,
                (user_id, model_slug),
            ).fetchall()

            results = []
            for r in rows:
                results.append(
                    {
                        "id": r["id"],
                        "version_id": r["id"],
                        "version_tag": r["version_tag"],
                        "custom_name": r["custom_name"],
                        "description": r["description"] or "",
                        "num_samples": int(r["num_samples"] or 0),
                        "accuracy": round(float(r["accuracy"] or 0.0), 2),
                        "final_loss": round(float(r["final_loss"] or 0.0), 4),
                        "delta_norm": round(float(r["delta_norm"] or 0.0), 4),
                        "base_model_sha256": r["base_model_sha256"],
                        "is_pinned": bool(r["is_pinned"]),
                        "is_active": bool(r["is_active"]),
                        "classes": json.loads(r["classes_json"] or "[]"),
                        "created_at": r["created_at"],
                    }
                )
            return results

    def get_expert_version(self, user_id: int, version_id: str) -> dict[str, Any] | None:
        """Fetch a specific expert version with its raw weights."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM expert_saved_versions WHERE user_id = ? AND id = ?",
                (user_id, version_id),
            ).fetchone()
            if not row:
                return None
            return {
                "id": row["id"],
                "version_id": row["id"],
                "model_slug": row["model_slug"],
                "version_tag": row["version_tag"],
                "custom_name": row["custom_name"],
                "description": row["description"] or "",
                "num_samples": int(row["num_samples"] or 0),
                "accuracy": float(row["accuracy"] or 0.0),
                "final_loss": float(row["final_loss"] or 0.0),
                "delta_norm": float(row["delta_norm"] or 0.0),
                "base_model_sha256": row["base_model_sha256"],
                "is_pinned": bool(row["is_pinned"]),
                "is_active": bool(row["is_active"]),
                "head_weights": json.loads(row["weights_json"] or "{}"),
                "classes": json.loads(row["classes_json"] or "[]"),
                "created_at": row["created_at"],
            }

    def delete_expert_version(self, user_id: int, version_id: str) -> bool:
        """Delete a saved version snapshot."""
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM expert_saved_versions WHERE user_id = ? AND id = ?",
                (user_id, version_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def set_pinned_expert_version(self, user_id: int, version_id: str, is_pinned: bool) -> bool:
        """Toggle pinned status for an expert version."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE expert_saved_versions SET is_pinned = ? WHERE user_id = ? AND id = ?",
                (1 if is_pinned else 0, user_id, version_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def set_active_expert_version(self, user_id: int, version_id: str, model_slug: str) -> bool:
        """Mark a saved version as active for this model, deactivating others."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE expert_saved_versions SET is_active = 0 "
                "WHERE user_id = ? AND model_slug = ?",
                (user_id, model_slug),
            )
            cur = conn.execute(
                "UPDATE expert_saved_versions SET is_active = 1 WHERE user_id = ? AND id = ?",
                (user_id, version_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def get_accounts_summary(self) -> dict[str, Any]:
        """Aggregate stats for the accounts administration dashboard."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
            active = conn.execute(
                "SELECT COUNT(*) AS c FROM users WHERE status = 'active' AND is_active = 1"
            ).fetchone()["c"]
            pending = conn.execute(
                "SELECT COUNT(*) AS c FROM users WHERE status = 'pending'"
            ).fetchone()["c"]
            banned = conn.execute(
                "SELECT COUNT(*) AS c FROM users WHERE status = 'banned' OR is_active = 0"
            ).fetchone()["c"]
            banned_patterns = conn.execute("SELECT COUNT(*) AS c FROM banned_entries").fetchone()[
                "c"
            ]
            return {
                "total_users": int(total),
                "active_users": int(active),
                "pending_approvals": int(pending),
                "banned_users": int(banned),
                "banned_patterns": int(banned_patterns),
                "policies": self.get_policies(),
            }
