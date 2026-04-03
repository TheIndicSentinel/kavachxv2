"""
KavachX Authentication & RBAC
DB-backed users, bcrypt passwords, JWT bearer tokens.
"""
import hashlib
import hmac
import re
import secrets
import time
import bcrypt
import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.database import get_db

# We use the native bcrypt library directly to avoid passlib 1.7.4 incompatibility with bcrypt 4.0+.
# This prevents AttributeErrors (about __about__) and strictly handles the 72-byte limit.
bearer_scheme = HTTPBearer(auto_error=False)

# ── RBAC roles ────────────────────────────────────────────────────────────────
ROLES = {
    "super_admin": {
        "label": "Super Admin",
        "permissions": ["*"],
        "description": "Full platform access including user management",
    },
    "compliance_officer": {
        "label": "Compliance Officer",
        "permissions": [
            "dashboard:read", "policies:read", "policies:write", "policies:delete",
            "audit:read", "audit:export", "models:read", "governance:read",
            "reports:read", "reports:generate", "alerts:read", "alerts:manage",
        ],
        "description": "Manages policies, compliance reports, audit logs",
    },
    "ml_engineer": {
        "label": "ML Engineer",
        "permissions": [
            "dashboard:read", "models:read", "models:write", "models:register",
            "governance:read", "governance:evaluate", "audit:read",
            "policies:read", "simulate:run", "alerts:read",
        ],
        "description": "Registers models, runs evaluations and simulations",
    },
    "executive": {
        "label": "Executive",
        "permissions": [
            "dashboard:read", "reports:read", "audit:read",
            "models:read", "policies:read", "alerts:read",
        ],
        "description": "Read-only executive overview and reports",
    },
    "auditor": {
        "label": "External Auditor",
        "permissions": [
            "audit:read", "audit:export", "policies:read",
            "models:read", "reports:read", "dashboard:read",
        ],
        "description": "Read-only audit access for external auditors",
    },
}

# ── Password policy ───────────────────────────────────────────────────────────
_PW_MIN_LEN  = 12
_PW_PATTERNS = [
    (r"[A-Z]",                  "at least one uppercase letter"),
    (r"[a-z]",                  "at least one lowercase letter"),
    (r"\d",                     "at least one digit"),
    (r'[!@#$%^&*()\-_=+\[\]{}|;:\'",.<>?/`~\\]', "at least one special character"),
]


def validate_password(password: str) -> list:
    """Returns a list of human-readable policy violations (empty = valid)."""
    errors = []
    if len(password) < _PW_MIN_LEN:
        errors.append(f"Minimum {_PW_MIN_LEN} characters")
    for pattern, msg in _PW_PATTERNS:
        if not re.search(pattern, password):
            errors.append(msg)
    return errors


# ── Password helpers ──────────────────────────────────────────────────────────
def hash_password(plain: str) -> str:
    # bcrypt limit is 72 bytes; truncate to avoid ValueErrors in newer libraries
    pw_bytes = plain[:72].encode("utf-8")
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pw_bytes, salt).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        pw_bytes = plain[:72].encode("utf-8")
        return bcrypt.checkpw(pw_bytes, hashed.encode("utf-8"))
    except Exception as _e:
        logger.error("Bcrypt verify error: %s", _e)
        return False


# ── JWT helpers ───────────────────────────────────────────────────────────────
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode["exp"] = expire
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")


# ── Rate limiting (in-process; use Redis in multi-process production) ─────────
_FAILED_LOGINS: dict = {}
_MAX_FAILURES  = 10
_LOCKOUT_WINDOW = 60   # seconds


def _check_rate_limit(identifier: str) -> None:
    now      = time.time()
    attempts = [t for t in _FAILED_LOGINS.get(identifier, []) if now - t < _LOCKOUT_WINDOW]
    _FAILED_LOGINS[identifier] = attempts
    if len(attempts) >= _MAX_FAILURES:
        raise HTTPException(
            status_code=429,
            detail=f"Too many failed attempts. Try again in {_LOCKOUT_WINDOW}s.",
        )


def _record_failure(identifier: str) -> None:
    _FAILED_LOGINS.setdefault(identifier, []).append(time.time())


def _clear_failures(identifier: str) -> None:
    _FAILED_LOGINS.pop(identifier, None)


# ── get_current_user ──────────────────────────────────────────────────────────
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
    db: AsyncSession = Depends(get_db),
) -> dict:
    from app.models.orm_models import User

    # Machine-to-machine via API key
    if x_api_key:
        api_key_val = getattr(settings, "KAVACHX_API_KEY", None)
        if api_key_val and hmac.compare_digest(x_api_key, api_key_val):
            return {
                "id": "system", "name": "API Client",
                "role": "ml_engineer", "email": "api@system",
                "permissions": ROLES["ml_engineer"]["permissions"],
                "is_active": True,
            }
        raise HTTPException(status_code=401, detail="Invalid API key")

    # JWT bearer
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        payload = jwt.decode(credentials.credentials, settings.SECRET_KEY, algorithms=["HS256"])
        email: str = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token payload")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account deactivated")

    role_info = ROLES.get(user.role, {})
    return {
        "id":          user.id,
        "name":        user.name,
        "email":       user.email,
        "role":        user.role,
        "role_label":  role_info.get("label", user.role),
        "permissions": role_info.get("permissions", []),
        "is_active":   user.is_active,
    }


# ── Permission guard ──────────────────────────────────────────────────────────
def require_permission(permission: str):
    async def _check(current_user: dict = Depends(get_current_user)) -> dict:
        perms = current_user.get("permissions", [])
        if "*" in perms or permission in perms:
            return current_user
        raise HTTPException(
            status_code=403,
            detail=f"Role '{current_user.get('role')}' lacks permission '{permission}'",
        )
    return _check


# ── Bootstrap token helpers ───────────────────────────────────────────────────
def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


async def ensure_bootstrap_token(db: AsyncSession) -> None:
    """
    Called at startup.  If no users exist and no valid token is active,
    generate a new bootstrap token, write it to disk (Jenkins-style), and
    print a clickable setup URL to the server log.

    The file path is controlled by the BOOTSTRAP_TOKEN_PATH environment
    variable (default: ./kavachx_setup_token.txt).  On cloud deployments
    where real-time console access is unavailable, operators can retrieve
    the token via the platform file manager or a secure SSH command.

    The file is automatically deleted once the bootstrap form is submitted.
    """
    import os
    from app.models.orm_models import User, BootstrapToken
    from sqlalchemy import func

    user_count = (await db.execute(select(func.count(User.id)))).scalar() or 0
    if user_count > 0:
        return   # Platform already has users — bootstrap not needed

    # Check for an existing valid token
    existing = (await db.execute(
        select(BootstrapToken)
        .where(BootstrapToken.used == False)
        .where(BootstrapToken.expires_at > datetime.utcnow())
    )).scalars().first()

    if existing:
        return   # Valid token already active; no need to regenerate

    token      = secrets.token_urlsafe(32)
    token_hash = _hash_token(token)
    expires    = datetime.utcnow() + timedelta(hours=24)

    db.add(BootstrapToken(token_hash=token_hash, expires_at=expires))
    await db.commit()

    # ── Write token to file (Jenkins / Grafana approach) ──────────────────
    token_path = os.environ.get("BOOTSTRAP_TOKEN_PATH", "./kavachx_setup_token.txt")
    token_abs  = ""
    try:
        with open(token_path, "w") as fh:
            fh.write("KavachX First-Run Bootstrap Token\n")
            fh.write("=" * 40 + "\n")
            fh.write(f"Token   : {token}\n")
            fh.write(f"Expires : {expires.strftime('%Y-%m-%d %H:%M UTC')}\n")
            fh.write("\n")
            fh.write("This file is deleted automatically after the Super Admin\n")
            fh.write("account is created.  Keep it confidential.\n")
        token_abs = os.path.abspath(token_path)
    except Exception as write_err:
        logger.warning("Could not write bootstrap token to file: %s", write_err)

    # ── Print setup banner to console ────────────────────────────────────
    border = "=" * 68
    print(f"\n{border}")
    print("  KAVACHX — FIRST RUN SETUP REQUIRED")
    print(f"")
    print(f"  Bootstrap Token : {token}")
    print(f"  Expires         : {expires.strftime('%Y-%m-%d %H:%M UTC')}")
    if token_abs:
        print(f"  Token file      : {token_abs}")
    print(f"")
    print(f"  Quick-setup URL  (pre-fills the token in the browser):")
    print(f"    /login?setup_token={token}")
    print(f"")
    print(f"  Or open the app, click 'First Run Setup', and paste the token.")
    print(f"  Token is single-use and is deleted after the account is created.")
    print(f"{border}\n")
