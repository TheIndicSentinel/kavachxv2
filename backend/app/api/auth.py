"""Auth API endpoints — login, setup-status, bootstrap, me, change-password."""
import hashlib
import hmac
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, EmailStr, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import (
    ROLES,
    _check_rate_limit, _record_failure, _clear_failures,
    _hash_token,
    create_access_token,
    get_current_user,
    hash_password,
    validate_password,
    verify_password,
)
from app.core.config import settings
from app.db.database import get_db
from app.models.orm_models import BootstrapToken, User

router = APIRouter()


# ── Schemas ───────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: str
    password: str


class BootstrapRequest(BaseModel):
    name: str
    email: str
    password: str
    bootstrap_token: str

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Name is required")
        return v.strip()


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


def _user_to_dict(user: User) -> dict:
    role_info = ROLES.get(user.role, {})
    return {
        "id":           user.id,
        "name":         user.name,
        "email":        user.email,
        "role":         user.role,
        "role_label":   role_info.get("label", user.role),
        "permissions":  role_info.get("permissions", []),
        "is_active":    user.is_active,
        "created_at":   user.created_at.isoformat() + "Z" if user.created_at else None,
        "last_login_at": user.last_login_at.isoformat() + "Z" if user.last_login_at else None,
    }


# ── GET /setup-status ─────────────────────────────────────────────────────────

@router.get("/setup-status", summary="Check whether first-run setup is needed")
async def setup_status(db: AsyncSession = Depends(get_db)):
    from sqlalchemy import func
    user_count = (await db.execute(select(func.count(User.id)))).scalar() or 0
    setup_required = user_count == 0

    token_active = False
    if setup_required:
        token_active = (await db.execute(
            select(BootstrapToken)
            .where(BootstrapToken.used == False)
            .where(BootstrapToken.expires_at > datetime.utcnow())
        )).scalars().first() is not None

    return {
        "setup_required":      setup_required,
        "bootstrap_token_active": token_active,
    }


# ── POST /bootstrap ───────────────────────────────────────────────────────────

@router.post("/bootstrap", summary="First-run Super Admin registration (requires bootstrap token)")
async def bootstrap(body: BootstrapRequest, request: Request, db: AsyncSession = Depends(get_db)):
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(f"bootstrap:{client_ip}")

    # Verify no users exist (race-condition guard)
    from sqlalchemy import func
    user_count = (await db.execute(select(func.count(User.id)))).scalar() or 0
    if user_count > 0:
        raise HTTPException(status_code=409, detail="Platform already has users. Bootstrap is disabled.")

    # Validate bootstrap token (timing-safe comparison on hash)
    token_hash = _hash_token(body.bootstrap_token)
    result = await db.execute(
        select(BootstrapToken)
        .where(BootstrapToken.token_hash == token_hash)
        .where(BootstrapToken.used == False)
        .where(BootstrapToken.expires_at > datetime.utcnow())
    )
    bt = result.scalars().first()
    if not bt:
        _record_failure(f"bootstrap:{client_ip}")
        raise HTTPException(status_code=401, detail="Invalid or expired bootstrap token")

    # Validate password strength
    email = body.email.strip().lower()
    pw_errors = validate_password(body.password)
    if pw_errors:
        raise HTTPException(status_code=422, detail={"password_errors": pw_errors})

    # Check email uniqueness (shouldn't be needed but defensive)
    existing = (await db.execute(select(User).where(User.email == email))).scalars().first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    # Create superadmin
    user = User(
        name            = body.name.strip(),
        email           = email,
        hashed_password = hash_password(body.password),
        role            = "super_admin",
        is_active       = True,
        created_by      = None,   # bootstrap — no creator
    )
    db.add(user)

    # Invalidate bootstrap token
    bt.used         = True
    bt.used_at      = datetime.utcnow()
    bt.used_by_email = email

    await db.commit()
    await db.refresh(user)

    # Clean up the token file written at startup (security hygiene)
    import os as _os
    _token_path = _os.environ.get("BOOTSTRAP_TOKEN_PATH", "./kavachx_setup_token.txt")
    try:
        if _os.path.exists(_token_path):
            _os.remove(_token_path)
    except Exception:
        pass  # Non-fatal — file may not exist in all deployment environments

    access_token = create_access_token(
        {"sub": user.email},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    _clear_failures(f"bootstrap:{client_ip}")

    return {
        "access_token": access_token,
        "token_type":   "bearer",
        "user":         _user_to_dict(user),
        "message":      "Super Admin account created. Welcome to KavachX.",
    }


# ── POST /login ───────────────────────────────────────────────────────────────

@router.post("/login", summary="Authenticate and receive JWT")
async def login(body: LoginRequest, request: Request, db: AsyncSession = Depends(get_db)):
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)

    email = body.email.strip().lower()
    result = await db.execute(select(User).where(User.email == email))
    user   = result.scalars().first()

    # Constant-time: always verify even if user not found (prevents user enumeration)
    dummy_hash = "$2b$12$invalidhashforenumerationprotection"
    pw_to_check = user.hashed_password if user else dummy_hash
    pw_valid    = verify_password(body.password, pw_to_check)

    if not user or not pw_valid:
        _record_failure(client_ip)
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account deactivated. Contact your administrator.")

    # Update last_login_at
    user.last_login_at = datetime.utcnow()
    await db.commit()

    access_token = create_access_token(
        {"sub": user.email},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    _clear_failures(client_ip)

    return {
        "access_token": access_token,
        "token_type":   "bearer",
        "user":         _user_to_dict(user),
    }


# ── GET /me ───────────────────────────────────────────────────────────────────

@router.get("/me", summary="Get current authenticated user profile")
async def me(current_user: dict = Depends(get_current_user)):
    return current_user


# ── POST /change-password ─────────────────────────────────────────────────────

@router.post("/change-password", summary="Change own password")
async def change_password(
    body: ChangePasswordRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.id == current_user["id"]))
    user   = result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not verify_password(body.current_password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Current password is incorrect")

    pw_errors = validate_password(body.new_password)
    if pw_errors:
        raise HTTPException(status_code=422, detail={"password_errors": pw_errors})

    if body.current_password == body.new_password:
        raise HTTPException(status_code=400, detail="New password must differ from current password")

    user.hashed_password = hash_password(body.new_password)
    await db.commit()
    return {"message": "Password updated successfully"}
