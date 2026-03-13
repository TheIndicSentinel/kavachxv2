"""
KavachX Authentication & RBAC
JWT-based auth with role-based access control
"""
from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from jose import JWTError, jwt
from app.core.config import settings

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)

# --- RBAC roles and permissions ---
ROLES = {
    "super_admin": {
        "label": "Super Admin",
        "permissions": ["*"],  # all
        "description": "Full platform access including user management"
    },
    "compliance_officer": {
        "label": "Compliance Officer",
        "permissions": [
            "dashboard:read", "policies:read", "policies:write", "policies:delete",
            "audit:read", "audit:export", "models:read", "governance:read",
            "reports:read", "reports:generate", "alerts:read", "alerts:manage"
        ],
        "description": "Manages policies, compliance reports, audit logs"
    },
    "ml_engineer": {
        "label": "ML Engineer",
        "permissions": [
            "dashboard:read", "models:read", "models:write", "models:register",
            "governance:read", "governance:evaluate", "audit:read",
            "policies:read", "simulate:run", "alerts:read"
        ],
        "description": "Registers models, runs evaluations and simulations"
    },
    "executive": {
        "label": "Executive",
        "permissions": [
            "dashboard:read", "reports:read", "audit:read",
            "models:read", "policies:read", "alerts:read"
        ],
        "description": "Read-only executive overview and reports"
    },
    "auditor": {
        "label": "External Auditor",
        "permissions": [
            "audit:read", "audit:export", "policies:read",
            "models:read", "reports:read", "dashboard:read"
        ],
        "description": "Read-only audit access for external auditors"
    },
}

# Seeded credentials (password: shown in value)
SEEDED_USERS = {
    "admin@kavachx.ai": {
        "id": "user-001",
        "name": "Admin User",
        "role": "super_admin",
        "hashed_password": None,  # set at startup
        "plain_password": "Admin@123",
        "avatar": "AU",
    },
    "compliance@kavachx.ai": {
        "id": "user-002",
        "name": "Priya Sharma",
        "role": "compliance_officer",
        "hashed_password": None,
        "plain_password": "Comply@123",
        "avatar": "PS",
    },
    "engineer@kavachx.ai": {
        "id": "user-003",
        "name": "Arjun Dev",
        "role": "ml_engineer",
        "hashed_password": None,
        "plain_password": "Eng@12345",
        "avatar": "AD",
    },
    "exec@kavachx.ai": {
        "id": "user-004",
        "name": "Kavita Menon",
        "role": "executive",
        "hashed_password": None,
        "plain_password": "Exec@1234",
        "avatar": "KM",
    },
    "auditor@kavachx.ai": {
        "id": "user-005",
        "name": "External Auditor",
        "role": "auditor",
        "hashed_password": None,
        "plain_password": "Audit@123",
        "avatar": "EA",
    },
}


def _init_users():
    for email, user in SEEDED_USERS.items():
        pw = user["plain_password"].encode("utf-8")[:72].decode("utf-8", errors="ignore")
        user["hashed_password"] = pwd_context.hash(pw)



_init_users()


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=8))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")


def authenticate_user(email: str, password: str):
    user = SEEDED_USERS.get(email)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


from fastapi import Header

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    # Optional header for machine-to-machine integrations (Kavach)
    x_api_key: Optional[str] = Header(None)
):
    # 1. API KEY Check (Kavach System-to-System Auth)
    if x_api_key:
        if x_api_key == "kavachx-demo-key":  # Simple static check for demo
            return {"id": "system", "name": "API Client", "role": "ml_engineer", "email": "api@kavachx.ai"}
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # 2. JWT Bearer Check (Web Browser SPA)
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated (Missing Bearer or x-api-key)")
    try:
        payload = jwt.decode(credentials.credentials, settings.SECRET_KEY, algorithms=["HS256"])
        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = SEEDED_USERS.get(email)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return {**user, "email": email}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def require_permission(permission: str):
    async def _check(current_user=Depends(get_current_user)):
        role = current_user["role"]
        role_config = ROLES.get(role, {})
        perms = role_config.get("permissions", [])
        if "*" in perms or permission in perms:
            return current_user
        raise HTTPException(
            status_code=403,
            detail=f"Role '{role}' lacks permission '{permission}'"
        )
    return _check
