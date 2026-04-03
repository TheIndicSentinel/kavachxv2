"""
KavachX AI Governance Platform - Main Application Entry Point v2.0
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from contextlib import asynccontextmanager
import logging

from app.db.database import init_db
from app.api import auth as auth_router
from app.api import users as users_api
from app.api import governance, policies, audit, dashboard, models, ws, proxy, settings as settings_api
from app.api import ledger as ledger_api
from app.api import nael as nael_api
from app.api import attestation as attestation_api
from app.api import registry as registry_api
from app.api import synthetic_shield as synthetic_shield_api
from app.api import bascg as bascg_api
from app.api import consensus as consensus_api
from app.api import distributed_tee as distributed_tee_api
from app.api import legal_export as legal_export_api
from app.core.config import settings
from app.core.crypto import crypto_service
from app.core.startup_checks import run_production_checks
from app.core.config import load_regulator_keys
from app.services.sovereign_ledger_sync import sovereign_ledger_sync
from app.services.nair_sync_service import nair_sync_worker
from app.services.distributed_tee_service import distributed_tee_worker

logger = logging.getLogger("kavachx")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Apply industry-standard security headers to every HTTP response.

    References:
      OWASP Secure Headers Project — https://owasp.org/www-project-secure-headers/
      MDN Content Security Policy  — https://developer.mozilla.org/en-US/docs/Web/HTTP/CSP
    """
    # Content-Security-Policy: restrict resource origins to same-site.
    # 'unsafe-inline' is required for React's style-injected CSS-in-JS.
    # 'unsafe-eval' is NOT included — no eval() in production bundles.
    _CSP = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob: https:; "
        "connect-src 'self' ws: wss: https:; "
        "font-src 'self' data:; "
        "media-src 'self' blob:; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "frame-ancestors 'none';"
    )

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"]  = "nosniff"
        response.headers["X-Frame-Options"]         = "DENY"
        response.headers["Referrer-Policy"]         = "strict-origin-when-cross-origin"
        response.headers["X-XSS-Protection"]        = "1; mode=block"
        response.headers["Content-Security-Policy"] = self._CSP
        response.headers["Permissions-Policy"]      = (
            "camera=(), microphone=(), geolocation=(), payment=(), "
            "usb=(), bluetooth=(), serial=()"
        )
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        if settings.ENVIRONMENT == "production":
            response.headers["Strict-Transport-Security"] = (
                "max-age=63072000; includeSubDomains; preload"
            )
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    # 1. Safety checks (raises RuntimeError in production if secrets missing)
    run_production_checks()
    # 2. Crypto service (Ed25519 key init) — must run before policy bundle
    try:
        crypto_service.initialize()
    except Exception as _e:
        logger.warning("Crypto service init warning (non-fatal in dev): %s", _e)
    # 3. Database tables (idempotent create_all)
    try:
        await init_db()
        # 3b. First-run bootstrap (if no users exist)
        from app.core.auth import ensure_bootstrap_token
        from app.db.database import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            await ensure_bootstrap_token(db)
        logger.info("Database initialised.")
    except Exception as _e:
        logger.error("Database init failed: %s — check DATABASE_URL", _e)
    # 4. Background workers
    try:
        await sovereign_ledger_sync.start()
    except Exception as _e:
        logger.warning("Sovereign ledger worker start warning: %s", _e)
    try:
        await nair_sync_worker.start()
    except Exception as _e:
        logger.warning("NAIR sync worker start warning: %s", _e)
    try:
        await distributed_tee_worker.start()
    except Exception as _e:
        logger.warning("Distributed TEE worker start warning: %s", _e)
    # 5. Bootstrap token (first-run setup)
    try:
        from app.core.auth import ensure_bootstrap_token
        from app.db.database import AsyncSessionLocal
        async with AsyncSessionLocal() as _boot_db:
            await ensure_bootstrap_token(_boot_db)
    except Exception as _e:
        logger.warning("Bootstrap token check warning: %s", _e)
    logger.info("KavachX startup complete — environment=%s", settings.ENVIRONMENT)

    yield  # ── Running ──────────────────────────────────────────────────────

    # ── Shutdown ─────────────────────────────────────────────────────────────
    for worker, name in [
        (sovereign_ledger_sync, "Sovereign Ledger"),
        (nair_sync_worker,      "NAIR Sync"),
        (distributed_tee_worker,"Distributed TEE"),
    ]:
        try:
            await worker.stop()
        except Exception as _e:
            logger.warning("%s worker stop warning: %s", name, _e)


app = FastAPI(
    title="KavachX AI Governance Platform",
    description="Real-time AI governance infrastructure",
    version="2.0.0-mvp",
    lifespan=lifespan,
)

# Use explicit allowed origins from config; wildcard + credentials is a browser security violation
_cors_origins = settings.get_cors_origins()
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
)

app.include_router(auth_router.router, prefix="/api/v1/auth", tags=["Auth"])
app.include_router(users_api.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(governance.router, prefix="/api/v1/governance", tags=["Governance"])
app.include_router(policies.router, prefix="/api/v1/policies", tags=["Policies"])
app.include_router(audit.router, prefix="/api/v1/audit", tags=["Audit"])
app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["Dashboard"])
app.include_router(models.router, prefix="/api/v1/models", tags=["Models"])
app.include_router(ws.router, prefix="/api/v1/ws", tags=["WebSockets"])
app.include_router(settings_api.router, prefix="/api/v1/settings", tags=["Settings"])
app.include_router(proxy.router, prefix="/api/v1/proxy", tags=["Proxy"])
app.include_router(ledger_api.router,          prefix="/api/v1/ledger",          tags=["Sovereign Ledger"])
app.include_router(nael_api.router,            prefix="/api/v1/nael",            tags=["NAEL Licensing"])
app.include_router(attestation_api.router,     prefix="/api/v1/attestation",     tags=["TEE Attestation"])
app.include_router(registry_api.router,        prefix="/api/v1/registry",        tags=["NAIR-I Registry"])
app.include_router(synthetic_shield_api.router, prefix="/api/v1/synthetic-shield", tags=["Synthetic Media Shield"])
app.include_router(bascg_api.router,            prefix="/api/v1/bascg",            tags=["BASCG Control Plane"])
app.include_router(consensus_api.router,        prefix="/api/v1/consensus",         tags=["Policy Consensus"])
app.include_router(distributed_tee_api.router,  prefix="/api/v1/attestation",        tags=["Distributed TEE"])
app.include_router(legal_export_api.router,     prefix="/api/v1/legal-export",        tags=["Legal Bundle Export"])


@app.get("/health")
async def health_check():
    from app.db.database import engine
    try:
        async with engine.connect():
            db_status = "healthy"
    except Exception:
        db_status = "degraded"
    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "service": "KavachX Governance Engine",
        "version": "2.0.0-mvp",
        "database": db_status,
        "environment": settings.ENVIRONMENT,
    }

# --- MONOLITHIC FRONTEND SERVING ---
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# Adjust path based on your deployment structure
frontend_path = os.path.join(os.path.dirname(__file__), "../../frontend/dist")

if os.path.exists(frontend_path):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_path, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # If the request is for an API route, let it fall through to 404 naturally
        if full_path.startswith("api/"):
            return {"detail": "Not Found"}
        
        # Serve index.html for all other routes (SPA)
        index_file = os.path.join(frontend_path, "index.html")
        return FileResponse(index_file)

