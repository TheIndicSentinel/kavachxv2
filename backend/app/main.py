"""
KavachX AI Governance Platform - Main Application Entry Point v2.0
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware
from contextlib import asynccontextmanager
import asyncio
import collections
import json
import logging
import logging.config
import time
from typing import Optional
import threading
import uuid

from app.db.database import init_db
from app.api import auth as auth_router
from app.api import users as users_api
from app.api import governance, policies, audit, dashboard, models, ws, proxy, settings as settings_api
from app.api import dpdp_moderator as dpdp_moderator_api
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

# ---------------------------------------------------------------------------
# Structured JSON logging — writes machine-readable log records in production.
# In development, standard text format is used for readability.
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line for log aggregation tools."""
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts":      self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _configure_logging(environment: str) -> None:
    root = logging.getLogger()
    if root.handlers:
        return  # already configured (e.g. uvicorn reloaded)
    handler = logging.StreamHandler()
    if environment == "production":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s  %(message)s",
            datefmt="%H:%M:%S",
        ))
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    # Suppress overly verbose third-party loggers
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


logger = logging.getLogger("kavachx")


# ---------------------------------------------------------------------------
# In-process metrics store — Prometheus-compatible text format at /metrics.
# Zero external dependencies; thread-safe counters using a lock.
# ---------------------------------------------------------------------------

class _Metrics:
    """Lightweight in-process counter store."""
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = collections.defaultdict(int)
        self._histograms: dict[str, list[float]] = collections.defaultdict(list)
        self._start = time.time()

    def inc(self, name: str, labels: Optional[dict] = None, amount: int = 1) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._counters[key] += amount

    def observe(self, name: str, value: float, labels: Optional[dict] = None) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._histograms[key].append(value)

    def _key(self, name: str, labels: Optional[dict]) -> str:
        if not labels:
            return name
        lstr = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{lstr}}}"

    def prometheus_text(self) -> str:
        lines: list[str] = [
            f"# HELP kavachx_uptime_seconds Seconds since process start",
            f"# TYPE kavachx_uptime_seconds gauge",
            f"kavachx_uptime_seconds {time.time() - self._start:.1f}",
        ]
        with self._lock:
            for key, val in sorted(self._counters.items()):
                name = key.split("{")[0]
                lines += [
                    f"# HELP {name} Counter",
                    f"# TYPE {name} counter",
                    f"{key} {val}",
                ]
            for key, observations in sorted(self._histograms.items()):
                if not observations:
                    continue
                name = key.split("{")[0]
                avg = sum(observations) / len(observations)
                lines += [
                    f"# HELP {name}_avg Average",
                    f"# TYPE {name}_avg gauge",
                    f"{key.replace(name, name + '_avg')} {avg:.4f}",
                    f"# HELP {name}_count Observation count",
                    f"# TYPE {name}_count counter",
                    f"{key.replace(name, name + '_count')} {len(observations)}",
                ]
        return "\n".join(lines) + "\n"


metrics = _Metrics()


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Propagate a unique request ID through every HTTP interaction.

    - If the caller sends ``X-Request-ID``, that value is used as-is
      (max 64 chars to prevent header injection).
    - Otherwise a fresh UUID4 is generated.
    - The final ID is always echoed back in the response header so
      callers can correlate requests with server-side logs.
    """
    async def dispatch(self, request: Request, call_next):
        req_id = (request.headers.get("X-Request-ID") or "")[:64] or str(uuid.uuid4())
        request.state.request_id = req_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response


class RequestSizeLimiter(BaseHTTPMiddleware):
    """Reject requests whose Content-Length exceeds MAX_REQUEST_SIZE_KB."""
    async def dispatch(self, request: Request, call_next):
        max_bytes = settings.MAX_REQUEST_SIZE_KB * 1024
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > max_bytes:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Request body too large (max {settings.MAX_REQUEST_SIZE_KB} KB)."},
            )
        return await call_next(request)


class LatencyMetricsMiddleware(BaseHTTPMiddleware):
    """Record per-route latency and HTTP status code metrics."""
    async def dispatch(self, request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - t0
        # Normalise path: strip IDs to avoid cardinality explosion
        path = request.url.path
        status = str(response.status_code)
        metrics.inc("http_requests_total", {"method": request.method, "status": status})
        metrics.observe("http_request_duration_seconds", elapsed, {"path": path})
        return response


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
    _configure_logging(settings.ENVIRONMENT)
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
    # 6. Eager ML model warmup — load weights before first request arrives
    #    (avoids crash-on-first-prompt caused by lazy model loading)
    try:
        import concurrent.futures
        _warmup_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="model_warmup")

        def _warmup_all():
            try:
                from app.modules.general_safety.inference import warmup as _gs_warmup
                _gs_warmup()
            except Exception as _e:
                logger.warning("General Safety warmup warning: %s", _e)
            try:
                from app.modules.dpdp_moderator.pipeline import warmup as _dpdp_warmup
                _dpdp_warmup()
            except Exception as _e:
                logger.warning("DPDP Moderator warmup warning: %s", _e)

        await asyncio.get_event_loop().run_in_executor(_warmup_executor, _warmup_all)
        _warmup_executor.shutdown(wait=False)
        logger.info("ML model warmup complete.")
    except Exception as _e:
        logger.warning("Model warmup error (non-fatal): %s", _e)

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
# Middleware stack — innermost first (LIFO execution order in Starlette)
app.add_middleware(LatencyMetricsMiddleware)
app.add_middleware(RequestSizeLimiter)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    # Also allow Chrome extension backgrounds to call the API.
    # Chrome extension IDs are 32 lowercase hex characters.
    allow_origin_regex=r"chrome-extension://[a-z0-9]{32}",
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
app.include_router(dpdp_moderator_api.router,  prefix="/api/v1",                      tags=["DPDP AI Moderator"])


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


@app.get("/ready")
async def readiness_check():
    """
    Kubernetes/load-balancer readiness probe.

    Returns 200 when the DPDP model is loaded and the database is reachable.
    Returns 503 when either subsystem is not yet ready — prevents traffic
    being routed to an instance that hasn't finished initialising.
    """
    checks: dict[str, str] = {}
    # Database
    from app.db.database import engine
    try:
        async with engine.connect():
            checks["database"] = "ready"
    except Exception:
        checks["database"] = "not_ready"
    # DPDP model
    try:
        from app.modules.dpdp_moderator.pipeline import get_pipeline
        p = get_pipeline()
        checks["dpdp_model"] = "ready" if (p._classifier._loaded and p._classifier._model is not None) else "loading"
    except Exception:
        checks["dpdp_model"] = "unavailable"

    overall = "ready" if all(v == "ready" for v in checks.values()) else "not_ready"
    status_code = 200 if overall == "ready" else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": overall, "checks": checks},
    )


@app.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
async def prometheus_metrics():
    """
    Prometheus-compatible text exposition format.

    Scrape with:
      prometheus.yml:
        scrape_configs:
          - job_name: kavachx
            static_configs:
              - targets: ['kavachx-backend:8001']
            metrics_path: /metrics

    Access should be restricted at the reverse proxy level —
    this endpoint is intentionally unauthenticated for scraper compatibility
    but should NOT be exposed to the public internet.
    """
    return metrics.prometheus_text()

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

