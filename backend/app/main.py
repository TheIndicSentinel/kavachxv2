"""
KavachX AI Governance Platform - Main Application Entry Point v2.0
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.db.database import init_db
from app.api import governance, policies, audit, dashboard, models, ws, proxy, settings as settings_api
print(f"DEBUG: GOVERNANCE MODULE LOADED FROM: {governance.__file__}")
from app.api import auth as auth_router
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="KavachX AI Governance Platform",
    description="Real-time AI governance, fairness monitoring, and compliance enforcement infrastructure",
    version="2.0.0-mvp",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router.router, prefix="/api/v1/auth", tags=["Auth"])
app.include_router(governance.router, prefix="/api/v1/governance", tags=["Governance"])
app.include_router(policies.router, prefix="/api/v1/policies", tags=["Policies"])
app.include_router(audit.router, prefix="/api/v1/audit", tags=["Audit"])
app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["Dashboard"])
app.include_router(models.router, prefix="/api/v1/models", tags=["Models"])
app.include_router(ws.router, prefix="/api/v1/ws", tags=["WebSockets"])
app.include_router(settings_api.router, prefix="/api/v1/settings", tags=["Settings"])
app.include_router(proxy.router, prefix="/api/v1/proxy", tags=["Proxy"])


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "KavachX Governance Engine",
        "version": "2.0.0-mvp",
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

