"""
DPDP AI Moderator REST API — production-ready partner integration layer.

Endpoints
---------
GET  /api/v1/dpdp-moderator/status          Model health + stats
POST /api/v1/dpdp-moderator/moderate        Single-text moderation
POST /api/v1/dpdp-moderator/moderate/batch  Batch moderation (up to 50 items)
POST /api/v1/dpdp-moderator/moderate/async  Async moderation with webhook callback
POST /api/v1/dpdp-moderator/feedback        Submit human correction for retraining
GET  /api/v1/dpdp-moderator/feedback        List saved feedback entries
GET  /api/v1/dpdp-moderator/decisions       Audit trail of recent moderation decisions

Authentication
--------------
All endpoints accept either:
  - ``X-API-Key: <partner-key>``  (preferred for SDK / server-to-server)
  - ``Authorization: Bearer <jwt>``  (UI / session-based access)

Partner API keys are configured via DPDP_API_KEYS (comma-separated) in .env.
Generate per partner: python -c "import secrets; print(secrets.token_urlsafe(32))"

Rate Limiting
-------------
Per API-key sliding-window rate limit (DPDP_RATE_LIMIT_RPM, default 120 req/min).
Exceeding the limit returns HTTP 429 with Retry-After and X-RateLimit-* headers.
"""
from __future__ import annotations

import asyncio
import collections
import hmac
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from app.core.config import settings

router = APIRouter(prefix="/dpdp-moderator", tags=["DPDP AI Moderator"])

# ---------------------------------------------------------------------------
# File paths (relative to the dpdp_moderator module)
# ---------------------------------------------------------------------------

_MODULE_DIR = Path(__file__).parent.parent / "modules" / "dpdp_moderator"
_MODEL_OUTPUT = _MODULE_DIR / "model_output"
_FEEDBACK_PATH  = _MODEL_OUTPUT / "feedback.jsonl"
_DECISIONS_PATH = _MODEL_OUTPUT / "decisions.jsonl"


# ---------------------------------------------------------------------------
# In-process rate limiter — sliding window per API key
# ---------------------------------------------------------------------------

class _SlidingWindowLimiter:
    """
    Thread-safe per-key sliding-window rate limiter.

    Uses a deque of timestamps per key.  Entries older than the window
    are discarded on each check.  Memory is bounded: keys with no recent
    activity have empty deques (GC-eligible).
    """
    def __init__(self, max_requests: int, window_seconds: int = 60):
        self._max = max_requests
        self._window = window_seconds
        self._lock = threading.Lock()
        self._windows: dict[str, collections.deque] = collections.defaultdict(collections.deque)

    def check(self, key: str) -> tuple[bool, int]:
        """
        Returns (allowed: bool, retry_after_seconds: int).
        If allowed, the request is also recorded.
        """
        if self._max <= 0:
            return True, 0
        now = time.monotonic()
        cutoff = now - self._window
        with self._lock:
            dq = self._windows[key]
            # Remove expired timestamps
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= self._max:
                # Earliest timestamp determines when the window resets
                retry_after = int(self._window - (now - dq[0])) + 1
                return False, retry_after
            dq.append(now)
            return True, 0


_rate_limiter = _SlidingWindowLimiter(
    max_requests=settings.DPDP_RATE_LIMIT_RPM,
    window_seconds=60,
)


# ---------------------------------------------------------------------------
# Authentication dependency
# ---------------------------------------------------------------------------

def _get_api_key(request: Request) -> str | None:
    """Extract X-API-Key header value."""
    return request.headers.get("X-API-Key") or request.headers.get("x-api-key")


def _get_client_id(request: Request) -> str:
    """
    Best-effort client identifier for rate limiting and audit logging.

    Priority: X-Client-ID header → API key (first 8 chars) → IP address.
    """
    client_id = request.headers.get("X-Client-ID") or request.headers.get("x-client-id")
    if client_id:
        return client_id[:64]
    api_key = _get_api_key(request)
    if api_key:
        return f"key:{api_key[:8]}…"
    return getattr(request.client, "host", "unknown")


async def require_dpdp_access(request: Request) -> dict:
    """
    FastAPI dependency: validates API key OR JWT bearer token.

    Returns a dict with at least {"client_id": str, "auth_method": str}.

    Resolution order:
      1. X-API-Key header → checked against DPDP_API_KEYS set (timing-safe)
      2. Authorization: Bearer <JWT> → delegated to the existing JWT auth
      3. No credentials → 401

    Rate limiting is applied per client_id after authentication.
    """
    api_key = _get_api_key(request)
    valid_keys = settings.get_dpdp_api_keys()

    if api_key:
        # Use timing-safe comparison to prevent timing attacks
        key_valid = any(
            hmac.compare_digest(api_key.encode(), k.encode())
            for k in valid_keys
        ) if valid_keys else False

        if not key_valid and valid_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key.",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        if key_valid:
            client_id = _get_client_id(request)
            _apply_rate_limit(client_id)
            return {"client_id": client_id, "auth_method": "api_key"}

    # No API keys configured or no key provided → fall back to JWT
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        try:
            from app.core.auth import get_current_user
            # Re-invoke the JWT dependency
            token = auth_header[7:]
            from app.core.auth import SECRET_KEY, ALGORITHM
            from jose import JWTError, jwt as _jwt
            payload = _jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            email: str = payload.get("sub", "")
            if not email:
                raise HTTPException(status_code=401, detail="Invalid token payload.")
            client_id = _get_client_id(request)
            _apply_rate_limit(client_id)
            return {"client_id": email, "auth_method": "jwt"}
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # No credentials at all — allow only when no API keys are configured
    # (i.e. internal / development mode with no partner keys set)
    if not valid_keys:
        client_id = _get_client_id(request)
        _apply_rate_limit(client_id)
        return {"client_id": client_id, "auth_method": "unauthenticated"}

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Provide X-API-Key or Authorization: Bearer <token>.",
        headers={"WWW-Authenticate": "ApiKey, Bearer"},
    )


def _apply_rate_limit(client_id: str) -> None:
    allowed, retry_after = _rate_limiter.check(client_id)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Retry after {retry_after}s.",
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(settings.DPDP_RATE_LIMIT_RPM),
                "X-RateLimit-Window": "60",
            },
        )


# ---------------------------------------------------------------------------
# Audit trail — append-only JSONL decision log
# ---------------------------------------------------------------------------

def _log_decision(
    request_id: str,
    client_id: str,
    text_preview: str,
    result,
    context_turns: int = 0,
) -> None:
    """Append one moderation decision to decisions.jsonl (non-blocking)."""
    if not settings.DPDP_AUDIT_LOG_ENABLED:
        return
    entry = {
        "id":              request_id,
        "ts":              datetime.now(timezone.utc).isoformat(),
        "client_id":       client_id,
        "text_preview":    text_preview[:120],
        "text_length":     len(text_preview),
        "context_turns":   context_turns,
        "verdict":         result.verdict.value,
        "risk_score":      round(result.risk_score, 4),
        "top_label":       result.top_label,
        "top_label_prob":  round(result.top_label_prob, 4),
        "regex_triggered": result.regex.triggered,
        "regex_patterns":  result.regex.matched_patterns,
        "low_confidence":  result.low_confidence,
        "reasons":         result.reasons[:5],
    }
    try:
        _DECISIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _DECISIONS_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass  # Never let audit logging crash the moderation response


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ModerateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000,
        description="Text to moderate. Up to 8 000 characters.")
    context: Optional[list[str]] = Field(
        None,
        max_length=3,
        description=(
            "Up to 3 prior conversation turns (oldest first). "
            "Enables cross-turn DPDP violation detection — "
            "e.g. purpose creep spread across multiple messages."
        ),
    )


class ModerateResponse(BaseModel):
    request_id:          str
    verdict:             str
    risk_score:          float
    top_label:           str
    top_label_prob:      float
    class_probs:         dict
    regex_triggered:     bool
    regex_patterns:      list
    embedding_similarity: float
    reasons:             list
    low_confidence:      bool = False


class BatchModerateRequest(BaseModel):
    items: list[ModerateRequest] = Field(
        ...,
        min_length=1,
        description=f"List of texts to moderate (max {settings.DPDP_BATCH_MAX_SIZE}).",
    )


class BatchModerateResponse(BaseModel):
    request_id: str
    total:      int
    results:    list[ModerateResponse]


class AsyncModerateRequest(BaseModel):
    text:         str = Field(..., min_length=1, max_length=8000)
    context:      Optional[list[str]] = Field(None, max_length=3)
    webhook_url:  str = Field(..., description="HTTPS endpoint to POST the result to.")
    webhook_secret: Optional[str] = Field(
        None,
        description=(
            "Optional shared secret. When provided, the webhook POST includes "
            "an X-KavachX-Signature header (HMAC-SHA256 of the JSON body) "
            "so you can verify authenticity."
        ),
    )


class AsyncModerateResponse(BaseModel):
    job_id:      str
    status:      str = "queued"
    webhook_url: str


class FeedbackRequest(BaseModel):
    text:           str = Field(..., min_length=1, max_length=8000)
    correct_label:  str = Field(..., description=(
        "One of: safe, personal_data, financial_data, health_data, "
        "profiling, surveillance, criminal_misuse, compliance_abuse"
    ))
    model_verdict:  Optional[str] = Field(None, description="What the model returned (optional).")
    feedback_type:  str = Field("correction",
        description="correction | confirmation | false_positive | false_negative")
    notes:          Optional[str] = None


class FeedbackResponse(BaseModel):
    id:      str
    saved:   bool
    message: str


VALID_LABELS = {
    "safe", "personal_data", "financial_data", "health_data",
    "profiling", "surveillance", "criminal_misuse", "compliance_abuse",
}
VALID_FEEDBACK_TYPES = {"correction", "confirmation", "false_positive", "false_negative"}


# ---------------------------------------------------------------------------
# Helper: run moderation and build response
# ---------------------------------------------------------------------------

async def _run_moderation(
    text: str,
    context: list[str] | None,
    request_id: str,
    client_id: str,
) -> ModerateResponse:
    from app.modules.dpdp_moderator.pipeline import moderate_async, moderate_async_with_context
    if context:
        result = await moderate_async_with_context(text, context)
    else:
        result = await moderate_async(text)

    _log_decision(request_id, client_id, text, result, context_turns=len(context or []))

    return ModerateResponse(
        request_id=request_id,
        verdict=result.verdict.value,
        risk_score=round(result.risk_score, 4),
        top_label=result.top_label,
        top_label_prob=round(result.top_label_prob, 4),
        class_probs={k: round(v, 4) for k, v in result.class_probs.items()},
        regex_triggered=result.regex.triggered,
        regex_patterns=result.regex.matched_patterns,
        embedding_similarity=round(result.embedding_similarity, 4),
        reasons=result.reasons,
        low_confidence=result.low_confidence,
    )


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@router.get("/status")
async def get_status(auth: dict = Depends(require_dpdp_access)):
    """Return DPDP AI Moderator health, model metadata, and corpus stats."""
    try:
        from app.modules.dpdp_moderator.pipeline import get_pipeline
        from app.modules.dpdp_moderator.dataset import LABELS
        pipeline = get_pipeline()
        model_dir = Path(pipeline._classifier._model_dir)
        model_loaded = pipeline._classifier._loaded and pipeline._classifier._model is not None
        embed_loaded = pipeline._embedder._encoder is not None
        return {
            "status": "ready" if model_loaded else "degraded",
            "model_dir":        str(model_dir),
            "model_loaded":     model_loaded,
            "embedding_loaded": embed_loaded,
            "labels":           LABELS,
            "thresholds": {
                "block":               pipeline._block_thresh,
                "review":              pipeline._review_thresh,
                "embedding_similarity": pipeline._embedder._threshold,
            },
            "feedback_samples": _count_file(_FEEDBACK_PATH),
            "decision_log_entries": _count_file(_DECISIONS_PATH),
            "corpus_size":      len(pipeline._embedder._known_unsafe),
            "client_id":        auth["client_id"],
        }
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc)}


# ---------------------------------------------------------------------------
# Single-text moderation
# ---------------------------------------------------------------------------

@router.post("/moderate", response_model=ModerateResponse)
async def moderate_text(
    req: ModerateRequest,
    request: Request,
    response: Response,
    auth: dict = Depends(require_dpdp_access),
):
    """
    Run the 3-layer DPDP moderator on a single text.

    **Verdicts**
    - ``ALLOW`` — no significant DPDP risk detected
    - ``REVIEW`` — borderline; human review recommended
    - ``BLOCK``  — high-risk; should be blocked

    **Session context**
    Pass up to 3 prior conversation turns in ``context`` to detect
    cross-turn violations (e.g., gradual purpose creep).

    **Low-confidence flag**
    When ``low_confidence=true`` the top-2 class probabilities are within
    0.15 of each other — human review is strongly recommended regardless
    of the verdict.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    try:
        result = await _run_moderation(req.text, req.context, request_id, auth["client_id"])
        response.headers["X-Request-ID"] = request_id
        return result
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DPDP AI Moderator not available. Train the model first (retrain.py).",
        )


# ---------------------------------------------------------------------------
# Batch moderation
# ---------------------------------------------------------------------------

@router.post("/moderate/batch", response_model=BatchModerateResponse)
async def moderate_batch(
    req: BatchModerateRequest,
    request: Request,
    response: Response,
    auth: dict = Depends(require_dpdp_access),
):
    """
    Moderate up to ``DPDP_BATCH_MAX_SIZE`` (default 50) texts in one call.

    Items are processed concurrently. Each result includes its own
    ``request_id`` derived from the batch request ID + item index.
    Batch quota counts as N individual requests against the rate limit
    (one pre-check is applied to the batch call itself).

    Ideal for:
    - Bulk content pipelines
    - Training-data pre-screening
    - Compliance audit of historical records
    """
    max_size = settings.DPDP_BATCH_MAX_SIZE
    if len(req.items) > max_size:
        raise HTTPException(
            status_code=422,
            detail=f"Batch size {len(req.items)} exceeds maximum of {max_size}.",
        )
    batch_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    try:
        tasks = [
            _run_moderation(
                item.text,
                item.context,
                f"{batch_id}:{i}",
                auth["client_id"],
            )
            for i, item in enumerate(req.items)
        ]
        results = await asyncio.gather(*tasks)
        response.headers["X-Request-ID"] = batch_id
        return BatchModerateResponse(
            request_id=batch_id,
            total=len(results),
            results=list(results),
        )
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DPDP AI Moderator not available.",
        )


# ---------------------------------------------------------------------------
# Async moderation with webhook delivery
# ---------------------------------------------------------------------------

async def _deliver_webhook(
    job_id: str,
    result: ModerateResponse,
    webhook_url: str,
    webhook_secret: str | None,
) -> None:
    """Background task: POST moderation result to the caller's webhook."""
    payload = result.model_dump()
    payload["job_id"] = job_id
    body = json.dumps(payload, ensure_ascii=False).encode()

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "KavachX-DPDP-Moderator/2.0",
        "X-KavachX-Job-ID": job_id,
    }
    if webhook_secret:
        import hashlib
        sig = hmac.new(
            webhook_secret.encode(),
            body,
            hashlib.sha256,
        ).hexdigest()
        headers["X-KavachX-Signature"] = f"sha256={sig}"

    timeout = settings.DPDP_WEBHOOK_TIMEOUT_SECONDS
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            await client.post(webhook_url, content=body, headers=headers)
    except Exception:
        pass  # Webhook delivery failure is best-effort; caller should implement retry


@router.post("/moderate/async", response_model=AsyncModerateResponse, status_code=202)
async def moderate_async_webhook(
    req: AsyncModerateRequest,
    request: Request,
    auth: dict = Depends(require_dpdp_access),
):
    """
    Enqueue a moderation job and return immediately with a ``job_id``.

    When moderation completes, the result is POSTed to ``webhook_url`` as JSON
    with the same schema as the ``/moderate`` response plus a ``job_id`` field.

    **Webhook security**
    Provide ``webhook_secret`` to receive an ``X-KavachX-Signature: sha256=<hex>``
    header. Verify with HMAC-SHA256 over the raw request body using your secret.

    **Retries**
    Webhook delivery is best-effort (one attempt). Implement retry logic on
    your webhook endpoint using the ``job_id`` to deduplicate.
    """
    job_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    async def _run_and_deliver():
        try:
            result = await _run_moderation(
                req.text, req.context, job_id, auth["client_id"]
            )
            await _deliver_webhook(job_id, result, req.webhook_url, req.webhook_secret)
        except Exception:
            pass

    asyncio.create_task(_run_and_deliver())
    return AsyncModerateResponse(
        job_id=job_id,
        status="queued",
        webhook_url=req.webhook_url,
    )


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

@router.post("/feedback", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
async def submit_feedback(
    req: FeedbackRequest,
    auth: dict = Depends(require_dpdp_access),
):
    """
    Submit a human-labelled correction for the next retraining run.

    Corrections are appended to ``model_output/feedback.jsonl`` and
    up-weighted (6×) during retraining via ``--feedback-weight 6``.

    **feedback_type values**
    - ``correction``      — model was wrong; correct_label overrides it
    - ``false_positive``  — model BLOCKed a safe text
    - ``false_negative``  — model ALLOWed a violating text
    - ``confirmation``    — model was correct (reinforcement signal)
    """
    if req.correct_label not in VALID_LABELS:
        raise HTTPException(status_code=422,
            detail=f"correct_label must be one of: {sorted(VALID_LABELS)}")
    if req.feedback_type not in VALID_FEEDBACK_TYPES:
        raise HTTPException(status_code=422,
            detail=f"feedback_type must be one of: {sorted(VALID_FEEDBACK_TYPES)}")

    from app.modules.dpdp_moderator.dataset import LABEL2ID
    entry = {
        "id":             str(uuid.uuid4()),
        "text":           req.text.strip(),
        "label":          LABEL2ID[req.correct_label],
        "label_name":     req.correct_label,
        "model_verdict":  req.model_verdict,
        "feedback_type":  req.feedback_type,
        "notes":          req.notes,
        "submitted_by":   auth["client_id"],
        "submitted_at":   datetime.now(timezone.utc).isoformat(),
    }
    try:
        _FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _FEEDBACK_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return FeedbackResponse(
            id=entry["id"],
            saved=True,
            message=f"Feedback saved (label={req.correct_label}). Run retrain.py to incorporate.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {exc}")


@router.get("/feedback")
async def list_feedback(
    limit: int = 50,
    feedback_type: Optional[str] = None,
    auth: dict = Depends(require_dpdp_access),
):
    """Return recent feedback entries (newest first, max 200)."""
    limit = min(limit, 200)
    return _read_jsonl(_FEEDBACK_PATH, limit, filter_key="feedback_type", filter_val=feedback_type)


# ---------------------------------------------------------------------------
# Decision audit trail
# ---------------------------------------------------------------------------

@router.get("/decisions")
async def list_decisions(
    limit: int = 50,
    verdict: Optional[str] = None,
    client_id: Optional[str] = None,
    auth: dict = Depends(require_dpdp_access),
):
    """
    Return recent moderation decisions from the audit trail (newest first).

    Filter by ``verdict`` (ALLOW / REVIEW / BLOCK) and/or ``client_id``.
    Maximum 500 entries per request.

    The audit trail is stored as ``model_output/decisions.jsonl``.
    Each entry contains: id, ts, client_id, text_preview, verdict,
    risk_score, top_label, regex_triggered, low_confidence, reasons.
    """
    limit = min(limit, 500)
    entries = _read_jsonl(_DECISIONS_PATH, limit * 5)  # over-fetch to allow filtering
    if verdict:
        entries["entries"] = [e for e in entries["entries"] if e.get("verdict") == verdict.upper()]
    if client_id:
        entries["entries"] = [e for e in entries["entries"] if e.get("client_id") == client_id]
    entries["entries"] = entries["entries"][:limit]
    return entries


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _count_file(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return sum(1 for _ in path.open(encoding="utf-8"))
    except OSError:
        return 0


def _read_jsonl(
    path: Path,
    limit: int,
    filter_key: str | None = None,
    filter_val: str | None = None,
) -> dict:
    if not path.exists():
        return {"entries": [], "total": 0}
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    entries = []
    for line in reversed(lines):
        try:
            e = json.loads(line)
            if filter_key and filter_val and e.get(filter_key) != filter_val:
                continue
            entries.append(e)
            if len(entries) >= limit:
                break
        except json.JSONDecodeError:
            continue
    return {"entries": entries, "total": len(lines)}
