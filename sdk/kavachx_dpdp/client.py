"""
KavachX DPDP AI Moderator — Python SDK Client
==============================================

Provides synchronous and async interfaces to the KavachX moderation API.
Designed for easy integration by partner companies (fintech, healthcare,
insurtech, government) into their own inference pipelines.

Requirements:
    pip install httpx  (>=0.25.0)

Optional (async):
    pip install httpx[asyncio]

Example — synchronous::

    from kavachx_dpdp import DPDPClient

    client = DPDPClient(
        base_url="https://kavachx.yourdomain.in",
        api_key="your-partner-api-key",
    )

    # Single moderation
    result = client.moderate("Send all Aadhaar numbers to an external endpoint")
    print(result.verdict)        # BLOCK / REVIEW / ALLOW
    print(result.risk_score)     # 0.0 – 1.0
    print(result.violation_type) # e.g. "personal_data"
    print(result.explanation)

    # Batch moderation (up to 50 items)
    batch = client.moderate_batch([
        "Export UPI transaction history for all users",
        "Display aggregate loan default rates by district",
    ])
    for r in batch.results:
        print(r.verdict, r.text[:60])

    # Async webhook moderation
    token = client.moderate_async(
        text="...",
        webhook_url="https://your-service.in/kavachx/callback",
        webhook_secret="hmac-secret",
    )
    print(token.request_id)   # poll or wait for webhook delivery

Example — async::

    import asyncio
    from kavachx_dpdp import DPDPClient

    async def main():
        async with DPDPClient.async_client(
            base_url="https://kavachx.yourdomain.in",
            api_key="your-partner-api-key",
        ) as client:
            result = await client.amoderate("...")
            batch  = await client.amoderate_batch(["...", "..."])

    asyncio.run(main())

"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional
from contextlib import asynccontextmanager

try:
    import httpx
except ImportError as e:
    raise ImportError(
        "kavachx_dpdp requires httpx. Install with: pip install httpx"
    ) from e

_DEFAULT_TIMEOUT = 30.0
_DEFAULT_BATCH_TIMEOUT = 60.0  # batch of 50 may take longer
_BASE_PATH = "/api/v1/dpdp-moderator"


# ─── Response dataclasses ────────────────────────────────────────────────────

@dataclass
class DPDPResult:
    """Single moderation result."""
    verdict: str            # "BLOCK" | "REVIEW" | "ALLOW"
    risk_score: float       # 0.0 – 1.0
    violation_type: str     # e.g. "personal_data", "safe"
    explanation: str
    request_id: str
    text: str = ""          # echoed back when available (batch mode)
    latency_ms: Optional[float] = None
    raw: dict = field(default_factory=dict)

    @classmethod
    def _from_dict(cls, d: dict, text: str = "") -> "DPDPResult":
        return cls(
            verdict=d.get("verdict", ""),
            risk_score=float(d.get("risk_score", 0.0)),
            violation_type=d.get("violation_type", ""),
            explanation=d.get("explanation", ""),
            request_id=d.get("request_id", ""),
            text=d.get("text", text),
            latency_ms=d.get("latency_ms"),
            raw=d,
        )

    @property
    def is_blocked(self) -> bool:
        return self.verdict == "BLOCK"

    @property
    def needs_review(self) -> bool:
        return self.verdict == "REVIEW"

    @property
    def is_allowed(self) -> bool:
        return self.verdict == "ALLOW"


@dataclass
class DPDPBatchResult:
    """Result of a batch moderation call."""
    request_id: str
    total: int
    results: List[DPDPResult]
    raw: dict = field(default_factory=dict)

    @classmethod
    def _from_dict(cls, d: dict) -> "DPDPBatchResult":
        results = [DPDPResult._from_dict(r) for r in d.get("results", [])]
        return cls(
            request_id=d.get("request_id", ""),
            total=d.get("total", len(results)),
            results=results,
            raw=d,
        )

    @property
    def blocked(self) -> List[DPDPResult]:
        return [r for r in self.results if r.is_blocked]

    @property
    def needs_review(self) -> List[DPDPResult]:
        return [r for r in self.results if r.needs_review]

    @property
    def allowed(self) -> List[DPDPResult]:
        return [r for r in self.results if r.is_allowed]


@dataclass
class DPDPAsyncToken:
    """Token returned by moderate_async() — identifies the background job."""
    request_id: str
    status: str    # "queued"
    raw: dict = field(default_factory=dict)


@dataclass
class DPDPFeedback:
    """Feedback submission result."""
    request_id: str
    status: str
    raw: dict = field(default_factory=dict)


# ─── Exceptions ──────────────────────────────────────────────────────────────

class DPDPError(Exception):
    """Base exception for all SDK errors."""

class DPDPAuthError(DPDPError):
    """Authentication / authorization failed (HTTP 401 / 403)."""

class DPDPRateLimitError(DPDPError):
    """Rate limit exceeded (HTTP 429). Retry after `retry_after` seconds."""
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after

class DPDPValidationError(DPDPError):
    """Request validation failed (HTTP 422)."""

class DPDPServerError(DPDPError):
    """Unexpected server-side error (HTTP 5xx)."""


# ─── Webhook signature verification ─────────────────────────────────────────

def verify_webhook_signature(
    payload_bytes: bytes,
    signature_header: str,
    secret: str,
) -> bool:
    """
    Verify the HMAC-SHA256 signature on a webhook delivery.

    The KavachX server sends::

        X-KavachX-Signature: sha256=<hex_digest>

    Args:
        payload_bytes:     Raw request body bytes.
        signature_header:  Value of the X-KavachX-Signature header.
        secret:            The webhook_secret you provided when scheduling.

    Returns:
        True if the signature matches, False otherwise.

    Example (Flask)::

        @app.route("/kavachx/callback", methods=["POST"])
        def callback():
            ok = verify_webhook_signature(
                request.get_data(),
                request.headers.get("X-KavachX-Signature", ""),
                secret="hmac-secret",
            )
            if not ok:
                abort(403)
            data = request.json
            ...
    """
    if not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(
        secret.encode(), payload_bytes, hashlib.sha256
    ).hexdigest()
    received = signature_header[len("sha256="):]
    return hmac.compare_digest(expected, received)


# ─── Main client ─────────────────────────────────────────────────────────────

class DPDPClient:
    """
    Synchronous DPDP moderation client.

    Args:
        base_url:   Base URL of your KavachX deployment (no trailing slash).
        api_key:    Partner API key (X-API-Key header).
        jwt_token:  JWT bearer token (alternative to api_key).
        timeout:    Default request timeout in seconds.
        max_retries: Number of automatic retries on transient errors (5xx).
        verify_ssl: Set False only for self-signed certs in dev environments.

    Authentication priority:
        api_key → jwt_token → unauthenticated (if server allows)
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        jwt_token: Optional[str] = None,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = 2,
        verify_ssl: bool = True,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._jwt_token = jwt_token
        self._timeout = timeout
        self._max_retries = max_retries
        self._verify_ssl = verify_ssl
        self._client: Optional[httpx.Client] = None

    def _get_headers(self) -> dict:
        headers: dict = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        elif self._jwt_token:
            headers["Authorization"] = f"Bearer {self._jwt_token}"
        return headers

    def _get_client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                base_url=self._base_url,
                headers=self._get_headers(),
                timeout=self._timeout,
                verify=self._verify_ssl,
                follow_redirects=True,
            )
        return self._client

    def _handle_response(self, resp: httpx.Response) -> dict:
        if resp.status_code == 200 or resp.status_code == 202:
            return resp.json()
        if resp.status_code == 401:
            raise DPDPAuthError(f"Authentication failed: {resp.text}")
        if resp.status_code == 403:
            raise DPDPAuthError(f"Authorization denied: {resp.text}")
        if resp.status_code == 422:
            raise DPDPValidationError(f"Validation error: {resp.text}")
        if resp.status_code == 429:
            retry_after = None
            try:
                retry_after = int(resp.headers.get("Retry-After", ""))
            except (ValueError, TypeError):
                pass
            raise DPDPRateLimitError(
                f"Rate limit exceeded. Retry after {retry_after}s.",
                retry_after=retry_after,
            )
        if resp.status_code >= 500:
            raise DPDPServerError(f"Server error {resp.status_code}: {resp.text}")
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, payload: dict, timeout: Optional[float] = None) -> dict:
        client = self._get_client()
        last_exc: Exception = DPDPError("No attempts made")
        for attempt in range(self._max_retries + 1):
            try:
                resp = client.post(
                    f"{_BASE_PATH}{path}",
                    json=payload,
                    timeout=timeout or self._timeout,
                )
                return self._handle_response(resp)
            except (DPDPServerError, httpx.TransportError) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise last_exc from None

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        client = self._get_client()
        resp = client.get(f"{_BASE_PATH}{path}", params=params)
        return self._handle_response(resp)

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        if self._client and not self._client.is_closed:
            self._client.close()

    def __enter__(self) -> "DPDPClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── Moderation endpoints ─────────────────────────────────────────────────

    def moderate(
        self,
        text: str,
        context: Optional[List[str]] = None,
    ) -> DPDPResult:
        """
        Moderate a single text prompt.

        Args:
            text:    The prompt or content to evaluate (max 8000 chars).
            context: Optional prior conversation turns (session context).
                     Providing context improves multi-turn escalation detection.

        Returns:
            DPDPResult with verdict, risk_score, violation_type, explanation.

        Example::

            result = client.moderate(
                "Share all user phone numbers with our marketing partner",
                context=["I need to run a campaign for our NBFC customers."],
            )
            if result.is_blocked:
                raise PermissionError(result.explanation)
        """
        payload: dict = {"text": text}
        if context:
            payload["context"] = context
        data = self._post("/moderate", payload)
        return DPDPResult._from_dict(data, text=text)

    def moderate_batch(
        self,
        texts: List[str],
        context: Optional[List[str]] = None,
    ) -> DPDPBatchResult:
        """
        Moderate a batch of up to 50 texts concurrently.

        Args:
            texts:   List of prompts (1–50 items, each max 8000 chars).
            context: Shared session context applied to every item in the batch.

        Returns:
            DPDPBatchResult containing individual DPDPResult per item.

        Example::

            batch = client.moderate_batch([
                "Export patient EHR records to S3",
                "Show aggregate lab test volumes by hospital",
                "Send Aadhaar-PAN combos to analytics warehouse",
            ])
            for r in batch.blocked:
                print(f"BLOCKED: {r.text[:80]}")
        """
        if not texts:
            raise ValueError("texts must be a non-empty list")
        if len(texts) > 50:
            raise ValueError(f"Batch size {len(texts)} exceeds maximum of 50")

        items = [{"text": t} for t in texts]
        if context:
            for item in items:
                item["context"] = context

        data = self._post("/moderate/batch", {"items": items}, timeout=_DEFAULT_BATCH_TIMEOUT)
        return DPDPBatchResult._from_dict(data)

    def moderate_async(
        self,
        text: str,
        webhook_url: str,
        webhook_secret: Optional[str] = None,
        context: Optional[List[str]] = None,
    ) -> DPDPAsyncToken:
        """
        Submit a text for moderation and receive the result via webhook (HTTP 202).

        The server will POST the result to `webhook_url` with an
        `X-KavachX-Signature: sha256=<hmac>` header if a `webhook_secret`
        is provided. Use `verify_webhook_signature()` to validate it.

        Args:
            text:           Text to moderate.
            webhook_url:    URL the server will POST the result to.
            webhook_secret: Optional HMAC-SHA256 signing secret.
            context:        Optional prior conversation turns.

        Returns:
            DPDPAsyncToken with `request_id` for correlation.

        Example::

            token = client.moderate_async(
                text="...",
                webhook_url="https://your-app.in/hooks/kavachx",
                webhook_secret="my-hmac-secret",
            )
            print("Job queued:", token.request_id)
        """
        payload: dict = {"text": text, "webhook_url": webhook_url}
        if webhook_secret:
            payload["webhook_secret"] = webhook_secret
        if context:
            payload["context"] = context
        data = self._post("/moderate/async", payload)
        return DPDPAsyncToken(
            request_id=data.get("request_id", ""),
            status=data.get("status", "queued"),
            raw=data,
        )

    # ── Feedback ─────────────────────────────────────────────────────────────

    def submit_feedback(
        self,
        text: str,
        correct_label: str,
        predicted_label: Optional[str] = None,
        note: Optional[str] = None,
    ) -> DPDPFeedback:
        """
        Submit a human-labelled correction to improve the model.

        Args:
            text:            The original prompt that was moderated.
            correct_label:   The correct class label (e.g. "personal_data", "safe").
            predicted_label: What the model predicted (optional, for tracking).
            note:            Free-text annotation from your compliance team.

        Returns:
            DPDPFeedback with status.

        Example::

            client.submit_feedback(
                text="Does your bank offer zero-balance accounts?",
                correct_label="safe",
                predicted_label="financial_data",
                note="Product enquiry — no PII involved",
            )
        """
        payload: dict = {"text": text, "correct_label": correct_label}
        if predicted_label:
            payload["predicted_label"] = predicted_label
        if note:
            payload["note"] = note
        data = self._post("/feedback", payload)
        return DPDPFeedback(
            request_id=data.get("request_id", ""),
            status=data.get("status", ""),
            raw=data,
        )

    # ── Audit / decisions ────────────────────────────────────────────────────

    def list_decisions(
        self,
        verdict: Optional[str] = None,
        client_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[dict]:
        """
        Retrieve the audit trail of past moderation decisions.

        Args:
            verdict:   Filter by "BLOCK", "REVIEW", or "ALLOW".
            client_id: Filter by originating partner client ID.
            limit:     Maximum number of records to return (default 50).

        Returns:
            List of raw decision dicts from the audit trail.
        """
        params: dict = {"limit": limit}
        if verdict:
            params["verdict"] = verdict
        if client_id:
            params["client_id"] = client_id
        data = self._get("/decisions", params=params)
        return data.get("decisions", data)

    # ── Service status ───────────────────────────────────────────────────────

    def status(self) -> dict:
        """
        Check whether the KavachX service is healthy.

        Calls the /ready endpoint. Returns the raw JSON dict.
        Raises DPDPServerError if the service is degraded (HTTP 503).

        Example::

            info = client.status()
            print(info["status"])       # "ok"
            print(info["dpdp_model"])   # True / False
        """
        client = self._get_client()
        resp = client.get("/ready")
        return self._handle_response(resp)

    # ── Async client (context manager) ───────────────────────────────────────

    @staticmethod
    @asynccontextmanager
    async def async_client(
        base_url: str,
        api_key: Optional[str] = None,
        jwt_token: Optional[str] = None,
        timeout: float = _DEFAULT_TIMEOUT,
        verify_ssl: bool = True,
    ):
        """
        Async context manager yielding an `AsyncDPDPClient`.

        Usage::

            async with DPDPClient.async_client(
                base_url="https://kavachx.yourdomain.in",
                api_key="your-key",
            ) as client:
                result = await client.amoderate("...")
        """
        async_client = AsyncDPDPClient(
            base_url=base_url,
            api_key=api_key,
            jwt_token=jwt_token,
            timeout=timeout,
            verify_ssl=verify_ssl,
        )
        try:
            yield async_client
        finally:
            await async_client.aclose()


# ─── Async client ────────────────────────────────────────────────────────────

class AsyncDPDPClient:
    """
    Async variant of DPDPClient.  Use with `async with DPDPClient.async_client(...)`.

    All methods mirror the synchronous DPDPClient but are coroutines (await them).
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        jwt_token: Optional[str] = None,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = 2,
        verify_ssl: bool = True,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._jwt_token = jwt_token
        self._timeout = timeout
        self._max_retries = max_retries
        self._verify_ssl = verify_ssl
        self._client: Optional[httpx.AsyncClient] = None

    def _get_headers(self) -> dict:
        headers: dict = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        elif self._jwt_token:
            headers["Authorization"] = f"Bearer {self._jwt_token}"
        return headers

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=self._get_headers(),
                timeout=self._timeout,
                verify=self._verify_ssl,
                follow_redirects=True,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def _handle_response(self, resp: httpx.Response) -> dict:
        if resp.status_code in (200, 202):
            return resp.json()
        if resp.status_code == 401:
            raise DPDPAuthError(f"Authentication failed: {resp.text}")
        if resp.status_code == 403:
            raise DPDPAuthError(f"Authorization denied: {resp.text}")
        if resp.status_code == 422:
            raise DPDPValidationError(f"Validation error: {resp.text}")
        if resp.status_code == 429:
            retry_after = None
            try:
                retry_after = int(resp.headers.get("Retry-After", ""))
            except (ValueError, TypeError):
                pass
            raise DPDPRateLimitError(
                f"Rate limit exceeded. Retry after {retry_after}s.",
                retry_after=retry_after,
            )
        if resp.status_code >= 500:
            raise DPDPServerError(f"Server error {resp.status_code}: {resp.text}")
        resp.raise_for_status()
        return resp.json()

    async def _post(self, path: str, payload: dict, timeout: Optional[float] = None) -> dict:
        import asyncio as _asyncio
        client = self._get_client()
        last_exc: Exception = DPDPError("No attempts made")
        for attempt in range(self._max_retries + 1):
            try:
                resp = await client.post(
                    f"{_BASE_PATH}{path}",
                    json=payload,
                    timeout=timeout or self._timeout,
                )
                return self._handle_response(resp)
            except (DPDPServerError, httpx.TransportError) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    await _asyncio.sleep(min(2 ** attempt, 8))
                    continue
                raise last_exc from None

    async def _get(self, path: str, params: Optional[dict] = None) -> dict:
        client = self._get_client()
        resp = await client.get(f"{_BASE_PATH}{path}", params=params)
        return self._handle_response(resp)

    async def amoderate(
        self,
        text: str,
        context: Optional[List[str]] = None,
    ) -> DPDPResult:
        """Async version of moderate()."""
        payload: dict = {"text": text}
        if context:
            payload["context"] = context
        data = await self._post("/moderate", payload)
        return DPDPResult._from_dict(data, text=text)

    async def amoderate_batch(
        self,
        texts: List[str],
        context: Optional[List[str]] = None,
    ) -> DPDPBatchResult:
        """Async version of moderate_batch()."""
        if not texts:
            raise ValueError("texts must be a non-empty list")
        if len(texts) > 50:
            raise ValueError(f"Batch size {len(texts)} exceeds maximum of 50")
        items = [{"text": t} for t in texts]
        if context:
            for item in items:
                item["context"] = context
        data = await self._post("/moderate/batch", {"items": items}, timeout=_DEFAULT_BATCH_TIMEOUT)
        return DPDPBatchResult._from_dict(data)

    async def amoderate_async(
        self,
        text: str,
        webhook_url: str,
        webhook_secret: Optional[str] = None,
        context: Optional[List[str]] = None,
    ) -> DPDPAsyncToken:
        """Async version of moderate_async()."""
        payload: dict = {"text": text, "webhook_url": webhook_url}
        if webhook_secret:
            payload["webhook_secret"] = webhook_secret
        if context:
            payload["context"] = context
        data = await self._post("/moderate/async", payload)
        return DPDPAsyncToken(
            request_id=data.get("request_id", ""),
            status=data.get("status", "queued"),
            raw=data,
        )

    async def asubmit_feedback(
        self,
        text: str,
        correct_label: str,
        predicted_label: Optional[str] = None,
        note: Optional[str] = None,
    ) -> DPDPFeedback:
        """Async version of submit_feedback()."""
        payload: dict = {"text": text, "correct_label": correct_label}
        if predicted_label:
            payload["predicted_label"] = predicted_label
        if note:
            payload["note"] = note
        data = await self._post("/feedback", payload)
        return DPDPFeedback(
            request_id=data.get("request_id", ""),
            status=data.get("status", ""),
            raw=data,
        )

    async def alist_decisions(
        self,
        verdict: Optional[str] = None,
        client_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[dict]:
        """Async version of list_decisions()."""
        params: dict = {"limit": limit}
        if verdict:
            params["verdict"] = verdict
        if client_id:
            params["client_id"] = client_id
        data = await self._get("/decisions", params=params)
        return data.get("decisions", data)

    async def astatus(self) -> dict:
        """Async version of status()."""
        client = self._get_client()
        resp = await client.get("/ready")
        return self._handle_response(resp)
