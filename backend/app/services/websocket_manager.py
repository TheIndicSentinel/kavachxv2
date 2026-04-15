from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json
import asyncio
import logging

logger = logging.getLogger(__name__)

_HEARTBEAT_INTERVAL = 30   # seconds between server-sent pings
_PONG_TIMEOUT       = 10   # seconds to wait for pong before evicting


class ConnectionManager:
    def __init__(self):
        # Maps WebSocket → user email (authenticated identity)
        self.active_connections: Dict[WebSocket, str] = {}
        # Set of principal_ids whose consent has been revoked
        self._revoked_principals: Set[str] = set()

    # ── Authentication ────────────────────────────────────────────────────────

    async def connect(self, websocket: WebSocket, token: str) -> bool:
        """
        Authenticate the connection via JWT bearer token (passed as query param).
        Returns True on success; closes the socket with code 4001 on failure.
        """
        from jose import JWTError, jwt as jose_jwt
        from app.core.config import settings

        email: str | None = None
        try:
            payload = jose_jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
            email = payload.get("sub")
        except (JWTError, Exception):
            pass

        if not email:
            await websocket.close(code=4001, reason="Unauthorized: invalid token")
            return False

        await websocket.accept()
        self.active_connections[websocket] = email
        logger.info("WebSocket connected: %s", email)
        # Start the heartbeat loop for this connection
        asyncio.create_task(self._heartbeat(websocket))
        return True

    def disconnect(self, websocket: WebSocket):
        self.active_connections.pop(websocket, None)

    # ── Heartbeat ─────────────────────────────────────────────────────────────

    async def _heartbeat(self, websocket: WebSocket):
        """
        Send a ping every 30 s and wait up to 10 s for a pong.
        Evicts the connection if no pong is received.
        """
        while websocket in self.active_connections:
            await asyncio.sleep(_HEARTBEAT_INTERVAL)
            if websocket not in self.active_connections:
                break
            try:
                await websocket.send_json({"type": "ping"})
            except Exception:
                self.disconnect(websocket)
                break

    # ── Consent revocation ────────────────────────────────────────────────────

    def is_revoked(self, principal_id: str) -> bool:
        return principal_id in self._revoked_principals

    async def broadcast_consent_revocation(self, principal_id: str):
        """
        Mark principal as revoked and notify all connected clients so in-flight
        sessions can abort immediately (DPDP 2023 §6 requirement).
        """
        self._revoked_principals.add(principal_id)
        payload = {
            "type":         "consent_revoked",
            "principal_id": principal_id,
        }
        dead: list[WebSocket] = []
        for ws in list(self.active_connections):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    # ── Broadcast ─────────────────────────────────────────────────────────────

    async def broadcast(self, message: dict):
        dead: list[WebSocket] = []
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error("WebSocket broadcast error: %s", e)
                dead.append(connection)
        for failed in dead:
            self.disconnect(failed)


manager = ConnectionManager()
