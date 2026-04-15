from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from app.services.websocket_manager import manager

router = APIRouter()


@router.websocket("/stream")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(default=""),
):
    """
    Authenticated WebSocket stream.

    Clients must supply a valid JWT bearer token as the `token` query parameter:
        ws://host/api/v1/ws/stream?token=<jwt>

    The server closes the connection with code 4001 on auth failure.
    After connecting, the server sends { "type": "ping" } every 30 s.
    Clients should respond with { "type": "pong" } to keep the connection alive.
    """
    authenticated = await manager.connect(websocket, token)
    if not authenticated:
        return

    try:
        while True:
            data = await websocket.receive_json()
            # Client pong — no-op (heartbeat is tracked server-side)
            if isinstance(data, dict) and data.get("type") == "pong":
                continue
            # Legacy text ping support
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)
