"""
WebSocket Chat Handler
======================
Handles real-time chat via WebSocket with streaming responses.
"""

import json
import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()
logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def send_json(self, websocket: WebSocket, data: dict):
        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")


manager = ConnectionManager()


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for chat."""
    import uuid
    connection_id = str(uuid.uuid4())  # Unique ID for this connection
    
    await manager.connect(websocket)
    logger.info(f"New connection: {connection_id}")

    try:
        # Session created lazily after we receive API keys
        from web.agent_wrapper import create_session, get_session, close_session
        session = None

        while True:
            data = await websocket.receive_json()
            message = data.get("message", "").strip()

            # Handle API key configuration from client
            if data.get("type") == "configure_keys":
                api_keys = {
                    "openai_api_key": data.get("openai_api_key", ""),
                    "arraylake_api_key": data.get("arraylake_api_key", ""),
                    "hf_token": data.get("hf_token", ""),
                }
                session = create_session(connection_id, api_keys=api_keys)
                ready = session.is_ready()
                await manager.send_json(websocket, {
                    "type": "keys_configured",
                    "ready": ready,
                })
                continue

            # Create default session if not yet created (keys from env)
            if session is None:
                session = create_session(connection_id)

            if not message:
                continue

            logger.info(f"[{connection_id[:8]}] Received: {message[:100]}...")

            # Handle /clear command — clear session memory + UI
            if message.strip() == "/clear":
                session = get_session(connection_id)
                if session:
                    session.clear_messages()
                await manager.send_json(websocket, {"type": "clear"})
                continue

            # Send thinking indicator
            await manager.send_json(websocket, {"type": "thinking"})

            try:
                # Get session for this connection (auto-recreate if lost)
                session = get_session(connection_id)
                if not session:
                    logger.warning(f"Session lost for {connection_id[:8]}, requesting keys...")
                    # Ask client to resend keys (e.g., after container restart)
                    await manager.send_json(websocket, {
                        "type": "request_keys",
                        "reason": "Session expired, please reconnect."
                    })
                    continue

                # Callback for streaming
                async def stream_callback(event_type: str, content: str, **kwargs):
                    msg = {"type": event_type, "content": content}
                    msg.update(kwargs)
                    await manager.send_json(websocket, msg)

                # Process message
                response = await session.process_message(message, stream_callback)

                # Send complete
                await manager.send_json(websocket, {
                    "type": "complete",
                    "content": response
                })

            except Exception as e:
                logger.exception(f"Error: {e}")
                await manager.send_json(websocket, {
                    "type": "error",
                    "content": str(e)
                })

    except WebSocketDisconnect:
        logger.info(f"Connection {connection_id[:8]} disconnected")
        manager.disconnect(websocket)
        close_session(connection_id)  # Clean up session
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
        manager.disconnect(websocket)
        close_session(connection_id)  # Clean up session
