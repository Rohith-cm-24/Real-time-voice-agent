"""
WebSocket connection management.
"""
import logging
from typing import List
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            f"New connection established. Total connections: {len(self.active_connections)}"
        )

    def disconnect(self, websocket: WebSocket) -> None:
        """Unregister a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(
            f"Connection closed. Total connections: {len(self.active_connections)}"
        )

    async def send_message(self, message: str, websocket: WebSocket) -> None:
        """Send a text message to a specific WebSocket connection"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    async def broadcast(self, message: str) -> None:
        """Broadcast a message to all active connections"""
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")

    def get_connection_count(self) -> int:
        """Get the number of active connections"""
        return len(self.active_connections)
