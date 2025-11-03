"""
Data models for session management and tracking.
"""
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from fastapi import WebSocket
from aiortc import RTCPeerConnection
import time


@dataclass
class LatencyMetrics:
    """Track latency metrics for the voice pipeline"""
    timestamps: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

    def mark(self, event: str) -> None:
        """Mark a timestamp for an event"""
        self.timestamps[event] = time.time()

    def measure(self, start_event: str, end_event: str, metric_name: str) -> Optional[float]:
        """Measure latency between two events in milliseconds"""
        if start_event in self.timestamps and end_event in self.timestamps:
            latency = (self.timestamps[end_event] -
                       self.timestamps[start_event]) * 1000
            self.metrics[metric_name] = latency
            return latency
        return None

    def get_metrics(self) -> Dict[str, float]:
        """Get all measured metrics"""
        return self.metrics.copy()

    def reset(self) -> None:
        """Reset all timestamps and metrics"""
        self.timestamps.clear()
        self.metrics.clear()


@dataclass
class ConversationMessage:
    """Represents a message in the conversation history"""
    role: str  # "system", "user", or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class WebRTCSession:
    """WebRTC session data"""
    session_id: str
    peer_connection: RTCPeerConnection
    websocket: Optional[WebSocket] = None
    latency_tracker: LatencyMetrics = field(default_factory=LatencyMetrics)
    audio_track: Optional[object] = None
    conversation_history: List[ConversationMessage] = field(
        default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            "pc": self.peer_connection,
            "ws": self.websocket,
            "latency_tracker": self.latency_tracker,
            "audio_track": self.audio_track
        }


@dataclass
class AudioStats:
    """Statistics for audio streaming"""
    chunks_received: int = 0
    total_bytes: int = 0
    start_time: Optional[float] = None

    def increment(self, bytes_count: int) -> None:
        """Increment audio statistics"""
        self.chunks_received += 1
        self.total_bytes += bytes_count
        if self.start_time is None:
            self.start_time = time.time()

    def reset(self) -> None:
        """Reset statistics"""
        self.chunks_received = 0
        self.total_bytes = 0
        self.start_time = None

    def get_duration(self) -> float:
        """Get duration in seconds"""
        if self.start_time:
            return time.time() - self.start_time
        return 0.0
