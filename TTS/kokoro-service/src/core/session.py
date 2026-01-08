"""
Session management for TTS service.

Implements:
- Session with cancellation support
- Fairness control via max inflight segments
- Request tracking for barge-in
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import uuid4

from fastapi import WebSocket

from ..config import settings
from ..inference.inference_queue import get_inference_queue
from ..protocol.errors import ClientSlowError, MaxSessionsReachedError, SessionExistsError
from .audio_streamer import SegmentMetrics, stream_audio_to_client
from .normalizer import normalizer
from .text_accumulator import DebouncedTextAccumulator

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """
    TTS session with cancellation and fairness support.
    
    Features:
    - Cancellation via cancelled flag and current_request_id
    - Fairness via max inflight segments limit
    - Debounced text accumulation
    - Text normalization
    """

    session_id: str
    voice: str
    speed: float
    websocket: WebSocket

    # Cancellation
    cancelled: bool = field(default=False, init=False)
    current_request_id: str | None = field(default=None, init=False)

    # Fairness/Backpressure
    inflight_segments: int = field(default=0, init=False)
    segment_counter: int = field(default=0, init=False)

    # Components (initialized in __post_init__)
    accumulator: DebouncedTextAccumulator = field(init=False)

    # Metrics
    total_segments: int = field(default=0, init=False)
    total_audio_duration_ms: float = field(default=0.0, init=False)
    total_ttfa_ms: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        """Initialize components after dataclass init."""
        # Create accumulator with callback to enqueue segments
        self.accumulator = DebouncedTextAccumulator(
            on_segment_ready=self._on_segment_ready,
            debounce_ms=settings.debounce_ms,
            min_tokens=settings.min_tokens,
            min_tokens_flush=settings.min_tokens_flush,
        )

    def cancel(self) -> None:
        """Mark session as cancelled - pending requests will be ignored."""
        self.cancelled = True
        logger.info(f"Session {self.session_id} cancelled")

    def generate_request_id(self) -> str:
        """Generate new request_id and update current."""
        self.segment_counter += 1
        request_id = f"{self.session_id}-seg-{self.segment_counter}"
        self.current_request_id = request_id
        return request_id

    async def _on_segment_ready(self, text: str) -> None:
        """
        Callback when accumulator has a segment ready.
        Respects fairness limit.
        """
        # Check cancellation
        if self.cancelled:
            return

        # Fairness: limit segments in flight PER SESSION
        if self.inflight_segments >= settings.max_inflight_segments:
            logger.warning(
                f"Session {self.session_id} hit inflight limit "
                f"({self.inflight_segments} segments)"
            )
            raise ClientSlowError(self.inflight_segments)

        await self.enqueue_segment(text)

    async def enqueue_segment(self, text: str) -> None:
        """
        Enqueue segment for inference with fairness control.
        
        Args:
            text: Text to synthesize
        """
        if self.cancelled:
            return

        # Normalize text
        normalized_text = normalizer.normalize(text)

        # Generate IDs
        request_id = self.generate_request_id()
        segment_id = f"seg-{uuid4().hex[:8]}"

        # Increment counter BEFORE enqueueing
        self.inflight_segments += 1

        try:
            # Get the inference queue
            queue = get_inference_queue()

            # Enqueue and wait for result (serialized by consumer)
            result = await queue.enqueue(
                text=normalized_text,
                voice=self.voice,
                speed=self.speed,
                session_id=self.session_id,
                request_id=request_id,
            )

            # Check if request is still valid (cancellation or newer request)
            if self.cancelled or request_id != self.current_request_id:
                logger.debug(
                    f"Discarding result for outdated request {request_id}"
                )
                return

            # Update metrics
            self.total_segments += 1
            self.total_audio_duration_ms += result.audio_duration_ms
            self.total_ttfa_ms += result.ttfa_ms

            # Stream audio with metrics
            metrics = SegmentMetrics(
                segment_id=segment_id,
                request_id=request_id,
                ttfa_ms=result.ttfa_ms,
                rtf=result.rtf,
                total_samples=len(result.audio),
            )
            await stream_audio_to_client(self, result.audio, metrics)

        finally:
            # ALWAYS decrement (try/finally guarantees)
            self.inflight_segments -= 1

    async def handle_send_text(self, text: str) -> None:
        """Handle send_text message."""
        await self.accumulator.add_text(text)

    async def handle_flush(self) -> None:
        """Handle flush message."""
        await self.accumulator.flush()

    async def handle_end_session(self) -> None:
        """Handle end_session message."""
        await self.accumulator.flush()

    async def handle_cancel(self) -> None:
        """Handle cancel message (barge-in)."""
        self.cancel()
        await self.accumulator.cancel()

    def get_metrics(self) -> dict:
        """Get session metrics for final report."""
        avg_ttfa = (
            self.total_ttfa_ms / self.total_segments
            if self.total_segments > 0
            else 0
        )
        avg_rtf = (
            self.total_audio_duration_ms / (self.total_ttfa_ms or 1)
            if self.total_segments > 0
            else 0
        )

        return {
            "segments_processed": self.total_segments,
            "audio_duration_ms": round(self.total_audio_duration_ms, 1),
            "ttfa_ms": round(avg_ttfa, 2),
            "rtf": round(avg_rtf, 3),
        }


class SessionManager:
    """
    Manages active TTS sessions.
    
    Thread-safe session tracking with max sessions limit.
    """

    def __init__(self, max_sessions: int = 100):
        """
        Initialize session manager.
        
        Args:
            max_sessions: Maximum concurrent sessions
        """
        self.max_sessions = max_sessions
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def create_session(
        self,
        session_id: str,
        voice: str,
        speed: float,
        websocket: WebSocket,
    ) -> Session:
        """
        Create a new session.
        
        Args:
            session_id: Unique session identifier
            voice: Voice to use
            speed: Speech speed
            websocket: WebSocket connection
            
        Returns:
            New Session instance
            
        Raises:
            SessionExistsError: If session already exists
            MaxSessionsReachedError: If max sessions limit reached
        """
        async with self._lock:
            if session_id in self._sessions:
                raise SessionExistsError(session_id)

            if len(self._sessions) >= self.max_sessions:
                raise MaxSessionsReachedError(self.max_sessions)

            session = Session(
                session_id=session_id,
                voice=voice,
                speed=speed,
                websocket=websocket,
            )
            self._sessions[session_id] = session
            logger.info(
                f"Session {session_id} created "
                f"(total: {len(self._sessions)})"
            )
            return session

    async def get_session(self, session_id: str) -> Session | None:
        """Get session by ID."""
        return self._sessions.get(session_id)

    async def remove_session(self, session_id: str) -> None:
        """Remove session."""
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(
                    f"Session {session_id} removed "
                    f"(total: {len(self._sessions)})"
                )

    @property
    def active_sessions(self) -> int:
        """Get number of active sessions."""
        return len(self._sessions)

    def get_all_sessions(self) -> list[Session]:
        """Get all active sessions."""
        return list(self._sessions.values())


# Global instance
session_manager = SessionManager(max_sessions=settings.max_sessions)
