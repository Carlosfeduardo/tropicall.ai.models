"""
Session management for STT service.

Implements:
- Session with audio processing pipeline
- Transcript delivery to client with simple debounce
- Detailed timing instrumentation
- Cancellation support

Simplified approach:
- Partials emit Whisper result directly (no incremental state)
- Simple debounce: only emit if text changed from last partial
- Finals reset debounce state for next utterance
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field

import numpy as np
from fastapi import WebSocket

from ..config import settings
from ..inference.inference_queue import (
    InferenceQueue,
    get_inference_queue,
)
from ..protocol.errors import (
    MaxSessionsReachedError,
    SessionExistsError,
    STTError,
)
from ..protocol.messages import (
    FinalTranscriptMessage,
    PartialTranscriptMessage,
    TimingInfo,
    VADEventMessage,
    VADState,
    WordInfo,
)
from .audio_buffer import AudioBuffer
from .utterance import FinalRequest, PartialRequest, UtteranceAggregator

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """
    STT session with audio processing pipeline.
    
    Features:
    - Audio buffering with backpressure
    - VAD-based utterance detection
    - Partial and final transcript delivery
    - Simple text debounce for partials
    - Timing metrics for each transcript
    """

    session_id: str
    websocket: WebSocket
    language: str = field(default_factory=lambda: settings.language)
    vad_enabled: bool = field(default_factory=lambda: settings.vad_enabled)
    vad_threshold: float = field(default_factory=lambda: settings.vad_threshold)
    partial_enabled: bool = True
    word_timestamps: bool = field(default_factory=lambda: settings.word_timestamps)

    # Internal state
    _audio_buffer: AudioBuffer = field(init=False)
    _aggregator: UtteranceAggregator = field(init=False)
    _cancelled: bool = field(default=False, init=False)
    _created_at: float = field(init=False)
    _last_partial_text: str = field(default="", init=False)
    
    # Metrics
    _total_audio_ms: float = field(default=0.0, init=False)
    _total_segments: int = field(default=0, init=False)
    _total_partial_transcripts: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Initialize session components."""
        self._created_at = time.monotonic()
        
        # Create audio buffer
        self._audio_buffer = AudioBuffer()
        
        # Create utterance aggregator with callbacks
        self._aggregator = UtteranceAggregator(
            on_partial=self._on_partial_request,
            on_final=self._on_final_request,
            on_vad_event=self._on_vad_event,
            vad_enabled=self.vad_enabled,
            vad_threshold=self.vad_threshold,
            partial_enabled=self.partial_enabled,
        )
        
        # Initialize VAD
        self._aggregator.init_vad()

    @property
    def is_cancelled(self) -> bool:
        """Check if session is cancelled."""
        return self._cancelled

    @property
    def session_duration_ms(self) -> float:
        """Session duration in milliseconds."""
        return (time.monotonic() - self._created_at) * 1000

    async def handle_audio_chunk(self, data: bytes) -> None:
        """
        Handle incoming audio chunk.
        
        Args:
            data: PCM16 LE audio bytes
        """
        if self._cancelled:
            return

        # Track timing
        preprocess_start = time.monotonic()

        # Add to buffer and check for overflow
        overflow = self._audio_buffer.append_pcm16(data)
        if overflow:
            logger.warning(f"Session {self.session_id}: buffer overflow")

        # Get samples as float32
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Track audio duration
        audio_ms = len(samples) / settings.sample_rate * 1000
        self._total_audio_ms += audio_ms

        preprocess_ms = (time.monotonic() - preprocess_start) * 1000

        # Process through aggregator
        await self._aggregator.process_audio(samples)

    async def handle_flush(self) -> None:
        """Handle flush message - force finalization."""
        if self._cancelled:
            return

        logger.debug(f"Session {self.session_id}: flush requested")
        await self._aggregator.flush()

    async def handle_cancel(self) -> None:
        """Handle cancel message - drop buffers."""
        self._cancelled = True
        await self._aggregator.cancel()
        logger.info(f"Session {self.session_id}: cancelled")

    async def handle_end_session(self) -> None:
        """Handle end session - flush and close."""
        if not self._cancelled:
            await self._aggregator.flush()

    async def _on_partial_request(self, request: PartialRequest) -> None:
        """
        Handle partial transcript request from aggregator.
        
        Emits Whisper result directly with simple debounce.
        """
        if self._cancelled:
            return

        try:
            # Get inference queue
            queue = get_inference_queue()
            
            # Enqueue for inference (no initial_prompt, no word timestamps for partials)
            result = await queue.enqueue_partial(
                audio=request.audio,
                language=self.language,
                session_id=self.session_id,
                segment_id=request.segment_id,
                word_timestamps=False,  # Not needed for partials - faster
            )
            
            # Result is None if partial was shed or coalesced
            if result is None:
                logger.debug(f"Partial [{request.segment_id}]: shed/coalesced")
                return
            
            # Simple debounce - only emit if text changed
            text = result.text.strip()
            if text == self._last_partial_text:
                logger.debug(f"Partial [{request.segment_id}]: debounced (no change)")
                return
            
            self._last_partial_text = text
            
            # Send partial transcript
            timing = TimingInfo(
                preprocess_ms=0.0,
                queue_wait_ms=result.queue_wait_ms,
                inference_ms=result.inference_ms,
                postprocess_ms=result.postprocess_ms,
                server_total_ms=result.server_total_ms,
            )
            
            message = PartialTranscriptMessage(
                session_id=self.session_id,
                segment_id=request.segment_id,
                text=text,
                confidence=result.confidence,
                timing=timing,
            )
            
            await self.websocket.send_json(message.model_dump())
            self._total_partial_transcripts += 1
            
            logger.debug(
                f"Partial [{request.segment_id}]: '{text[:50]}...' "
                f"(inf={result.inference_ms:.0f}ms)"
            )
            
        except STTError as e:
            # Admission control errors - log but don't fail session
            logger.debug(f"Partial skipped due to {e.code}: {e.message}")
        except Exception as e:
            logger.exception(f"Error processing partial request: {e}")

    async def _on_final_request(self, request: FinalRequest) -> None:
        """
        Handle final transcript request from aggregator.
        
        Resets debounce state after emitting final.
        """
        if self._cancelled:
            return

        try:
            # Get inference queue
            queue = get_inference_queue()
            
            # Enqueue for inference (finals always accepted, never shed)
            result = await queue.enqueue_final(
                audio=request.audio,
                language=self.language,
                session_id=self.session_id,
                segment_id=request.segment_id,
                word_timestamps=self.word_timestamps,
            )
            
            # Convert word timestamps
            words = []
            if self.word_timestamps and result.words:
                for w in result.words:
                    words.append(
                        WordInfo(
                            word=w.word,
                            start=w.start,
                            end=w.end,
                            probability=w.probability,
                        )
                    )
            
            # Send final transcript
            timing = TimingInfo(
                preprocess_ms=0.0,
                queue_wait_ms=result.queue_wait_ms,
                inference_ms=result.inference_ms,
                postprocess_ms=result.postprocess_ms,
                server_total_ms=result.server_total_ms,
            )
            
            message = FinalTranscriptMessage(
                session_id=self.session_id,
                segment_id=request.segment_id,
                text=result.text,
                confidence=result.confidence,
                language=result.language,
                language_probability=result.language_probability,
                words=words,
                audio_duration_ms=result.audio_duration_ms,
                timing=timing,
            )
            
            await self.websocket.send_json(message.model_dump())
            self._total_segments += 1
            
            # Reset debounce for next utterance
            self._last_partial_text = ""
            
            logger.info(
                f"Final [{request.segment_id}]: '{result.text[:80]}' "
                f"(dur={result.audio_duration_ms:.0f}ms, inf={result.inference_ms:.0f}ms)"
            )
            
        except STTError as e:
            logger.warning(f"Final transcript failed: {e.code}: {e.message}")
            raise
        except Exception as e:
            logger.exception(f"Error processing final request: {e}")
            raise

    async def _on_vad_event(self, state: str, timestamp_ms: float) -> None:
        """Handle VAD state change event."""
        if self._cancelled:
            return

        try:
            vad_state = VADState.SPEECH_START if state == "speech_start" else VADState.SPEECH_END
            
            message = VADEventMessage(
                session_id=self.session_id,
                state=vad_state,
                t_ms=timestamp_ms,
            )
            
            await self.websocket.send_json(message.model_dump())
            
        except Exception as e:
            logger.exception(f"Error sending VAD event: {e}")

    def get_metrics(self) -> dict:
        """Get session metrics."""
        return {
            "session_id": self.session_id,
            "duration_ms": round(self.session_duration_ms, 2),
            "total_audio_ms": round(self._total_audio_ms, 2),
            "total_segments": self._total_segments,
            "total_partial_transcripts": self._total_partial_transcripts,
            "buffer_stats": self._audio_buffer.get_stats().__dict__,
            "aggregator_stats": self._aggregator.get_stats(),
        }


class SessionManager:
    """
    Manages active STT sessions.
    
    Thread-safe session tracking with max sessions limit.
    """

    def __init__(self, max_sessions: int | None = None):
        """
        Initialize session manager.
        
        Args:
            max_sessions: Maximum concurrent sessions
        """
        self.max_sessions = max_sessions or settings.max_sessions
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def create_session(
        self,
        session_id: str,
        websocket: WebSocket,
        language: str | None = None,
        vad_enabled: bool | None = None,
        vad_threshold: float | None = None,
        partial_enabled: bool | None = None,
        word_timestamps: bool | None = None,
    ) -> Session:
        """
        Create a new session.
        
        Args:
            session_id: Unique session identifier
            websocket: WebSocket connection
            language: Language code
            vad_enabled: Enable VAD
            vad_threshold: VAD threshold
            partial_enabled: Enable partial results
            word_timestamps: Enable word timestamps
            
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
                websocket=websocket,
                language=language or settings.language,
                vad_enabled=vad_enabled if vad_enabled is not None else settings.vad_enabled,
                vad_threshold=vad_threshold if vad_threshold is not None else settings.vad_threshold,
                partial_enabled=partial_enabled if partial_enabled is not None else True,
                word_timestamps=word_timestamps if word_timestamps is not None else settings.word_timestamps,
            )
            
            self._sessions[session_id] = session
            logger.info(f"Session {session_id} created (total: {len(self._sessions)})")
            return session

    async def get_session(self, session_id: str) -> Session | None:
        """Get session by ID."""
        return self._sessions.get(session_id)

    async def remove_session(self, session_id: str) -> None:
        """Remove session."""
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"Session {session_id} removed (total: {len(self._sessions)})")

    @property
    def active_sessions(self) -> int:
        """Get number of active sessions."""
        return len(self._sessions)

    def get_all_sessions(self) -> list[Session]:
        """Get all active sessions."""
        return list(self._sessions.values())


# Global instance
session_manager = SessionManager()
