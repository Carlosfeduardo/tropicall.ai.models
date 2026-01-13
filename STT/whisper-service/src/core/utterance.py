"""
Utterance aggregator for streaming STT.

Manages the lifecycle of utterances:
- Collects audio samples during speech
- Triggers partial decodes at regular intervals
- Triggers final decode on end-of-speech or timeout
- Handles forced flush and cancellation

Simplified approach:
- Always sends ALL audio from current utterance for transcription
- No incremental/cursor logic - Whisper gets full context
- Simple debounce handled at session level
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Awaitable
from uuid import uuid4

import numpy as np

from ..config import settings
from .audio_buffer import UtteranceBuffer
from .vad import SileroVAD

logger = logging.getLogger(__name__)


@dataclass
class PartialRequest:
    """Request for partial transcription."""

    segment_id: str
    audio: np.ndarray
    timestamp_ms: float


@dataclass
class FinalRequest:
    """Request for final transcription."""

    segment_id: str
    audio: np.ndarray
    duration_ms: float
    timestamp_ms: float


@dataclass
class UtteranceAggregator:
    """
    Aggregates audio samples into utterances for transcription.
    
    Features:
    - VAD-based speech detection
    - Partial transcription at regular intervals (sends ALL audio)
    - Final transcription on end-of-speech
    - Max utterance duration limit
    """

    on_partial: Callable[[PartialRequest], Awaitable[None]] | None = None
    on_final: Callable[[FinalRequest], Awaitable[None]] | None = None
    on_vad_event: Callable[[str, float], Awaitable[None]] | None = None
    on_buffer_overflow: Callable[[float, float], Awaitable[None]] | None = None

    vad_enabled: bool = field(default_factory=lambda: settings.vad_enabled)
    vad_threshold: float = field(default_factory=lambda: settings.vad_threshold)
    partial_enabled: bool = True
    partial_interval_ms: int = field(default_factory=lambda: settings.partial_interval_ms)
    min_utterance_ms: int = field(default_factory=lambda: settings.min_utterance_ms)
    sample_rate: int = field(default_factory=lambda: settings.sample_rate)
    max_utterance_s: float = field(default_factory=lambda: settings.max_utterance_s)

    _vad: SileroVAD | None = field(default=None, init=False)
    _utterance_buffer: UtteranceBuffer = field(init=False)
    _current_segment_id: str | None = field(default=None, init=False)
    _session_start_time: float = field(init=False)
    _last_partial_time: float = field(default=0.0, init=False)
    _utterance_start_time: float = field(default=0.0, init=False)
    _in_speech: bool = field(default=False, init=False)
    _cancelled: bool = field(default=False, init=False)
    _total_audio_ms: float = field(default=0.0, init=False)
    _total_segments: int = field(default=0, init=False)
    _partial_task: asyncio.Task | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._utterance_buffer = UtteranceBuffer(sample_rate=self.sample_rate)
        self._session_start_time = time.monotonic()

    def init_vad(self) -> None:
        """Initialize VAD if enabled."""
        if not self.vad_enabled:
            return
        self._vad = SileroVAD(
            threshold=self.vad_threshold,
            sample_rate=self.sample_rate,
            on_speech_start=self._handle_vad_speech_start,
            on_speech_end=self._handle_vad_speech_end,
        )
        self._vad.load_model()

    def _handle_vad_speech_start(self, timestamp_ms: float) -> None:
        """Handle VAD speech start event."""
        if self._cancelled:
            return
        self._in_speech = True
        self._current_segment_id = f"seg-{uuid4().hex[:8]}"
        self._utterance_buffer.mark_speech_start()
        self._last_partial_time = time.monotonic()
        self._utterance_start_time = time.monotonic()
        logger.debug(f"Speech started: segment={self._current_segment_id}")
        if self.on_vad_event:
            asyncio.create_task(self.on_vad_event("speech_start", timestamp_ms))

    def _handle_vad_speech_end(self, timestamp_ms: float) -> None:
        """Handle VAD speech end event."""
        if self._cancelled:
            return
        self._in_speech = False
        logger.debug(f"Speech ended: segment={self._current_segment_id}")
        if self.on_vad_event:
            asyncio.create_task(self.on_vad_event("speech_end", timestamp_ms))

    async def process_audio(self, audio: np.ndarray) -> bool:
        """
        Process incoming audio chunk.
        
        Args:
            audio: Float32 audio samples
            
        Returns:
            True if buffer overflow occurred
        """
        if self._cancelled:
            return False

        overflow = False
        num_samples = len(audio)
        audio_duration_ms = num_samples / self.sample_rate * 1000
        self._total_audio_ms += audio_duration_ms

        # Process through VAD if enabled
        if self._vad is not None:
            self._vad.process_chunk(audio)

        # Accumulate audio if in speech or VAD disabled
        if self._in_speech or not self.vad_enabled:
            at_max = self._utterance_buffer.append(audio)

            if at_max:
                logger.info(f"Max utterance duration reached for {self._current_segment_id}")
                await self._emit_final()
            elif self.partial_enabled and self._should_emit_partial():
                await self._emit_partial()
            elif not self.vad_enabled and self._should_force_periodic_final():
                logger.info("Periodic final triggered (no VAD)")
                await self._emit_final()

        # Emit final when speech ends
        elif not self._utterance_buffer.is_empty:
            await self._emit_final()

        return overflow

    def _should_emit_partial(self) -> bool:
        """Check if we should emit a partial based on timing."""
        if self._utterance_buffer.is_empty:
            return False
        if self._utterance_buffer.duration_ms < self.min_utterance_ms:
            return False
        elapsed_ms = (time.monotonic() - self._last_partial_time) * 1000
        return elapsed_ms >= self.partial_interval_ms

    def _should_force_periodic_final(self) -> bool:
        """Check if we should force a final when VAD is disabled."""
        if self.vad_enabled:
            return False
        if self._utterance_buffer.is_empty:
            return False
        # Force final after max_utterance_s without VAD
        utterance_duration_s = self._utterance_buffer.duration_ms / 1000
        return utterance_duration_s >= self.max_utterance_s

    async def _emit_partial(self) -> None:
        """
        Emit partial transcription request.
        
        Sends ALL audio from current utterance (limited to max window).
        """
        if self._cancelled or self.on_partial is None:
            return

        if self._current_segment_id is None:
            self._current_segment_id = f"seg-{uuid4().hex[:8]}"

        # Get ALL audio from utterance (limited to max window internally)
        audio = self._utterance_buffer.get_samples_for_partial()
        if len(audio) == 0:
            return

        request = PartialRequest(
            segment_id=self._current_segment_id,
            audio=audio,
            timestamp_ms=self._get_session_time_ms(),
        )

        self._last_partial_time = time.monotonic()
        self._utterance_buffer.mark_partial_sent()

        try:
            await self.on_partial(request)
        except Exception as e:
            logger.exception(f"Error emitting partial: {e}")

    async def _emit_final(self) -> None:
        """
        Emit final transcription request.
        
        Sends ALL audio from current utterance.
        """
        if self._cancelled:
            return

        audio = self._utterance_buffer.get_samples_for_final()

        # Skip if utterance is too short
        if len(audio) < int(self.sample_rate * self.min_utterance_ms / 1000):
            logger.debug("Utterance too short, skipping")
            self._utterance_buffer.clear()
            self._current_segment_id = None
            return

        segment_id = self._current_segment_id or f"seg-{uuid4().hex[:8]}"
        duration_ms = len(audio) / self.sample_rate * 1000

        request = FinalRequest(
            segment_id=segment_id,
            audio=audio,
            duration_ms=duration_ms,
            timestamp_ms=self._get_session_time_ms(),
        )

        # Reset state for next utterance
        self._utterance_buffer.clear()
        self._current_segment_id = None
        self._total_segments += 1

        if self._vad is not None:
            self._vad.reset()

        if self.on_final is not None:
            try:
                await self.on_final(request)
            except Exception as e:
                logger.exception(f"Error emitting final: {e}")

    async def flush(self) -> None:
        """Force finalization of current utterance."""
        if self._cancelled:
            return
        if not self._utterance_buffer.is_empty:
            await self._emit_final()

    async def cancel(self) -> None:
        """Cancel current processing and drop buffers."""
        self._cancelled = True
        self._utterance_buffer.clear()
        self._current_segment_id = None
        if self._partial_task is not None:
            self._partial_task.cancel()
            try:
                await self._partial_task
            except asyncio.CancelledError:
                pass
        logger.debug("Utterance aggregator cancelled")

    def reset(self) -> None:
        """Reset aggregator state for new session."""
        self._utterance_buffer.clear()
        self._current_segment_id = None
        self._in_speech = False
        self._cancelled = False
        self._last_partial_time = 0.0
        self._utterance_start_time = 0.0
        self._total_audio_ms = 0.0
        self._total_segments = 0
        self._session_start_time = time.monotonic()
        if self._vad is not None:
            self._vad.reset()

    def _get_session_time_ms(self) -> float:
        """Get elapsed time since session start in milliseconds."""
        return (time.monotonic() - self._session_start_time) * 1000

    @property
    def is_in_speech(self) -> bool:
        """Whether currently in speech state."""
        return self._in_speech

    @property
    def has_pending_utterance(self) -> bool:
        """Whether there is audio buffered."""
        return not self._utterance_buffer.is_empty

    @property
    def pending_duration_ms(self) -> float:
        """Duration of buffered audio in milliseconds."""
        return self._utterance_buffer.duration_ms

    @property
    def total_audio_ms(self) -> float:
        """Total audio processed in session."""
        return self._total_audio_ms

    @property
    def total_segments(self) -> int:
        """Total segments (finals) emitted."""
        return self._total_segments

    def get_stats(self) -> dict:
        """Get aggregator statistics."""
        return {
            "in_speech": self._in_speech,
            "has_pending_utterance": self.has_pending_utterance,
            "pending_duration_ms": round(self.pending_duration_ms, 2),
            "total_audio_ms": round(self._total_audio_ms, 2),
            "total_segments": self._total_segments,
            "vad_enabled": self.vad_enabled,
            "partial_enabled": self.partial_enabled,
        }
