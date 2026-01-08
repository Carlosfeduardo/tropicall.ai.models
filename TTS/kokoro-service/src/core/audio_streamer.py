"""
Audio streaming utilities.

Handles chunking audio into WebSocket frames and sending
segment_done messages with detailed timing telemetry.

Features:
- Optional pacing (send chunks at real-time rate to avoid bursts)
- Detailed timing metrics
- Cancellation support
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..config import settings

if TYPE_CHECKING:
    from .session import Session

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000


@dataclass
class SegmentMetrics:
    """Detailed timing metrics for a completed segment."""

    segment_id: str
    request_id: str
    rtf: float
    total_samples: int
    
    # Detailed timing breakdown (all in milliseconds)
    preprocess_ms: float = 0.0  # Text normalization time
    queue_wait_ms: float = 0.0  # Time waiting in inference queue
    inference_ms: float = 0.0  # Pure GPU inference time
    postprocess_ms: float = 0.0  # Audio conversion to int16
    send_first_chunk_ms: float = 0.0  # Time to send first chunk (set during streaming)

    @property
    def audio_duration_ms(self) -> float:
        """Calculate audio duration in milliseconds."""
        return (self.total_samples / SAMPLE_RATE) * 1000
    
    @property
    def server_total_ms(self) -> float:
        """Total server-side processing time (excluding network)."""
        return (
            self.preprocess_ms
            + self.queue_wait_ms
            + self.inference_ms
            + self.postprocess_ms
            + self.send_first_chunk_ms
        )


async def stream_audio_to_client(
    session: "Session",
    audio: np.ndarray,
    metrics: SegmentMetrics,
) -> None:
    """
    Stream audio in chunks to client with optional pacing.
    
    Features:
    - Sends chunks with real size (no padding with zeros)
    - Optional pacing: sends chunks at real-time rate to avoid bursts
    - Measures send_first_chunk_ms for latency breakdown
    - Sends segment_done with detailed timing telemetry at the end
    - Checks for cancellation at each chunk
    
    Pacing:
    - First chunk is sent immediately (for low TTFA)
    - Subsequent chunks are paced at ~real-time rate (configurable via pacing_factor)
    - pacing_factor=0.8 means 80% of real-time (sends slightly faster than playback)
    
    Args:
        session: The session to stream to
        audio: Audio data as int16 numpy array
        metrics: Segment metrics for telemetry (will be updated with send timing)
    """
    # Check if request is still valid (cancellation)
    if session.cancelled or metrics.request_id != session.current_request_id:
        logger.debug(
            f"Discarding audio for cancelled/outdated request {metrics.request_id}"
        )
        return

    chunk_size = settings.chunk_size_samples
    total_samples = len(audio)
    
    # Calculate chunk duration for pacing
    # chunk_size_ms is the duration of audio in each chunk (e.g., 20ms)
    chunk_duration_s = settings.chunk_size_ms / 1000.0
    pacing_delay_s = chunk_duration_s * settings.pacing_factor if settings.audio_pacing_enabled else 0
    
    # Track timing for first chunk send
    stream_start = time.monotonic()
    first_chunk_sent = False
    chunk_index = 0

    for i in range(0, total_samples, chunk_size):
        # Re-check cancellation at each chunk
        if session.cancelled or metrics.request_id != session.current_request_id:
            logger.debug(f"Stopping stream for cancelled request {metrics.request_id}")
            return

        chunk = audio[i : i + chunk_size]

        # Send binary chunk (real size, no padding)
        await session.websocket.send_bytes(chunk.tobytes())
        
        # Record time for first chunk
        if not first_chunk_sent:
            metrics.send_first_chunk_ms = (time.monotonic() - stream_start) * 1000
            first_chunk_sent = True
        
        # Pacing: wait before sending next chunk (except first for low TTFA)
        # This prevents bursting all audio at once and allows smoother playback
        if settings.audio_pacing_enabled and chunk_index > 0 and pacing_delay_s > 0:
            await asyncio.sleep(pacing_delay_s)
        
        chunk_index += 1

    # Send segment_done with detailed timing breakdown
    await session.websocket.send_json(
        {
            "type": "segment_done",
            "segment_id": metrics.segment_id,
            "request_id": metrics.request_id,
            # Detailed timing breakdown
            "timing": {
                "preprocess_ms": round(metrics.preprocess_ms, 2),
                "queue_wait_ms": round(metrics.queue_wait_ms, 2),
                "inference_ms": round(metrics.inference_ms, 2),
                "postprocess_ms": round(metrics.postprocess_ms, 2),
                "send_first_chunk_ms": round(metrics.send_first_chunk_ms, 2),
                "server_total_ms": round(metrics.server_total_ms, 2),
            },
            "rtf": round(metrics.rtf, 3),
            "audio_duration_ms": round(metrics.audio_duration_ms, 1),
            "total_samples": metrics.total_samples,
        }
    )
