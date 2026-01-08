"""
Audio streaming utilities.

Handles chunking audio into WebSocket frames and sending
segment_done messages with telemetry.
"""

import logging
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
    """Metrics for a completed segment."""

    segment_id: str
    request_id: str
    ttfa_ms: float
    rtf: float
    total_samples: int

    @property
    def audio_duration_ms(self) -> float:
        """Calculate audio duration in milliseconds."""
        return (self.total_samples / SAMPLE_RATE) * 1000


async def stream_audio_to_client(
    session: "Session",
    audio: np.ndarray,
    metrics: SegmentMetrics,
) -> None:
    """
    Stream audio in chunks to client.
    
    - Sends chunks with real size (no padding with zeros)
    - Sends segment_done with telemetry metrics at the end
    - Checks for cancellation at each chunk
    
    Args:
        session: The session to stream to
        audio: Audio data as int16 numpy array
        metrics: Segment metrics for telemetry
    """
    # Check if request is still valid (cancellation)
    if session.cancelled or metrics.request_id != session.current_request_id:
        logger.debug(
            f"Discarding audio for cancelled/outdated request {metrics.request_id}"
        )
        return

    chunk_size = settings.chunk_size_samples
    total_samples = len(audio)

    for i in range(0, total_samples, chunk_size):
        # Re-check cancellation at each chunk
        if session.cancelled or metrics.request_id != session.current_request_id:
            logger.debug(f"Stopping stream for cancelled request {metrics.request_id}")
            return

        chunk = audio[i : i + chunk_size]

        # Send binary chunk (real size, no padding)
        await session.websocket.send_bytes(chunk.tobytes())

    # Send segment_done with telemetry metrics
    await session.websocket.send_json(
        {
            "type": "segment_done",
            "segment_id": metrics.segment_id,
            "request_id": metrics.request_id,
            "ttfa_ms": round(metrics.ttfa_ms, 2),
            "rtf": round(metrics.rtf, 3),
            "audio_duration_ms": round(metrics.audio_duration_ms, 1),
            "total_samples": metrics.total_samples,
        }
    )
