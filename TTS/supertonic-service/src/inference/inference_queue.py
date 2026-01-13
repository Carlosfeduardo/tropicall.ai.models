"""
Global inference queue with single consumer for serialized GPU access.

This module implements the InferenceQueue pattern to ensure that only one
thread accesses the GPU at a time, preventing contention and providing
predictable latency.

Features:
- Single-threaded GPU access (no contention)
- Queue wait time tracking for SLO monitoring
- Admission control based on queue depth and estimated SERVER_TOTAL time
- EMA-based server_total tracking (queue_wait + inference + postprocess)
- Server-only SLO (network latency is not our control)
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .supertonic_worker import SupertonicWorker

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Request for TTS inference with Future for async result."""

    text: str
    voice: str
    speed: float
    session_id: str  # For cancellation tracking
    request_id: str  # For request correlation
    created_at: float  # Timestamp for TTFA calculation
    future: asyncio.Future  # For returning result to caller


@dataclass
class InferenceResult:
    """Result of TTS inference including audio and detailed timing metrics."""

    audio: np.ndarray
    rtf: float  # Real-time factor
    audio_duration_ms: float  # Duration of generated audio
    
    # Detailed timing breakdown
    queue_wait_ms: float  # Time spent waiting in queue (created_at → inference_start)
    inference_ms: float  # Pure GPU inference time
    postprocess_ms: float  # Audio conversion to int16
    
    # Computed totals
    @property
    def server_total_ms(self) -> float:
        """Total server-side processing time."""
        return self.queue_wait_ms + self.inference_ms + self.postprocess_ms


class InferenceQueue:
    """
    Global queue with SINGLE CONSUMER for serialized GPU access.
    
    Benefits:
    - Determinism: requests processed in FIFO order
    - No contention: only 1 thread accesses GPU at a time
    - Predictability: consistent and measurable latency
    - Admission control: reject requests when queue is congested
    - EMA tracking: estimate wait time for new requests
    
    Usage:
        queue = InferenceQueue(worker)
        queue.start()
        
        # From any session:
        result = await queue.enqueue(text, voice, speed, session_id, request_id)
        
        # On shutdown:
        await queue.stop()
    """

    SAMPLE_RATE = 44100  # Supertonic uses 44.1kHz
    
    # Admission control defaults (can be overridden via config)
    DEFAULT_MAX_QUEUE_DEPTH = 30
    DEFAULT_MAX_ESTIMATED_SERVER_TOTAL_MS = 200.0
    
    # EMA smoothing factor for server_total tracking
    EMA_ALPHA = 0.1

    def __init__(
        self,
        worker: "SupertonicWorker",
        max_queue_depth: int | None = None,
        max_estimated_server_total_ms: float | None = None,
    ):
        """
        Initialize the inference queue.
        
        Args:
            worker: The SupertonicWorker instance for inference
            max_queue_depth: Maximum queue depth before rejection (None = use default)
            max_estimated_server_total_ms: Max estimated server_total time in ms (None = use default)
        """
        self.worker = worker
        self.queue: asyncio.Queue[InferenceRequest] = asyncio.Queue()
        self._consumer_task: asyncio.Task | None = None
        # CRITICAL: max_workers=1 ensures single thread for GPU access
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="gpu-inference",
        )
        self._running = False
        
        # Admission control settings (server-only SLO)
        self._max_queue_depth = max_queue_depth or self.DEFAULT_MAX_QUEUE_DEPTH
        self._max_estimated_server_total_ms = (
            max_estimated_server_total_ms or self.DEFAULT_MAX_ESTIMATED_SERVER_TOTAL_MS
        )
        
        # EMA-tracked average processing time (inference + postprocess)
        # This is used to estimate server_total for new requests
        self._avg_processing_ms: float = 40.0  # ~40ms typical for Supertonic (faster than Kokoro)
        
        # Stats tracking
        self._total_requests: int = 0
        self._rejected_requests: int = 0

    def start(self) -> None:
        """Start the consumer loop."""
        if self._running:
            return
        
        self._running = True
        self._consumer_task = asyncio.create_task(self._consumer_loop())
        logger.info("InferenceQueue consumer started")

    async def stop(self) -> None:
        """Stop the consumer gracefully."""
        if not self._running:
            return
        
        self._running = False
        
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
        
        self._executor.shutdown(wait=True)
        logger.info("InferenceQueue consumer stopped")

    async def _consumer_loop(self) -> None:
        """
        SINGLE point of inference execution.
        Runs continuously processing requests from the queue.
        """
        loop = asyncio.get_event_loop()

        while self._running:
            try:
                # Wait for a request
                req = await self.queue.get()
                
                try:
                    # Mark inference start time - queue wait ends here
                    inference_start = time.monotonic()
                    queue_wait_ms = (inference_start - req.created_at) * 1000
                    
                    # Execute in dedicated thread (single-threaded executor)
                    # Returns (audio, timings) with detailed breakdown
                    audio, timings = await loop.run_in_executor(
                        self._executor,
                        self.worker.generate_segment,
                        req.text,
                        req.voice,
                        req.speed,
                    )
                    
                    # Flatten audio in case it has extra dimensions
                    audio = audio.flatten()
                    
                    # Calculate derived metrics
                    audio_duration = audio.size / self.SAMPLE_RATE
                    rtf = (timings.inference_ms / 1000) / audio_duration if audio_duration > 0 else 0
                    audio_duration_ms = audio_duration * 1000
                    
                    # Update EMA for average processing time (inference + postprocess)
                    # This is used to estimate server_total for admission control
                    processing_ms = timings.inference_ms + timings.postprocess_ms
                    self._avg_processing_ms = (
                        (1 - self.EMA_ALPHA) * self._avg_processing_ms
                        + self.EMA_ALPHA * processing_ms
                    )
                    
                    # Return result with detailed timing breakdown
                    result = InferenceResult(
                        audio=audio,
                        rtf=rtf,
                        audio_duration_ms=audio_duration_ms,
                        queue_wait_ms=queue_wait_ms,
                        inference_ms=timings.inference_ms,
                        postprocess_ms=timings.postprocess_ms,
                    )
                    req.future.set_result(result)
                    
                    # Log timing breakdown if significant queue wait
                    if queue_wait_ms > 100:
                        logger.debug(
                            f"Request {req.request_id} queue={queue_wait_ms:.1f}ms "
                            f"inf={timings.inference_ms:.1f}ms post={timings.postprocess_ms:.1f}ms"
                        )
                    
                except Exception as e:
                    logger.exception(f"Inference error for request {req.request_id}")
                    req.future.set_exception(e)
                
                finally:
                    self.queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Unexpected error in consumer loop")

    async def enqueue(
        self,
        text: str,
        voice: str,
        speed: float,
        session_id: str,
        request_id: str,
    ) -> InferenceResult:
        """
        Enqueue request and wait for result.
        
        Can be called from multiple sessions concurrently,
        but actual execution is serialized by the consumer.
        
        Admission Control (Server-Only SLO):
        - Rejects if queue depth >= max_queue_depth
        - Rejects if estimated_server_total > max_estimated_server_total_ms
        
        Note: Admission is based on SERVER-ONLY metrics. Network latency
        is not controllable and should not affect admission decisions.
        
        Args:
            text: Text to synthesize
            voice: Voice to use
            speed: Speech speed multiplier
            session_id: Session identifier (for tracking)
            request_id: Request identifier (for correlation)
            
        Returns:
            InferenceResult with audio and metrics
            
        Raises:
            QueueFullError: If queue depth exceeds limit
            QueueCongestionError: If estimated server_total exceeds limit
        """
        from ..protocol.errors import QueueCongestionError, QueueFullError
        
        self._total_requests += 1
        
        # Admission control: check queue depth
        current_depth = self.depth
        if current_depth >= self._max_queue_depth:
            self._rejected_requests += 1
            logger.warning(
                f"Queue full: depth={current_depth} >= max={self._max_queue_depth}"
            )
            raise QueueFullError(current_depth, self._max_queue_depth)
        
        # Admission control: check estimated server_total
        # estimated_server_total = queue_wait (depth * avg_processing) + own processing
        estimated_server_total_ms = self.estimated_server_total_ms
        if estimated_server_total_ms > self._max_estimated_server_total_ms:
            self._rejected_requests += 1
            logger.warning(
                f"Queue congested: estimated_server_total={estimated_server_total_ms:.1f}ms "
                f"> max={self._max_estimated_server_total_ms}ms"
            )
            raise QueueCongestionError(estimated_server_total_ms, self._max_estimated_server_total_ms)
        
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        
        req = InferenceRequest(
            text=text,
            voice=voice,
            speed=speed,
            session_id=session_id,
            request_id=request_id,
            created_at=time.monotonic(),
            future=future,
        )
        
        await self.queue.put(req)
        
        # Wait for result (doesn't block event loop)
        return await future

    @property
    def depth(self) -> int:
        """Current queue depth (for metrics)."""
        return self.queue.qsize()

    @property
    def is_running(self) -> bool:
        """Check if the consumer is running."""
        return self._running

    @property
    def avg_processing_ms(self) -> float:
        """EMA-tracked average processing time (inference + postprocess) in ms."""
        return self._avg_processing_ms
    
    @property
    def estimated_queue_wait_ms(self) -> float:
        """Estimated queue wait time for a new request in milliseconds."""
        return self.depth * self._avg_processing_ms
    
    @property
    def estimated_server_total_ms(self) -> float:
        """
        Estimated server_total for a new request.
        
        server_total = queue_wait + processing (inference + postprocess)
        """
        return self.estimated_queue_wait_ms + self._avg_processing_ms
    
    @property
    def max_queue_depth(self) -> int:
        """Maximum queue depth before rejection."""
        return self._max_queue_depth
    
    @property
    def max_estimated_server_total_ms(self) -> float:
        """Maximum estimated server_total before rejection."""
        return self._max_estimated_server_total_ms
    
    @property
    def total_requests(self) -> int:
        """Total requests received."""
        return self._total_requests
    
    @property
    def rejected_requests(self) -> int:
        """Total requests rejected by admission control."""
        return self._rejected_requests
    
    @property
    def accepting_requests(self) -> bool:
        """Whether the queue is accepting new requests."""
        return (
            self.depth < self._max_queue_depth
            and self.estimated_server_total_ms <= self._max_estimated_server_total_ms
        )
    
    def get_stats(self) -> dict:
        """Get queue statistics for monitoring."""
        return {
            "depth": self.depth,
            "avg_processing_ms": round(self._avg_processing_ms, 2),
            "estimated_queue_wait_ms": round(self.estimated_queue_wait_ms, 2),
            "estimated_server_total_ms": round(self.estimated_server_total_ms, 2),
            "max_queue_depth": self._max_queue_depth,
            "max_estimated_server_total_ms": self._max_estimated_server_total_ms,
            "total_requests": self._total_requests,
            "rejected_requests": self._rejected_requests,
            "accepting_requests": self.accepting_requests,
        }


# Global instance (initialized in lifespan)
inference_queue: InferenceQueue | None = None


def get_inference_queue() -> InferenceQueue:
    """Get the global inference queue instance."""
    if inference_queue is None:
        raise RuntimeError("InferenceQueue not initialized")
    return inference_queue
