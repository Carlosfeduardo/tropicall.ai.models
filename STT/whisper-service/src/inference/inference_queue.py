"""
Global inference queue with priority-based processing and partial shedding.

This module implements the InferenceQueue pattern with:
- Priority queue: Finals processed before Partials
- Per-session coalescing: Only latest partial per session kept
- Partial shedding: Drop partials when congested, never drop finals
- Single-threaded GPU access (no contention)

Features:
- Determinism: requests processed by priority, then FIFO
- No contention: only 1 thread accesses GPU at a time
- Predictability: consistent and measurable latency
- Admission control: shed partials when congested, reject all when overloaded
- EMA tracking: estimate wait time for new requests
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .whisper_worker import WhisperWorker

logger = logging.getLogger(__name__)


# Priority values (lower = higher priority)
PRIORITY_FINAL = 0
PRIORITY_PARTIAL = 1


class RequestType(str, Enum):
    """Type of transcription request."""

    PARTIAL = "partial"  # Fast, streaming partial results
    FINAL = "final"  # Full transcription with word timestamps


@dataclass
class InferenceRequest:
    """Request for STT inference with Future for async result."""

    audio: np.ndarray
    language: str
    request_type: RequestType
    session_id: str
    segment_id: str
    created_at: float
    future: asyncio.Future
    word_timestamps: bool = True
    priority: int = PRIORITY_PARTIAL

    def __lt__(self, other: "InferenceRequest") -> bool:
        """Compare by priority, then by creation time (for PriorityQueue)."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created_at < other.created_at


@dataclass
class InferenceResult:
    """Result of STT inference including text and timing metrics."""

    text: str
    words: list  # List of WordTimestamp from worker
    language: str
    language_probability: float
    audio_duration_ms: float
    no_speech_probability: float
    confidence: float  # Derived from language_probability and no_speech_prob
    
    # Timing breakdown
    queue_wait_ms: float
    inference_ms: float
    postprocess_ms: float
    
    @property
    def server_total_ms(self) -> float:
        """Total server-side processing time."""
        return self.queue_wait_ms + self.inference_ms + self.postprocess_ms


class InferenceQueue:
    """
    Priority queue with SINGLE CONSUMER for serialized GPU access.
    
    Key Features:
    - Finals have higher priority than Partials
    - Per-session partial coalescing (only latest kept)
    - Partial shedding when congested (finals always pass)
    - Admission control for overload protection
    
    Benefits:
    - Determinism: requests processed by priority, then FIFO
    - No contention: only 1 thread accesses GPU at a time
    - Predictability: consistent and measurable latency
    - Low latency finals: never blocked by partial backlog
    
    Usage:
        queue = InferenceQueue(worker)
        queue.start()
        
        # From sessions:
        result = await queue.enqueue_final(audio, "pt", session_id, segment_id)
        result = await queue.enqueue_partial(audio, "pt", session_id, segment_id)
        
        # On shutdown:
        await queue.stop()
    """

    SAMPLE_RATE = 16000
    
    # EMA smoothing factor for tracking
    EMA_ALPHA = 0.1

    def __init__(
        self,
        worker: "WhisperWorker",
        max_queue_depth: int | None = None,
        max_estimated_server_total_ms: float | None = None,
    ):
        """
        Initialize the inference queue.
        
        Args:
            worker: The WhisperWorker instance for inference
            max_queue_depth: Maximum queue depth before rejection
            max_estimated_server_total_ms: Max estimated server_total in ms
        """
        from ..config import settings
        
        self.worker = worker
        
        # Priority queue for request ordering
        self.queue: asyncio.PriorityQueue[InferenceRequest] = asyncio.PriorityQueue()
        
        # Per-session coalescing: track pending partials
        self._pending_partials: dict[str, InferenceRequest] = {}
        self._coalescing_enabled = settings.partial_coalescing_enabled
        
        # Partial shedding settings
        self._shedding_enabled = settings.partial_shedding_enabled
        self._partial_shedding_depth = settings.partial_shedding_queue_depth
        self._partial_shedding_wait_ms = settings.partial_shedding_wait_ms
        
        self._consumer_task: asyncio.Task | None = None
        # CRITICAL: max_workers=1 ensures single thread for GPU access
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="gpu-inference",
        )
        self._running = False
        
        # Admission control settings (for finals/hard limit)
        self._max_queue_depth = max_queue_depth or settings.max_queue_depth
        self._max_estimated_server_total_ms = (
            max_estimated_server_total_ms or settings.max_estimated_server_total_ms
        )
        
        # EMA-tracked average processing times
        self._avg_partial_ms: float = 60.0  # Typical for turbo partial
        self._avg_final_ms: float = 100.0  # Typical for turbo final
        
        # Stats tracking
        self._total_requests: int = 0
        self._rejected_requests: int = 0
        self._partial_requests: int = 0
        self._final_requests: int = 0
        self._shed_partial_count: int = 0
        self._coalesced_count: int = 0

    def start(self) -> None:
        """Start the consumer loop."""
        if self._running:
            return
        
        self._running = True
        self._consumer_task = asyncio.create_task(self._consumer_loop())
        logger.info(
            f"InferenceQueue started (shedding={self._shedding_enabled}, "
            f"coalescing={self._coalescing_enabled})"
        )

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
        Runs continuously processing requests from the priority queue.
        """
        loop = asyncio.get_event_loop()

        while self._running:
            try:
                # Wait for a request (highest priority first)
                req = await self.queue.get()
                
                # If this was a coalesced partial, remove from tracking
                if req.request_type == RequestType.PARTIAL and self._coalescing_enabled:
                    if self._pending_partials.get(req.session_id) is req:
                        del self._pending_partials[req.session_id]
                
                try:
                    # Mark inference start time
                    inference_start = time.monotonic()
                    queue_wait_ms = (inference_start - req.created_at) * 1000
                    
                    # Execute in dedicated thread
                    if req.request_type == RequestType.PARTIAL:
                        result = await loop.run_in_executor(
                            self._executor,
                            self.worker.transcribe_partial,
                            req.audio,
                            req.language,
                            None,  # max_window_samples (use default)
                            req.word_timestamps,
                        )
                        self._partial_requests += 1
                        
                        # Update EMA for partial
                        processing_ms = result.timings.inference_ms + result.timings.postprocess_ms
                        self._avg_partial_ms = (
                            (1 - self.EMA_ALPHA) * self._avg_partial_ms
                            + self.EMA_ALPHA * processing_ms
                        )
                    else:
                        result = await loop.run_in_executor(
                            self._executor,
                            self.worker.transcribe_final,
                            req.audio,
                            req.language,
                            req.word_timestamps,
                        )
                        self._final_requests += 1
                        
                        # Update EMA for final
                        processing_ms = result.timings.inference_ms + result.timings.postprocess_ms
                        self._avg_final_ms = (
                            (1 - self.EMA_ALPHA) * self._avg_final_ms
                            + self.EMA_ALPHA * processing_ms
                        )
                    
                    # Calculate confidence from language probability and no_speech
                    confidence = result.language_probability * (1 - result.no_speech_probability)
                    
                    # Return result
                    inference_result = InferenceResult(
                        text=result.text,
                        words=result.words,
                        language=result.language,
                        language_probability=result.language_probability,
                        audio_duration_ms=result.duration * 1000,
                        no_speech_probability=result.no_speech_probability,
                        confidence=confidence,
                        queue_wait_ms=queue_wait_ms,
                        inference_ms=result.timings.inference_ms,
                        postprocess_ms=result.timings.postprocess_ms,
                    )
                    
                    if not req.future.done():
                        req.future.set_result(inference_result)
                    
                    # Log if significant queue wait
                    if queue_wait_ms > 100:
                        logger.debug(
                            f"Request {req.segment_id} [{req.request_type.value}] "
                            f"queue={queue_wait_ms:.1f}ms inf={result.timings.inference_ms:.1f}ms"
                        )
                    
                except Exception as e:
                    logger.exception(f"Inference error for request {req.segment_id}")
                    if not req.future.done():
                        req.future.set_exception(e)
                
                finally:
                    self.queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Unexpected error in consumer loop")

    async def enqueue_final(
        self,
        audio: np.ndarray,
        language: str,
        session_id: str,
        segment_id: str,
        word_timestamps: bool = True,
    ) -> InferenceResult:
        """
        Enqueue a FINAL transcription request (high priority, never shed).
        
        Finals are always enqueued unless the queue is at absolute max capacity.
        Each utterance is transcribed independently (no context from previous).
        
        Args:
            audio: Audio samples as numpy array
            language: Language code
            session_id: Session identifier
            segment_id: Segment identifier
            word_timestamps: Include word timestamps
            
        Returns:
            InferenceResult with text and metrics
            
        Raises:
            QueueFullError: Only if queue at absolute max capacity
        """
        from ..protocol.errors import QueueFullError
        
        self._total_requests += 1
        
        # Only reject finals if at absolute max (hard limit)
        current_depth = self.depth
        if current_depth >= self._max_queue_depth:
            self._rejected_requests += 1
            logger.warning(
                f"Queue full for FINAL: depth={current_depth} >= max={self._max_queue_depth}"
            )
            raise QueueFullError(current_depth, self._max_queue_depth)
        
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        
        req = InferenceRequest(
            audio=audio,
            language=language,
            request_type=RequestType.FINAL,
            session_id=session_id,
            segment_id=segment_id,
            created_at=time.monotonic(),
            future=future,
            word_timestamps=word_timestamps,
            priority=PRIORITY_FINAL,
        )
        
        await self.queue.put(req)
        
        # Wait for result
        return await future

    async def enqueue_partial(
        self,
        audio: np.ndarray,
        language: str,
        session_id: str,
        segment_id: str,
        word_timestamps: bool = False,
    ) -> InferenceResult | None:
        """
        Enqueue a PARTIAL transcription request (low priority, may be shed).
        
        Partials are:
        - Shed (dropped) when queue is congested
        - Coalesced (replaced by newer) per session
        - Transcribed without context (no initial_prompt) to avoid loops
        
        Args:
            audio: Audio samples as numpy array
            language: Language code
            session_id: Session identifier
            segment_id: Segment identifier
            word_timestamps: Include word timestamps (default False for speed)
            
        Returns:
            InferenceResult if processed, None if shed
        """
        self._total_requests += 1
        
        # Shedding check: drop partials when congested
        if self._shedding_enabled:
            should_shed = (
                self.depth > self._partial_shedding_depth
                or self.estimated_queue_wait_ms > self._partial_shedding_wait_ms
            )
            if should_shed:
                self._shed_partial_count += 1
                logger.debug(
                    f"Partial shed: depth={self.depth}, wait={self.estimated_queue_wait_ms:.1f}ms"
                )
                return None
        
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        
        req = InferenceRequest(
            audio=audio,
            language=language,
            request_type=RequestType.PARTIAL,
            session_id=session_id,
            segment_id=segment_id,
            created_at=time.monotonic(),
            future=future,
            word_timestamps=word_timestamps,
            priority=PRIORITY_PARTIAL,
        )
        
        # Coalescing: replace existing partial for this session
        if self._coalescing_enabled:
            old_req = self._pending_partials.get(session_id)
            if old_req is not None:
                # Cancel the old future (it will be ignored)
                if not old_req.future.done():
                    old_req.future.cancel()
                self._coalesced_count += 1
                logger.debug(f"Partial coalesced for session {session_id}")
            
            # Track new pending partial
            self._pending_partials[session_id] = req
        
        await self.queue.put(req)
        
        # Wait for result
        try:
            return await future
        except asyncio.CancelledError:
            # This partial was coalesced (replaced by newer one)
            return None

    # Legacy method for backward compatibility
    async def enqueue(
        self,
        audio: np.ndarray,
        language: str,
        request_type: RequestType,
        session_id: str,
        segment_id: str,
        word_timestamps: bool = True,
    ) -> InferenceResult:
        """
        Legacy enqueue method - routes to appropriate method.
        
        Prefer using enqueue_final() or enqueue_partial() directly.
        """
        if request_type == RequestType.FINAL:
            return await self.enqueue_final(
                audio, language, session_id, segment_id, word_timestamps
            )
        else:
            result = await self.enqueue_partial(audio, language, session_id, segment_id)
            if result is None:
                # Shed - raise error for legacy callers
                from ..protocol.errors import QueueCongestionError
                raise QueueCongestionError(
                    self.estimated_server_total_ms,
                    self._partial_shedding_wait_ms,
                )
            return result

    @property
    def depth(self) -> int:
        """Current queue depth."""
        return self.queue.qsize()

    @property
    def is_running(self) -> bool:
        """Check if consumer is running."""
        return self._running

    @property
    def avg_processing_ms(self) -> float:
        """Average processing time (weighted by request types)."""
        # Simple average weighted toward final (more common for actual output)
        return (self._avg_partial_ms + self._avg_final_ms * 2) / 3

    @property
    def estimated_queue_wait_ms(self) -> float:
        """Estimated queue wait time for a new request."""
        return self.depth * self.avg_processing_ms

    @property
    def estimated_server_total_ms(self) -> float:
        """Estimated server_total for a new request."""
        return self.estimated_queue_wait_ms + self.avg_processing_ms

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
    def shed_partial_count(self) -> int:
        """Total partials shed due to congestion."""
        return self._shed_partial_count

    @property
    def coalesced_count(self) -> int:
        """Total partials coalesced (replaced by newer)."""
        return self._coalesced_count

    @property
    def accepting_requests(self) -> bool:
        """Whether the queue is accepting new requests."""
        return self.depth < self._max_queue_depth

    @property
    def accepting_partials(self) -> bool:
        """Whether the queue is accepting partials (not congested)."""
        if not self._shedding_enabled:
            return self.accepting_requests
        return (
            self.depth <= self._partial_shedding_depth
            and self.estimated_queue_wait_ms <= self._partial_shedding_wait_ms
        )

    def get_stats(self) -> dict:
        """Get queue statistics for monitoring."""
        return {
            "depth": self.depth,
            "avg_partial_ms": round(self._avg_partial_ms, 2),
            "avg_final_ms": round(self._avg_final_ms, 2),
            "avg_processing_ms": round(self.avg_processing_ms, 2),
            "estimated_queue_wait_ms": round(self.estimated_queue_wait_ms, 2),
            "estimated_server_total_ms": round(self.estimated_server_total_ms, 2),
            "max_queue_depth": self._max_queue_depth,
            "max_estimated_server_total_ms": self._max_estimated_server_total_ms,
            "total_requests": self._total_requests,
            "rejected_requests": self._rejected_requests,
            "partial_requests": self._partial_requests,
            "final_requests": self._final_requests,
            "shed_partials": self._shed_partial_count,
            "coalesced_partials": self._coalesced_count,
            "accepting_requests": self.accepting_requests,
            "accepting_partials": self.accepting_partials,
            "shedding_enabled": self._shedding_enabled,
            "coalescing_enabled": self._coalescing_enabled,
        }


# Global instance (initialized in lifespan)
inference_queue: InferenceQueue | None = None


def get_inference_queue() -> InferenceQueue:
    """Get the global inference queue instance."""
    if inference_queue is None:
        raise RuntimeError("InferenceQueue not initialized")
    return inference_queue
