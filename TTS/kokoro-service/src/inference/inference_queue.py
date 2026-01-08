"""
Global inference queue with single consumer for serialized GPU access.

This module implements the InferenceQueue pattern to ensure that only one
thread accesses the GPU at a time, preventing contention and providing
predictable latency.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .kokoro_worker import KokoroWorker

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
    """Result of TTS inference including audio and metrics."""

    audio: np.ndarray
    ttfa_ms: float  # Time to first audio
    rtf: float  # Real-time factor
    audio_duration_ms: float  # Duration of generated audio


class InferenceQueue:
    """
    Global queue with SINGLE CONSUMER for serialized GPU access.
    
    Benefits:
    - Determinism: requests processed in FIFO order
    - No contention: only 1 thread accesses GPU at a time
    - Predictability: consistent and measurable latency
    - Extensible: easy to add micro-batching later
    
    Usage:
        queue = InferenceQueue(worker)
        queue.start()
        
        # From any session:
        result = await queue.enqueue(text, voice, speed, session_id, request_id)
        
        # On shutdown:
        await queue.stop()
    """

    SAMPLE_RATE = 24000

    def __init__(self, worker: "KokoroWorker"):
        """
        Initialize the inference queue.
        
        Args:
            worker: The KokoroWorker instance for inference
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
                    # Mark inference start time
                    inference_start = time.monotonic()
                    
                    # Execute in dedicated thread (single-threaded executor)
                    audio = await loop.run_in_executor(
                        self._executor,
                        self.worker.generate_segment,
                        req.text,
                        req.voice,
                        req.speed,
                    )
                    
                    inference_end = time.monotonic()
                    
                    # Calculate metrics
                    ttfa_ms = (inference_end - req.created_at) * 1000
                    inference_time = inference_end - inference_start
                    audio_duration = len(audio) / self.SAMPLE_RATE
                    rtf = inference_time / audio_duration if audio_duration > 0 else 0
                    audio_duration_ms = audio_duration * 1000
                    
                    # Return result with metrics
                    result = InferenceResult(
                        audio=audio,
                        ttfa_ms=ttfa_ms,
                        rtf=rtf,
                        audio_duration_ms=audio_duration_ms,
                    )
                    req.future.set_result(result)
                    
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
        
        Args:
            text: Text to synthesize
            voice: Voice to use
            speed: Speech speed multiplier
            session_id: Session identifier (for tracking)
            request_id: Request identifier (for correlation)
            
        Returns:
            InferenceResult with audio and metrics
        """
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


# Global instance (initialized in lifespan)
inference_queue: InferenceQueue | None = None


def get_inference_queue() -> InferenceQueue:
    """Get the global inference queue instance."""
    if inference_queue is None:
        raise RuntimeError("InferenceQueue not initialized")
    return inference_queue
