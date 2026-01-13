"""
Request handler for managing inference requests.

Provides:
- Request queuing with admission control
- Timeout handling
- Metrics collection
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator

from ..config import settings
from ..protocol.errors import QueueCongestionError, QueueFullError
from .vllm_engine import GenerationOutput, GenerationResult, get_vllm_engine

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Request for LLM inference."""

    request_id: str
    messages: list[dict[str, str]]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    stop: list[str] | None = None
    created_at: float = field(default_factory=time.monotonic)


class RequestHandler:
    """
    Handles inference requests with admission control.
    
    Features:
    - Queue depth limiting
    - Estimated wait time tracking
    - Request timeout handling
    - Metrics collection
    """

    # EMA smoothing factor for latency tracking
    EMA_ALPHA = 0.1

    def __init__(
        self,
        max_queue_depth: int | None = None,
        max_concurrent: int | None = None,
    ):
        """
        Initialize request handler.
        
        Args:
            max_queue_depth: Maximum pending requests
            max_concurrent: Maximum concurrent requests
        """
        self._max_queue_depth = max_queue_depth or settings.max_queue_depth
        self._max_concurrent = max_concurrent or settings.max_concurrent_requests
        
        # Semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        
        # Request tracking
        self._pending_count = 0
        self._active_count = 0
        
        # EMA-tracked metrics
        self._avg_ttft_ms: float = 100.0  # Initial estimate
        self._avg_generation_ms: float = 1000.0  # Initial estimate
        
        # Stats
        self._total_requests = 0
        self._rejected_requests = 0
        self._completed_requests = 0
        self._failed_requests = 0

    def _check_admission(self) -> None:
        """
        Check if we can accept a new request.
        
        Raises:
            QueueFullError: If queue is at capacity
            QueueCongestionError: If estimated wait is too high
        """
        # Check queue depth
        if self._pending_count >= self._max_queue_depth:
            self._rejected_requests += 1
            raise QueueFullError(self._pending_count, self._max_queue_depth)

    async def generate_stream(
        self,
        request: InferenceRequest,
    ) -> AsyncGenerator[GenerationOutput, None]:
        """
        Process a streaming generation request with admission control.
        
        Args:
            request: InferenceRequest with generation parameters
            
        Yields:
            GenerationOutput for each token
        """
        self._total_requests += 1
        
        # Admission control
        self._check_admission()
        
        self._pending_count += 1
        
        try:
            # Wait for concurrency slot
            async with self._semaphore:
                self._pending_count -= 1
                self._active_count += 1
                
                start_time = time.monotonic()
                ttft: float | None = None
                token_count = 0
                
                engine = get_vllm_engine()
                
                try:
                    async for output in engine.generate_stream(
                        messages=request.messages,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        top_k=request.top_k,
                        repetition_penalty=request.repetition_penalty,
                        stop=request.stop,
                        request_id=request.request_id,
                    ):
                        # Track TTFT
                        if ttft is None:
                            ttft = (time.monotonic() - start_time) * 1000
                            self._update_avg_ttft(ttft)
                        
                        token_count += 1
                        yield output
                    
                    # Update metrics on completion
                    total_time = (time.monotonic() - start_time) * 1000
                    self._update_avg_generation(total_time)
                    self._completed_requests += 1
                    
                except Exception as e:
                    self._failed_requests += 1
                    raise
                    
                finally:
                    self._active_count -= 1
                    
        except (QueueFullError, QueueCongestionError):
            self._pending_count -= 1
            raise

    async def generate(
        self,
        request: InferenceRequest,
    ) -> GenerationResult:
        """
        Process a non-streaming generation request.
        
        Args:
            request: InferenceRequest with generation parameters
            
        Returns:
            GenerationResult with full response and metrics
        """
        self._total_requests += 1
        
        # Admission control
        self._check_admission()
        
        self._pending_count += 1
        
        try:
            # Wait for concurrency slot
            async with self._semaphore:
                self._pending_count -= 1
                self._active_count += 1
                
                engine = get_vllm_engine()
                
                try:
                    result = await engine.generate(
                        messages=request.messages,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        top_k=request.top_k,
                        repetition_penalty=request.repetition_penalty,
                        stop=request.stop,
                        request_id=request.request_id,
                    )
                    
                    # Update metrics
                    self._update_avg_ttft(result.ttft_ms)
                    self._update_avg_generation(result.total_time_ms)
                    self._completed_requests += 1
                    
                    return result
                    
                except Exception as e:
                    self._failed_requests += 1
                    raise
                    
                finally:
                    self._active_count -= 1
                    
        except (QueueFullError, QueueCongestionError):
            self._pending_count -= 1
            raise

    def _update_avg_ttft(self, ttft_ms: float) -> None:
        """Update EMA-tracked average TTFT."""
        self._avg_ttft_ms = (
            (1 - self.EMA_ALPHA) * self._avg_ttft_ms
            + self.EMA_ALPHA * ttft_ms
        )

    def _update_avg_generation(self, total_ms: float) -> None:
        """Update EMA-tracked average generation time."""
        self._avg_generation_ms = (
            (1 - self.EMA_ALPHA) * self._avg_generation_ms
            + self.EMA_ALPHA * total_ms
        )

    @property
    def pending_count(self) -> int:
        """Number of pending requests."""
        return self._pending_count

    @property
    def active_count(self) -> int:
        """Number of active requests."""
        return self._active_count

    @property
    def estimated_wait_ms(self) -> float:
        """Estimated wait time for a new request."""
        # Simple estimate: pending * avg_generation / concurrency
        if self._pending_count == 0:
            return 0
        return (self._pending_count * self._avg_generation_ms) / self._max_concurrent

    @property
    def accepting_requests(self) -> bool:
        """Whether the handler is accepting new requests."""
        return self._pending_count < self._max_queue_depth

    def get_stats(self) -> dict:
        """Get handler statistics."""
        return {
            "pending_count": self._pending_count,
            "active_count": self._active_count,
            "max_queue_depth": self._max_queue_depth,
            "max_concurrent": self._max_concurrent,
            "avg_ttft_ms": round(self._avg_ttft_ms, 2),
            "avg_generation_ms": round(self._avg_generation_ms, 2),
            "estimated_wait_ms": round(self.estimated_wait_ms, 2),
            "total_requests": self._total_requests,
            "completed_requests": self._completed_requests,
            "rejected_requests": self._rejected_requests,
            "failed_requests": self._failed_requests,
            "accepting_requests": self.accepting_requests,
        }


# Global instance (initialized in lifespan)
request_handler: RequestHandler | None = None


def get_request_handler() -> RequestHandler:
    """Get the global request handler instance."""
    if request_handler is None:
        raise RuntimeError("Request handler not initialized")
    return request_handler
