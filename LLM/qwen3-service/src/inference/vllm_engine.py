"""
vLLM AsyncLLMEngine wrapper for LLM inference.

Provides:
- Async initialization with warmup
- Streaming token generation
- Request cancellation
- Metrics collection
- Automatic chat template handling via tokenizer
- LMCache integration for KV cache CPU offloading
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import AsyncGenerator

from transformers import AutoTokenizer
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.outputs import RequestOutput

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class GenerationOutput:
    """Output from token generation."""

    token: str
    token_id: int
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int


@dataclass
class GenerationResult:
    """Final result of generation with metrics."""

    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    total_time_ms: float
    tokens_per_second: float


class VLLMEngine:
    """
    Wrapper around vLLM AsyncLLMEngine for LLM inference.
    
    Features:
    - Single instance (initialized at startup)
    - Streaming token generation
    - Request cancellation via abort()
    - Automatic chat template formatting via tokenizer
    """

    def __init__(self):
        """Initialize engine configuration (engine created in start())."""
        self._engine: AsyncLLMEngine | None = None
        self._tokenizer: AutoTokenizer | None = None
        self._ready = False
        self._model_id = settings.model_id
        
        # Request tracking
        self._active_requests: set[str] = set()
        self._request_counter = 0

    @property
    def is_ready(self) -> bool:
        """Check if engine is initialized and ready."""
        return self._ready and self._engine is not None

    async def start(self) -> None:
        """
        Initialize the vLLM engine and warm up.
        
        This should be called once at startup in the lifespan handler.
        """
        if self._engine is not None:
            logger.warning("Engine already started")
            return

        logger.info(f"Initializing vLLM engine with model: {self._model_id}")
        start_time = time.monotonic()

        # Configure LMCache for KV cache CPU offloading if enabled
        if settings.lmcache_enabled:
            logger.info(
                f"Enabling LMCache: offloading KV cache to CPU RAM "
                f"({settings.lmcache_cpu_size_gb}GB)"
            )
            os.environ["LMCACHE_LOCAL_CPU"] = "True"
            os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(settings.lmcache_cpu_size_gb)
            # LMCache requires prefix caching to be disabled
            if settings.enable_prefix_caching:
                logger.warning(
                    "LMCache requires prefix caching to be disabled. "
                    "Set LLM_ENABLE_PREFIX_CACHING=false for optimal performance."
                )

        # Load tokenizer first (for chat template)
        logger.info("Loading tokenizer...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id,
            revision=settings.model_revision,
            trust_remote_code=True,
        )
        logger.info("Tokenizer loaded")

        # Configure engine arguments
        engine_args = AsyncEngineArgs(
            model=self._model_id,
            revision=settings.model_revision,
            max_model_len=settings.max_model_len,
            gpu_memory_utilization=settings.gpu_memory_utilization,
            tensor_parallel_size=settings.tensor_parallel_size,
            dtype=settings.dtype,
            enforce_eager=settings.enforce_eager,
            enable_prefix_caching=settings.enable_prefix_caching,
            trust_remote_code=True,
            disable_log_stats=False,
        )

        # Create engine
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)

        init_time = time.monotonic() - start_time
        logger.info(f"vLLM engine initialized in {init_time:.2f}s")

        # Warmup with a simple generation
        await self._warmup()
        
        self._ready = True
        logger.info("vLLM engine ready")

    async def _warmup(self) -> None:
        """Warm up the engine with a simple generation."""
        logger.info("Warming up vLLM engine...")
        
        warmup_messages = [
            {"role": "user", "content": "Hello"}
        ]
        
        sampling_params = SamplingParams(
            max_tokens=10,
            temperature=0.7,
        )
        
        prompt = self._format_chat_messages(warmup_messages)
        request_id = "warmup-0"
        
        try:
            async for _ in self._generate_stream(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                pass  # Just consume the warmup tokens
            logger.info("Warmup complete")
        except Exception as e:
            logger.warning(f"Warmup generation failed (non-fatal): {e}")

    async def stop(self) -> None:
        """Shutdown the engine gracefully."""
        if self._engine is None:
            return

        logger.info("Shutting down vLLM engine...")
        
        # Cancel all active requests
        for request_id in list(self._active_requests):
            await self.abort(request_id)
        
        # vLLM doesn't have explicit shutdown, but we mark as not ready
        self._ready = False
        self._engine = None
        logger.info("vLLM engine shutdown complete")

    def _format_chat_messages(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Format messages using the model's chat template via tokenizer.
        
        Uses the tokenizer's apply_chat_template method which automatically
        handles the correct format for any model (ChatML, Llama, etc.)
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        
        # Use tokenizer's chat template (works for any model)
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        
        return prompt

    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        self._request_counter += 1
        return f"req-{self._request_counter}-{int(time.time() * 1000)}"

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        stop: list[str] | None = None,
        request_id: str | None = None,
    ) -> AsyncGenerator[GenerationOutput, None]:
        """
        Generate tokens from chat messages with streaming.
        
        Args:
            messages: List of chat messages [{"role": "...", "content": "..."}]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling
            top_k: Top-k sampling
            repetition_penalty: Repetition penalty
            stop: Stop sequences
            request_id: Optional request ID (generated if not provided)
            
        Yields:
            GenerationOutput with each token and metadata
        """
        if not self.is_ready:
            raise RuntimeError("vLLM engine not ready")

        # Get stop tokens from tokenizer if not provided
        default_stop = []
        if self._tokenizer is not None:
            # Common stop tokens - get EOS token from tokenizer
            if self._tokenizer.eos_token:
                default_stop.append(self._tokenizer.eos_token)
            # Add common chat template end tokens
            for token in ["<|im_end|>", "<|eot_id|>", "</s>", "<|end|>"]:
                if token not in default_stop:
                    default_stop.append(token)

        # Use defaults from settings if not provided
        sampling_params = SamplingParams(
            max_tokens=max_tokens or settings.max_tokens,
            temperature=temperature if temperature is not None else settings.temperature,
            top_p=top_p if top_p is not None else settings.top_p,
            top_k=top_k if top_k is not None else settings.top_k,
            repetition_penalty=repetition_penalty or settings.repetition_penalty,
            stop=stop or default_stop or None,
        )

        # Format messages to prompt
        prompt = self._format_chat_messages(messages)
        
        # Generate request ID
        req_id = request_id or self._generate_request_id()

        async for output in self._generate_stream(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=req_id,
        ):
            yield output

    async def _generate_stream(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str,
    ) -> AsyncGenerator[GenerationOutput, None]:
        """
        Internal streaming generation.
        
        Args:
            prompt: Formatted prompt string
            sampling_params: vLLM sampling parameters
            request_id: Request ID for tracking
            
        Yields:
            GenerationOutput for each token
        """
        if self._engine is None:
            raise RuntimeError("Engine not initialized")

        self._active_requests.add(request_id)
        prev_text = ""
        prompt_tokens = 0

        try:
            results_generator = self._engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            )

            async for request_output in results_generator:
                output = request_output.outputs[0]
                
                # Get the new token(s)
                new_text = output.text[len(prev_text):]
                prev_text = output.text
                
                # Get token count from first iteration
                if prompt_tokens == 0 and hasattr(request_output, 'prompt_token_ids'):
                    prompt_tokens = len(request_output.prompt_token_ids)

                # Get the last token ID
                token_ids = output.token_ids
                last_token_id = token_ids[-1] if token_ids else 0

                yield GenerationOutput(
                    token=new_text,
                    token_id=last_token_id,
                    finish_reason=output.finish_reason,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=len(token_ids),
                )

        finally:
            self._active_requests.discard(request_id)

    async def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        stop: list[str] | None = None,
        request_id: str | None = None,
    ) -> GenerationResult:
        """
        Generate complete response (non-streaming).
        
        Returns:
            GenerationResult with full text and metrics
        """
        start_time = time.monotonic()
        ttft: float | None = None
        
        text_parts: list[str] = []
        finish_reason = "stop"
        prompt_tokens = 0
        completion_tokens = 0

        async for output in self.generate_stream(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop=stop,
            request_id=request_id,
        ):
            if ttft is None:
                ttft = (time.monotonic() - start_time) * 1000
            
            text_parts.append(output.token)
            prompt_tokens = output.prompt_tokens
            completion_tokens = output.completion_tokens
            
            if output.finish_reason:
                finish_reason = output.finish_reason

        total_time_ms = (time.monotonic() - start_time) * 1000
        full_text = "".join(text_parts)
        
        # Calculate tokens per second
        tokens_per_second = (
            completion_tokens / (total_time_ms / 1000)
            if total_time_ms > 0 else 0
        )

        return GenerationResult(
            text=full_text,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=ttft or 0,
            total_time_ms=total_time_ms,
            tokens_per_second=tokens_per_second,
        )

    async def abort(self, request_id: str) -> bool:
        """
        Abort an active request.
        
        Args:
            request_id: Request ID to abort
            
        Returns:
            True if request was found and aborted
        """
        if self._engine is None:
            return False

        if request_id not in self._active_requests:
            return False

        try:
            await self._engine.abort(request_id)
            self._active_requests.discard(request_id)
            logger.debug(f"Aborted request: {request_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to abort request {request_id}: {e}")
            return False

    @property
    def active_requests(self) -> int:
        """Number of currently active requests."""
        return len(self._active_requests)

    def get_stats(self) -> dict:
        """Get engine statistics."""
        return {
            "model_id": self._model_id,
            "ready": self._ready,
            "active_requests": len(self._active_requests),
            "total_requests": self._request_counter,
            "max_model_len": settings.max_model_len,
            "gpu_memory_utilization": settings.gpu_memory_utilization,
        }


# Global instance (initialized in lifespan)
vllm_engine: VLLMEngine | None = None


def get_vllm_engine() -> VLLMEngine:
    """Get the global vLLM engine instance."""
    if vllm_engine is None:
        raise RuntimeError("vLLM engine not initialized")
    return vllm_engine
