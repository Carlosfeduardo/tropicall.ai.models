"""
Token streaming to WebSocket clients.

Handles:
- Streaming tokens to client as they're generated
- Sending completion messages with metrics
- Error handling and cleanup
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from fastapi import WebSocket

from ..inference.request_handler import InferenceRequest, get_request_handler
from ..inference.vllm_engine import get_vllm_engine
from ..protocol.errors import GenerationCancelledError, LLMError
from ..protocol.messages import (
    ErrorCode,
    ErrorMessage,
    GenerationMetrics,
    MessageDoneMessage,
    TokenMessage,
    UsageInfo,
)

if TYPE_CHECKING:
    from .session import Session

logger = logging.getLogger(__name__)


async def stream_tokens_to_client(
    session: "Session",
    request_id: str,
) -> str | None:
    """
    Stream tokens from generation to WebSocket client.
    
    Args:
        session: Active chat session
        request_id: Request ID for this generation
        
    Returns:
        Full generated text, or None if cancelled/error
    """
    websocket = session.websocket
    start_time = time.monotonic()
    ttft: float | None = None
    
    text_parts: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    finish_reason = "stop"
    
    try:
        # Get generation parameters from session
        gen_config = session.get_generation_config()
        messages = session.get_messages_for_generation()
        
        # Create inference request
        request = InferenceRequest(
            request_id=request_id,
            messages=messages,
            **gen_config,
        )
        
        # Get request handler
        handler = get_request_handler()
        
        # Stream tokens
        async for output in handler.generate_stream(request):
            # Check if still valid (not cancelled)
            if not session.is_request_valid(request_id):
                # Abort the request
                engine = get_vllm_engine()
                await engine.abort(request_id)
                raise GenerationCancelledError(request_id)
            
            # Track TTFT
            if ttft is None:
                ttft = (time.monotonic() - start_time) * 1000
            
            # Send token to client
            token_msg = TokenMessage(
                token=output.token,
                request_id=request_id,
            )
            await websocket.send_json(token_msg.model_dump())
            
            # Accumulate text
            text_parts.append(output.token)
            prompt_tokens = output.prompt_tokens
            completion_tokens = output.completion_tokens
            
            if output.finish_reason:
                finish_reason = output.finish_reason
        
        # Calculate final metrics
        total_time_ms = (time.monotonic() - start_time) * 1000
        full_text = "".join(text_parts)
        
        tokens_per_second = (
            completion_tokens / (total_time_ms / 1000)
            if total_time_ms > 0 else 0
        )
        
        # Record metrics in session
        session.record_generation_metrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=ttft or 0,
            total_time_ms=total_time_ms,
        )
        
        # Send completion message
        done_msg = MessageDoneMessage(
            request_id=request_id,
            content=full_text,
            finish_reason=finish_reason,
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            metrics=GenerationMetrics(
                ttft_ms=round(ttft or 0, 2),
                total_time_ms=round(total_time_ms, 2),
                tokens_per_second=round(tokens_per_second, 2),
            ),
        )
        await websocket.send_json(done_msg.model_dump())
        
        # Add to conversation history
        session.add_assistant_message(full_text)
        
        logger.debug(
            f"Generation complete: {completion_tokens} tokens in {total_time_ms:.0f}ms "
            f"({tokens_per_second:.1f} tok/s)"
        )
        
        return full_text
        
    except GenerationCancelledError:
        logger.info(f"Generation cancelled: {request_id}")
        return None
        
    except LLMError as e:
        # Send error to client
        error_msg = ErrorMessage(
            code=e.code,
            message=e.message,
            request_id=request_id,
        )
        await websocket.send_json(error_msg.model_dump())
        session.generating = False
        raise
        
    except Exception as e:
        logger.exception(f"Error during generation: {request_id}")
        # Send error to client
        error_msg = ErrorMessage(
            code=ErrorCode.INTERNAL_ERROR,
            message=str(e),
            request_id=request_id,
        )
        try:
            await websocket.send_json(error_msg.model_dump())
        except Exception:
            pass
        session.generating = False
        raise


async def generate_response(
    session: "Session",
    request_id: str,
) -> str | None:
    """
    Generate complete response (non-streaming internally but streams to client).
    
    This is a convenience wrapper that handles the full generation flow.
    
    Args:
        session: Active chat session
        request_id: Request ID for this generation
        
    Returns:
        Full generated text, or None if cancelled/error
    """
    return await stream_tokens_to_client(session, request_id)
