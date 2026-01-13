"""
OpenAI-compatible chat completions API.

Provides /v1/chat/completions endpoint compatible with OpenAI SDK.
Supports both streaming (SSE) and non-streaming responses.
"""

import json
import logging
import time
from typing import AsyncGenerator
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..config import settings
from ..inference.request_handler import InferenceRequest, get_request_handler
from ..protocol.errors import AuthenticationError, LLMError
from ..protocol.messages import (
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionChunkChoice,
    OpenAIChatCompletionChunkDelta,
    OpenAIChatCompletionChoice,
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIChatCompletionUsage,
    OpenAIChatMessage,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai"])


async def verify_api_key(request: Request) -> None:
    """
    Verify API key from Authorization header.
    
    Expected format: Authorization: Bearer <api_key>
    """
    if not settings.require_auth:
        return
    
    if not settings.api_key:
        # No API key configured, skip auth
        return
    
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header format. Expected: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = auth_header[7:]  # Remove "Bearer " prefix
    if token != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


def generate_completion_id() -> str:
    """Generate a unique completion ID."""
    return f"chatcmpl-{uuid4().hex[:24]}"


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    request: OpenAIChatCompletionRequest,
    _: None = Depends(verify_api_key),
) -> OpenAIChatCompletionResponse | StreamingResponse:
    """
    OpenAI-compatible chat completions endpoint.
    
    Supports:
    - Streaming responses (stream=true) via SSE
    - Non-streaming responses (stream=false)
    - Temperature, top_p, max_tokens parameters
    - Stop sequences
    """
    completion_id = generate_completion_id()
    created = int(time.time())
    model_name = "qwen3-32b"
    
    # Convert messages to internal format
    messages = [
        {"role": msg.role, "content": msg.content}
        for msg in request.messages
    ]
    
    # Handle stop sequences
    stop = None
    if request.stop:
        if isinstance(request.stop, str):
            stop = [request.stop]
        else:
            stop = request.stop
    
    # Create inference request
    request_id = f"openai-{uuid4().hex[:8]}"
    inference_request = InferenceRequest(
        request_id=request_id,
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=stop,
    )
    
    if request.stream:
        # Streaming response via SSE
        return StreamingResponse(
            stream_chat_completions(
                inference_request=inference_request,
                completion_id=completion_id,
                created=created,
                model_name=model_name,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        # Non-streaming response
        return await generate_chat_completion(
            inference_request=inference_request,
            completion_id=completion_id,
            created=created,
            model_name=model_name,
        )


async def generate_chat_completion(
    inference_request: InferenceRequest,
    completion_id: str,
    created: int,
    model_name: str,
) -> OpenAIChatCompletionResponse:
    """Generate non-streaming chat completion."""
    try:
        handler = get_request_handler()
        result = await handler.generate(inference_request)
        
        return OpenAIChatCompletionResponse(
            id=completion_id,
            created=created,
            model=model_name,
            choices=[
                OpenAIChatCompletionChoice(
                    index=0,
                    message=OpenAIChatMessage(
                        role="assistant",
                        content=result.text,
                    ),
                    finish_reason=result.finish_reason,
                )
            ],
            usage=OpenAIChatCompletionUsage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.prompt_tokens + result.completion_tokens,
            ),
        )
        
    except LLMError as e:
        logger.warning(f"LLM error: {e.message}")
        raise HTTPException(
            status_code=503 if e.code.value.startswith("QUEUE") else 500,
            detail=e.message,
        )
    except Exception as e:
        logger.exception("Error generating completion")
        raise HTTPException(status_code=500, detail=str(e))


async def stream_chat_completions(
    inference_request: InferenceRequest,
    completion_id: str,
    created: int,
    model_name: str,
) -> AsyncGenerator[str, None]:
    """
    Generate streaming chat completion via SSE.
    
    Format: data: {json}\n\n
    Final message: data: [DONE]\n\n
    """
    try:
        handler = get_request_handler()
        
        # Send initial chunk with role
        initial_chunk = OpenAIChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model_name,
            choices=[
                OpenAIChatCompletionChunkChoice(
                    index=0,
                    delta=OpenAIChatCompletionChunkDelta(role="assistant"),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {initial_chunk.model_dump_json()}\n\n"
        
        # Stream tokens
        finish_reason = None
        async for output in handler.generate_stream(inference_request):
            if output.token:
                chunk = OpenAIChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=model_name,
                    choices=[
                        OpenAIChatCompletionChunkChoice(
                            index=0,
                            delta=OpenAIChatCompletionChunkDelta(
                                content=output.token
                            ),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
            
            if output.finish_reason:
                finish_reason = output.finish_reason
        
        # Send final chunk with finish_reason
        final_chunk = OpenAIChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model_name,
            choices=[
                OpenAIChatCompletionChunkChoice(
                    index=0,
                    delta=OpenAIChatCompletionChunkDelta(),
                    finish_reason=finish_reason or "stop",
                )
            ],
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        
        # Send [DONE] marker
        yield "data: [DONE]\n\n"
        
    except LLMError as e:
        logger.warning(f"LLM error during streaming: {e.message}")
        error_data = {"error": {"message": e.message, "code": e.code.value}}
        yield f"data: {json.dumps(error_data)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.exception("Error during streaming")
        error_data = {"error": {"message": str(e), "code": "internal_error"}}
        yield f"data: {json.dumps(error_data)}\n\n"
        yield "data: [DONE]\n\n"


@router.get("/models")
async def list_models(_: None = Depends(verify_api_key)) -> dict:
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen3-32b",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "tropicall",
                "permission": [],
                "root": "qwen3-32b",
                "parent": None,
            }
        ],
    }


@router.get("/models/{model_id}")
async def get_model(
    model_id: str,
    _: None = Depends(verify_api_key),
) -> dict:
    """Get model details (OpenAI-compatible)."""
    if model_id != "qwen3-32b":
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    
    return {
        "id": "qwen3-32b",
        "object": "model",
        "created": int(time.time()),
        "owned_by": "tropicall",
        "permission": [],
        "root": "qwen3-32b",
        "parent": None,
    }
