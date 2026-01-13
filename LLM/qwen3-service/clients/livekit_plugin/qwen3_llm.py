"""
LiveKit Agents LLM plugin for Qwen3 service.

Implements the LLM interface for use with LiveKit Agents pipelines.
Supports both WebSocket and OpenAI-compatible HTTP connections.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import AsyncIterable, Optional
from uuid import uuid4

import httpx
from livekit.agents import llm
from livekit.agents.llm import (
    ChatChunk,
    ChatContext,
    ChatRole,
    ChoiceDelta,
    LLMStream,
)

logger = logging.getLogger(__name__)


@dataclass
class Qwen3LLMOptions:
    """Options for Qwen3 LLM client."""

    # Connection
    base_url: str = "http://localhost:8001"
    api_key: Optional[str] = None
    timeout: float = 120.0

    # Generation
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9

    # Mode
    use_websocket: bool = False  # Use HTTP by default for simplicity


class Qwen3LLM(llm.LLM):
    """
    LiveKit Agents LLM implementation for Qwen3 service.
    
    Connects to the Qwen3 LLM microservice via OpenAI-compatible HTTP API.
    
    Usage:
        from clients.livekit_plugin import Qwen3LLM
        
        llm = Qwen3LLM(
            base_url="http://qwen3-service:8001",
            temperature=0.7,
        )
        
        # Use with VoicePipelineAgent
        agent = VoicePipelineAgent(
            llm=llm,
            ...
        )
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8001",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
    ):
        """
        Initialize Qwen3 LLM client.
        
        Args:
            base_url: Base URL of the Qwen3 service
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p (nucleus) sampling
        """
        super().__init__()
        
        self._opts = Qwen3LLMOptions(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        
        # HTTP client for OpenAI-compatible API
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        self._client = httpx.AsyncClient(
            base_url=f"{self._opts.base_url}/v1",
            headers=headers,
            timeout=timeout,
        )

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        fnc_ctx: Optional[llm.FunctionContext] = None,
        temperature: Optional[float] = None,
        n: int = 1,
        parallel_tool_calls: bool = True,
    ) -> "Qwen3LLMStream":
        """
        Start a chat completion stream.
        
        Args:
            chat_ctx: Chat context with message history
            fnc_ctx: Function context (not supported yet)
            temperature: Override temperature
            n: Number of completions (only 1 supported)
            parallel_tool_calls: Not applicable
            
        Returns:
            Qwen3LLMStream for streaming tokens
        """
        if fnc_ctx is not None:
            logger.warning("Function calling not yet supported by Qwen3 plugin")
        
        return Qwen3LLMStream(
            client=self._client,
            chat_ctx=chat_ctx,
            opts=self._opts,
            temperature=temperature,
        )


class Qwen3LLMStream(LLMStream):
    """
    Streaming chat completion from Qwen3 service.
    
    Implements the LiveKit Agents LLMStream interface.
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        chat_ctx: ChatContext,
        opts: Qwen3LLMOptions,
        temperature: Optional[float] = None,
    ):
        super().__init__(chat_ctx=chat_ctx, fnc_ctx=None)
        
        self._client = client
        self._opts = opts
        self._temperature = temperature or opts.temperature
        self._request_id = f"lk-{uuid4().hex[:8]}"
        
        # Response tracking
        self._response_text = ""
        self._input_tokens = 0
        self._output_tokens = 0

    def _convert_messages(self) -> list[dict]:
        """Convert LiveKit ChatContext to OpenAI message format."""
        messages = []
        
        for msg in self._chat_ctx.messages:
            role = "user"
            if msg.role == ChatRole.ASSISTANT:
                role = "assistant"
            elif msg.role == ChatRole.SYSTEM:
                role = "system"
            
            # Handle text content
            content = ""
            if isinstance(msg.content, str):
                content = msg.content
            elif isinstance(msg.content, list):
                # Extract text from content parts
                for part in msg.content:
                    if hasattr(part, "text"):
                        content += part.text
            
            if content:
                messages.append({"role": role, "content": content})
        
        return messages

    async def _run(self) -> None:
        """Execute the streaming request."""
        messages = self._convert_messages()
        
        request_body = {
            "model": "qwen3-32b",
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._opts.max_tokens,
            "top_p": self._opts.top_p,
            "stream": True,
        }
        
        try:
            async with self._client.stream(
                "POST",
                "/chat/completions",
                json=request_body,
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix
                        
                        if data == "[DONE]":
                            break
                        
                        try:
                            chunk_data = json.loads(data)
                            
                            # Check for errors
                            if "error" in chunk_data:
                                logger.error(f"LLM error: {chunk_data['error']}")
                                break
                            
                            # Extract token from chunk
                            if "choices" in chunk_data and chunk_data["choices"]:
                                choice = chunk_data["choices"][0]
                                delta = choice.get("delta", {})
                                content = delta.get("content", "")
                                
                                if content:
                                    self._response_text += content
                                    
                                    # Create ChatChunk with delta
                                    chunk = ChatChunk(
                                        request_id=self._request_id,
                                        choices=[
                                            ChoiceDelta(
                                                delta=llm.ChoiceDelta(
                                                    role="assistant",
                                                    content=content,
                                                ),
                                                index=0,
                                            )
                                        ],
                                    )
                                    self._event_ch.send_nowait(chunk)
                                
                                # Check for finish
                                if choice.get("finish_reason"):
                                    break
                                    
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse SSE chunk: {data}")
                            continue
                            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from Qwen3 service: {e.response.status_code}")
            raise
        except Exception as e:
            logger.exception(f"Error during Qwen3 streaming: {e}")
            raise

    async def __anext__(self) -> ChatChunk:
        """Get next chunk from the stream."""
        return await self._event_ch.receive()

    @property
    def input_tokens(self) -> int:
        """Number of input tokens (estimated)."""
        return self._input_tokens

    @property
    def output_tokens(self) -> int:
        """Number of output tokens."""
        return self._output_tokens


# Convenience function for creating LLM instance
def create_qwen3_llm(
    base_url: str = "http://localhost:8001",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> Qwen3LLM:
    """
    Create a Qwen3 LLM instance for LiveKit Agents.
    
    Args:
        base_url: Base URL of the Qwen3 service
        api_key: Optional API key
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        
    Returns:
        Configured Qwen3LLM instance
    """
    return Qwen3LLM(
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
