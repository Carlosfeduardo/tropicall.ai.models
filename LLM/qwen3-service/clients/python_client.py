"""
Python client for Qwen3 LLM service.

Provides both WebSocket and HTTP clients for the Qwen3 service.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import AsyncGenerator, Optional
from uuid import uuid4

import httpx
import websockets
from websockets.client import WebSocketClientProtocol

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stop: Optional[list[str]] = None


@dataclass
class ChatMessage:
    """A message in the conversation."""

    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class GenerationResult:
    """Result of a generation request."""

    content: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    total_time_ms: float
    tokens_per_second: float


class Qwen3WebSocketClient:
    """
    WebSocket client for Qwen3 LLM service.
    
    Provides real-time streaming chat interface.
    
    Usage:
        async with Qwen3WebSocketClient("ws://localhost:8001") as client:
            await client.start_session("my-session")
            
            async for token in client.send_message("Hello!"):
                print(token, end="", flush=True)
            
            await client.end_session()
    """

    def __init__(
        self,
        url: str = "ws://localhost:8001",
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ):
        """
        Initialize WebSocket client.
        
        Args:
            url: WebSocket URL (ws:// or wss://)
            system_prompt: Optional system prompt
            config: Generation configuration
        """
        self._url = url.rstrip("/")
        self._ws_url = f"{self._url}/ws/chat"
        self._system_prompt = system_prompt
        self._config = config or GenerationConfig()
        self._ws: Optional[WebSocketClientProtocol] = None
        self._session_id: Optional[str] = None

    async def __aenter__(self) -> "Qwen3WebSocketClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """Connect to the WebSocket server."""
        if self._ws is not None:
            return
        
        self._ws = await websockets.connect(self._ws_url)
        logger.info(f"Connected to {self._ws_url}")

    async def close(self) -> None:
        """Close the WebSocket connection."""
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
            logger.info("WebSocket connection closed")

    async def start_session(
        self,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Start a new chat session.
        
        Args:
            session_id: Optional session ID (generated if not provided)
            system_prompt: Override system prompt for this session
            
        Returns:
            Session ID
        """
        if self._ws is None:
            await self.connect()
        
        self._session_id = session_id or f"session-{uuid4().hex[:8]}"
        
        start_msg = {
            "type": "start_session",
            "session_id": self._session_id,
            "config": {
                "system_prompt": system_prompt or self._system_prompt,
                "generation": {
                    "max_tokens": self._config.max_tokens,
                    "temperature": self._config.temperature,
                    "top_p": self._config.top_p,
                    "top_k": self._config.top_k,
                    "stop": self._config.stop,
                },
            },
        }
        
        await self._ws.send(json.dumps(start_msg))
        
        # Wait for session_started response
        response = await self._ws.recv()
        data = json.loads(response)
        
        if data["type"] == "error":
            raise Exception(f"Failed to start session: {data['message']}")
        
        if data["type"] != "session_started":
            raise Exception(f"Unexpected response: {data}")
        
        logger.info(f"Session started: {self._session_id}")
        return self._session_id

    async def send_message(
        self,
        content: str,
    ) -> AsyncGenerator[str, None]:
        """
        Send a message and stream the response.
        
        Args:
            content: User message content
            
        Yields:
            Generated tokens
        """
        if self._ws is None or self._session_id is None:
            raise RuntimeError("Session not started. Call start_session() first.")
        
        msg = {
            "type": "send_message",
            "content": content,
        }
        await self._ws.send(json.dumps(msg))
        
        # Stream tokens
        full_response = ""
        while True:
            response = await self._ws.recv()
            data = json.loads(response)
            
            if data["type"] == "token":
                token = data["token"]
                full_response += token
                yield token
                
            elif data["type"] == "message_done":
                logger.debug(
                    f"Generation complete: {data['usage']['completion_tokens']} tokens, "
                    f"TTFT={data['metrics']['ttft_ms']:.0f}ms, "
                    f"{data['metrics']['tokens_per_second']:.1f} tok/s"
                )
                break
                
            elif data["type"] == "error":
                raise Exception(f"Generation error: {data['message']}")

    async def cancel(self) -> None:
        """Cancel the current generation."""
        if self._ws is None:
            return
        
        await self._ws.send(json.dumps({"type": "cancel"}))
        logger.info("Sent cancel request")

    async def end_session(self) -> dict:
        """
        End the current session.
        
        Returns:
            Session metrics
        """
        if self._ws is None or self._session_id is None:
            return {}
        
        await self._ws.send(json.dumps({"type": "end_session"}))
        
        # Wait for session_ended response
        response = await self._ws.recv()
        data = json.loads(response)
        
        metrics = data.get("metrics", {})
        logger.info(f"Session ended: {self._session_id}")
        
        self._session_id = None
        return metrics


class Qwen3HTTPClient:
    """
    HTTP client for Qwen3 LLM service (OpenAI-compatible).
    
    Provides both streaming and non-streaming chat completions.
    
    Usage:
        client = Qwen3HTTPClient("http://localhost:8001")
        
        # Non-streaming
        result = await client.chat([
            ChatMessage(role="user", content="Hello!")
        ])
        print(result.content)
        
        # Streaming
        async for token in client.chat_stream([
            ChatMessage(role="user", content="Hello!")
        ]):
            print(token, end="", flush=True)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        api_key: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ):
        """
        Initialize HTTP client.
        
        Args:
            base_url: Base URL of the service
            api_key: Optional API key for authentication
            config: Generation configuration
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._config = config or GenerationConfig()
        
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        self._client = httpx.AsyncClient(
            base_url=f"{self._base_url}/v1",
            headers=headers,
            timeout=120.0,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def chat(
        self,
        messages: list[ChatMessage],
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """
        Send chat completion request (non-streaming).
        
        Args:
            messages: List of chat messages
            config: Override generation config
            
        Returns:
            GenerationResult with response and metrics
        """
        cfg = config or self._config
        
        request_body = {
            "model": "qwen3-32b",
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "top_p": cfg.top_p,
            "stream": False,
        }
        if cfg.stop:
            request_body["stop"] = cfg.stop
        
        response = await self._client.post("/chat/completions", json=request_body)
        response.raise_for_status()
        
        data = response.json()
        choice = data["choices"][0]
        usage = data["usage"]
        
        return GenerationResult(
            content=choice["message"]["content"],
            finish_reason=choice["finish_reason"],
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            ttft_ms=0,  # Not available in non-streaming
            total_time_ms=0,
            tokens_per_second=0,
        )

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        config: Optional[GenerationConfig] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Send streaming chat completion request.
        
        Args:
            messages: List of chat messages
            config: Override generation config
            
        Yields:
            Generated tokens
        """
        cfg = config or self._config
        
        request_body = {
            "model": "qwen3-32b",
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "top_p": cfg.top_p,
            "stream": True,
        }
        if cfg.stop:
            request_body["stop"] = cfg.stop
        
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
                    data = line[6:]
                    
                    if data == "[DONE]":
                        break
                    
                    try:
                        chunk = json.loads(data)
                        if "choices" in chunk and chunk["choices"]:
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue

    async def health_check(self) -> bool:
        """Check if the service is healthy."""
        try:
            response = await self._client.get(
                f"{self._base_url}/health/ready",
            )
            return response.status_code == 200
        except Exception:
            return False


# Example usage
async def main():
    """Example usage of Qwen3 clients."""
    
    # HTTP Client example
    print("=== HTTP Client ===")
    http_client = Qwen3HTTPClient("http://localhost:8001")
    
    # Check health
    if await http_client.health_check():
        print("Service is healthy!")
    
    # Streaming chat
    print("\nStreaming response:")
    async for token in http_client.chat_stream([
        ChatMessage(role="user", content="Tell me a short joke.")
    ]):
        print(token, end="", flush=True)
    print("\n")
    
    await http_client.close()
    
    # WebSocket Client example
    print("=== WebSocket Client ===")
    async with Qwen3WebSocketClient(
        "ws://localhost:8001",
        system_prompt="You are a helpful assistant.",
    ) as ws_client:
        await ws_client.start_session()
        
        print("Streaming response:")
        async for token in ws_client.send_message("What is the capital of France?"):
            print(token, end="", flush=True)
        print("\n")
        
        metrics = await ws_client.end_session()
        print(f"Session metrics: {metrics}")


if __name__ == "__main__":
    asyncio.run(main())
