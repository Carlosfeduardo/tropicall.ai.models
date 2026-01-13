"""
Tests for OpenAI-compatible API.

These are integration tests that require a running Qwen3 service.
Run with: pytest tests/test_openai_api.py -v
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check(async_client: AsyncClient):
    """Test health endpoint."""
    response = await async_client.get("/health/ready")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"


@pytest.mark.asyncio
async def test_list_models(async_client: AsyncClient):
    """Test models list endpoint."""
    response = await async_client.get("/v1/models")
    
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert len(data["data"]) > 0
    assert data["data"][0]["id"] == "qwen3-32b"


@pytest.mark.asyncio
async def test_chat_completion_non_streaming(async_client: AsyncClient):
    """Test non-streaming chat completion."""
    response = await async_client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3-32b",
            "messages": [
                {"role": "user", "content": "Say 'hello' and nothing else."}
            ],
            "max_tokens": 50,
            "stream": False,
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "choices" in data
    assert len(data["choices"]) == 1
    assert "message" in data["choices"][0]
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert len(data["choices"][0]["message"]["content"]) > 0
    
    assert "usage" in data
    assert data["usage"]["prompt_tokens"] > 0
    assert data["usage"]["completion_tokens"] > 0


@pytest.mark.asyncio
async def test_chat_completion_streaming(async_client: AsyncClient):
    """Test streaming chat completion."""
    async with async_client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "qwen3-32b",
            "messages": [
                {"role": "user", "content": "Say 'hello' and nothing else."}
            ],
            "max_tokens": 50,
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        
        chunks = []
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = line[6:]
                if data != "[DONE]":
                    chunks.append(data)
        
        # Should have received some chunks
        assert len(chunks) > 0


@pytest.mark.asyncio
async def test_chat_with_system_prompt(async_client: AsyncClient):
    """Test chat with system prompt."""
    response = await async_client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3-32b",
            "messages": [
                {"role": "system", "content": "You always respond in Portuguese."},
                {"role": "user", "content": "Say hello."},
            ],
            "max_tokens": 50,
            "stream": False,
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["choices"][0]["message"]["content"]) > 0


@pytest.mark.asyncio
async def test_temperature_parameter(async_client: AsyncClient):
    """Test temperature parameter."""
    # Low temperature - more deterministic
    response = await async_client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3-32b",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 20,
            "temperature": 0.0,
            "stream": False,
        },
    )
    
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_max_tokens_limit(async_client: AsyncClient):
    """Test max_tokens limits response length."""
    response = await async_client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3-32b",
            "messages": [
                {"role": "user", "content": "Write a very long story."}
            ],
            "max_tokens": 10,  # Very short
            "stream": False,
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Completion tokens should be around max_tokens
    assert data["usage"]["completion_tokens"] <= 15  # Allow some margin
