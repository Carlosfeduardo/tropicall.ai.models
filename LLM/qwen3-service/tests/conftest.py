"""
Pytest fixtures for Qwen3 LLM service tests.
"""

import asyncio
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Note: These tests require a running vLLM engine
# For unit tests, mock the vLLM engine


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_messages():
    """Sample messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]


@pytest.fixture
def generation_config():
    """Sample generation config."""
    return {
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
    }


# Integration test fixtures (require running service)

@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """
    Async HTTP client for integration tests.
    
    Requires: Running Qwen3 service at localhost:8001
    """
    async with AsyncClient(
        base_url="http://localhost:8001",
        timeout=120.0,
    ) as client:
        yield client


@pytest.fixture
def ws_url():
    """WebSocket URL for testing."""
    return "ws://localhost:8001/ws/chat"
