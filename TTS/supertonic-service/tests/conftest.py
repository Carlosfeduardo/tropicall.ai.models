"""
Pytest fixtures for Kokoro TTS tests.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket for testing."""
    ws = AsyncMock()
    ws.send_bytes = AsyncMock()
    ws.send_json = AsyncMock()
    ws.send_text = AsyncMock()
    ws.receive_text = AsyncMock()
    ws.receive_bytes = AsyncMock()
    ws.close = AsyncMock()
    return ws


@pytest.fixture
def mock_kokoro_worker():
    """Create a mock KokoroWorker for testing."""
    worker = MagicMock()
    worker.is_ready = True
    worker.SAMPLE_RATE = 24000
    
    # Generate fake audio (1 second of silence)
    fake_audio = np.zeros(24000, dtype=np.int16)
    worker.generate_segment = MagicMock(return_value=fake_audio)
    worker.warmup = MagicMock()
    
    return worker


@pytest.fixture
def sample_audio():
    """Generate sample audio for testing."""
    # 0.5 seconds of sine wave at 440Hz
    sample_rate = 24000
    duration = 0.5
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * 440 * t)
    return (audio * 32767).astype(np.int16)
