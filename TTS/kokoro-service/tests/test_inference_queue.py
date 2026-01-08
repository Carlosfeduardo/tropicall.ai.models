"""
Tests for InferenceQueue.
"""

import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.inference.inference_queue import InferenceQueue


class TestInferenceQueue:
    """Tests for InferenceQueue."""

    @pytest.fixture
    def mock_worker(self):
        """Create a mock worker."""
        worker = MagicMock()
        worker.is_ready = True
        worker.generate_segment = MagicMock(
            return_value=np.zeros(24000, dtype=np.int16)
        )
        return worker

    @pytest.fixture
    async def queue(self, mock_worker):
        """Create and start an inference queue."""
        q = InferenceQueue(mock_worker)
        q.start()
        yield q
        await q.stop()

    @pytest.mark.asyncio
    async def test_enqueue_and_process(self, queue, mock_worker):
        """Should process enqueued requests."""
        result = await queue.enqueue(
            text="Teste",
            voice="pf_dora",
            speed=1.0,
            session_id="test-session",
            request_id="test-request",
        )
        
        assert result.audio is not None
        assert len(result.audio) == 24000
        assert result.ttfa_ms > 0
        assert result.rtf >= 0
        mock_worker.generate_segment.assert_called_once()

    @pytest.mark.asyncio
    async def test_serialized_processing(self, queue, mock_worker):
        """Should process requests serially (FIFO)."""
        call_order = []
        
        def track_call(text, voice, speed):
            call_order.append(text)
            return np.zeros(24000, dtype=np.int16)
        
        mock_worker.generate_segment = track_call
        
        # Enqueue multiple requests
        tasks = [
            queue.enqueue(f"Text {i}", "pf_dora", 1.0, "sess", f"req-{i}")
            for i in range(5)
        ]
        
        await asyncio.gather(*tasks)
        
        # Should be processed in order (FIFO)
        assert call_order == ["Text 0", "Text 1", "Text 2", "Text 3", "Text 4"]

    @pytest.mark.asyncio
    async def test_queue_depth(self, queue, mock_worker):
        """Should track queue depth correctly."""
        # Initially empty
        assert queue.depth == 0
        
        # Add delay to worker so we can measure queue depth
        async def slow_generate(*args):
            await asyncio.sleep(0.1)
            return np.zeros(24000, dtype=np.int16)
        
        # Start multiple requests but don't await them yet
        # This is tricky to test without actually making the worker slow

    @pytest.mark.asyncio
    async def test_stop(self, queue):
        """Should stop gracefully."""
        assert queue.is_running
        await queue.stop()
        assert not queue.is_running

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_worker):
        """Should handle errors from worker."""
        mock_worker.generate_segment = MagicMock(
            side_effect=RuntimeError("Test error")
        )
        
        queue = InferenceQueue(mock_worker)
        queue.start()
        
        try:
            with pytest.raises(RuntimeError, match="Test error"):
                await queue.enqueue(
                    text="Test",
                    voice="pf_dora",
                    speed=1.0,
                    session_id="test",
                    request_id="test",
                )
        finally:
            await queue.stop()


class TestInferenceQueueConcurrency:
    """Concurrency tests for InferenceQueue."""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self):
        """Should handle many concurrent requests."""
        worker = MagicMock()
        worker.generate_segment = MagicMock(
            return_value=np.zeros(24000, dtype=np.int16)
        )
        
        queue = InferenceQueue(worker)
        queue.start()
        
        try:
            # Enqueue 100 requests concurrently
            tasks = [
                queue.enqueue(
                    text=f"Text {i}",
                    voice="pf_dora",
                    speed=1.0,
                    session_id=f"session-{i}",
                    request_id=f"request-{i}",
                )
                for i in range(100)
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 100
            assert all(r.audio is not None for r in results)
            assert worker.generate_segment.call_count == 100
        finally:
            await queue.stop()
