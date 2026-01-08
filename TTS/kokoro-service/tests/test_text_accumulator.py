"""
Tests for DebouncedTextAccumulator.
"""

import asyncio

import pytest

from src.core.text_accumulator import DebouncedTextAccumulator


class TestDebouncedTextAccumulator:
    """Tests for DebouncedTextAccumulator."""

    @pytest.fixture
    def segments(self):
        """List to collect emitted segments."""
        return []

    @pytest.fixture
    def accumulator(self, segments):
        """Create accumulator with callback that collects segments."""
        async def on_segment(text: str):
            segments.append(text)
        
        return DebouncedTextAccumulator(
            on_segment_ready=on_segment,
            debounce_ms=50,  # Short for testing
            min_tokens=5,
            min_tokens_flush=2,
        )

    @pytest.mark.asyncio
    async def test_emits_on_punctuation(self, accumulator, segments):
        """Should emit segment when punctuation is found with enough tokens."""
        await accumulator.add_text("Olá, como você está? ")
        
        # Wait a bit for processing
        await asyncio.sleep(0.01)
        
        assert len(segments) == 1
        assert "Olá, como você está?" in segments[0]

    @pytest.mark.asyncio
    async def test_debounce_timer(self, accumulator, segments):
        """Should emit after debounce timeout if not enough punctuation."""
        # Add text without punctuation
        await accumulator.add_text("olá como vai")
        
        # Should not emit immediately
        await asyncio.sleep(0.01)
        assert len(segments) == 0
        
        # Wait for debounce to expire
        await asyncio.sleep(0.1)
        assert len(segments) == 1
        assert "olá como vai" in segments[0]

    @pytest.mark.asyncio
    async def test_debounce_timer_restarts(self, accumulator, segments):
        """Debounce timer should restart when new text arrives."""
        await accumulator.add_text("olá ")
        await asyncio.sleep(0.03)
        
        # Add more text - should restart timer
        await accumulator.add_text("como ")
        await asyncio.sleep(0.03)
        
        # Still shouldn't have emitted
        assert len(segments) == 0
        
        # Add more text - should restart timer again
        await accumulator.add_text("vai")
        
        # Wait for debounce
        await asyncio.sleep(0.1)
        assert len(segments) == 1
        assert "olá como vai" in segments[0]

    @pytest.mark.asyncio
    async def test_flush(self, accumulator, segments):
        """Flush should emit remaining buffer."""
        await accumulator.add_text("texto pendente")
        
        # Should not emit immediately (no punctuation, waiting for debounce)
        await asyncio.sleep(0.01)
        assert len(segments) == 0
        
        # Force flush
        await accumulator.flush()
        assert len(segments) == 1
        assert "texto pendente" in segments[0]

    @pytest.mark.asyncio
    async def test_cancel(self, accumulator, segments):
        """Cancel should discard buffer and stop processing."""
        await accumulator.add_text("texto que será descartado")
        await accumulator.cancel()
        
        # Should not emit after cancel
        await asyncio.sleep(0.1)
        assert len(segments) == 0
        assert accumulator.is_cancelled

    @pytest.mark.asyncio
    async def test_empty_buffer_not_emitted(self, accumulator, segments):
        """Empty buffer should not emit segment."""
        await accumulator.flush()
        assert len(segments) == 0

    @pytest.mark.asyncio
    async def test_multiple_segments(self, accumulator, segments):
        """Should handle multiple segments correctly."""
        await accumulator.add_text("Primeira frase. Segunda frase. ")
        
        await asyncio.sleep(0.01)
        
        # Should have emitted at least one segment
        assert len(segments) >= 1


class TestDebouncedTextAccumulatorEdgeCases:
    """Edge case tests for DebouncedTextAccumulator."""

    @pytest.mark.asyncio
    async def test_very_long_text(self):
        """Should handle very long text."""
        segments = []
        
        async def on_segment(text: str):
            segments.append(text)
        
        accumulator = DebouncedTextAccumulator(
            on_segment_ready=on_segment,
            debounce_ms=50,
            min_tokens=5,
            min_tokens_flush=2,
        )
        
        # Add a very long text
        long_text = "Esta é uma frase muito longa. " * 10
        await accumulator.add_text(long_text)
        
        await asyncio.sleep(0.1)
        
        # Should have emitted
        assert len(segments) >= 1

    @pytest.mark.asyncio
    async def test_only_whitespace(self):
        """Should not emit only whitespace."""
        segments = []
        
        async def on_segment(text: str):
            segments.append(text)
        
        accumulator = DebouncedTextAccumulator(
            on_segment_ready=on_segment,
            debounce_ms=50,
            min_tokens=5,
            min_tokens_flush=2,
        )
        
        await accumulator.add_text("   \n\t   ")
        await accumulator.flush()
        
        assert len(segments) == 0
