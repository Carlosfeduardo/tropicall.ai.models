"""
Debounced text accumulator for intelligent segment bundling.

The Kokoro model generates audio per complete segment, not token-by-token.
This accumulator bundles text intelligently to create segments with
good quality (avoiding too-short utterances).
"""

import asyncio
import logging
import re
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


class DebouncedTextAccumulator:
    """
    Accumulator with debounce timer - restarts timer on each text received.
    
    Pattern:
    1. Receive text -> add to buffer
    2. Check if complete segment (punctuation) -> emit immediately
    3. If not, (re)start 150ms timer
    4. If timer expires without new text -> emit buffer if sufficient content
    
    This avoids the problem of await sleep() inline which blocks the flow
    and creates unpredictable jitter.
    """

    PUNCTUATION = re.compile(r"[.!?;:]")

    def __init__(
        self,
        on_segment_ready: Callable[[str], Awaitable[None]],
        debounce_ms: int = 150,
        min_tokens: int = 12,
        min_tokens_flush: int = 5,
    ):
        """
        Initialize the accumulator.
        
        Args:
            on_segment_ready: Async callback when a segment is ready
            debounce_ms: Debounce timer in milliseconds
            min_tokens: Minimum tokens for quality segment
            min_tokens_flush: Minimum tokens for forced flush
        """
        self.on_segment_ready = on_segment_ready
        self.debounce_ms = debounce_ms
        self.min_tokens = min_tokens
        self.min_tokens_flush = min_tokens_flush
        
        self.buffer = ""
        self._timer_task: asyncio.Task | None = None
        self._cancelled = False

    async def add_text(self, text: str) -> None:
        """
        Add text and manage debounce timer.
        
        Args:
            text: Text to add to the buffer
        """
        if self._cancelled:
            return
        
        self.buffer += text
        
        # 1. Check if we have a complete segment (punctuation + enough tokens)
        segment = self._try_extract_complete_segment()
        if segment:
            self._cancel_timer()
            await self.on_segment_ready(segment)
            return
        
        # 2. Restart debounce timer (cancel previous if exists)
        self._restart_timer()

    def _try_extract_complete_segment(self) -> str | None:
        """
        Try to extract segment if we have punctuation AND enough tokens.
        
        Returns:
            Extracted segment or None
        """
        token_count = len(self.buffer.split())
        
        if token_count >= self.min_tokens and self.PUNCTUATION.search(self.buffer):
            # Find the last punctuation
            match = None
            for m in self.PUNCTUATION.finditer(self.buffer):
                match = m
            
            if match:
                segment = self.buffer[: match.end()]
                self.buffer = self.buffer[match.end() :].lstrip()
                return segment.strip()
        
        return None

    def _restart_timer(self) -> None:
        """Cancel previous timer and create new one."""
        self._cancel_timer()
        self._timer_task = asyncio.create_task(self._timer_expired())

    def _cancel_timer(self) -> None:
        """Cancel active timer."""
        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()
            self._timer_task = None

    async def _timer_expired(self) -> None:
        """
        Called when timer expires without new text.
        Emits buffer if it has minimum content.
        """
        try:
            await asyncio.sleep(self.debounce_ms / 1000)
            
            if self._cancelled:
                return
            
            # Timer expired - flush if we have minimum content
            token_count = len(self.buffer.split())
            if self.buffer.strip() and token_count >= self.min_tokens_flush:
                segment = self.buffer.strip()
                self.buffer = ""
                await self.on_segment_ready(segment)
                
        except asyncio.CancelledError:
            pass  # Timer was restarted or cancelled - normal

    async def flush(self) -> None:
        """Force flush of buffer (called on end_session)."""
        self._cancel_timer()
        if self.buffer.strip():
            segment = self.buffer.strip()
            self.buffer = ""
            await self.on_segment_ready(segment)

    async def cancel(self) -> None:
        """Cancel accumulator (called on cancel message)."""
        self._cancelled = True
        self._cancel_timer()
        self.buffer = ""

    @property
    def is_cancelled(self) -> bool:
        """Check if accumulator is cancelled."""
        return self._cancelled

    @property
    def pending_text(self) -> str:
        """Get current pending text in buffer."""
        return self.buffer
