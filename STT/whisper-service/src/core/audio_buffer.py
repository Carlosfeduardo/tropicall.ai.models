"""
Audio buffer with backpressure management.

Provides:
- Ring buffer for efficient audio storage
- Backpressure handling (trim/drop when overflow)
- PCM16 to float32 conversion
- Buffer statistics for monitoring
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class BufferStats:
    """Statistics about buffer state."""

    current_samples: int
    current_ms: float
    max_samples: int
    max_ms: float
    total_received_samples: int
    total_dropped_samples: int
    overflow_count: int


@dataclass
class AudioBuffer:
    """
    Audio buffer with backpressure handling.
    
    Features:
    - Efficient numpy-based storage
    - Automatic overflow handling with trim
    - PCM16 to float32 conversion
    - Statistics tracking
    
    Usage:
        buffer = AudioBuffer(max_buffer_ms=5000)
        
        # Add audio (PCM16 bytes)
        overflow = buffer.append_pcm16(audio_bytes)
        
        # Get samples for processing
        audio = buffer.get_samples()
        
        # Clear after processing
        buffer.clear()
    """

    max_buffer_ms: int = field(default_factory=lambda: settings.max_buffer_ms)
    sample_rate: int = field(default_factory=lambda: settings.sample_rate)
    trim_percent: float = field(default_factory=lambda: settings.buffer_trim_percent)

    # Internal state
    _samples: np.ndarray = field(init=False)
    _write_pos: int = field(default=0, init=False)
    _total_received: int = field(default=0, init=False)
    _total_dropped: int = field(default=0, init=False)
    _overflow_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Initialize the buffer array."""
        self._max_samples = int(self.sample_rate * self.max_buffer_ms / 1000)
        # Pre-allocate buffer
        self._samples = np.zeros(self._max_samples, dtype=np.float32)
        self._write_pos = 0

    @property
    def current_samples(self) -> int:
        """Number of samples currently in buffer."""
        return self._write_pos

    @property
    def current_ms(self) -> float:
        """Current buffer duration in milliseconds."""
        return self._write_pos / self.sample_rate * 1000

    @property
    def max_samples(self) -> int:
        """Maximum buffer size in samples."""
        return self._max_samples

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self._write_pos == 0

    @property
    def is_full(self) -> bool:
        """Check if buffer is at max capacity."""
        return self._write_pos >= self._max_samples

    @property
    def available_samples(self) -> int:
        """Available space in buffer (samples)."""
        return self._max_samples - self._write_pos

    def append_pcm16(self, data: bytes) -> bool:
        """
        Append PCM16 little-endian audio data to buffer.
        
        Args:
            data: PCM16 LE bytes (2 bytes per sample)
            
        Returns:
            True if overflow occurred (data was trimmed)
        """
        # Convert PCM16 to float32
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        return self.append_float32(samples)

    def append_float32(self, samples: np.ndarray) -> bool:
        """
        Append float32 audio samples to buffer.
        
        Args:
            samples: Float32 numpy array (normalized to [-1, 1])
            
        Returns:
            True if overflow occurred (data was trimmed)
        """
        num_samples = len(samples)
        self._total_received += num_samples
        
        overflow = False
        
        # Check if we need to trim
        if self._write_pos + num_samples > self._max_samples:
            overflow = True
            self._overflow_count += 1
            
            # Calculate how much to trim (from the beginning)
            trim_samples = int(self._write_pos * self.trim_percent)
            if trim_samples > 0:
                # Shift buffer contents
                remaining = self._write_pos - trim_samples
                self._samples[:remaining] = self._samples[trim_samples:self._write_pos]
                self._write_pos = remaining
                self._total_dropped += trim_samples
                
                logger.warning(
                    f"Buffer overflow: trimmed {trim_samples} samples "
                    f"({trim_samples / self.sample_rate * 1000:.0f}ms)"
                )
            
            # If still not enough space, drop incoming
            available = self._max_samples - self._write_pos
            if num_samples > available:
                dropped = num_samples - available
                samples = samples[-available:]
                num_samples = available
                self._total_dropped += dropped
                logger.warning(f"Dropped {dropped} incoming samples due to buffer full")
        
        # Append samples
        if num_samples > 0:
            self._samples[self._write_pos:self._write_pos + num_samples] = samples
            self._write_pos += num_samples
        
        return overflow

    def get_samples(self, start: int = 0, end: int | None = None) -> np.ndarray:
        """
        Get samples from buffer.
        
        Args:
            start: Start index (default: 0)
            end: End index (default: write position)
            
        Returns:
            Copy of samples in range
        """
        if end is None:
            end = self._write_pos
        
        end = min(end, self._write_pos)
        start = max(0, start)
        
        if start >= end:
            return np.array([], dtype=np.float32)
        
        return self._samples[start:end].copy()

    def get_last_n_samples(self, n: int) -> np.ndarray:
        """
        Get the last N samples from buffer.
        
        Args:
            n: Number of samples to get
            
        Returns:
            Copy of last N samples
        """
        n = min(n, self._write_pos)
        if n <= 0:
            return np.array([], dtype=np.float32)
        
        return self._samples[self._write_pos - n:self._write_pos].copy()

    def get_last_n_ms(self, ms: float) -> np.ndarray:
        """
        Get the last N milliseconds of audio.
        
        Args:
            ms: Duration in milliseconds
            
        Returns:
            Copy of samples
        """
        n_samples = int(self.sample_rate * ms / 1000)
        return self.get_last_n_samples(n_samples)

    def get_all(self) -> np.ndarray:
        """Get all samples in buffer."""
        return self.get_samples()

    def clear(self) -> None:
        """Clear the buffer."""
        self._write_pos = 0

    def consume(self, n_samples: int) -> np.ndarray:
        """
        Consume (get and remove) samples from the beginning.
        
        Args:
            n_samples: Number of samples to consume
            
        Returns:
            Consumed samples
        """
        n = min(n_samples, self._write_pos)
        if n <= 0:
            return np.array([], dtype=np.float32)
        
        consumed = self._samples[:n].copy()
        
        # Shift remaining samples
        remaining = self._write_pos - n
        if remaining > 0:
            self._samples[:remaining] = self._samples[n:self._write_pos]
        
        self._write_pos = remaining
        
        return consumed

    def consume_all(self) -> np.ndarray:
        """Consume all samples in buffer."""
        return self.consume(self._write_pos)

    def get_stats(self) -> BufferStats:
        """Get buffer statistics."""
        return BufferStats(
            current_samples=self._write_pos,
            current_ms=self.current_ms,
            max_samples=self._max_samples,
            max_ms=self.max_buffer_ms,
            total_received_samples=self._total_received,
            total_dropped_samples=self._total_dropped,
            overflow_count=self._overflow_count,
        )

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._total_received = 0
        self._total_dropped = 0
        self._overflow_count = 0


class UtteranceBuffer:
    """
    Buffer for accumulating audio samples for an utterance.
    
    Tracks speech start/end positions for partial/final extraction.
    """

    def __init__(
        self,
        max_utterance_s: float | None = None,
        sample_rate: int | None = None,
    ):
        """
        Initialize utterance buffer.
        
        Args:
            max_utterance_s: Maximum utterance duration (seconds)
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate or settings.sample_rate
        self.max_utterance_s = max_utterance_s or settings.max_utterance_s
        self._max_samples = int(self.sample_rate * self.max_utterance_s)
        
        # Dynamic array for utterance
        self._samples: list[np.ndarray] = []
        self._total_samples = 0
        
        # Tracking
        self._speech_start_sample: int | None = None
        self._last_partial_pos: int = 0

    @property
    def duration_samples(self) -> int:
        """Total samples in utterance."""
        return self._total_samples

    @property
    def duration_ms(self) -> float:
        """Total duration in milliseconds."""
        return self._total_samples / self.sample_rate * 1000

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self._total_samples == 0

    @property
    def is_at_max(self) -> bool:
        """Check if utterance reached maximum duration."""
        return self._total_samples >= self._max_samples

    def append(self, samples: np.ndarray) -> bool:
        """
        Append samples to utterance.
        
        Args:
            samples: Float32 audio samples
            
        Returns:
            True if max duration reached
        """
        # Check max duration
        space_available = self._max_samples - self._total_samples
        if len(samples) > space_available:
            samples = samples[:space_available]
        
        if len(samples) > 0:
            self._samples.append(samples)
            self._total_samples += len(samples)
        
        return self._total_samples >= self._max_samples

    def mark_speech_start(self) -> None:
        """Mark the current position as speech start."""
        self._speech_start_sample = self._total_samples

    def get_samples_for_partial(self, max_window_s: float | None = None) -> np.ndarray:
        """
        Get samples for partial transcription.
        
        Limits to last max_window_s seconds to bound latency.
        
        Args:
            max_window_s: Maximum window in seconds
            
        Returns:
            Audio samples for partial decode
        """
        if self._total_samples == 0:
            return np.array([], dtype=np.float32)
        
        # Get all samples
        all_samples = np.concatenate(self._samples)
        
        # Limit window if needed
        max_samples = settings.max_partial_window_samples
        if max_window_s is not None:
            max_samples = int(self.sample_rate * max_window_s)
        
        if len(all_samples) > max_samples:
            return all_samples[-max_samples:]
        
        return all_samples

    def get_samples_for_final(self) -> np.ndarray:
        """
        Get all samples for final transcription.
        
        Returns:
            All audio samples in utterance
        """
        if self._total_samples == 0:
            return np.array([], dtype=np.float32)
        
        return np.concatenate(self._samples)

    def get_samples_since_last_partial(self) -> np.ndarray:
        """Get samples added since last partial."""
        if self._total_samples <= self._last_partial_pos:
            return np.array([], dtype=np.float32)
        
        all_samples = np.concatenate(self._samples)
        return all_samples[self._last_partial_pos:]

    def mark_partial_sent(self) -> None:
        """Mark that a partial was sent at current position."""
        self._last_partial_pos = self._total_samples

    def clear(self) -> None:
        """Clear the utterance buffer."""
        self._samples.clear()
        self._total_samples = 0
        self._speech_start_sample = None
        self._last_partial_pos = 0
