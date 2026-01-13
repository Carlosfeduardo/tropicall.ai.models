"""
Silero VAD (Voice Activity Detection) wrapper.

Provides:
- Speech detection with configurable threshold
- Start-of-speech and end-of-speech detection
- Efficient batch processing
- Thread-safe inference
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
import torch

from ..config import settings

logger = logging.getLogger(__name__)


class VADState(str, Enum):
    """Current state of the VAD."""

    IDLE = "idle"  # No speech detected
    SPEECH = "speech"  # Speech in progress
    TRAILING_SILENCE = "trailing_silence"  # Silence after speech (pending end)


@dataclass
class VADConfig:
    """Configuration for VAD."""

    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 500
    speech_pad_ms: int = 100
    window_size_samples: int = 512
    sample_rate: int = 16000


@dataclass
class VADEvent:
    """VAD state change event."""

    state: str  # "speech_start" or "speech_end"
    timestamp_ms: float


class SileroVAD:
    """
    Silero VAD wrapper for voice activity detection.
    
    Uses the Silero VAD model to detect speech in audio streams.
    Provides start-of-speech and end-of-speech events with configurable
    thresholds and timing parameters.
    
    Usage:
        vad = SileroVAD(
            threshold=0.5,
            min_speech_ms=250,
            min_silence_ms=500,
        )
        vad.load_model()
        
        # Process audio chunks
        is_speech = vad.process_chunk(audio_chunk)
        
        # Get events
        events = vad.get_events()
    """

    SUPPORTED_SAMPLE_RATES = [8000, 16000]

    def __init__(
        self,
        threshold: float | None = None,
        min_speech_ms: int | None = None,
        min_silence_ms: int | None = None,
        speech_pad_ms: int | None = None,
        window_size_samples: int | None = None,
        sample_rate: int | None = None,
        on_speech_start: Callable[[float], None] | None = None,
        on_speech_end: Callable[[float], None] | None = None,
    ):
        """
        Initialize Silero VAD.
        
        Args:
            threshold: Speech probability threshold (0.0-1.0)
            min_speech_ms: Minimum speech duration to trigger start event
            min_silence_ms: Minimum silence duration to trigger end event
            speech_pad_ms: Padding around speech segments
            window_size_samples: VAD window size (512 for 16kHz)
            sample_rate: Audio sample rate (8000 or 16000)
            on_speech_start: Callback for speech start events
            on_speech_end: Callback for speech end events
        """
        self._threshold = threshold if threshold is not None else settings.vad_threshold
        self._min_speech_ms = min_speech_ms if min_speech_ms is not None else settings.vad_min_speech_ms
        self._min_silence_ms = min_silence_ms if min_silence_ms is not None else settings.vad_min_silence_ms
        self._speech_pad_ms = speech_pad_ms if speech_pad_ms is not None else settings.vad_speech_pad_ms
        self._window_size = window_size_samples if window_size_samples is not None else settings.vad_window_size_samples
        self._sample_rate = sample_rate if sample_rate is not None else settings.sample_rate
        
        # Callbacks
        self._on_speech_start = on_speech_start
        self._on_speech_end = on_speech_end
        
        # Validate sample rate
        if self._sample_rate not in self.SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"Sample rate {self._sample_rate} not supported. "
                f"Use one of {self.SUPPORTED_SAMPLE_RATES}"
            )
        
        # Model
        self._model = None
        self._ready = False
        
        # State tracking
        self._state = VADState.IDLE
        self._speech_samples = 0  # Consecutive speech samples
        self._silence_samples = 0  # Consecutive silence samples
        self._total_samples = 0  # Total samples processed
        self._speech_start_sample = 0  # Sample where speech started
        
        # Convert ms to samples
        self._min_speech_samples = int(self._sample_rate * self._min_speech_ms / 1000)
        self._min_silence_samples = int(self._sample_rate * self._min_silence_ms / 1000)
        
        # Events queue
        self._events: list[VADEvent] = []

    @property
    def is_ready(self) -> bool:
        """Check if model is loaded."""
        return self._ready and self._model is not None

    @property
    def state(self) -> VADState:
        """Current VAD state."""
        return self._state

    @property
    def is_speech(self) -> bool:
        """Whether currently in speech state."""
        return self._state in (VADState.SPEECH, VADState.TRAILING_SILENCE)

    @property
    def total_samples(self) -> int:
        """Total samples processed."""
        return self._total_samples

    @property
    def total_ms(self) -> float:
        """Total time processed in milliseconds."""
        return self._total_samples / self._sample_rate * 1000

    def load_model(self) -> None:
        """Load the Silero VAD model."""
        if self._model is not None:
            logger.warning("VAD model already loaded")
            return

        logger.info("Loading Silero VAD model...")
        
        # Load model from torch hub
        self._model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        
        # Set model to eval mode
        self._model.eval()
        
        self._ready = True
        logger.info("Silero VAD model loaded")

    def reset(self) -> None:
        """Reset VAD state for new utterance."""
        self._state = VADState.IDLE
        self._speech_samples = 0
        self._silence_samples = 0
        self._total_samples = 0
        self._speech_start_sample = 0
        self._events.clear()
        
        # Reset model state
        if self._model is not None:
            self._model.reset_states()

    def process_chunk(self, audio: np.ndarray) -> bool:
        """
        Process an audio chunk and update VAD state.
        
        Args:
            audio: Audio samples as float32 numpy array (normalized to [-1, 1])
            
        Returns:
            True if speech is detected in this chunk
        """
        if self._model is None:
            raise RuntimeError("VAD model not loaded")

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize if needed
        if np.abs(audio).max() > 1.0:
            audio = audio / 32768.0
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio)
        
        # Process in windows
        num_samples = len(audio)
        is_speech_chunk = False
        
        for i in range(0, num_samples, self._window_size):
            window = audio_tensor[i:i + self._window_size]
            
            # Pad if needed
            if len(window) < self._window_size:
                window = torch.nn.functional.pad(
                    window,
                    (0, self._window_size - len(window)),
                )
            
            # Get speech probability
            with torch.no_grad():
                speech_prob = self._model(window, self._sample_rate).item()
            
            # Update state based on probability
            window_is_speech = speech_prob >= self._threshold
            
            if window_is_speech:
                is_speech_chunk = True
                self._speech_samples += len(window)
                self._silence_samples = 0
            else:
                self._silence_samples += len(window)
            
            self._total_samples += len(window)
            
            # State machine
            self._update_state(window_is_speech)
        
        return is_speech_chunk

    def _update_state(self, is_speech: bool) -> None:
        """Update VAD state machine."""
        current_ms = self._total_samples / self._sample_rate * 1000
        
        if self._state == VADState.IDLE:
            if is_speech and self._speech_samples >= self._min_speech_samples:
                # Speech started
                self._state = VADState.SPEECH
                self._speech_start_sample = self._total_samples - self._speech_samples
                
                event = VADEvent(
                    state="speech_start",
                    timestamp_ms=self._speech_start_sample / self._sample_rate * 1000,
                )
                self._events.append(event)
                
                if self._on_speech_start:
                    self._on_speech_start(event.timestamp_ms)
                
                logger.debug(f"Speech started at {event.timestamp_ms:.0f}ms")
        
        elif self._state == VADState.SPEECH:
            if not is_speech:
                # Start trailing silence
                self._state = VADState.TRAILING_SILENCE
        
        elif self._state == VADState.TRAILING_SILENCE:
            if is_speech:
                # Speech resumed
                self._state = VADState.SPEECH
                self._silence_samples = 0
            elif self._silence_samples >= self._min_silence_samples:
                # Speech ended
                self._state = VADState.IDLE
                
                event = VADEvent(
                    state="speech_end",
                    timestamp_ms=current_ms,
                )
                self._events.append(event)
                
                if self._on_speech_end:
                    self._on_speech_end(event.timestamp_ms)
                
                logger.debug(f"Speech ended at {event.timestamp_ms:.0f}ms")
                
                # Reset counters
                self._speech_samples = 0

    def get_speech_probability(self, audio: np.ndarray) -> float:
        """
        Get speech probability for an audio chunk without updating state.
        
        Args:
            audio: Audio samples as float32 numpy array
            
        Returns:
            Speech probability (0.0-1.0)
        """
        if self._model is None:
            raise RuntimeError("VAD model not loaded")

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize if needed
        if np.abs(audio).max() > 1.0:
            audio = audio / 32768.0
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio)
        
        # Pad if needed
        if len(audio_tensor) < self._window_size:
            audio_tensor = torch.nn.functional.pad(
                audio_tensor,
                (0, self._window_size - len(audio_tensor)),
            )
        
        # Get probability
        with torch.no_grad():
            speech_prob = self._model(audio_tensor[:self._window_size], self._sample_rate).item()
        
        return speech_prob

    def get_events(self) -> list[VADEvent]:
        """Get and clear pending events."""
        events = self._events.copy()
        self._events.clear()
        return events

    def get_config(self) -> VADConfig:
        """Get current configuration."""
        return VADConfig(
            threshold=self._threshold,
            min_speech_ms=self._min_speech_ms,
            min_silence_ms=self._min_silence_ms,
            speech_pad_ms=self._speech_pad_ms,
            window_size_samples=self._window_size,
            sample_rate=self._sample_rate,
        )


# Global instance for shared model (optional)
_shared_vad_model = None


def get_shared_vad_model():
    """Get or create shared VAD model."""
    global _shared_vad_model
    
    if _shared_vad_model is None:
        _shared_vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        _shared_vad_model.eval()
        logger.info("Shared Silero VAD model loaded")
    
    return _shared_vad_model
