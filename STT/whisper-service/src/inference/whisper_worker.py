"""
Faster-Whisper worker for STT inference.

Provides:
- Model loading with configurable compute type
- Warmup for consistent first-request latency
- Transcription with word timestamps
- Timing metrics for observability
"""

import logging
import time
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionTimings:
    """Timing breakdown for transcription."""

    inference_ms: float
    postprocess_ms: float


@dataclass
class WordTimestamp:
    """Word-level timestamp information."""

    word: str
    start: float
    end: float
    probability: float


@dataclass
class TranscriptionResult:
    """Result of transcription with metadata."""

    text: str
    words: list[WordTimestamp]
    language: str
    language_probability: float
    duration: float  # Audio duration in seconds
    no_speech_probability: float
    timings: TranscriptionTimings


class WhisperWorker:
    """
    Wrapper around Faster-Whisper for STT inference.
    
    Features:
    - Single model instance (GPU memory efficient)
    - Configurable compute type (float16, int8_float16, etc.)
    - Warmup for consistent latency
    - Detailed timing metrics
    
    Usage:
        worker = WhisperWorker()
        worker.load_model()
        worker.warmup()
        
        result = worker.transcribe(audio_samples)
    """

    SAMPLE_RATE = 16000  # Whisper expects 16kHz
    
    # Map custom model names to HuggingFace model IDs
    MODEL_MAP = {
        "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
        "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    }

    def __init__(
        self,
        model_size: str | None = None,
        device: str | None = None,
        device_index: int | None = None,
        compute_type: str | None = None,
        num_workers: int | None = None,
        cpu_threads: int | None = None,
    ):
        """
        Initialize the Whisper worker.
        
        Args:
            model_size: Model size (e.g., "distil-whisper-large-v3", "large-v3")
            device: Device to use ("cuda", "cpu", "auto")
            device_index: GPU device index
            compute_type: Compute type ("float16", "int8_float16", etc.)
            num_workers: Number of CPU workers for feature extraction
            cpu_threads: Number of CPU threads per worker
        """
        self._model_size = model_size or settings.whisper_model
        self._device = device or settings.device
        self._device_index = device_index if device_index is not None else settings.device_index
        self._compute_type = compute_type or settings.compute_type
        self._num_workers = num_workers if num_workers is not None else settings.num_workers
        self._cpu_threads = cpu_threads if cpu_threads is not None else settings.cpu_threads
        
        self._model: WhisperModel | None = None
        self._ready = False

    @property
    def is_ready(self) -> bool:
        """Check if model is loaded and ready."""
        return self._ready and self._model is not None

    @property
    def model_info(self) -> dict:
        """Get model information."""
        return {
            "model_size": self._model_size,
            "device": self._device,
            "device_index": self._device_index,
            "compute_type": self._compute_type,
            "ready": self._ready,
        }

    def load_model(self) -> None:
        """
        Load the Whisper model.
        
        This should be called once at startup.
        """
        if self._model is not None:
            logger.warning("Model already loaded")
            return

        # Map custom model names to HuggingFace model IDs
        model_path = self.MODEL_MAP.get(self._model_size, self._model_size)
        
        logger.info(
            f"Loading Whisper model: {self._model_size} ({model_path}) "
            f"(device={self._device}, compute_type={self._compute_type})"
        )
        
        start_time = time.monotonic()
        
        self._model = WhisperModel(
            model_size_or_path=model_path,
            device=self._device,
            device_index=self._device_index,
            compute_type=self._compute_type,
            num_workers=self._num_workers,
            cpu_threads=self._cpu_threads,
        )
        
        load_time = time.monotonic() - start_time
        logger.info(f"Model loaded in {load_time:.2f}s")

    def warmup(self) -> None:
        """
        Warm up the model with a short audio sample.
        
        This ensures consistent latency on first real request.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        logger.info("Warming up Whisper model...")
        
        # Create a short silent audio sample (1 second)
        warmup_audio = np.zeros(self.SAMPLE_RATE, dtype=np.float32)
        
        # Run a transcription to warm up
        start_time = time.monotonic()
        
        segments, _ = self._model.transcribe(
            warmup_audio,
            language=settings.language,
            task=settings.task,
            beam_size=1,  # Fast warmup
            vad_filter=False,
            word_timestamps=False,
        )
        
        # Consume the generator
        for _ in segments:
            pass
        
        warmup_time = time.monotonic() - start_time
        logger.info(f"Warmup complete in {warmup_time:.2f}s")
        
        self._ready = True

    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None = None,
        task: str | None = None,
        beam_size: int | None = None,
        best_of: int | None = None,
        patience: float | None = None,
        temperature: float | None = None,
        word_timestamps: bool | None = None,
        vad_filter: bool = False,
        initial_prompt: str | None = None,
        is_final: bool = False,
        condition_on_previous_text: bool | None = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio samples as float32 numpy array (16kHz mono)
            language: Language code (default from settings)
            task: Task (transcribe or translate)
            beam_size: Beam size for decoding
            best_of: Number of candidates
            patience: Beam search patience
            temperature: Sampling temperature
            word_timestamps: Include word-level timestamps
            vad_filter: Use Whisper's internal VAD (we handle externally)
            initial_prompt: Initial prompt for context
            is_final: Whether this is a final transcription (uses different params)
            condition_on_previous_text: Override for condition_on_previous_text
            
        Returns:
            TranscriptionResult with text, words, and timing metrics
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")

        # Use defaults from settings
        language = language or settings.language
        task = task or settings.task
        beam_size = beam_size if beam_size is not None else settings.beam_size
        best_of = best_of if best_of is not None else settings.best_of
        patience = patience if patience is not None else settings.patience
        word_timestamps = word_timestamps if word_timestamps is not None else settings.word_timestamps
        
        # Different params for partial vs final to reduce hallucination
        if is_final:
            # Final: slightly more flexible, can use previous text conditioning
            temperature = temperature if temperature is not None else 0.1
            cond_prev = condition_on_previous_text if condition_on_previous_text is not None else True
        else:
            # Partial: strict settings to avoid loops
            temperature = temperature if temperature is not None else 0.0
            cond_prev = condition_on_previous_text if condition_on_previous_text is not None else settings.condition_on_previous_text
        
        # Use initial_prompt from parameter, or settings if not provided
        if initial_prompt is None and settings.initial_prompt:
            initial_prompt = settings.initial_prompt

        # Ensure audio is float32 and normalized
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize if needed (Whisper expects [-1, 1] range)
        if np.abs(audio).max() > 1.0:
            audio = audio / 32768.0

        # Track inference time
        inference_start = time.monotonic()
        
        segments, info = self._model.transcribe(
            audio,
            language=language,
            task=task,
            beam_size=beam_size,
            best_of=best_of,
            patience=patience,
            temperature=temperature,
            compression_ratio_threshold=settings.compression_ratio_threshold,
            log_prob_threshold=settings.log_prob_threshold,
            no_speech_threshold=settings.no_speech_prob_threshold,
            condition_on_previous_text=cond_prev,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
            initial_prompt=initial_prompt if initial_prompt else None,
        )
        
        # Collect segments (generator)
        text_parts: list[str] = []
        words: list[WordTimestamp] = []
        no_speech_prob = 0.0
        segment_count = 0
        
        for segment in segments:
            text_parts.append(segment.text.strip())
            no_speech_prob = max(no_speech_prob, segment.no_speech_prob)
            segment_count += 1
            
            if word_timestamps and segment.words:
                for word in segment.words:
                    words.append(
                        WordTimestamp(
                            word=word.word,
                            start=word.start,
                            end=word.end,
                            probability=word.probability,
                        )
                    )
        
        inference_ms = (time.monotonic() - inference_start) * 1000
        
        # Postprocess: combine text
        postprocess_start = time.monotonic()
        full_text = " ".join(text_parts).strip()
        postprocess_ms = (time.monotonic() - postprocess_start) * 1000
        
        # Calculate audio duration
        audio_duration = len(audio) / self.SAMPLE_RATE
        
        return TranscriptionResult(
            text=full_text,
            words=words,
            language=info.language,
            language_probability=info.language_probability,
            duration=audio_duration,
            no_speech_probability=no_speech_prob,
            timings=TranscriptionTimings(
                inference_ms=inference_ms,
                postprocess_ms=postprocess_ms,
            ),
        )

    def transcribe_partial(
        self,
        audio: np.ndarray,
        language: str | None = None,
        max_window_samples: int | None = None,
        word_timestamps: bool = False,
    ) -> TranscriptionResult:
        """
        Transcribe audio for partial results (optimized for speed).
        
        Uses reduced beam size and strict anti-hallucination settings:
        - NO initial_prompt (causes loops)
        - condition_on_previous_text=False (critical for avoiding loops)
        - beam_size=1 (greedy, fast)
        - temperature=0 (deterministic)
        
        Args:
            audio: Audio samples as float32 numpy array
            language: Language code
            max_window_samples: Maximum samples to transcribe (sliding window)
            word_timestamps: Include word timestamps (default False for partials)
            
        Returns:
            TranscriptionResult optimized for streaming partials
        """
        # Limit window size for partial decoding
        max_samples = max_window_samples or settings.max_partial_window_samples
        if len(audio) > max_samples:
            audio = audio[-max_samples:]
        
        return self.transcribe(
            audio=audio,
            language=language,
            beam_size=1,  # Greedy for speed
            best_of=1,
            word_timestamps=word_timestamps,
            initial_prompt=None,  # NEVER use initial_prompt for partials - causes loops
            is_final=False,
            condition_on_previous_text=False,  # Critical for avoiding loops
        )

    def transcribe_final(
        self,
        audio: np.ndarray,
        language: str | None = None,
        word_timestamps: bool = True,
    ) -> TranscriptionResult:
        """
        Transcribe audio for final results (full quality).
        
        Uses full beam search for best quality.
        Each utterance is independent - no context from previous utterances.
        
        Args:
            audio: Audio samples as float32 numpy array
            language: Language code
            word_timestamps: Include word timestamps
            
        Returns:
            TranscriptionResult with full quality transcription
        """
        return self.transcribe(
            audio=audio,
            language=language,
            word_timestamps=word_timestamps,
            initial_prompt=None,  # Each utterance is independent
            is_final=True,
            condition_on_previous_text=False,  # Safer default - avoids propagating errors
        )


# Global instance (initialized in lifespan)
whisper_worker: WhisperWorker | None = None


def get_whisper_worker() -> WhisperWorker:
    """Get the global Whisper worker instance."""
    if whisper_worker is None:
        raise RuntimeError("WhisperWorker not initialized")
    return whisper_worker
