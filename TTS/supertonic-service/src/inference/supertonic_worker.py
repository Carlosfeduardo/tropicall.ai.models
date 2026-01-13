"""GPU worker for Supertonic 2 TTS inference via ONNX Runtime."""

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

# IMPORTANT: Patch supertonic config BEFORE importing TTS
# to enable CUDA execution provider for GPU acceleration
# The supertonic package hardcodes CPUExecutionProvider as default
# We need to patch BOTH config AND loader because loader does:
#   from .config import DEFAULT_ONNX_PROVIDERS (creates a local copy)
_CUDA_PROVIDERS = [
    "CUDAExecutionProvider",  # Try CUDA first
    "CPUExecutionProvider",   # Fall back to CPU if CUDA not available
]
import supertonic.config
supertonic.config.DEFAULT_ONNX_PROVIDERS = _CUDA_PROVIDERS
import supertonic.loader
supertonic.loader.DEFAULT_ONNX_PROVIDERS = _CUDA_PROVIDERS

if TYPE_CHECKING:
    from supertonic import TTS, Style

logger = logging.getLogger(__name__)


@dataclass
class InferenceTimings:
    """Detailed timing breakdown for inference."""
    
    inference_ms: float  # Pure model inference time
    postprocess_ms: float  # Audio conversion to int16


class SupertonicWorker:
    """
    Worker for Supertonic 2 TTS inference via ONNX Runtime.
    
    This worker should be used via InferenceQueue to ensure
    serialized access to the GPU (single consumer pattern).
    
    Supertonic 2 features:
    - 167x faster than real-time (M4 Pro with WebGPU)
    - 66M parameters (lightweight)
    - Multilingual: en, ko, es, pt, fr
    - ONNX Runtime for inference
    """

    SAMPLE_RATE = 44100  # Supertonic uses 44.1kHz

    def __init__(
        self,
        default_voice: str = "F1",
        model_dir: str | None = None,
        num_inference_steps: int = 5,
    ):
        """
        Initialize the Supertonic worker.
        
        Args:
            default_voice: Default voice style name (e.g., 'F1', 'M1')
            model_dir: Path to ONNX model directory (if None, downloads from HF)
            num_inference_steps: Number of inference steps (2 for fastest, 5 for quality)
        """
        self.default_voice = default_voice
        self.model_dir = model_dir
        self.num_inference_steps = num_inference_steps
        self.tts: "TTS | None" = None
        self._voice_styles: dict[str, "Style"] = {}
        self._is_ready = False

    @property
    def is_ready(self) -> bool:
        """Check if the worker is ready for inference."""
        return self._is_ready and self.tts is not None

    def warmup(self) -> None:
        """
        Load the model and run a warmup inference.
        
        This should be called during startup to reduce TTFA
        for the first real requests.
        """
        logger.info("Loading Supertonic 2 model...")
        
        # Import here to avoid loading at module import time
        from supertonic import TTS
        
        # Initialize TTS
        # If model_dir is None, it will download from HuggingFace
        self.tts = TTS(
            model_dir=self.model_dir,
            auto_download=True,
        )
        
        logger.info("Pre-loading voice styles...")
        
        # Pre-load common voice styles
        voice_names = ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]
        for voice_name in voice_names:
            try:
                self._voice_styles[voice_name] = self.tts.get_voice_style(voice_name)
                logger.debug(f"Loaded voice style: {voice_name}")
            except Exception as e:
                logger.warning(f"Could not load voice style {voice_name}: {e}")
        
        logger.info(f"Loaded {len(self._voice_styles)} voice styles")
        
        logger.info("Running warmup inference...")
        
        # Get default voice style
        default_style = self._get_voice_style(self.default_voice)
        
        # Run warmup inference
        warmup_text = "System initialization test."
        _, _ = self.tts.synthesize(
            warmup_text,
            voice_style=default_style,
            total_steps=self.num_inference_steps,
            speed=1.0,
        )
        
        self._is_ready = True
        logger.info("Supertonic worker ready")

    def _get_voice_style(self, voice: str) -> "Style":
        """Get voice style by name, with caching."""
        if voice in self._voice_styles:
            return self._voice_styles[voice]
        
        # Try to load it dynamically
        if self.tts is not None:
            try:
                style = self.tts.get_voice_style(voice)
                self._voice_styles[voice] = style
                return style
            except Exception as e:
                logger.warning(f"Could not load voice style {voice}: {e}")
        
        # Fallback to default
        if self.default_voice in self._voice_styles:
            logger.warning(f"Falling back to default voice {self.default_voice}")
            return self._voice_styles[self.default_voice]
        
        raise RuntimeError(f"No voice style available for {voice}")

    def generate_segment(
        self,
        text: str,
        voice: str = "F1",
        speed: float = 1.0,
    ) -> tuple[np.ndarray, InferenceTimings]:
        """
        Generate audio for a text segment.
        
        Args:
            text: Text to synthesize
            voice: Voice style name (e.g., 'F1', 'M1')
            speed: Speech speed multiplier (0.5-2.0)
            
        Returns:
            Tuple of (audio as int16 numpy array, timing breakdown)
        """
        if not self.is_ready:
            raise RuntimeError("Worker not ready - call warmup() first")
        
        if not text.strip():
            return np.array([], dtype=np.int16), InferenceTimings(0.0, 0.0)
        
        # Get voice style
        voice_style = self._get_voice_style(voice)
        
        # Start timing inference
        inference_start = time.monotonic()
        
        # Supertonic TTS.synthesize returns (audio_chunks, durations)
        # audio_chunks is np.ndarray of float32
        audio_chunks, durations = self.tts.synthesize(
            text,
            voice_style=voice_style,
            total_steps=self.num_inference_steps,
            speed=speed,
        )
        
        inference_end = time.monotonic()
        inference_ms = (inference_end - inference_start) * 1000
        
        # Start timing postprocess
        postprocess_start = time.monotonic()
        
        # audio_chunks is float32 in range [-1, 1]
        # Convert to int16 for PCM output
        if isinstance(audio_chunks, np.ndarray):
            # Clip to prevent overflow
            audio_clipped = np.clip(audio_chunks, -1.0, 1.0)
            audio_int16 = (audio_clipped * 32767).astype(np.int16)
        else:
            # If it's a tensor, convert to numpy first
            audio_np = np.array(audio_chunks)
            audio_clipped = np.clip(audio_np, -1.0, 1.0)
            audio_int16 = (audio_clipped * 32767).astype(np.int16)
        
        postprocess_ms = (time.monotonic() - postprocess_start) * 1000
        
        timings = InferenceTimings(
            inference_ms=inference_ms,
            postprocess_ms=postprocess_ms,
        )
        
        return audio_int16, timings

    def get_sample_rate(self) -> int:
        """Get the audio sample rate."""
        return self.SAMPLE_RATE
