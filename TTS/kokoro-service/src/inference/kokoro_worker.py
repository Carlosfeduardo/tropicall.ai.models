"""GPU worker for Kokoro-82M inference."""

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from kokoro import KPipeline

logger = logging.getLogger(__name__)


@dataclass
class InferenceTimings:
    """Detailed timing breakdown for inference."""
    
    inference_ms: float  # Pure model inference time
    postprocess_ms: float  # Audio conversion to int16


class KokoroWorker:
    """
    Worker for Kokoro-82M TTS inference.
    
    This worker should be used via InferenceQueue to ensure
    serialized access to the GPU (single consumer pattern).
    """

    SAMPLE_RATE = 24000

    def __init__(
        self,
        lang_code: str = "p",
        repo_id: str = "hexgrad/Kokoro-82M",
        device: str = "cuda",
    ):
        """
        Initialize the Kokoro worker.
        
        Args:
            lang_code: Language code ('p' for pt-BR)
            repo_id: HuggingFace model repository
            device: Device to use ('cuda' or 'cpu')
        """
        self.lang_code = lang_code
        self.repo_id = repo_id
        self.device = device
        self.pipeline: "KPipeline | None" = None
        self._is_ready = False

    @property
    def is_ready(self) -> bool:
        """Check if the worker is ready for inference."""
        return self._is_ready and self.pipeline is not None

    def warmup(self) -> None:
        """
        Load the model and run a warmup inference.
        
        This should be called during startup to reduce TTFA
        for the first real requests.
        """
        # Detect and log device
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"CUDA available: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            device = "cpu"
            logger.warning("CUDA NOT available - using CPU (slower inference)")
        
        self.device = device
        logger.info(f"Device selected: {device.upper()}")
        
        logger.info("Loading Kokoro pipeline...")
        
        # Import here to avoid loading at module import time
        from kokoro import KPipeline
        
        self.pipeline = KPipeline(
            lang_code=self.lang_code,
            repo_id=self.repo_id,
            device=device,  # Explicitly pass device!
        )
        
        logger.info("Running warmup inference...")
        
        # Run a dummy inference to warm up the model
        warmup_text = "Teste de inicialização do sistema."
        _ = list(self.pipeline(warmup_text, voice="pf_dora"))
        
        # Synchronize CUDA to ensure warmup is complete
        if cuda_available:
            torch.cuda.synchronize()
            # Log GPU memory usage after loading
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            logger.info(f"GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        
        self._is_ready = True
        logger.info(f"Kokoro worker ready on {device.upper()}")

    def generate_segment(
        self,
        text: str,
        voice: str = "pf_dora",
        speed: float = 1.0,
    ) -> tuple[np.ndarray, InferenceTimings]:
        """
        Generate audio for a text segment.
        
        Args:
            text: Text to synthesize
            voice: Voice to use (pf_dora, pm_alex, pm_santa)
            speed: Speech speed multiplier (0.5-2.0)
            
        Returns:
            Tuple of (audio as int16 numpy array, timing breakdown)
        """
        if not self.is_ready:
            raise RuntimeError("Worker not ready - call warmup() first")
        
        if not text.strip():
            return np.array([], dtype=np.int16), InferenceTimings(0.0, 0.0)
        
        # Start timing inference
        inference_start = time.monotonic()
        
        # KPipeline returns generator of (graphemes, phonemes, audio)
        # For a single segment, we take the first result
        for _gs, _ps, audio in self.pipeline(text, voice=voice, speed=speed):
            inference_end = time.monotonic()
            inference_ms = (inference_end - inference_start) * 1000
            
            # Start timing postprocess
            postprocess_start = time.monotonic()
            
            # audio can be PyTorch tensor or numpy array, depending on version
            # Convert to numpy if it's a tensor
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            
            # audio is float32 in range [-1, 1]
            # Convert to int16 for PCM output
            audio_int16 = (audio * 32767).astype(np.int16)
            
            postprocess_ms = (time.monotonic() - postprocess_start) * 1000
            
            timings = InferenceTimings(
                inference_ms=inference_ms,
                postprocess_ms=postprocess_ms,
            )
            return audio_int16, timings
        
        # If generator is empty, return empty array
        return np.array([], dtype=np.int16), InferenceTimings(0.0, 0.0)

    def get_sample_rate(self) -> int:
        """Get the audio sample rate."""
        return self.SAMPLE_RATE
