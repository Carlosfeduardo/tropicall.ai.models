"""Configuration settings for the STT service using pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="STT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ==========================================================================
    # Server
    # ==========================================================================
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=8000, description="Port to bind to")

    # ==========================================================================
    # Session limits
    # ==========================================================================
    max_sessions: int = Field(default=100, description="Maximum concurrent sessions")
    max_active_talkers: int = Field(
        default=10,
        description="Maximum sessions actively transcribing simultaneously",
    )

    # ==========================================================================
    # Whisper Model Configuration
    # ==========================================================================
    whisper_model: str = Field(
        default="large-v3-turbo",
        description="Whisper model to use (large-v3-turbo, large-v3, distil-large-v3, etc.)",
    )
    compute_type: Literal["float16", "int8_float16", "int8", "float32"] = Field(
        default="float16",
        description="Compute type for inference (float16 recommended for GPU)",
    )
    device: Literal["cuda", "cpu", "auto"] = Field(
        default="cuda",
        description="Device for inference",
    )
    device_index: int = Field(
        default=0,
        description="GPU device index",
    )
    num_workers: int = Field(
        default=1,
        description="Number of CPU workers for feature extraction",
    )
    cpu_threads: int = Field(
        default=4,
        description="Number of CPU threads per worker",
    )

    # ==========================================================================
    # Transcription Settings (pt-BR optimized)
    # ==========================================================================
    language: str = Field(
        default="pt",
        description="Language code for transcription (pt for Portuguese)",
    )
    task: Literal["transcribe", "translate"] = Field(
        default="transcribe",
        description="Task: transcribe or translate to English",
    )
    beam_size: int = Field(
        default=5,
        description="Beam size for decoding (higher = better quality, slower)",
    )
    best_of: int = Field(
        default=1,
        description="Number of candidates to consider",
    )
    patience: float = Field(
        default=1.0,
        description="Beam search patience factor",
    )
    temperature: float = Field(
        default=0.0,
        description="Sampling temperature (0 = greedy)",
    )
    compression_ratio_threshold: float = Field(
        default=2.4,
        description="Threshold for compression ratio to detect hallucination",
    )
    log_prob_threshold: float = Field(
        default=-1.0,
        description="Threshold for log probability to detect low confidence",
    )
    no_speech_prob_threshold: float = Field(
        default=0.6,
        description="Threshold for no_speech probability",
    )
    initial_prompt: str = Field(
        default="",
        description="Initial prompt to guide language and style (leave empty for pure transcription)",
    )
    word_timestamps: bool = Field(
        default=True,
        description="Enable word-level timestamps in final transcripts",
    )

    # ==========================================================================
    # Audio Settings
    # ==========================================================================
    sample_rate: int = Field(
        default=16000,
        description="Expected audio sample rate in Hz",
    )
    chunk_size_ms: int = Field(
        default=20,
        description="Expected audio chunk size in milliseconds (20ms = 640 samples)",
    )

    # ==========================================================================
    # VAD (Voice Activity Detection) Settings
    # ==========================================================================
    vad_enabled: bool = Field(
        default=True,
        description="Enable VAD for endpointing",
    )
    vad_threshold: float = Field(
        default=0.5,
        description="VAD probability threshold for speech detection (0.0-1.0)",
    )
    vad_min_speech_ms: int = Field(
        default=250,
        description="Minimum speech duration to trigger start-of-speech (ms)",
    )
    vad_min_silence_ms: int = Field(
        default=500,
        description="Minimum silence duration to trigger end-of-speech (ms)",
    )
    vad_speech_pad_ms: int = Field(
        default=100,
        description="Padding around speech segments (ms)",
    )
    vad_window_size_samples: int = Field(
        default=512,
        description="VAD window size in samples (512 for 16kHz = 32ms)",
    )

    # ==========================================================================
    # Streaming / Partial Transcript Settings
    # ==========================================================================
    partial_interval_ms: int = Field(
        default=300,
        description="Interval between partial transcript updates (ms)",
    )
    max_partial_window_s: float = Field(
        default=8.0,
        description="Maximum audio window for partial decoding (seconds)",
    )
    max_utterance_s: float = Field(
        default=30.0,
        description="Maximum utterance duration before forced finalization (seconds)",
    )
    min_utterance_ms: int = Field(
        default=200,
        description="Minimum audio duration before attempting decode (ms)",
    )

    # ==========================================================================
    # Backpressure / Buffer Settings
    # ==========================================================================
    max_buffer_ms: int = Field(
        default=5000,
        description="Maximum audio buffer per session before dropping (ms)",
    )
    buffer_trim_percent: float = Field(
        default=0.5,
        description="Percentage of buffer to trim when overflow occurs",
    )

    # ==========================================================================
    # Inference Queue / Admission Control
    # ==========================================================================
    max_queue_depth: int = Field(
        default=20,
        description="Maximum queue depth before rejecting new requests",
    )
    max_estimated_server_total_ms: float = Field(
        default=500.0,
        description="Maximum estimated server_total (ms) before rejecting requests",
    )

    # ==========================================================================
    # Partial Shedding / Coalescing (Priority Queue Optimization)
    # ==========================================================================
    partial_shedding_enabled: bool = Field(
        default=True,
        description="Enable shedding (dropping) partials when queue is congested",
    )
    partial_shedding_queue_depth: int = Field(
        default=5,
        description="Queue depth threshold for shedding partials (finals always pass)",
    )
    partial_shedding_wait_ms: float = Field(
        default=200.0,
        description="Estimated wait threshold (ms) for shedding partials",
    )
    partial_coalescing_enabled: bool = Field(
        default=True,
        description="Enable per-session partial coalescing (keep only latest)",
    )

    # ==========================================================================
    # Whisper Anti-Hallucination Settings
    # ==========================================================================
    condition_on_previous_text: bool = Field(
        default=False,
        description="Whisper condition_on_previous_text (False reduces loops in streaming)",
    )

    # ==========================================================================
    # SLO (Service Level Objectives) - Server-Only Metrics
    # ==========================================================================
    slo_partial_p95_ms: float = Field(
        default=250.0,
        description="Target partial transcript latency p95 in ms",
    )
    slo_final_p95_ms: float = Field(
        default=1000.0,
        description="Target final transcript latency p95 in ms",
    )

    # ==========================================================================
    # Observability
    # ==========================================================================
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )
    prometheus_enabled: bool = Field(
        default=True,
        description="Enable Prometheus metrics endpoint",
    )

    # ==========================================================================
    # Computed Properties
    # ==========================================================================
    @property
    def chunk_size_samples(self) -> int:
        """Calculate chunk size in samples based on sample rate and chunk_size_ms."""
        return int(self.sample_rate * self.chunk_size_ms / 1000)

    @property
    def chunk_size_bytes(self) -> int:
        """Calculate chunk size in bytes (16-bit PCM = 2 bytes per sample)."""
        return self.chunk_size_samples * 2

    @property
    def max_buffer_samples(self) -> int:
        """Maximum buffer size in samples."""
        return int(self.sample_rate * self.max_buffer_ms / 1000)

    @property
    def max_partial_window_samples(self) -> int:
        """Maximum partial window in samples."""
        return int(self.sample_rate * self.max_partial_window_s)

    @property
    def max_utterance_samples(self) -> int:
        """Maximum utterance duration in samples."""
        return int(self.sample_rate * self.max_utterance_s)

    @property
    def partial_interval_samples(self) -> int:
        """Partial interval in samples."""
        return int(self.sample_rate * self.partial_interval_ms / 1000)

    @property
    def vad_min_speech_samples(self) -> int:
        """Minimum speech duration in samples."""
        return int(self.sample_rate * self.vad_min_speech_ms / 1000)

    @property
    def vad_min_silence_samples(self) -> int:
        """Minimum silence duration in samples."""
        return int(self.sample_rate * self.vad_min_silence_ms / 1000)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Export a default instance for convenience
settings = get_settings()
