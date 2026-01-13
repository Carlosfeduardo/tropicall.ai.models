"""Configuration settings for the Supertonic TTS service using pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="TTS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Server
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=8000, description="Port to bind to")

    # Session limits
    max_sessions: int = Field(default=100, description="Maximum concurrent sessions")
    max_inflight_segments: int = Field(
        default=2,
        description="Maximum segments in flight per session (fairness control)",
    )

    # Audio settings
    sample_rate: int = Field(default=44100, description="Audio sample rate in Hz (Supertonic uses 44.1kHz)")
    chunk_size_ms: int = Field(default=20, description="Audio chunk size in milliseconds")
    
    # Audio pacing (send chunks at real-time rate)
    # DISABLED by default: Most TTS services (ElevenLabs, Play.ht, OpenAI) send
    # audio as fast as possible. The CLIENT is responsible for buffering and
    # playing at real-time. Server-side pacing creates backlog issues.
    audio_pacing_enabled: bool = Field(
        default=False,
        description="Enable pacing of audio chunks (send at real-time rate)",
    )
    pacing_factor: float = Field(
        default=0.8,
        description="Pacing speed factor (0.8 = 80% of real-time, faster than playback)",
    )

    # Text bundling
    debounce_ms: int = Field(
        default=150,
        description="Debounce timer for text bundling in milliseconds",
    )
    min_tokens: int = Field(default=12, description="Minimum tokens for a segment")
    max_tokens: int = Field(default=25, description="Maximum tokens before forced emit")
    min_tokens_flush: int = Field(
        default=5,
        description="Minimum tokens for forced flush",
    )

    # Voice settings - Supertonic 2 voices
    # Female: F1, F2, F3, F4, F5
    # Male: M1, M2, M3, M4, M5
    default_voice: Literal["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"] = Field(
        default="F1",
        description="Default voice for TTS",
    )
    
    # Model settings
    model_path: str | None = Field(
        default=None,
        description="Path to local ONNX model directory (if None, downloads from HuggingFace)",
    )
    num_inference_steps: int = Field(
        default=5,
        description="Number of inference steps (2 for fastest, 5 for quality)",
    )

    # Inference
    cuda_visible_devices: str = Field(default="0", description="CUDA device to use")

    # Observability
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )
    prometheus_port: int = Field(default=9090, description="Prometheus metrics port")

    # ==========================================================================
    # SLO (Service Level Objectives) - Server-Only Metrics
    # ==========================================================================
    # Note: These are SERVER-ONLY SLOs. Network latency is not our control.
    # Client-side TTFA = server_total + network_overhead
    
    # Server-only SLO targets (what we control)
    # Supertonic uses 5 inference steps by default, which is slower (~1000ms)
    # For 2 inference steps, it would be faster (~400ms)
    slo_server_total_p95_ms: float = Field(
        default=1500.0,
        description="Target server_total p95 in ms (preprocess+queue+inference+postprocess+send)",
    )
    slo_queue_wait_p95_ms: float = Field(
        default=500.0,
        description="Target queue_wait p95 in ms",
    )
    slo_inference_p95_ms: float = Field(
        default=1200.0,
        description="Target inference p95 in ms",
    )
    slo_rtf_max: float = Field(
        default=0.5,
        description="Maximum acceptable RTF (should be < 1.0 for realtime)",
    )
    
    # Admission Control - proactive rejection to protect server SLO
    max_queue_depth: int = Field(
        default=30,
        description="Maximum queue depth before rejecting new requests",
    )
    max_estimated_server_total_ms: float = Field(
        default=2000.0,
        description="Maximum estimated server_total (ms) before rejecting requests",
    )

    @property
    def chunk_size_samples(self) -> int:
        """Calculate chunk size in samples based on sample rate and chunk_size_ms."""
        return int(self.sample_rate * self.chunk_size_ms / 1000)

    @property
    def chunk_size_bytes(self) -> int:
        """Calculate chunk size in bytes (16-bit PCM = 2 bytes per sample)."""
        return self.chunk_size_samples * 2


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Export a default instance for convenience
settings = get_settings()
