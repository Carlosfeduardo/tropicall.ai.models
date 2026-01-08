"""Configuration settings for the TTS service using pydantic-settings."""

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
    sample_rate: int = Field(default=24000, description="Audio sample rate in Hz")
    chunk_size_ms: int = Field(default=20, description="Audio chunk size in milliseconds")

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

    # Voice settings
    default_voice: Literal["pf_dora", "pm_alex", "pm_santa"] = Field(
        default="pf_dora",
        description="Default voice for pt-BR",
    )
    lang_code: str = Field(default="p", description="Language code for Kokoro")
    model_repo: str = Field(
        default="hexgrad/Kokoro-82M",
        description="HuggingFace model repository",
    )

    # Inference
    cuda_visible_devices: str = Field(default="0", description="CUDA device to use")

    # Observability
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )
    prometheus_port: int = Field(default=9090, description="Prometheus metrics port")

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
