"""Configuration settings for the LLM service using pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Server
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=8001, description="Port to bind to")

    # Model settings
    model_id: str = Field(
        default="Qwen/Qwen3-32B-Instruct",
        description="HuggingFace model ID",
    )
    model_revision: str = Field(
        default="main",
        description="Model revision/commit SHA for reproducibility",
    )

    # vLLM Engine settings
    max_model_len: int = Field(
        default=8192,
        description="Maximum context length",
    )
    gpu_memory_utilization: float = Field(
        default=0.9,
        ge=0.1,
        le=0.99,
        description="Fraction of GPU memory to use",
    )
    tensor_parallel_size: int = Field(
        default=1,
        description="Number of GPUs for tensor parallelism",
    )
    dtype: Literal["auto", "half", "float16", "bfloat16", "float32"] = Field(
        default="auto",
        description="Data type for model weights",
    )
    enforce_eager: bool = Field(
        default=False,
        description="Disable CUDA graphs for debugging",
    )
    enable_prefix_caching: bool = Field(
        default=True,
        description="Enable automatic prefix caching",
    )

    # LMCache settings (KV cache CPU offloading)
    # When enabled, KV cache is offloaded to CPU RAM, allowing larger models
    lmcache_enabled: bool = Field(
        default=False,
        description="Enable LMCache for KV cache offloading to CPU RAM",
    )
    lmcache_cpu_size_gb: int = Field(
        default=20,
        ge=1,
        le=256,
        description="CPU RAM size in GB allocated for KV cache offload",
    )

    # Generation defaults
    max_tokens: int = Field(
        default=2048,
        description="Default max tokens per response",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Default temperature",
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Default top_p (nucleus sampling)",
    )
    top_k: int = Field(
        default=50,
        description="Default top_k sampling",
    )
    repetition_penalty: float = Field(
        default=1.0,
        ge=1.0,
        le=2.0,
        description="Repetition penalty",
    )

    # Qwen3 specific
    enable_thinking: bool = Field(
        default=False,
        description="Enable thinking/reasoning mode (adds thinking tokens)",
    )
    thinking_budget: int = Field(
        default=1024,
        description="Max tokens for thinking when enabled",
    )

    # Session limits
    max_sessions: int = Field(
        default=50,
        description="Maximum concurrent WebSocket sessions",
    )
    max_history_turns: int = Field(
        default=20,
        description="Maximum conversation turns to keep in history",
    )
    session_timeout_seconds: int = Field(
        default=300,
        description="Session timeout in seconds (5 minutes)",
    )

    # Request limits
    max_concurrent_requests: int = Field(
        default=100,
        description="Maximum concurrent inference requests",
    )
    request_timeout_seconds: float = Field(
        default=120.0,
        description="Request timeout in seconds",
    )

    # Authentication
    api_key: str | None = Field(
        default=None,
        description="API key for authentication (optional)",
    )
    require_auth: bool = Field(
        default=False,
        description="Require authentication for all requests",
    )

    # Observability
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )
    prometheus_port: int = Field(
        default=9091,
        description="Prometheus metrics port",
    )

    # ==========================================================================
    # SLO (Service Level Objectives)
    # ==========================================================================

    slo_ttft_p95_ms: float = Field(
        default=500.0,
        description="Target time-to-first-token p95 in ms",
    )
    slo_tokens_per_second_min: float = Field(
        default=30.0,
        description="Minimum acceptable tokens per second",
    )

    # Admission Control
    max_queue_depth: int = Field(
        default=50,
        description="Maximum queue depth before rejecting new requests",
    )
    max_pending_tokens: int = Field(
        default=50000,
        description="Maximum pending tokens across all requests",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Export a default instance for convenience
settings = get_settings()
