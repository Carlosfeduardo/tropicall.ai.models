"""
Prometheus metrics for the TTS service.

Exports:
- TTFA (Time to First Audio) histogram
- RTF (Real-Time Factor) histogram
- Queue depth gauge
- Active sessions gauge
- GPU utilization gauge
"""

from prometheus_client import Counter, Gauge, Histogram

# =============================================================================
# Latency metrics
# =============================================================================

tts_ttfa = Histogram(
    "tts_time_to_first_audio_seconds",
    "Time to first audio in seconds",
    ["voice"],
    buckets=(0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0),
)

tts_chunk_latency = Histogram(
    "tts_chunk_latency_seconds",
    "Per-chunk streaming latency in seconds",
    buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1),
)

tts_rtf = Histogram(
    "tts_realtime_factor",
    "Real-time factor (< 1.0 = faster than realtime)",
    ["voice"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0),
)

tts_segment_duration = Histogram(
    "tts_segment_duration_seconds",
    "Duration of generated audio segments",
    buckets=(0.5, 1.0, 2.0, 3.0, 5.0, 10.0),
)

# =============================================================================
# Throughput metrics
# =============================================================================

tts_requests_total = Counter(
    "tts_requests_total",
    "Total TTS segment requests",
    ["status"],  # success, error, cancelled
)

tts_audio_seconds_total = Counter(
    "tts_audio_seconds_generated_total",
    "Total audio seconds generated",
)

tts_sessions_total = Counter(
    "tts_sessions_total",
    "Total sessions created",
    ["end_reason"],  # completed, cancelled, error
)

# =============================================================================
# State metrics
# =============================================================================

tts_active_sessions = Gauge(
    "tts_active_sessions",
    "Currently active sessions",
)

tts_queue_depth = Gauge(
    "tts_queue_depth",
    "Inference queue depth",
)

tts_inflight_segments = Gauge(
    "tts_inflight_segments_total",
    "Total segments currently in flight across all sessions",
)

tts_gpu_utilization = Gauge(
    "tts_gpu_utilization_percent",
    "GPU utilization percentage",
)

tts_gpu_memory_used = Gauge(
    "tts_gpu_memory_used_bytes",
    "GPU memory used in bytes",
)

tts_gpu_memory_total = Gauge(
    "tts_gpu_memory_total_bytes",
    "GPU memory total in bytes",
)


# =============================================================================
# Helper functions
# =============================================================================


def record_segment_metrics(
    voice: str,
    ttfa_seconds: float,
    rtf: float,
    duration_seconds: float,
    success: bool = True,
) -> None:
    """
    Record metrics for a completed segment.
    
    Args:
        voice: Voice used for synthesis
        ttfa_seconds: Time to first audio in seconds
        rtf: Real-time factor
        duration_seconds: Duration of generated audio
        success: Whether the segment was successful
    """
    tts_ttfa.labels(voice=voice).observe(ttfa_seconds)
    tts_rtf.labels(voice=voice).observe(rtf)
    tts_segment_duration.observe(duration_seconds)
    tts_audio_seconds_total.inc(duration_seconds)
    
    status = "success" if success else "error"
    tts_requests_total.labels(status=status).inc()


def record_session_ended(reason: str) -> None:
    """Record a session ending."""
    tts_sessions_total.labels(end_reason=reason).inc()


def update_gpu_metrics() -> None:
    """
    Update GPU metrics from nvidia-smi.
    
    This should be called periodically (e.g., every 10 seconds).
    """
    try:
        import subprocess
        
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 3:
                gpu_util = float(parts[0].strip())
                memory_used = float(parts[1].strip()) * 1024 * 1024  # MiB to bytes
                memory_total = float(parts[2].strip()) * 1024 * 1024
                
                tts_gpu_utilization.set(gpu_util)
                tts_gpu_memory_used.set(memory_used)
                tts_gpu_memory_total.set(memory_total)
                
    except Exception:
        # nvidia-smi not available or error
        pass
