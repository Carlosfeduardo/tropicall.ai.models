"""
Prometheus metrics for the STT service.

Exports:
- Transcription latency histograms
- Queue metrics
- Session metrics
- GPU utilization gauges
"""

from prometheus_client import Counter, Gauge, Histogram

# =============================================================================
# Latency metrics
# =============================================================================

stt_partial_latency = Histogram(
    "stt_partial_latency_seconds",
    "Partial transcript latency in seconds (queue_wait + inference)",
    buckets=(0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0),
)

stt_final_latency = Histogram(
    "stt_final_latency_seconds",
    "Final transcript latency in seconds (queue_wait + inference)",
    buckets=(0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0),
)

stt_queue_wait = Histogram(
    "stt_queue_wait_seconds",
    "Time request spent waiting in queue",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0),
)

stt_inference_time = Histogram(
    "stt_inference_time_seconds",
    "Pure inference time (excluding queue wait)",
    buckets=(0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0),
)

stt_audio_duration = Histogram(
    "stt_audio_duration_seconds",
    "Duration of audio segments transcribed",
    buckets=(0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 30.0),
)

stt_rtf = Histogram(
    "stt_realtime_factor",
    "Real-time factor (inference_time / audio_duration, < 1.0 = faster than realtime)",
    buckets=(0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0),
)

# =============================================================================
# Throughput metrics
# =============================================================================

stt_transcripts_total = Counter(
    "stt_transcripts_total",
    "Total transcripts generated",
    ["type"],  # partial, final
)

stt_audio_seconds_total = Counter(
    "stt_audio_seconds_processed_total",
    "Total audio seconds processed",
)

stt_sessions_total = Counter(
    "stt_sessions_total",
    "Total sessions created",
    ["end_reason"],  # completed, cancelled, error
)

stt_requests_rejected = Counter(
    "stt_requests_rejected_total",
    "Requests rejected by admission control",
    ["reason"],  # queue_full, queue_congestion
)

stt_vad_events = Counter(
    "stt_vad_events_total",
    "Total VAD events",
    ["state"],  # speech_start, speech_end
)

# =============================================================================
# State metrics
# =============================================================================

stt_active_sessions = Gauge(
    "stt_active_sessions",
    "Currently active sessions",
)

stt_queue_depth = Gauge(
    "stt_queue_depth",
    "Inference queue depth",
)

stt_estimated_server_total = Gauge(
    "stt_estimated_server_total_ms",
    "Estimated server_total for new requests (queue_wait + processing)",
)

stt_avg_inference = Gauge(
    "stt_avg_inference_ms",
    "EMA-tracked average inference time in ms",
)

stt_gpu_utilization = Gauge(
    "stt_gpu_utilization_percent",
    "GPU utilization percentage",
)

stt_gpu_memory_used = Gauge(
    "stt_gpu_memory_used_bytes",
    "GPU memory used in bytes",
)

stt_gpu_memory_total = Gauge(
    "stt_gpu_memory_total_bytes",
    "GPU memory total in bytes",
)


# =============================================================================
# Helper functions
# =============================================================================


def record_partial_metrics(
    latency_seconds: float,
    queue_wait_seconds: float,
    inference_seconds: float,
    audio_duration_seconds: float,
) -> None:
    """Record metrics for a partial transcript."""
    stt_partial_latency.observe(latency_seconds)
    stt_queue_wait.observe(queue_wait_seconds)
    stt_inference_time.observe(inference_seconds)
    stt_transcripts_total.labels(type="partial").inc()
    
    if audio_duration_seconds > 0:
        rtf = inference_seconds / audio_duration_seconds
        stt_rtf.observe(rtf)


def record_final_metrics(
    latency_seconds: float,
    queue_wait_seconds: float,
    inference_seconds: float,
    audio_duration_seconds: float,
) -> None:
    """Record metrics for a final transcript."""
    stt_final_latency.observe(latency_seconds)
    stt_queue_wait.observe(queue_wait_seconds)
    stt_inference_time.observe(inference_seconds)
    stt_audio_duration.observe(audio_duration_seconds)
    stt_audio_seconds_total.inc(audio_duration_seconds)
    stt_transcripts_total.labels(type="final").inc()
    
    if audio_duration_seconds > 0:
        rtf = inference_seconds / audio_duration_seconds
        stt_rtf.observe(rtf)


def record_request_rejected(reason: str) -> None:
    """Record a request rejected by admission control."""
    stt_requests_rejected.labels(reason=reason).inc()


def record_session_ended(reason: str) -> None:
    """Record a session ending."""
    stt_sessions_total.labels(end_reason=reason).inc()


def record_vad_event(state: str) -> None:
    """Record a VAD event."""
    stt_vad_events.labels(state=state).inc()


def update_queue_metrics(
    queue_depth: int,
    estimated_server_total_ms: float,
    avg_inference_ms: float,
) -> None:
    """Update queue-related gauge metrics."""
    stt_queue_depth.set(queue_depth)
    stt_estimated_server_total.set(estimated_server_total_ms)
    stt_avg_inference.set(avg_inference_ms)


def update_gpu_metrics() -> None:
    """
    Update GPU metrics from nvidia-smi.
    
    Should be called periodically (e.g., every 10 seconds).
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
                
                stt_gpu_utilization.set(gpu_util)
                stt_gpu_memory_used.set(memory_used)
                stt_gpu_memory_total.set(memory_total)
                
    except Exception:
        # nvidia-smi not available or error
        pass
