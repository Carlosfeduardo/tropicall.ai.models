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
    "Time to first audio in seconds (queue_wait + inference)",
    ["voice"],
    buckets=(0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0),
)

tts_queue_wait = Histogram(
    "tts_queue_wait_seconds",
    "Time request spent waiting in queue (before inference starts)",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0),
)

tts_inference_time = Histogram(
    "tts_inference_time_seconds",
    "Pure inference time (excluding queue wait)",
    ["voice"],
    buckets=(0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5),
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

tts_requests_rejected = Counter(
    "tts_requests_rejected_total",
    "Requests rejected by admission control",
    ["reason"],  # queue_full, queue_congestion
)

tts_slo_violations = Counter(
    "tts_slo_violations_total",
    "SLO violations detected",
    ["type"],  # ttfa_p95, ttfa_p99, rtf
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

tts_estimated_server_total = Gauge(
    "tts_estimated_server_total_ms",
    "Estimated server_total for new requests (queue_wait + processing)",
)

tts_avg_processing = Gauge(
    "tts_avg_processing_ms",
    "EMA-tracked average processing time (inference + postprocess) in ms",
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
    queue_wait_seconds: float = 0.0,
    inference_seconds: float = 0.0,
    success: bool = True,
) -> None:
    """
    Record metrics for a completed segment.
    
    Args:
        voice: Voice used for synthesis
        ttfa_seconds: Time to first audio in seconds (queue_wait + inference)
        rtf: Real-time factor
        duration_seconds: Duration of generated audio
        queue_wait_seconds: Time spent waiting in queue
        inference_seconds: Pure inference time
        success: Whether the segment was successful
    """
    tts_ttfa.labels(voice=voice).observe(ttfa_seconds)
    tts_rtf.labels(voice=voice).observe(rtf)
    tts_segment_duration.observe(duration_seconds)
    tts_audio_seconds_total.inc(duration_seconds)
    
    # Record queue wait and inference time
    if queue_wait_seconds > 0:
        tts_queue_wait.observe(queue_wait_seconds)
    if inference_seconds > 0:
        tts_inference_time.labels(voice=voice).observe(inference_seconds)
    
    status = "success" if success else "error"
    tts_requests_total.labels(status=status).inc()


def record_request_rejected(reason: str) -> None:
    """Record a request rejected by admission control."""
    tts_requests_rejected.labels(reason=reason).inc()


def record_slo_violation(violation_type: str) -> None:
    """Record an SLO violation."""
    tts_slo_violations.labels(type=violation_type).inc()


def update_queue_metrics(
    queue_depth: int,
    estimated_server_total_ms: float,
    avg_processing_ms: float,
) -> None:
    """Update queue-related gauge metrics (server-only SLO)."""
    tts_queue_depth.set(queue_depth)
    tts_estimated_server_total.set(estimated_server_total_ms)
    tts_avg_processing.set(avg_processing_ms)


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
