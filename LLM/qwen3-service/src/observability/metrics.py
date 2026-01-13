"""
Prometheus metrics for the LLM service.

Exports:
- TTFT (Time to First Token) histogram
- Tokens per second histogram
- Request latency histogram
- Active sessions gauge
- Queue depth gauge
- GPU metrics
"""

from prometheus_client import Counter, Gauge, Histogram

# =============================================================================
# Latency metrics
# =============================================================================

llm_ttft = Histogram(
    "llm_time_to_first_token_seconds",
    "Time to first token in seconds",
    ["model"],
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0),
)

llm_tokens_per_second = Histogram(
    "llm_tokens_per_second",
    "Token generation rate (tokens per second)",
    ["model"],
    buckets=(10, 20, 30, 40, 50, 75, 100, 150, 200, 300),
)

llm_request_latency = Histogram(
    "llm_request_latency_seconds",
    "Total request latency in seconds",
    ["model", "stream"],
    buckets=(0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 60.0),
)

llm_queue_wait = Histogram(
    "llm_queue_wait_seconds",
    "Time request spent waiting in queue",
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
)

# =============================================================================
# Throughput metrics
# =============================================================================

llm_requests_total = Counter(
    "llm_requests_total",
    "Total LLM requests",
    ["status", "stream"],  # success, error, cancelled
)

llm_tokens_generated_total = Counter(
    "llm_tokens_generated_total",
    "Total tokens generated",
)

llm_prompt_tokens_total = Counter(
    "llm_prompt_tokens_total",
    "Total prompt tokens processed",
)

llm_sessions_total = Counter(
    "llm_sessions_total",
    "Total WebSocket sessions",
    ["end_reason"],  # completed, cancelled, error, timeout
)

llm_requests_rejected = Counter(
    "llm_requests_rejected_total",
    "Requests rejected by admission control",
    ["reason"],  # queue_full, queue_congestion, rate_limited
)

llm_slo_violations = Counter(
    "llm_slo_violations_total",
    "SLO violations detected",
    ["type"],  # ttft_p95, tokens_per_second
)

# =============================================================================
# State metrics
# =============================================================================

llm_active_sessions = Gauge(
    "llm_active_sessions",
    "Currently active WebSocket sessions",
)

llm_active_requests = Gauge(
    "llm_active_requests",
    "Currently active inference requests",
)

llm_pending_requests = Gauge(
    "llm_pending_requests",
    "Requests pending in queue",
)

llm_estimated_wait = Gauge(
    "llm_estimated_wait_ms",
    "Estimated wait time for new requests in ms",
)

llm_avg_ttft = Gauge(
    "llm_avg_ttft_ms",
    "EMA-tracked average TTFT in ms",
)

# =============================================================================
# GPU metrics
# =============================================================================

llm_gpu_utilization = Gauge(
    "llm_gpu_utilization_percent",
    "GPU utilization percentage",
    ["gpu_id"],
)

llm_gpu_memory_used = Gauge(
    "llm_gpu_memory_used_bytes",
    "GPU memory used in bytes",
    ["gpu_id"],
)

llm_gpu_memory_total = Gauge(
    "llm_gpu_memory_total_bytes",
    "GPU memory total in bytes",
    ["gpu_id"],
)

llm_gpu_temperature = Gauge(
    "llm_gpu_temperature_celsius",
    "GPU temperature in Celsius",
    ["gpu_id"],
)


# =============================================================================
# Helper functions
# =============================================================================


def record_generation_metrics(
    model: str,
    ttft_seconds: float,
    tokens_per_second: float,
    total_latency_seconds: float,
    prompt_tokens: int,
    completion_tokens: int,
    stream: bool,
    success: bool = True,
) -> None:
    """
    Record metrics for a completed generation.
    
    Args:
        model: Model name
        ttft_seconds: Time to first token in seconds
        tokens_per_second: Token generation rate
        total_latency_seconds: Total request latency
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        stream: Whether request was streamed
        success: Whether generation was successful
    """
    llm_ttft.labels(model=model).observe(ttft_seconds)
    llm_tokens_per_second.labels(model=model).observe(tokens_per_second)
    llm_request_latency.labels(
        model=model,
        stream=str(stream).lower(),
    ).observe(total_latency_seconds)
    
    llm_tokens_generated_total.inc(completion_tokens)
    llm_prompt_tokens_total.inc(prompt_tokens)
    
    status = "success" if success else "error"
    llm_requests_total.labels(
        status=status,
        stream=str(stream).lower(),
    ).inc()


def record_request_rejected(reason: str) -> None:
    """Record a request rejected by admission control."""
    llm_requests_rejected.labels(reason=reason).inc()


def record_slo_violation(violation_type: str) -> None:
    """Record an SLO violation."""
    llm_slo_violations.labels(type=violation_type).inc()


def record_session_ended(reason: str) -> None:
    """Record a session ending."""
    llm_sessions_total.labels(end_reason=reason).inc()


def update_queue_metrics(
    pending_count: int,
    active_count: int,
    estimated_wait_ms: float,
    avg_ttft_ms: float,
) -> None:
    """Update queue-related gauge metrics."""
    llm_pending_requests.set(pending_count)
    llm_active_requests.set(active_count)
    llm_estimated_wait.set(estimated_wait_ms)
    llm_avg_ttft.set(avg_ttft_ms)


def update_session_metrics(active_sessions: int) -> None:
    """Update session-related gauge metrics."""
    llm_active_sessions.set(active_sessions)


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
                "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    gpu_id = parts[0]
                    gpu_util = float(parts[1])
                    memory_used = float(parts[2]) * 1024 * 1024  # MiB to bytes
                    memory_total = float(parts[3]) * 1024 * 1024
                    temperature = float(parts[4])
                    
                    llm_gpu_utilization.labels(gpu_id=gpu_id).set(gpu_util)
                    llm_gpu_memory_used.labels(gpu_id=gpu_id).set(memory_used)
                    llm_gpu_memory_total.labels(gpu_id=gpu_id).set(memory_total)
                    llm_gpu_temperature.labels(gpu_id=gpu_id).set(temperature)
                    
    except Exception:
        # nvidia-smi not available or error
        pass
