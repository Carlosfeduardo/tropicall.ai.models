"""
Load Test for Kokoro TTS Service.

Usage:
    python load_test.py --sessions 10
    python load_test.py --sessions 50 --url wss://your-url/ws/tts
"""

import argparse
import asyncio
import json
import ssl
import statistics
import time
from dataclasses import dataclass, field
from uuid import uuid4

import websockets


# Default URL (update with your RunPod endpoint)
DEFAULT_URL = "wss://qg03vkmajpeg2d-8000.proxy.runpod.net/ws/tts"

# Test sentences (varied lengths for realistic testing)
TEST_SENTENCES = [
    "Olá! Como posso ajudar você hoje?",
    "Bom dia! Estou aqui para responder suas perguntas.",
    "Claro, posso te ajudar com isso. Me conte mais sobre o que você precisa.",
    "Entendi sua solicitação. Vou processar isso para você agora mesmo.",
    "Perfeito! Sua requisição foi processada com sucesso. Posso ajudar em algo mais?",
]


@dataclass
class SessionResult:
    """Result from a single TTS session."""
    session_id: str
    success: bool
    # TTFA metrics (separated for SLO monitoring)
    ttfa_from_text_ms: float = 0.0  # From send_text to first audio (for SLO)
    ttfa_total_ms: float = 0.0  # From connection to first audio
    # Server-reported metrics
    queue_wait_ms: float = 0.0  # Time in queue (from segment_done)
    inference_ms: float = 0.0  # Pure inference time (from segment_done)
    rtf: float = 0.0
    # Session metrics
    duration_ms: float = 0.0
    audio_bytes: int = 0
    audio_duration_ms: float = 0.0
    error: str | None = None


@dataclass
class LoadTestResults:
    """Aggregated results from load test."""
    total_sessions: int
    results: list[SessionResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def successful(self) -> list[SessionResult]:
        return [r for r in self.results if r.success]

    @property
    def failed(self) -> list[SessionResult]:
        return [r for r in self.results if not r.success]

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return len(self.successful) / len(self.results) * 100

    @property
    def total_duration_s(self) -> float:
        return self.end_time - self.start_time

    @property
    def throughput(self) -> float:
        if self.total_duration_s <= 0:
            return 0.0
        return len(self.successful) / self.total_duration_s


def percentile(data: list[float], p: int) -> float:
    """Calculate percentile of a list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


async def run_session(
    url: str,
    session_id: str,
    text: str,
    ssl_context: ssl.SSLContext,
) -> SessionResult:
    """Run a single TTS session and collect metrics."""
    result = SessionResult(session_id=session_id, success=False)
    
    # Timing points
    connect_time = time.monotonic()
    session_started_time: float | None = None
    text_sent_time: float | None = None
    first_audio_time: float | None = None

    try:
        async with websockets.connect(url, ssl=ssl_context, close_timeout=30) as ws:
            # Start session
            await ws.send(json.dumps({
                "type": "start_session",
                "session_id": session_id,
                "config": {"voice": "pf_dora", "lang_code": "p"}
            }))

            resp = json.loads(await ws.recv())
            if resp.get("type") != "session_started":
                result.error = f"Unexpected response: {resp}"
                return result
            
            session_started_time = time.monotonic()

            # Send text - record time for TTFA calculation
            text_sent_time = time.monotonic()
            await ws.send(json.dumps({"type": "send_text", "text": text}))
            await ws.send(json.dumps({"type": "flush"}))
            await ws.send(json.dumps({"type": "end_session"}))

            # Receive audio
            async for msg in ws:
                if isinstance(msg, bytes):
                    if first_audio_time is None:
                        first_audio_time = time.monotonic()
                    result.audio_bytes += len(msg)
                else:
                    data = json.loads(msg)
                    if data["type"] == "segment_done":
                        result.rtf = data.get("rtf", 0)
                        result.audio_duration_ms = data.get("audio_duration_ms", 0)
                        # Capture server-reported queue metrics
                        result.queue_wait_ms = data.get("queue_wait_ms", 0)
                        result.inference_ms = data.get("inference_ms", 0)
                    elif data["type"] == "session_ended":
                        break
                    elif data["type"] == "error":
                        result.error = data.get("message", "Unknown error")
                        return result

            end_time = time.monotonic()
            result.duration_ms = (end_time - connect_time) * 1000

            # Calculate TTFA metrics
            if first_audio_time and text_sent_time:
                # TTFA from text sent (for SLO monitoring)
                result.ttfa_from_text_ms = (first_audio_time - text_sent_time) * 1000
            if first_audio_time:
                # TTFA from connection (total latency)
                result.ttfa_total_ms = (first_audio_time - connect_time) * 1000

            result.success = True

    except websockets.exceptions.ConnectionClosed as e:
        result.error = f"Connection closed: {e}"
    except asyncio.TimeoutError:
        result.error = "Timeout"
    except Exception as e:
        result.error = str(e)

    return result


async def run_load_test(url: str, num_sessions: int) -> LoadTestResults:
    """Run load test with specified number of concurrent sessions."""
    results = LoadTestResults(total_sessions=num_sessions)

    # SSL context
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    # Create session tasks
    tasks = []
    for i in range(num_sessions):
        session_id = f"load-{uuid4().hex[:8]}"
        text = TEST_SENTENCES[i % len(TEST_SENTENCES)]
        task = run_session(url, session_id, text, ssl_context)
        tasks.append(task)

    print(f"\n{'='*50}")
    print(f"Load Test: {num_sessions} sessões simultâneas")
    print(f"{'='*50}\n")

    # Run all sessions concurrently
    results.start_time = time.monotonic()

    # Use gather with return_exceptions to handle errors gracefully
    completed = await asyncio.gather(*tasks, return_exceptions=True)

    results.end_time = time.monotonic()

    # Process results
    for i, result in enumerate(completed):
        if isinstance(result, BaseException):
            results.results.append(SessionResult(
                session_id=f"load-error-{i}",
                success=False,
                error=str(result)
            ))
        elif isinstance(result, SessionResult):
            results.results.append(result)

    return results


def print_results(results: LoadTestResults) -> None:
    """Print formatted load test results."""
    successful = results.successful
    failed = results.failed

    print(f"\n{'='*60}")
    print("RESULTADOS")
    print(f"{'='*60}\n")

    # Success rate
    print(f"Sessões: {len(successful)}/{results.total_sessions} sucesso ({results.success_rate:.1f}%)")
    if failed:
        print(f"Falhas:  {len(failed)}")
        for f in failed[:5]:  # Show first 5 errors
            print(f"  - {f.session_id}: {f.error}")
        if len(failed) > 5:
            print(f"  ... e mais {len(failed) - 5} erros")

    if not successful:
        print("\nNenhuma sessão bem sucedida para calcular métricas.")
        return

    # TTFA from text (for SLO monitoring)
    ttfas_from_text = [r.ttfa_from_text_ms for r in successful if r.ttfa_from_text_ms > 0]
    if ttfas_from_text:
        print(f"\nTTFA from Text (para SLO - send_text → primeiro áudio):")
        print(f"  Média: {statistics.mean(ttfas_from_text):>8.1f} ms")
        print(f"  p50:   {percentile(ttfas_from_text, 50):>8.1f} ms")
        print(f"  p95:   {percentile(ttfas_from_text, 95):>8.1f} ms  {'✓ OK' if percentile(ttfas_from_text, 95) < 250 else '✗ SLO VIOLADO (target: 250ms)'}")
        print(f"  p99:   {percentile(ttfas_from_text, 99):>8.1f} ms  {'✓ OK' if percentile(ttfas_from_text, 99) < 500 else '✗ SLO VIOLADO (target: 500ms)'}")
        print(f"  Min:   {min(ttfas_from_text):>8.1f} ms")
        print(f"  Max:   {max(ttfas_from_text):>8.1f} ms")

    # TTFA total (from connection)
    ttfas_total = [r.ttfa_total_ms for r in successful if r.ttfa_total_ms > 0]
    if ttfas_total:
        print(f"\nTTFA Total (conexão → primeiro áudio):")
        print(f"  Média: {statistics.mean(ttfas_total):>8.1f} ms")
        print(f"  p95:   {percentile(ttfas_total, 95):>8.1f} ms")

    # Queue wait (server-reported)
    queue_waits = [r.queue_wait_ms for r in successful if r.queue_wait_ms > 0]
    if queue_waits:
        print(f"\nQueue Wait (tempo na fila do servidor):")
        print(f"  Média: {statistics.mean(queue_waits):>8.1f} ms")
        print(f"  p50:   {percentile(queue_waits, 50):>8.1f} ms")
        print(f"  p95:   {percentile(queue_waits, 95):>8.1f} ms")
        print(f"  Max:   {max(queue_waits):>8.1f} ms")

    # Inference time (server-reported)
    inference_times = [r.inference_ms for r in successful if r.inference_ms > 0]
    if inference_times:
        print(f"\nInference Time (tempo de inferência puro):")
        print(f"  Média: {statistics.mean(inference_times):>8.1f} ms")
        print(f"  p50:   {percentile(inference_times, 50):>8.1f} ms")
        print(f"  p95:   {percentile(inference_times, 95):>8.1f} ms")

    # RTF metrics
    rtfs = [r.rtf for r in successful if r.rtf > 0]
    if rtfs:
        print(f"\nRTF (Real-Time Factor):")
        print(f"  Média: {statistics.mean(rtfs):>8.3f}  {'✓ OK' if statistics.mean(rtfs) < 0.8 else '✗ ALTO'}")
        print(f"  p95:   {percentile(rtfs, 95):>8.3f}")
        print(f"  Max:   {max(rtfs):>8.3f}")

    # Duration metrics
    durations = [r.duration_ms for r in successful]
    if durations:
        print(f"\nDuração da Sessão:")
        print(f"  Média: {statistics.mean(durations):>8.1f} ms")
        print(f"  p95:   {percentile(durations, 95):>8.1f} ms")
        print(f"  Max:   {max(durations):>8.1f} ms")

    # Audio stats
    total_audio_bytes = sum(r.audio_bytes for r in successful)
    total_audio_duration = sum(r.audio_duration_ms for r in successful) / 1000

    print(f"\nÁudio Gerado:")
    print(f"  Total:    {total_audio_bytes / 1024:.1f} KB")
    print(f"  Duração:  {total_audio_duration:.1f} s")

    # Overall stats
    print(f"\nPerformance Geral:")
    print(f"  Tempo total:  {results.total_duration_s:.2f} s")
    print(f"  Throughput:   {results.throughput:.2f} sessões/s")

    # Capacity estimate
    if inference_times:
        avg_inference = statistics.mean(inference_times)
        max_talkers_p95 = int(250 / avg_inference) if avg_inference > 0 else 0
        max_talkers_p99 = int(500 / avg_inference) if avg_inference > 0 else 0
        print(f"\nCapacidade Estimada:")
        print(f"  Com SLO TTFA p95 < 250ms: ~{max_talkers_p95} talkers simultâneos")
        print(f"  Com SLO TTFA p99 < 500ms: ~{max_talkers_p99} talkers simultâneos")

    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Load test for Kokoro TTS")
    parser.add_argument(
        "--sessions",
        type=int,
        default=10,
        help="Number of concurrent sessions (10-100)",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="WebSocket URL of the TTS service",
    )

    args = parser.parse_args()

    # Validate sessions
    if args.sessions < 1 or args.sessions > 100:
        print("Error: --sessions must be between 1 and 100")
        return

    print(f"URL: {args.url}")

    # Run load test
    results = asyncio.run(run_load_test(args.url, args.sessions))

    # Print results
    print_results(results)


if __name__ == "__main__":
    main()
