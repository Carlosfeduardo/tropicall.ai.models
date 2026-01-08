"""
Async load test for Kokoro TTS service.

Measures TTFA (Time To First Audio), RTF, and handles barge-in scenarios.

Usage:
    python tests/load_test_async.py --sessions 30 --duration 60
    python tests/load_test_async.py --url ws://remote:8000/ws/tts --sessions 100
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field

import websockets


@dataclass
class SessionMetrics:
    """Metrics for a single session."""
    session_id: str
    ttfa_ms: float = 0
    total_duration_ms: float = 0
    audio_bytes: int = 0
    segments: int = 0
    error: str | None = None
    cancelled: bool = False


@dataclass
class LoadTestResults:
    """Aggregated load test results."""
    sessions: list[SessionMetrics] = field(default_factory=list)
    
    def summary(self) -> dict:
        """Generate summary statistics."""
        successful = [s for s in self.sessions if not s.error]
        failed = [s for s in self.sessions if s.error]
        cancelled = [s for s in self.sessions if s.cancelled]
        
        ttfas = [s.ttfa_ms for s in successful if s.ttfa_ms > 0]
        
        return {
            "total_sessions": len(self.sessions),
            "successful": len(successful),
            "failed": len(failed),
            "cancelled": len(cancelled),
            "ttfa_p50_ms": round(statistics.median(ttfas), 2) if ttfas else 0,
            "ttfa_p95_ms": round(
                statistics.quantiles(ttfas, n=20)[18] if len(ttfas) > 20 
                else max(ttfas, default=0), 2
            ),
            "ttfa_p99_ms": round(
                statistics.quantiles(ttfas, n=100)[98] if len(ttfas) > 100 
                else max(ttfas, default=0), 2
            ),
            "ttfa_mean_ms": round(statistics.mean(ttfas), 2) if ttfas else 0,
            "total_audio_bytes": sum(s.audio_bytes for s in successful),
            "total_segments": sum(s.segments for s in successful),
        }


async def run_session(
    session_id: str,
    url: str,
    text: str,
    cancel_after_ms: int | None = None,
) -> SessionMetrics:
    """
    Run a single TTS session.
    
    Args:
        session_id: Unique session identifier
        url: WebSocket URL
        text: Text to synthesize
        cancel_after_ms: If set, cancel the session after this many ms (barge-in test)
    """
    metrics = SessionMetrics(session_id=session_id)
    start_time = time.monotonic()
    first_audio_time = None
    cancel_task = None

    try:
        async with websockets.connect(url, close_timeout=10) as ws:
            # Start session
            await ws.send(json.dumps({
                "type": "start_session",
                "session_id": session_id,
                "config": {"voice": "pf_dora", "lang_code": "p"}
            }))

            resp = json.loads(await ws.recv())
            if resp["type"] != "session_started":
                metrics.error = f"Unexpected: {resp}"
                return metrics

            # Send text
            await ws.send(json.dumps({"type": "send_text", "text": text}))
            await ws.send(json.dumps({"type": "flush"}))
            
            # Schedule cancellation if testing barge-in
            if cancel_after_ms:
                async def cancel_session():
                    await asyncio.sleep(cancel_after_ms / 1000)
                    await ws.send(json.dumps({"type": "cancel"}))
                    metrics.cancelled = True
                
                cancel_task = asyncio.create_task(cancel_session())
            else:
                await ws.send(json.dumps({"type": "end_session"}))

            # Receive audio
            async for msg in ws:
                if isinstance(msg, bytes):
                    if first_audio_time is None:
                        first_audio_time = time.monotonic()
                        metrics.ttfa_ms = (first_audio_time - start_time) * 1000
                    metrics.audio_bytes += len(msg)
                else:
                    data = json.loads(msg)
                    if data["type"] == "segment_done":
                        metrics.segments += 1
                    elif data["type"] == "session_ended":
                        break
                    elif data["type"] == "error":
                        metrics.error = data.get("message")
                        break

            metrics.total_duration_ms = (time.monotonic() - start_time) * 1000

    except asyncio.TimeoutError:
        metrics.error = "Timeout"
    except websockets.exceptions.ConnectionClosed as e:
        metrics.error = f"Connection closed: {e}"
    except Exception as e:
        metrics.error = str(e)
    finally:
        if cancel_task:
            cancel_task.cancel()
            try:
                await cancel_task
            except asyncio.CancelledError:
                pass

    return metrics


async def run_load_test(
    url: str,
    num_sessions: int,
    duration_seconds: int,
    test_barge_in: bool = False,
) -> LoadTestResults:
    """
    Run load test.
    
    Args:
        url: WebSocket URL
        num_sessions: Number of concurrent sessions
        duration_seconds: Test duration in seconds
        test_barge_in: If True, randomly cancel some sessions (barge-in test)
    """
    results = LoadTestResults()
    test_text = (
        "Olá, este é um teste de carga do sistema de síntese de voz. "
        "Estamos verificando a latência e a capacidade de processamento. "
        "O sistema deve responder de forma rápida e consistente."
    )

    end_time = time.monotonic() + duration_seconds
    session_counter = 0
    active_tasks: set[asyncio.Task] = set()

    print(f"Starting load test: {num_sessions} concurrent sessions for {duration_seconds}s")
    print(f"URL: {url}")
    if test_barge_in:
        print("Barge-in testing enabled (random cancellations)")
    print()

    while time.monotonic() < end_time or active_tasks:
        # Spawn new sessions up to limit
        while len(active_tasks) < num_sessions and time.monotonic() < end_time:
            session_counter += 1
            
            # Randomly cancel some sessions for barge-in testing
            cancel_after = None
            if test_barge_in and session_counter % 5 == 0:
                cancel_after = 500  # Cancel after 500ms
            
            task = asyncio.create_task(
                run_session(
                    f"load-{session_counter}",
                    url,
                    test_text,
                    cancel_after,
                )
            )
            active_tasks.add(task)
            await asyncio.sleep(0.1)  # Stagger sessions

        # Collect completed sessions
        if active_tasks:
            done, active_tasks = await asyncio.wait(
                active_tasks,
                timeout=1.0,
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                try:
                    metrics = task.result()
                    results.sessions.append(metrics)
                except Exception as e:
                    print(f"Task error: {e}")

        # Progress
        elapsed = time.monotonic() - (end_time - duration_seconds)
        print(
            f"\rActive: {len(active_tasks):3d} | "
            f"Completed: {len(results.sessions):4d} | "
            f"Elapsed: {elapsed:.0f}s",
            end="",
            flush=True,
        )

    print("\n")
    return results


def print_results(results: LoadTestResults) -> None:
    """Print formatted results."""
    summary = results.summary()

    print("=" * 60)
    print("LOAD TEST RESULTS")
    print("=" * 60)
    print()
    print(f"Total Sessions:    {summary['total_sessions']}")
    print(f"  Successful:      {summary['successful']}")
    print(f"  Failed:          {summary['failed']}")
    print(f"  Cancelled:       {summary['cancelled']}")
    print()
    print("TTFA (Time To First Audio):")
    print(f"  p50:             {summary['ttfa_p50_ms']:.1f} ms")
    print(f"  p95:             {summary['ttfa_p95_ms']:.1f} ms")
    print(f"  p99:             {summary['ttfa_p99_ms']:.1f} ms")
    print(f"  mean:            {summary['ttfa_mean_ms']:.1f} ms")
    print()
    print(f"Total Audio:       {summary['total_audio_bytes'] / 1024:.1f} KB")
    print(f"Total Segments:    {summary['total_segments']}")
    print("=" * 60)

    # Print errors if any
    errors = [s for s in results.sessions if s.error]
    if errors:
        print("\nErrors:")
        error_counts: dict[str, int] = {}
        for s in errors:
            error_counts[s.error or "Unknown"] = error_counts.get(s.error or "Unknown", 0) + 1
        for error, count in error_counts.items():
            print(f"  {error}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Async load test for Kokoro TTS")
    parser.add_argument("--url", default="ws://localhost:8000/ws/tts", help="WebSocket URL")
    parser.add_argument("--sessions", type=int, default=10, help="Concurrent sessions")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--barge-in", action="store_true", help="Test barge-in (cancellation)")

    args = parser.parse_args()

    results = asyncio.run(
        run_load_test(
            args.url,
            args.sessions,
            args.duration,
            args.barge_in,
        )
    )

    print_results(results)


if __name__ == "__main__":
    main()
