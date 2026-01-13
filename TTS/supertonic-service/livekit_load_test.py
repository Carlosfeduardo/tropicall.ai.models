"""
LiveKit Simulation Load Test for Kokoro TTS Service.

Simulates realistic LiveKit agent behavior:
- Long-running sessions (60-120s)
- Periodic text segments (every 800-1500ms)
- Multiple concurrent sessions
- Optional wait-for-done mode (realistic agent behavior)

Modes:
- --wait-for-done (default): Sender waits for segment_done before sending next.
  This simulates a real agent that doesn't talk over itself.
- --no-wait-for-done: Sender fires segments as fast as intervals allow.
  This creates maximum stress but is unrealistic.

Usage:
    python livekit_load_test.py --sessions 3 --duration 60
    python livekit_load_test.py --sessions 50 --duration 120 --no-wait-for-done
"""

import argparse
import asyncio
import json
import logging
import random
import ssl
import statistics
import sys
import time
from dataclasses import dataclass, field
from uuid import uuid4

import websockets

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# Default URL (update with your RunPod endpoint)
DEFAULT_URL = "wss://3xob8ka3v5feuf-8000.proxy.runpod.net/ws/tts"

# Test sentences (varied lengths for realistic testing)
TEST_SENTENCES = [
    "Olá! Como posso ajudar você hoje?",
    "Claro, vou verificar isso para você.",
    "Um momento, por favor.",
    "Entendi sua solicitação.",
    "Perfeito! Posso ajudar em algo mais?",
    "Bom dia! Estou aqui para responder suas perguntas.",
    "Deixe-me explicar melhor.",
    "Essa é uma ótima pergunta.",
    "Vou processar isso agora mesmo.",
    "Obrigado por aguardar.",
    "Posso confirmar que sua requisição foi processada.",
    "Há mais alguma coisa que eu possa fazer por você?",
    "Entendo perfeitamente sua preocupação.",
    "Vou te ajudar com isso agora.",
    "Só um instante enquanto verifico.",
]


@dataclass
class SegmentResult:
    """Result from a single segment sent with detailed timing breakdown."""
    session_id: str
    segment_num: int
    timestamp: float  # When it was sent (relative to test start)
    text_sent_time: float = 0.0  # Absolute monotonic time when text was sent
    
    # Client-side measurement
    ttfa_from_text_ms: float = 0.0  # send_text → first audio received
    
    # Server-side timing breakdown (from segment_done)
    preprocess_ms: float = 0.0     # Text normalization
    queue_wait_ms: float = 0.0     # Waiting in inference queue
    inference_ms: float = 0.0      # GPU inference time
    postprocess_ms: float = 0.0    # Audio conversion
    send_first_chunk_ms: float = 0.0  # Time to send first chunk
    server_total_ms: float = 0.0   # Total server processing
    
    # Other metrics
    rtf: float = 0.0
    audio_duration_ms: float = 0.0
    success: bool = False
    error: str | None = None
    
    @property
    def network_overhead_ms(self) -> float:
        """Estimated network overhead (client TTFA - server total)."""
        if self.ttfa_from_text_ms > 0 and self.server_total_ms > 0:
            return max(0, self.ttfa_from_text_ms - self.server_total_ms)
        return 0.0


@dataclass
class SessionResult:
    """Aggregated results from a session."""
    session_id: str
    segments: list[SegmentResult] = field(default_factory=list)
    connection_error: str | None = None
    
    @property
    def segments_sent(self) -> int:
        return len(self.segments)
    
    @property
    def segments_success(self) -> int:
        return len([s for s in self.segments if s.success])
    
    @property
    def segments_failed(self) -> int:
        return len([s for s in self.segments if not s.success])


def percentile(data: list[float], p: int) -> float:
    """Calculate percentile of a list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


async def sender_loop(
    ws,
    session_id: str,
    duration_s: float,
    min_interval_ms: int,
    max_interval_ms: int,
    test_start_time: float,
    pending_segments: dict[int, SegmentResult],
    stop_event: asyncio.Event,
    wait_for_done: bool = True,
    done_event: asyncio.Event | None = None,
) -> None:
    """
    Sender task: sends text segments at intervals.
    
    Args:
        wait_for_done: If True, waits for segment_done before sending next segment.
                      This simulates realistic agent behavior.
        done_event: Event set by receiver when segment_done is received.
    """
    segment_num = 0
    session_start = time.monotonic()
    
    try:
        while not stop_event.is_set():
            # Check duration
            elapsed = time.monotonic() - session_start
            if elapsed >= duration_s:
                break
            
            segment_num += 1
            text = random.choice(TEST_SENTENCES)
            
            # Create pending segment
            segment_result = SegmentResult(
                session_id=session_id,
                segment_num=segment_num,
                timestamp=time.monotonic() - test_start_time,
                text_sent_time=time.monotonic(),
            )
            pending_segments[segment_num] = segment_result
            
            # Clear done_event before sending (if wait mode)
            if wait_for_done and done_event:
                done_event.clear()
            
            try:
                # Send text and flush
                await ws.send(json.dumps({"type": "send_text", "text": text}))
                await ws.send(json.dumps({"type": "flush"}))
            except Exception as e:
                segment_result.error = f"Send error: {e}"
                stop_event.set()
                break
            
            # If wait_for_done mode, wait for segment completion before next send
            if wait_for_done and done_event:
                remaining = duration_s - (time.monotonic() - session_start)
                try:
                    # Wait for segment_done with timeout
                    await asyncio.wait_for(
                        done_event.wait(),
                        timeout=max(0, min(remaining, 30.0))  # 30s max per segment
                    )
                except asyncio.TimeoutError:
                    # Segment took too long, continue anyway
                    logger.debug(f"{session_id}: Timeout waiting for segment {segment_num}")
            else:
                # No wait mode: just use interval
                interval_ms = random.randint(min_interval_ms, max_interval_ms)
                remaining = duration_s - (time.monotonic() - session_start)
                wait_time = min(interval_ms / 1000.0, max(0, remaining))
                
                if wait_time > 0:
                    try:
                        await asyncio.wait_for(
                            stop_event.wait(),
                            timeout=wait_time
                        )
                        break
                    except asyncio.TimeoutError:
                        pass
                    
    except asyncio.CancelledError:
        pass


async def receiver_loop(
    ws,
    session_id: str,
    pending_segments: dict[int, SegmentResult],
    completed_segments: list[SegmentResult],
    stop_event: asyncio.Event,
    done_event: asyncio.Event | None = None,
) -> None:
    """
    Receiver task: continuously reads from WebSocket.
    
    Correlates audio and segment_done with pending segments.
    Sets done_event when segment_done is received (for wait-for-done mode).
    Does NOT close connection on errors - just records and continues.
    """
    current_segment_num = 0
    first_audio_time: float | None = None
    
    try:
        async for msg in ws:
            if stop_event.is_set():
                break
                
            if isinstance(msg, bytes):
                # Audio chunk
                if first_audio_time is None:
                    first_audio_time = time.monotonic()
                    # Find the oldest pending segment
                    if pending_segments:
                        current_segment_num = min(pending_segments.keys())
                        
            else:
                # JSON message
                try:
                    data = json.loads(msg)
                except json.JSONDecodeError:
                    continue
                    
                msg_type = data.get("type")
                
                if msg_type == "segment_done":
                    # Complete the current segment
                    if current_segment_num in pending_segments:
                        seg = pending_segments.pop(current_segment_num)
                        
                        # Parse timing breakdown from server
                        timing = data.get("timing", {})
                        seg.preprocess_ms = timing.get("preprocess_ms", 0)
                        seg.queue_wait_ms = timing.get("queue_wait_ms", 0)
                        seg.inference_ms = timing.get("inference_ms", 0)
                        seg.postprocess_ms = timing.get("postprocess_ms", 0)
                        seg.send_first_chunk_ms = timing.get("send_first_chunk_ms", 0)
                        seg.server_total_ms = timing.get("server_total_ms", 0)
                        
                        # Other metrics
                        seg.rtf = data.get("rtf", 0)
                        seg.audio_duration_ms = data.get("audio_duration_ms", 0)
                        
                        # Calculate client-side TTFA
                        if first_audio_time and seg.text_sent_time:
                            seg.ttfa_from_text_ms = (first_audio_time - seg.text_sent_time) * 1000
                        
                        seg.success = True
                        completed_segments.append(seg)
                    
                    # Reset for next segment
                    first_audio_time = None
                    current_segment_num = 0
                    
                    # Signal sender that segment is done (for wait-for-done mode)
                    if done_event:
                        done_event.set()
                    
                elif msg_type == "error":
                    error_msg = data.get("message", "Unknown error")
                    error_code = data.get("code", "")
                    
                    # For admission control errors (QUEUE_FULL, QUEUE_CONGESTION),
                    # the error is for the NEWEST segment (the one just queued).
                    # For other errors, it's for the current segment being processed.
                    if error_code in ["QUEUE_FULL", "QUEUE_CONGESTION"]:
                        # Mark newest pending segment as failed
                        if pending_segments:
                            newest = max(pending_segments.keys())
                            seg = pending_segments.pop(newest)
                            seg.error = error_msg
                            seg.success = False
                            completed_segments.append(seg)
                        # DON'T reset first_audio_time - current segment is still streaming
                        # Signal done so sender can retry
                        if done_event:
                            done_event.set()
                    else:
                        # Other errors - mark current segment as failed
                        if current_segment_num in pending_segments:
                            seg = pending_segments.pop(current_segment_num)
                            seg.error = error_msg
                            seg.success = False
                            completed_segments.append(seg)
                        
                        # Reset for next segment
                        first_audio_time = None
                        current_segment_num = 0
                        
                        logger.debug(f"Session {session_id}: Error {error_code}: {error_msg}")
                        
                        # Signal done so sender can continue
                        if done_event:
                            done_event.set()
                    
                elif msg_type == "session_ended":
                    break
                    
    except websockets.exceptions.ConnectionClosed:
        pass
    except asyncio.CancelledError:
        pass
    finally:
        # Mark any remaining pending segments as failed
        for seg_num, seg in list(pending_segments.items()):
            seg.error = "Session ended before completion"
            seg.success = False
            completed_segments.append(seg)
        pending_segments.clear()
        
        # Signal done in case sender is waiting
        if done_event:
            done_event.set()


async def run_session(
    url: str,
    session_id: str,
    duration_s: float,
    min_interval_ms: int,
    max_interval_ms: int,
    test_start_time: float,
    ssl_context: ssl.SSLContext,
    wait_for_done: bool = True,
) -> SessionResult:
    """
    Run a single long-running TTS session with async sender/receiver.
    
    Architecture:
    - Sender task: sends text at intervals
    - Receiver task: reads responses
    - Shared pending_segments dict for correlation
    - Optional done_event for wait-for-done mode
    
    Args:
        wait_for_done: If True (default), sender waits for segment_done before
                      sending next segment. This is realistic agent behavior.
    """
    result = SessionResult(session_id=session_id)
    pending_segments: dict[int, SegmentResult] = {}
    completed_segments: list[SegmentResult] = []
    stop_event = asyncio.Event()
    done_event = asyncio.Event() if wait_for_done else None
    
    try:
        async with websockets.connect(
            url,
            ssl=ssl_context,
            close_timeout=30,
            ping_interval=20,
            ping_timeout=20,
        ) as ws:
            # Start session
            # Supertonic voices: F1-F5 (female), M1-M5 (male)
            await ws.send(json.dumps({
                "type": "start_session",
                "session_id": session_id,
                "config": {"voice": "F1"}
            }))
            
            resp = json.loads(await ws.recv())
            if resp.get("type") != "session_started":
                result.connection_error = f"Failed to start session: {resp}"
                return result
            
            # Create sender and receiver tasks
            sender_task = asyncio.create_task(
                sender_loop(
                    ws=ws,
                    session_id=session_id,
                    duration_s=duration_s,
                    min_interval_ms=min_interval_ms,
                    max_interval_ms=max_interval_ms,
                    test_start_time=test_start_time,
                    pending_segments=pending_segments,
                    stop_event=stop_event,
                    wait_for_done=wait_for_done,
                    done_event=done_event,
                )
            )
            
            receiver_task = asyncio.create_task(
                receiver_loop(
                    ws=ws,
                    session_id=session_id,
                    pending_segments=pending_segments,
                    completed_segments=completed_segments,
                    stop_event=stop_event,
                    done_event=done_event,
                )
            )
            
            # Wait for sender to complete (duration expired)
            await sender_task
            
            # Give receiver a moment to process remaining responses
            # Wait up to 5 seconds for pending segments to complete
            wait_start = time.monotonic()
            while pending_segments and (time.monotonic() - wait_start) < 5.0:
                await asyncio.sleep(0.1)
            
            # Signal receiver to stop
            stop_event.set()
            
            # End session gracefully
            try:
                await ws.send(json.dumps({"type": "end_session"}))
                # Give a moment for session_ended
                await asyncio.sleep(0.5)
            except Exception:
                pass
            
            # Cancel receiver
            receiver_task.cancel()
            try:
                await receiver_task
            except asyncio.CancelledError:
                pass
                
    except websockets.exceptions.ConnectionClosed as e:
        result.connection_error = f"Connection closed: {e}"
    except Exception as e:
        result.connection_error = str(e)
    
    # Collect results
    result.segments = completed_segments
    return result


async def run_test(
    url: str,
    num_sessions: int,
    duration_s: float,
    min_interval_ms: int,
    max_interval_ms: int,
    wait_for_done: bool = True,
) -> list[SessionResult]:
    """Run the load test with multiple concurrent sessions."""
    
    # SSL context
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    test_start_time = time.monotonic()
    
    # Create session tasks
    tasks = []
    for i in range(num_sessions):
        session_id = f"lk-sim-{uuid4().hex[:8]}"
        task = run_session(
            url=url,
            session_id=session_id,
            duration_s=duration_s,
            min_interval_ms=min_interval_ms,
            max_interval_ms=max_interval_ms,
            test_start_time=test_start_time,
            ssl_context=ssl_context,
            wait_for_done=wait_for_done,
        )
        tasks.append(task)
    
    # Progress indicator
    progress_task = asyncio.create_task(show_progress(duration_s))
    
    # Run all sessions concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Stop progress
    progress_task.cancel()
    try:
        await progress_task
    except asyncio.CancelledError:
        pass
    
    # Process results
    session_results = []
    for result in results:
        if isinstance(result, BaseException):
            session_results.append(SessionResult(
                session_id="error",
                connection_error=str(result)
            ))
        elif isinstance(result, SessionResult):
            session_results.append(result)
    
    return session_results


async def show_progress(duration_s: float) -> None:
    """Show a progress bar during the test."""
    start = time.monotonic()
    bar_width = 40
    
    while True:
        elapsed = time.monotonic() - start
        if elapsed >= duration_s:
            break
        
        progress = min(elapsed / duration_s, 1.0)
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        sys.stdout.write(f"\rProgresso: [{bar}] {int(elapsed)}/{int(duration_s)}s")
        sys.stdout.flush()
        
        await asyncio.sleep(0.5)
    
    # Final update
    bar = "█" * bar_width
    sys.stdout.write(f"\rProgresso: [{bar}] {int(duration_s)}/{int(duration_s)}s\n")
    sys.stdout.flush()


def print_summary(
    results: list[SessionResult],
    num_sessions: int,
    duration_s: float,
    min_interval_ms: int,
    max_interval_ms: int,
) -> None:
    """Print formatted test summary."""
    
    print(f"\n{'='*80}")
    print("RESUMO")
    print(f"{'='*80}\n")
    
    # Collect all segments
    all_segments = []
    for session in results:
        all_segments.extend(session.segments)
    
    successful_segments = [s for s in all_segments if s.success]
    failed_segments = [s for s in all_segments if not s.success]
    
    # Connection errors
    connection_errors = [r for r in results if r.connection_error]
    
    # Basic stats
    total_sent = len(all_segments)
    total_success = len(successful_segments)
    success_rate = (total_success / total_sent * 100) if total_sent > 0 else 0
    
    print(f"Sessões: {num_sessions} | Duração: {duration_s}s | Intervalo: {min_interval_ms}-{max_interval_ms}ms")
    print(f"Segmentos: {total_sent} enviados, {total_success} sucesso ({success_rate:.1f}%)")
    
    if failed_segments:
        # Count error types
        error_counts: dict[str, int] = {}
        for seg in failed_segments:
            error = seg.error or "Unknown"
            # Simplify error message
            if "Queue congested" in error:
                error = "Queue congested (admission control)"
            elif "Queue full" in error:
                error = "Queue full (admission control)"
            elif "estimated_server_total" in error:
                error = "Server SLO exceeded (admission control)"
            error_counts[error] = error_counts.get(error, 0) + 1
        
        print(f"Falhas: {len(failed_segments)}")
        for error, count in sorted(error_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"  - {error}: {count}")
    
    if connection_errors:
        print(f"Erros de conexão: {len(connection_errors)}")
        for r in connection_errors[:3]:
            print(f"  - {r.session_id}: {r.connection_error}")
    
    if not successful_segments:
        print("\nNenhum segmento bem sucedido para calcular métricas.")
        return
    
    # ==========================================
    # DETAILED TIMING BREAKDOWN (AVERAGES)
    # ==========================================
    print(f"\n{'─'*60}")
    print("BREAKDOWN DE LATÊNCIA (médias)")
    print(f"{'─'*60}")
    
    # Collect all timing values
    preprocess_vals = [s.preprocess_ms for s in successful_segments if s.preprocess_ms > 0]
    queue_vals = [s.queue_wait_ms for s in successful_segments if s.queue_wait_ms > 0]
    inference_vals = [s.inference_ms for s in successful_segments if s.inference_ms > 0]
    postprocess_vals = [s.postprocess_ms for s in successful_segments if s.postprocess_ms > 0]
    send_vals = [s.send_first_chunk_ms for s in successful_segments if s.send_first_chunk_ms > 0]
    server_total_vals = [s.server_total_ms for s in successful_segments if s.server_total_ms > 0]
    network_vals = [s.network_overhead_ms for s in successful_segments if s.network_overhead_ms > 0]
    
    avg_preprocess = statistics.mean(preprocess_vals) if preprocess_vals else 0
    avg_queue = statistics.mean(queue_vals) if queue_vals else 0
    avg_inference = statistics.mean(inference_vals) if inference_vals else 0
    avg_postprocess = statistics.mean(postprocess_vals) if postprocess_vals else 0
    avg_send = statistics.mean(send_vals) if send_vals else 0
    avg_server_total = statistics.mean(server_total_vals) if server_total_vals else 0
    avg_network = statistics.mean(network_vals) if network_vals else 0
    
    print(f"  preprocess:        {avg_preprocess:>7.1f} ms  (text normalization)")
    print(f"  queue_wait:        {avg_queue:>7.1f} ms  (waiting in queue)")
    print(f"  inference:         {avg_inference:>7.1f} ms  (GPU)")
    print(f"  postprocess:       {avg_postprocess:>7.1f} ms  (audio conversion)")
    print(f"  send_first_chunk:  {avg_send:>7.1f} ms  (WebSocket)")
    print(f"  {'─'*28}")
    print(f"  server_total:      {avg_server_total:>7.1f} ms")
    print(f"  network_overhead:  {avg_network:>7.1f} ms  (estimado)")
    
    # ==========================================
    # SERVER-ONLY SLO CHECK
    # ==========================================
    if server_total_vals:
        print(f"\n{'─'*60}")
        print("SERVER-ONLY SLO (o que controlamos)")
        print(f"{'─'*60}")
        st_p95 = percentile(server_total_vals, 95)
        st_p99 = percentile(server_total_vals, 99)
        print(f"  server_total p50:  {percentile(server_total_vals, 50):>6.1f} ms")
        print(f"  server_total p95:  {st_p95:>6.1f} ms  {'✓ OK' if st_p95 < 250 else '✗ SLO VIOLADO (target: 250ms)'}")
        print(f"  server_total p99:  {st_p99:>6.1f} ms")
    
    # ==========================================
    # TTFA FROM TEXT (includes network)
    # ==========================================
    ttfas = [s.ttfa_from_text_ms for s in successful_segments if s.ttfa_from_text_ms > 0]
    if ttfas:
        print(f"\n{'─'*60}")
        print("TTFA from Text (inclui network - apenas referência)")
        print(f"{'─'*60}")
        print(f"  p50:   {percentile(ttfas, 50):>7.1f} ms")
        print(f"  p95:   {percentile(ttfas, 95):>7.1f} ms")
        print(f"  p99:   {percentile(ttfas, 99):>7.1f} ms")
        print(f"  max:   {max(ttfas):>7.1f} ms")
    
    # ==========================================
    # SERVER-SIDE DETAILED PERCENTILES
    # ==========================================
    print(f"\n{'─'*60}")
    print("PERCENTIS POR COMPONENTE")
    print(f"{'─'*60}")
    
    if queue_vals:
        print(f"\n  Queue Wait:")
        print(f"    p50: {percentile(queue_vals, 50):>6.1f} ms  |  p95: {percentile(queue_vals, 95):>6.1f} ms  |  p99: {percentile(queue_vals, 99):>6.1f} ms")
    
    if inference_vals:
        print(f"\n  Inference:")
        print(f"    p50: {percentile(inference_vals, 50):>6.1f} ms  |  p95: {percentile(inference_vals, 95):>6.1f} ms  |  p99: {percentile(inference_vals, 99):>6.1f} ms")
    
    if server_total_vals:
        print(f"\n  Server Total:")
        print(f"    p50: {percentile(server_total_vals, 50):>6.1f} ms  |  p95: {percentile(server_total_vals, 95):>6.1f} ms  |  p99: {percentile(server_total_vals, 99):>6.1f} ms")
    
    if network_vals:
        print(f"\n  Network Overhead:")
        print(f"    p50: {percentile(network_vals, 50):>6.1f} ms  |  p95: {percentile(network_vals, 95):>6.1f} ms  |  p99: {percentile(network_vals, 99):>6.1f} ms")
    
    # RTF
    rtfs = [s.rtf for s in successful_segments if s.rtf > 0]
    if rtfs:
        print(f"\n{'─'*60}")
        print("RTF (Real-Time Factor)")
        print(f"{'─'*60}")
        print(f"  média: {statistics.mean(rtfs):>7.3f}  {'✓ OK' if statistics.mean(rtfs) < 0.8 else '✗ ALTO'}")
        print(f"  p95:   {percentile(rtfs, 95):>7.3f}")
    
    # ==========================================
    # TIMELINE ANALYSIS
    # ==========================================
    if successful_segments:
        print(f"\n{'─'*60}")
        print("TIMELINE (médias a cada 10s)")
        print(f"{'─'*60}")
        
        bucket_size = 10.0
        max_timestamp = max(s.timestamp for s in successful_segments)
        num_buckets = int(max_timestamp / bucket_size) + 1
        
        for i in range(min(num_buckets, 12)):  # Limit to 12 buckets (2 min)
            start_t = i * bucket_size
            end_t = (i + 1) * bucket_size
            
            bucket_segments = [
                s for s in successful_segments
                if start_t <= s.timestamp < end_t
            ]
            
            if bucket_segments:
                b_ttfa = statistics.mean([s.ttfa_from_text_ms for s in bucket_segments if s.ttfa_from_text_ms > 0] or [0])
                b_queue = statistics.mean([s.queue_wait_ms for s in bucket_segments if s.queue_wait_ms > 0] or [0])
                b_inf = statistics.mean([s.inference_ms for s in bucket_segments if s.inference_ms > 0] or [0])
                b_server = statistics.mean([s.server_total_ms for s in bucket_segments if s.server_total_ms > 0] or [0])
                
                print(f"  {int(start_t):>3}-{int(end_t):<3}s: server={b_server:>4.0f}ms  queue={b_queue:>4.0f}ms  inf={b_inf:>4.0f}ms  ttfa={b_ttfa:>4.0f}ms  ({len(bucket_segments)} segs)")
    
    # Throughput
    total_audio_ms = sum(s.audio_duration_ms for s in successful_segments)
    print(f"\nThroughput:")
    print(f"  Total áudio gerado: {total_audio_ms / 1000:.1f}s")
    print(f"  Segmentos/segundo:  {total_success / duration_s:.2f}")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="LiveKit simulation load test for Kokoro TTS"
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=3,
        help="Number of concurrent sessions (default: 3)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Test duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--min-interval",
        type=int,
        default=800,
        help="Minimum interval between segments in ms (default: 800, only used with --no-wait-for-done)",
    )
    parser.add_argument(
        "--max-interval",
        type=int,
        default=1500,
        help="Maximum interval between segments in ms (default: 1500, only used with --no-wait-for-done)",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="WebSocket URL of the TTS service",
    )
    parser.add_argument(
        "--wait-for-done",
        dest="wait_for_done",
        action="store_true",
        default=True,
        help="Wait for segment_done before sending next (default: enabled, realistic behavior)",
    )
    parser.add_argument(
        "--no-wait-for-done",
        dest="wait_for_done",
        action="store_false",
        help="Don't wait for segment_done, fire at intervals (stress test mode)",
    )
    
    args = parser.parse_args()
    
    # Validate
    if args.sessions < 1 or args.sessions > 100:
        print("Error: --sessions must be between 1 and 100")
        return
    if args.duration < 10 or args.duration > 300:
        print("Error: --duration must be between 10 and 300 seconds")
        return
    if args.min_interval < 100 or args.max_interval > 5000:
        print("Error: intervals must be between 100 and 5000 ms")
        return
    if args.min_interval > args.max_interval:
        print("Error: --min-interval must be <= --max-interval")
        return
    
    mode = "Wait-for-Done (realistic)" if args.wait_for_done else f"Fire-at-Interval ({args.min_interval}-{args.max_interval}ms)"
    
    print(f"\n{'='*80}")
    print("LiveKit Simulation Load Test")
    print(f"  Sessões: {args.sessions}  |  Duração: {args.duration}s  |  Modo: {mode}")
    print(f"  URL: {args.url}")
    print(f"{'='*80}\n")
    
    # Run test
    results = asyncio.run(run_test(
        url=args.url,
        num_sessions=args.sessions,
        duration_s=args.duration,
        min_interval_ms=args.min_interval,
        max_interval_ms=args.max_interval,
        wait_for_done=args.wait_for_done,
    ))
    
    # Print summary
    print_summary(
        results=results,
        num_sessions=args.sessions,
        duration_s=args.duration,
        min_interval_ms=args.min_interval,
        max_interval_ms=args.max_interval,
    )


if __name__ == "__main__":
    main()
