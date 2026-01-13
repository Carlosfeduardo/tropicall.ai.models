"""
LiveKit Simulation Load Test for Whisper STT Service.

Simulates realistic LiveKit agent behavior:
- Long-running sessions (configurable duration)
- Continuous audio streaming with speech patterns
- Multiple concurrent sessions (multiple active talkers)
- VAD triggering and transcript generation

Modes:
- Speech patterns: Simulates realistic talk/pause cycles
- Continuous: Streams audio continuously without pauses

Usage:
    python livekit_load_test.py --sessions 3 --duration 60
    python livekit_load_test.py --sessions 10 --duration 120 --continuous
"""

import argparse
import asyncio
import json
import logging
import math
import random
import ssl
import statistics
import struct
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import numpy as np
import websockets
from scipy.signal import resample

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Default URL (update with your RunPod endpoint)
DEFAULT_URL = "wss://aujk9ttkh71fbe-8000.proxy.runpod.net/ws/stt"

# Audio parameters
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 20  # 20ms chunks (standard for real-time audio)
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # 320 samples
CHUNK_BYTES = CHUNK_SAMPLES * 2  # 640 bytes (16-bit PCM)


@dataclass
class TranscriptResult:
    """Result from a transcript (partial or final)."""
    session_id: str
    segment_id: int
    is_final: bool
    text: str
    timestamp: float  # Relative to test start
    
    # Timing breakdown from server
    preprocess_ms: float = 0.0
    queue_wait_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    server_total_ms: float = 0.0
    
    # Audio info
    audio_duration_ms: float = 0.0
    
    # Client tracking
    speech_start_time: float = 0.0  # When VAD speech_start was received
    transcript_received_time: float = 0.0  # When this transcript was received
    
    @property
    def client_latency_ms(self) -> float:
        """Time from speech_start to transcript received."""
        if self.speech_start_time and self.transcript_received_time:
            return (self.transcript_received_time - self.speech_start_time) * 1000
        return 0.0


@dataclass
class SessionResult:
    """Aggregated results from a session."""
    session_id: str
    partials: list[TranscriptResult] = field(default_factory=list)
    finals: list[TranscriptResult] = field(default_factory=list)
    vad_events: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    connection_error: str | None = None
    
    total_audio_sent_ms: float = 0.0
    total_speech_segments: int = 0
    
    @property
    def total_transcripts(self) -> int:
        return len(self.partials) + len(self.finals)


def generate_silence(duration_ms: int) -> bytes:
    """Generate silence (zeros) for given duration."""
    num_samples = int(SAMPLE_RATE * duration_ms / 1000)
    return b'\x00' * (num_samples * 2)


def generate_tone(duration_ms: int, frequency: float = 440.0, amplitude: float = 0.3) -> bytes:
    """Generate a sine wave tone (simulates speech-like audio for VAD triggering)."""
    num_samples = int(SAMPLE_RATE * duration_ms / 1000)
    samples = []
    
    for i in range(num_samples):
        # Add some variation to make it more speech-like
        t = i / SAMPLE_RATE
        # Mix multiple frequencies for richer signal
        value = amplitude * (
            0.5 * math.sin(2 * math.pi * frequency * t) +
            0.3 * math.sin(2 * math.pi * (frequency * 1.5) * t) +
            0.2 * math.sin(2 * math.pi * (frequency * 2) * t)
        )
        # Add slight noise
        value += random.uniform(-0.05, 0.05) * amplitude
        # Clamp and convert to 16-bit
        value = max(-1.0, min(1.0, value))
        sample = int(value * 32767)
        samples.append(struct.pack('<h', sample))
    
    return b''.join(samples)


def resample_audio(audio_data: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    if original_rate == target_rate:
        return audio_data
    
    # Calculate the number of samples in the resampled audio
    num_samples = int(len(audio_data) * target_rate / original_rate)
    
    # Resample using scipy
    resampled = resample(audio_data, num_samples)
    
    return resampled.astype(np.int16)


def load_test_audio(wav_path: str | None) -> bytes | None:
    """Load a WAV file for testing, or return None to use synthetic audio.
    
    Automatically resamples to 16kHz if needed.
    """
    if not wav_path or not Path(wav_path).exists():
        return None
    
    try:
        with wave.open(wav_path, 'rb') as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                logger.warning(f"WAV file must be mono 16-bit, using synthetic audio")
                return None
            
            original_rate = wf.getframerate()
            audio_bytes = wf.readframes(wf.getnframes())
            
            # Convert to numpy array
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Resample if needed
            if original_rate != SAMPLE_RATE:
                print(f"⚠️  Resampling audio from {original_rate}Hz to {SAMPLE_RATE}Hz...")
                audio_data = resample_audio(audio_data, original_rate, SAMPLE_RATE)
                print(f"✓ Resampled: {len(audio_bytes)//2} → {len(audio_data)} samples")
            
            return audio_data.tobytes()
    except Exception as e:
        logger.warning(f"Failed to load WAV: {e}")
        return None


async def audio_sender_loop(
    ws,
    session_id: str,
    duration_s: float,
    test_start_time: float,
    stop_event: asyncio.Event,
    continuous: bool = False,
    test_audio: bytes | None = None,
    session_result: SessionResult | None = None,
) -> None:
    """
    Audio sender task: streams audio chunks in real-time.
    
    Args:
        continuous: If True, streams audio continuously.
                   If False, simulates talk/pause patterns.
        test_audio: Optional real audio to stream (loops if needed).
    """
    session_start = time.monotonic()
    audio_sent_ms = 0.0
    speech_segments = 0
    
    # Speech pattern parameters (when not continuous)
    min_speech_ms = 1500  # 1.5s minimum speech
    max_speech_ms = 5000  # 5s maximum speech
    min_pause_ms = 500    # 0.5s minimum pause
    max_pause_ms = 2000   # 2s maximum pause
    
    try:
        while not stop_event.is_set():
            elapsed = time.monotonic() - session_start
            if elapsed >= duration_s:
                break
            
            if continuous:
                # Continuous mode: stream audio non-stop
                speech_duration_ms = int((duration_s - elapsed) * 1000)
            else:
                # Speech pattern mode: talk, then pause
                speech_duration_ms = random.randint(min_speech_ms, max_speech_ms)
                remaining_ms = int((duration_s - elapsed) * 1000)
                speech_duration_ms = min(speech_duration_ms, remaining_ms)
            
            if speech_duration_ms <= 0:
                break
            
            # Generate or get audio for this speech segment
            speech_segments += 1
            
            if test_audio:
                # Use real audio (loop if needed)
                audio_data = test_audio
                # Trim to speech duration
                needed_bytes = int(SAMPLE_RATE * speech_duration_ms / 1000) * 2
                if len(audio_data) < needed_bytes:
                    # Loop the audio
                    repeats = (needed_bytes // len(audio_data)) + 1
                    audio_data = audio_data * repeats
                audio_data = audio_data[:needed_bytes]
            else:
                # Generate synthetic speech-like audio
                frequency = random.uniform(200, 400)  # Vary frequency
                audio_data = generate_tone(speech_duration_ms, frequency=frequency)
            
            # Stream in chunks (real-time)
            chunk_count = len(audio_data) // CHUNK_BYTES
            
            for i in range(chunk_count):
                if stop_event.is_set():
                    break
                
                chunk = audio_data[i * CHUNK_BYTES:(i + 1) * CHUNK_BYTES]
                try:
                    await ws.send(chunk)
                    audio_sent_ms += CHUNK_DURATION_MS
                except Exception as e:
                    logger.debug(f"{session_id}: Send error: {e}")
                    stop_event.set()
                    break
                
                # Real-time pacing
                await asyncio.sleep(CHUNK_DURATION_MS / 1000)
            
            if not continuous and not stop_event.is_set():
                # Pause between speech segments
                pause_ms = random.randint(min_pause_ms, max_pause_ms)
                remaining = duration_s - (time.monotonic() - session_start)
                pause_s = min(pause_ms / 1000, max(0, remaining))
                
                if pause_s > 0:
                    # Send silence during pause
                    silence_chunks = int(pause_s * 1000 / CHUNK_DURATION_MS)
                    silence_chunk = generate_silence(CHUNK_DURATION_MS)
                    
                    for _ in range(silence_chunks):
                        if stop_event.is_set():
                            break
                        try:
                            await ws.send(silence_chunk)
                            audio_sent_ms += CHUNK_DURATION_MS
                        except Exception:
                            break
                        await asyncio.sleep(CHUNK_DURATION_MS / 1000)
    
    except asyncio.CancelledError:
        pass
    finally:
        if session_result:
            session_result.total_audio_sent_ms = audio_sent_ms
            session_result.total_speech_segments = speech_segments


async def receiver_loop(
    ws,
    session_id: str,
    test_start_time: float,
    session_result: SessionResult,
    stop_event: asyncio.Event,
) -> None:
    """
    Receiver task: continuously reads transcripts and events from WebSocket.
    """
    current_speech_start: float | None = None
    
    try:
        async for msg in ws:
            if stop_event.is_set():
                break
            
            if isinstance(msg, bytes):
                # Shouldn't receive binary from STT, but ignore if we do
                continue
            
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue
            
            msg_type = data.get("type")
            now = time.monotonic()
            
            if msg_type == "vad_event":
                state = data.get("state")
                session_result.vad_events.append({
                    "state": state,
                    "timestamp": now - test_start_time,
                    "t_ms": data.get("t_ms", 0)
                })
                
                if state == "speech_start":
                    current_speech_start = now
                elif state == "speech_end":
                    current_speech_start = None
            
            elif msg_type == "partial_transcript":
                timing = data.get("timing", {})
                result = TranscriptResult(
                    session_id=session_id,
                    segment_id=data.get("segment_id", 0),
                    is_final=False,
                    text=data.get("text", ""),
                    timestamp=now - test_start_time,
                    preprocess_ms=timing.get("preprocess_ms", 0),
                    queue_wait_ms=timing.get("queue_wait_ms", 0),
                    inference_ms=timing.get("inference_ms", 0),
                    postprocess_ms=timing.get("postprocess_ms", 0),
                    server_total_ms=timing.get("server_total_ms", 0),
                    audio_duration_ms=data.get("audio_duration_ms", 0),
                    speech_start_time=current_speech_start or 0,
                    transcript_received_time=now,
                )
                session_result.partials.append(result)
            
            elif msg_type == "final_transcript":
                timing = data.get("timing", {})
                result = TranscriptResult(
                    session_id=session_id,
                    segment_id=data.get("segment_id", 0),
                    is_final=True,
                    text=data.get("text", ""),
                    timestamp=now - test_start_time,
                    preprocess_ms=timing.get("preprocess_ms", 0),
                    queue_wait_ms=timing.get("queue_wait_ms", 0),
                    inference_ms=timing.get("inference_ms", 0),
                    postprocess_ms=timing.get("postprocess_ms", 0),
                    server_total_ms=timing.get("server_total_ms", 0),
                    audio_duration_ms=data.get("audio_duration_ms", 0),
                    speech_start_time=current_speech_start or 0,
                    transcript_received_time=now,
                )
                session_result.finals.append(result)
                current_speech_start = None  # Reset for next segment
            
            elif msg_type == "error":
                error_msg = f"{data.get('code', 'UNKNOWN')}: {data.get('message', '')}"
                session_result.errors.append(error_msg)
                logger.debug(f"{session_id}: Error: {error_msg}")
            
            elif msg_type == "session_ended":
                break
    
    except websockets.exceptions.ConnectionClosed:
        pass
    except asyncio.CancelledError:
        pass


async def run_session(
    url: str,
    session_id: str,
    duration_s: float,
    test_start_time: float,
    ssl_context: ssl.SSLContext,
    continuous: bool = False,
    test_audio: bytes | None = None,
) -> SessionResult:
    """
    Run a single STT session with async sender/receiver.
    """
    result = SessionResult(session_id=session_id)
    stop_event = asyncio.Event()
    
    try:
        async with websockets.connect(
            url,
            ssl=ssl_context,
            close_timeout=30,
            ping_interval=20,
            ping_timeout=20,
            max_size=10 * 1024 * 1024,  # 10MB max message
        ) as ws:
            # Start session
            await ws.send(json.dumps({
                "type": "start_session",
                "session_id": session_id,
                "config": {
                    "lang_code": "pt",
                    "sample_rate": SAMPLE_RATE,
                    "vad_enabled": True,
                    "partial_results": True,
                    "word_timestamps": True,
                }
            }))
            
            resp = json.loads(await ws.recv())
            if resp.get("type") != "session_started":
                result.connection_error = f"Failed to start session: {resp}"
                return result
            
            # Create sender and receiver tasks
            sender_task = asyncio.create_task(
                audio_sender_loop(
                    ws=ws,
                    session_id=session_id,
                    duration_s=duration_s,
                    test_start_time=test_start_time,
                    stop_event=stop_event,
                    continuous=continuous,
                    test_audio=test_audio,
                    session_result=result,
                )
            )
            
            receiver_task = asyncio.create_task(
                receiver_loop(
                    ws=ws,
                    session_id=session_id,
                    test_start_time=test_start_time,
                    session_result=result,
                    stop_event=stop_event,
                )
            )
            
            # Wait for sender to complete (duration expired)
            await sender_task
            
            # Send flush and wait for final transcripts
            try:
                await ws.send(json.dumps({"type": "flush"}))
                await asyncio.sleep(2.0)  # Wait for final processing
            except Exception:
                pass
            
            # Signal receiver to stop
            stop_event.set()
            
            # End session gracefully
            try:
                await ws.send(json.dumps({"type": "end_session"}))
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
    
    return result


async def run_test(
    url: str,
    num_sessions: int,
    duration_s: float,
    continuous: bool = False,
    test_audio: bytes | None = None,
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
        session_id = f"stt-load-{uuid4().hex[:8]}"
        task = run_session(
            url=url,
            session_id=session_id,
            duration_s=duration_s,
            test_start_time=test_start_time,
            ssl_context=ssl_context,
            continuous=continuous,
            test_audio=test_audio,
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


def percentile(data: list[float], p: int) -> float:
    """Calculate percentile of a list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def print_summary(
    results: list[SessionResult],
    num_sessions: int,
    duration_s: float,
    continuous: bool,
) -> None:
    """Print formatted test summary."""
    
    print(f"\n{'='*80}")
    print("RESUMO DO TESTE STT")
    print(f"{'='*80}\n")
    
    # Collect all transcripts
    all_partials = []
    all_finals = []
    all_vad_events = []
    all_errors = []
    
    for session in results:
        all_partials.extend(session.partials)
        all_finals.extend(session.finals)
        all_vad_events.extend(session.vad_events)
        all_errors.extend(session.errors)
    
    # Connection errors
    connection_errors = [r for r in results if r.connection_error]
    
    # Basic stats
    total_audio_ms = sum(r.total_audio_sent_ms for r in results)
    total_speech_segments = sum(r.total_speech_segments for r in results)
    
    mode = "Contínuo" if continuous else "Padrão de fala (talk/pause)"
    print(f"Sessões: {num_sessions} | Duração: {duration_s}s | Modo: {mode}")
    print(f"Áudio total enviado: {total_audio_ms / 1000:.1f}s")
    print(f"Segmentos de fala: {total_speech_segments}")
    print(f"VAD events: {len(all_vad_events)} (speech_start: {len([e for e in all_vad_events if e['state'] == 'speech_start'])})")
    print(f"Partials recebidos: {len(all_partials)}")
    print(f"Finals recebidos: {len(all_finals)}")
    
    if all_errors:
        error_counts: dict[str, int] = {}
        for error in all_errors:
            key = error.split(":")[0] if ":" in error else error[:50]
            error_counts[key] = error_counts.get(key, 0) + 1
        print(f"\nErros: {len(all_errors)}")
        for error, count in sorted(error_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"  - {error}: {count}")
    
    if connection_errors:
        print(f"\nErros de conexão: {len(connection_errors)}")
        for r in connection_errors[:3]:
            print(f"  - {r.session_id}: {r.connection_error}")
    
    # ==========================================
    # PARTIAL TRANSCRIPT LATENCY
    # ==========================================
    if all_partials:
        print(f"\n{'─'*60}")
        print("LATÊNCIA - PARTIAL TRANSCRIPTS")
        print(f"{'─'*60}")
        
        server_totals = [p.server_total_ms for p in all_partials if p.server_total_ms > 0]
        inference_times = [p.inference_ms for p in all_partials if p.inference_ms > 0]
        queue_times = [p.queue_wait_ms for p in all_partials if p.queue_wait_ms > 0]
        
        if server_totals:
            print(f"\n  Server Total (o que controlamos):")
            st_p50 = percentile(server_totals, 50)
            st_p95 = percentile(server_totals, 95)
            st_p99 = percentile(server_totals, 99)
            print(f"    média: {statistics.mean(server_totals):>6.1f} ms")
            print(f"    p50:   {st_p50:>6.1f} ms")
            print(f"    p95:   {st_p95:>6.1f} ms  {'✓ OK' if st_p95 < 250 else '✗ SLO VIOLADO (target: 250ms)'}")
            print(f"    p99:   {st_p99:>6.1f} ms")
            print(f"    max:   {max(server_totals):>6.1f} ms")
        
        if inference_times:
            print(f"\n  Inference (GPU):")
            print(f"    média: {statistics.mean(inference_times):>6.1f} ms")
            print(f"    p95:   {percentile(inference_times, 95):>6.1f} ms")
        
        if queue_times:
            print(f"\n  Queue Wait:")
            print(f"    média: {statistics.mean(queue_times):>6.1f} ms")
            print(f"    p95:   {percentile(queue_times, 95):>6.1f} ms")
    
    # ==========================================
    # FINAL TRANSCRIPT LATENCY
    # ==========================================
    if all_finals:
        print(f"\n{'─'*60}")
        print("LATÊNCIA - FINAL TRANSCRIPTS")
        print(f"{'─'*60}")
        
        server_totals = [f.server_total_ms for f in all_finals if f.server_total_ms > 0]
        inference_times = [f.inference_ms for f in all_finals if f.inference_ms > 0]
        audio_durations = [f.audio_duration_ms for f in all_finals if f.audio_duration_ms > 0]
        
        if server_totals:
            print(f"\n  Server Total:")
            st_p50 = percentile(server_totals, 50)
            st_p95 = percentile(server_totals, 95)
            st_p99 = percentile(server_totals, 99)
            print(f"    média: {statistics.mean(server_totals):>6.1f} ms")
            print(f"    p50:   {st_p50:>6.1f} ms")
            print(f"    p95:   {st_p95:>6.1f} ms  {'✓ OK' if st_p95 < 1000 else '✗ SLO VIOLADO (target: 1000ms)'}")
            print(f"    p99:   {st_p99:>6.1f} ms")
            print(f"    max:   {max(server_totals):>6.1f} ms")
        
        if audio_durations:
            print(f"\n  Duração do áudio transcrito:")
            print(f"    média: {statistics.mean(audio_durations):>6.0f} ms")
            print(f"    max:   {max(audio_durations):>6.0f} ms")
        
        # RTF (Real-Time Factor) for finals
        if server_totals and audio_durations:
            rtfs = [s / a for s, a in zip(server_totals, audio_durations) if a > 0]
            if rtfs:
                print(f"\n  RTF (tempo_processamento / duração_áudio):")
                avg_rtf = statistics.mean(rtfs)
                print(f"    média: {avg_rtf:>6.3f}  {'✓ OK' if avg_rtf < 0.5 else '✗ ALTO'}")
                print(f"    p95:   {percentile(rtfs, 95):>6.3f}")
    
    # ==========================================
    # BREAKDOWN POR COMPONENTE
    # ==========================================
    all_transcripts = all_partials + all_finals
    if all_transcripts:
        print(f"\n{'─'*60}")
        print("BREAKDOWN DE LATÊNCIA (médias)")
        print(f"{'─'*60}")
        
        preprocess = [t.preprocess_ms for t in all_transcripts if t.preprocess_ms > 0]
        queue = [t.queue_wait_ms for t in all_transcripts if t.queue_wait_ms > 0]
        inference = [t.inference_ms for t in all_transcripts if t.inference_ms > 0]
        postprocess = [t.postprocess_ms for t in all_transcripts if t.postprocess_ms > 0]
        server_total = [t.server_total_ms for t in all_transcripts if t.server_total_ms > 0]
        
        if preprocess:
            print(f"  preprocess:    {statistics.mean(preprocess):>7.1f} ms  (VAD + audio prep)")
        if queue:
            print(f"  queue_wait:    {statistics.mean(queue):>7.1f} ms  (waiting in queue)")
        if inference:
            print(f"  inference:     {statistics.mean(inference):>7.1f} ms  (Whisper GPU)")
        if postprocess:
            print(f"  postprocess:   {statistics.mean(postprocess):>7.1f} ms  (text formatting)")
        if server_total:
            print(f"  {'─'*28}")
            print(f"  server_total:  {statistics.mean(server_total):>7.1f} ms")
    
    # ==========================================
    # TIMELINE ANALYSIS
    # ==========================================
    if all_transcripts:
        print(f"\n{'─'*60}")
        print("TIMELINE (médias a cada 10s)")
        print(f"{'─'*60}")
        
        bucket_size = 10.0
        max_timestamp = max(t.timestamp for t in all_transcripts)
        num_buckets = int(max_timestamp / bucket_size) + 1
        
        for i in range(min(num_buckets, 12)):  # Limit to 12 buckets
            start_t = i * bucket_size
            end_t = (i + 1) * bucket_size
            
            bucket = [t for t in all_transcripts if start_t <= t.timestamp < end_t]
            
            if bucket:
                b_partials = len([t for t in bucket if not t.is_final])
                b_finals = len([t for t in bucket if t.is_final])
                b_server = statistics.mean([t.server_total_ms for t in bucket if t.server_total_ms > 0] or [0])
                b_queue = statistics.mean([t.queue_wait_ms for t in bucket if t.queue_wait_ms > 0] or [0])
                
                print(f"  {int(start_t):>3}-{int(end_t):<3}s: server={b_server:>4.0f}ms  queue={b_queue:>4.0f}ms  partials={b_partials}  finals={b_finals}")
    
    # ==========================================
    # PER-SESSION SUMMARY
    # ==========================================
    print(f"\n{'─'*60}")
    print("POR SESSÃO")
    print(f"{'─'*60}")
    
    for i, session in enumerate(results[:10]):  # Limit to first 10
        if session.connection_error:
            print(f"  {session.session_id}: ❌ {session.connection_error}")
        else:
            print(f"  {session.session_id}: audio={session.total_audio_sent_ms/1000:.1f}s  partials={len(session.partials)}  finals={len(session.finals)}  errors={len(session.errors)}")
    
    if len(results) > 10:
        print(f"  ... e mais {len(results) - 10} sessões")
    
    # Throughput
    total_finals = len(all_finals)
    print(f"\nThroughput:")
    print(f"  Finals/segundo: {total_finals / duration_s:.2f}")
    print(f"  Partials/segundo: {len(all_partials) / duration_s:.2f}")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="LiveKit simulation load test for Whisper STT"
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
        "--url",
        default=DEFAULT_URL,
        help="WebSocket URL of the STT service",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Stream audio continuously (no pauses between speech)",
    )
    parser.add_argument(
        "--wav",
        type=str,
        default=None,
        help="Optional WAV file to use as test audio (mono 16kHz 16-bit)",
    )
    
    args = parser.parse_args()
    
    # Validate
    if args.sessions < 1 or args.sessions > 50:
        print("Error: --sessions must be between 1 and 50")
        return
    if args.duration < 10 or args.duration > 300:
        print("Error: --duration must be between 10 and 300 seconds")
        return
    
    # Load test audio if provided
    test_audio = load_test_audio(args.wav) if args.wav else None
    if args.wav and test_audio:
        print(f"Using audio file: {args.wav}")
    elif args.wav:
        print(f"Failed to load {args.wav}, using synthetic audio")
    
    mode = "Contínuo" if args.continuous else "Padrão de fala"
    
    print(f"\n{'='*80}")
    print("LiveKit Simulation Load Test - STT")
    print(f"  Sessões: {args.sessions}  |  Duração: {args.duration}s  |  Modo: {mode}")
    print(f"  URL: {args.url}")
    print(f"{'='*80}\n")
    
    # Run test
    results = asyncio.run(run_test(
        url=args.url,
        num_sessions=args.sessions,
        duration_s=args.duration,
        continuous=args.continuous,
        test_audio=test_audio,
    ))
    
    # Print summary
    print_summary(
        results=results,
        num_sessions=args.sessions,
        duration_s=args.duration,
        continuous=args.continuous,
    )


if __name__ == "__main__":
    main()
