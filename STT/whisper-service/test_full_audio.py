#!/usr/bin/env python3
"""
Test full audio transcription - streams complete audio without interruptions.
"""

import asyncio
import json
import ssl
import wave
from pathlib import Path

import websockets

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# RunPod URL
RUNPOD_URL = "wss://aujk9ttkh71fbe-8000.proxy.runpod.net/ws/stt"

# Audio settings
CHUNK_MS = 20  # 20ms chunks
SAMPLE_RATE = 16000
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)  # 320 samples
CHUNK_BYTES = CHUNK_SAMPLES * 2  # 640 bytes


def resample_audio(audio_bytes: bytes, orig_rate: int, target_rate: int = 16000) -> bytes:
    """Resample audio from orig_rate to target_rate using linear interpolation."""
    if not HAS_NUMPY:
        raise RuntimeError("numpy required for resampling. Install with: pip install numpy")
    
    # Convert bytes to int16 array
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    
    # Calculate resampling ratio
    ratio = target_rate / orig_rate
    new_length = int(len(audio) * ratio)
    
    # Linear interpolation
    old_indices = np.arange(len(audio))
    new_indices = np.linspace(0, len(audio) - 1, new_length)
    resampled = np.interp(new_indices, old_indices, audio)
    
    # Convert back to int16 bytes
    return resampled.astype(np.int16).tobytes()


async def transcribe_audio(wav_path: str, url: str = RUNPOD_URL):
    """Stream a WAV file and get full transcription."""
    
    # Load WAV file
    wav_file = Path(wav_path)
    if not wav_file.exists():
        print(f"❌ Arquivo não encontrado: {wav_path}")
        return
    
    with wave.open(str(wav_file), 'rb') as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)
    
    duration_s = n_frames / framerate
    print(f"📁 Arquivo: {wav_path}")
    print(f"   Duração: {duration_s:.2f}s")
    print(f"   Sample rate: {framerate}Hz, Channels: {channels}, Bits: {sample_width * 8}")
    
    if channels != 1 or sample_width != 2:
        print("❌ Áudio deve ser mono 16-bit PCM")
        return
    
    # Resample if needed
    if framerate != SAMPLE_RATE:
        print(f"   ⚠️  Resampleando de {framerate}Hz para {SAMPLE_RATE}Hz...")
        audio_data = resample_audio(audio_data, framerate, SAMPLE_RATE)
        duration_s = len(audio_data) / 2 / SAMPLE_RATE  # Update duration
        print(f"   ✅ Áudio resampleado: {duration_s:.2f}s")
    
    # SSL context for RunPod
    ssl_context = ssl._create_unverified_context()
    
    print(f"\n🔌 Conectando a {url}...")
    
    async with websockets.connect(url, ssl=ssl_context) as ws:
        # Start session
        start_msg = {
            "type": "start_session",
            "session_id": "full-audio-test",
            "config": {
                "language": "pt",
                "sample_rate": SAMPLE_RATE,
                "vad_enabled": False,  # Disable VAD - transcribe everything
                "vad_threshold": 0.3,
                "partial_results": True,
                "word_timestamps": True,
            }
        }
        await ws.send(json.dumps(start_msg))
        
        # Wait for session_started
        response = await ws.recv()
        data = json.loads(response)
        if data.get("type") != "session_started":
            print(f"❌ Erro: {data}")
            return
        print("✅ Sessão iniciada")
        
        # Transcription results
        partials = []
        finals = []
        
        async def receive_messages():
            """Receive all messages until session ends."""
            try:
                async for msg in ws:
                    data = json.loads(msg)
                    msg_type = data.get("type")
                    
                    if msg_type == "partial_transcript":
                        text = data.get("text", "")
                        if text:
                            partials.append(text)
                            print(f"   📝 Partial: \"{text}\"")
                    
                    elif msg_type == "final_transcript":
                        text = data.get("text", "")
                        timing = data.get("timing", {})
                        audio_ms = data.get("audio_duration_ms", 0)
                        if text:
                            finals.append({
                                "text": text,
                                "audio_ms": audio_ms,
                                "server_ms": timing.get("server_total_ms", 0)
                            })
                            print(f"   ✅ Final [{timing.get('server_total_ms', 0):.0f}ms]: \"{text}\"")
                    
                    elif msg_type == "vad_event":
                        state = data.get("state")
                        print(f"   🎤 VAD: {state}")
                    
                    elif msg_type == "session_ended":
                        print("✅ Sessão finalizada")
                        return
                    
                    elif msg_type == "error":
                        print(f"❌ Erro: {data.get('message')}")
            except websockets.exceptions.ConnectionClosed:
                pass
        
        # Start receiver task
        receiver_task = asyncio.create_task(receive_messages())
        
        # Stream audio in chunks (real-time simulation)
        print(f"\n🎵 Enviando áudio ({duration_s:.1f}s)...\n")
        
        offset = 0
        chunks_sent = 0
        
        while offset < len(audio_data):
            chunk = audio_data[offset:offset + CHUNK_BYTES]
            if chunk:
                await ws.send(chunk)
                chunks_sent += 1
                offset += CHUNK_BYTES
                # Simulate real-time: wait 20ms per chunk
                await asyncio.sleep(CHUNK_MS / 1000)
        
        print(f"\n📤 Áudio enviado: {chunks_sent} chunks ({offset / 2 / SAMPLE_RATE:.2f}s)")
        
        # Send flush to finalize
        await ws.send(json.dumps({"type": "flush"}))
        print("📤 Flush enviado")
        
        # Wait a bit for final transcripts
        await asyncio.sleep(2)
        
        # End session
        await ws.send(json.dumps({"type": "end_session"}))
        
        # Wait for session to end
        try:
            await asyncio.wait_for(receiver_task, timeout=5)
        except asyncio.TimeoutError:
            pass
        
        # Print final results
        print(f"\n{'='*60}")
        print("RESULTADO FINAL")
        print(f"{'='*60}")
        
        if finals:
            print(f"\n📜 TRANSCRIÇÃO COMPLETA:\n")
            full_text = " ".join(f["text"] for f in finals)
            
            # Word wrap
            words = full_text.split()
            lines = []
            current = ""
            for word in words:
                if len(current) + len(word) + 1 <= 70:
                    current += (" " if current else "") + word
                else:
                    lines.append(current)
                    current = word
            if current:
                lines.append(current)
            
            for line in lines:
                print(f"   {line}")
            
            print(f"\n📊 Estatísticas:")
            print(f"   - Segmentos finais: {len(finals)}")
            print(f"   - Partials recebidos: {len(partials)}")
            total_audio = sum(f["audio_ms"] for f in finals)
            print(f"   - Áudio transcrito: {total_audio:.0f}ms")
            
            if finals:
                avg_latency = sum(f["server_ms"] for f in finals) / len(finals)
                print(f"   - Latência média (server): {avg_latency:.0f}ms")
        else:
            print("❌ Nenhuma transcrição final recebida")
        
        print(f"\n{'='*60}\n")


async def transcribe_with_vad(wav_path: str, url: str = RUNPOD_URL):
    """Stream a WAV file with VAD enabled to test segmentation."""
    
    # Load WAV file
    wav_file = Path(wav_path)
    if not wav_file.exists():
        print(f"❌ Arquivo não encontrado: {wav_path}")
        return
    
    with wave.open(str(wav_file), 'rb') as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)
    
    duration_s = n_frames / framerate
    print(f"📁 Arquivo: {wav_path}")
    print(f"   Duração: {duration_s:.2f}s")
    print(f"   Sample rate: {framerate}Hz, Channels: {channels}, Bits: {sample_width * 8}")
    
    if channels != 1 or sample_width != 2:
        print("❌ Áudio deve ser mono 16-bit PCM")
        return
    
    # Resample if needed
    if framerate != SAMPLE_RATE:
        print(f"   ⚠️  Resampleando de {framerate}Hz para {SAMPLE_RATE}Hz...")
        audio_data = resample_audio(audio_data, framerate, SAMPLE_RATE)
        duration_s = len(audio_data) / 2 / SAMPLE_RATE
        print(f"   ✅ Áudio resampleado: {duration_s:.2f}s")
    
    # SSL context for RunPod
    ssl_context = ssl._create_unverified_context()
    
    print(f"\n🔌 Conectando a {url}...")
    print("🎤 VAD HABILITADO - testando segmentação automática\n")
    
    async with websockets.connect(url, ssl=ssl_context) as ws:
        # Start session with VAD enabled
        start_msg = {
            "type": "start_session",
            "session_id": "vad-test",
            "config": {
                "language": "pt",
                "sample_rate": SAMPLE_RATE,
                "vad_enabled": True,  # Enable VAD for segmentation
                "vad_threshold": 0.5,
                "partial_results": True,
                "word_timestamps": True,
            }
        }
        await ws.send(json.dumps(start_msg))
        
        # Wait for session_started
        response = await ws.recv()
        data = json.loads(response)
        if data.get("type") != "session_started":
            print(f"❌ Erro: {data}")
            return
        print("✅ Sessão iniciada\n")
        
        # Transcription results
        partials = []
        finals = []
        vad_events = []
        
        async def receive_messages():
            """Receive all messages until session ends."""
            try:
                async for msg in ws:
                    data = json.loads(msg)
                    msg_type = data.get("type")
                    
                    if msg_type == "partial_transcript":
                        text = data.get("text", "")
                        if text:
                            # Detect potential loops (repeated patterns)
                            is_loop = len(partials) > 0 and (
                                text.count(text[-10:]) > 2 if len(text) > 10 else False
                            )
                            loop_warn = " ⚠️ LOOP?" if is_loop else ""
                            partials.append(text)
                            print(f"   📝 Partial: \"{text[:80]}...\"{loop_warn}" if len(text) > 80 else f"   📝 Partial: \"{text}\"{loop_warn}")
                    
                    elif msg_type == "final_transcript":
                        text = data.get("text", "")
                        timing = data.get("timing", {})
                        audio_ms = data.get("audio_duration_ms", 0)
                        if text:
                            finals.append({
                                "text": text,
                                "audio_ms": audio_ms,
                                "server_ms": timing.get("server_total_ms", 0)
                            })
                            print(f"\n   ✅ FINAL [{timing.get('server_total_ms', 0):.0f}ms]: \"{text}\"\n")
                    
                    elif msg_type == "vad_event":
                        state = data.get("state")
                        t_ms = data.get("t_ms", 0)
                        vad_events.append({"state": state, "t_ms": t_ms})
                        icon = "🗣️" if state == "speech_start" else "🔇"
                        print(f"   {icon} VAD: {state} @ {t_ms:.0f}ms")
                    
                    elif msg_type == "session_ended":
                        print("\n✅ Sessão finalizada")
                        return
                    
                    elif msg_type == "error":
                        print(f"❌ Erro: {data.get('message')}")
            except websockets.exceptions.ConnectionClosed:
                pass
        
        # Start receiver task
        receiver_task = asyncio.create_task(receive_messages())
        
        # Stream audio in chunks (real-time simulation)
        print(f"🎵 Enviando áudio ({duration_s:.1f}s)...\n")
        
        offset = 0
        chunks_sent = 0
        
        while offset < len(audio_data):
            chunk = audio_data[offset:offset + CHUNK_BYTES]
            if chunk:
                await ws.send(chunk)
                chunks_sent += 1
                offset += CHUNK_BYTES
                # Simulate real-time: wait 20ms per chunk
                await asyncio.sleep(CHUNK_MS / 1000)
        
        print(f"\n📤 Áudio enviado: {chunks_sent} chunks ({offset / 2 / SAMPLE_RATE:.2f}s)")
        
        # Wait for processing to complete
        await asyncio.sleep(3)
        
        # End session
        await ws.send(json.dumps({"type": "end_session"}))
        
        # Wait for session to end
        try:
            await asyncio.wait_for(receiver_task, timeout=5)
        except asyncio.TimeoutError:
            pass
        
        # Print final results
        print(f"\n{'='*60}")
        print("RESULTADO FINAL (VAD HABILITADO)")
        print(f"{'='*60}")
        
        if finals:
            print(f"\n📜 TRANSCRIÇÃO COMPLETA:\n")
            full_text = " ".join(f["text"] for f in finals)
            
            # Word wrap
            words = full_text.split()
            lines = []
            current = ""
            for word in words:
                if len(current) + len(word) + 1 <= 70:
                    current += (" " if current else "") + word
                else:
                    lines.append(current)
                    current = word
            if current:
                lines.append(current)
            
            for line in lines:
                print(f"   {line}")
            
            print(f"\n📊 Estatísticas:")
            print(f"   - Segmentos finais: {len(finals)}")
            print(f"   - Partials recebidos: {len(partials)}")
            print(f"   - Eventos VAD: {len(vad_events)}")
            total_audio = sum(f["audio_ms"] for f in finals)
            print(f"   - Áudio transcrito: {total_audio:.0f}ms")
            
            if finals:
                avg_latency = sum(f["server_ms"] for f in finals) / len(finals)
                print(f"   - Latência média (server): {avg_latency:.0f}ms")
        else:
            print("❌ Nenhuma transcrição final recebida")
        
        print(f"\n{'='*60}\n")


if __name__ == "__main__":
    import sys
    
    # Parse arguments
    vad_mode = "--vad" in sys.argv
    wav_path = None
    
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            wav_path = arg
            break
    
    wav_path = wav_path or "whatsapp_test.wav"
    
    print(f"\n{'='*60}")
    print("TESTE DE TRANSCRIÇÃO STT")
    print(f"{'='*60}")
    print(f"URL: {RUNPOD_URL}")
    print(f"Modo: {'VAD habilitado' if vad_mode else 'VAD desabilitado (flush manual)'}")
    print(f"{'='*60}\n")
    
    if vad_mode:
        asyncio.run(transcribe_with_vad(wav_path))
    else:
        asyncio.run(transcribe_audio(wav_path))
