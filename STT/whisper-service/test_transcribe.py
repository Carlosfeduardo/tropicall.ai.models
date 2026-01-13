#!/usr/bin/env python3
"""
Simple transcription test - shows full transcripts from audio file.
"""

import asyncio
import json
import ssl
import sys
import time
import wave
from pathlib import Path

import websockets

RUNPOD_URL = "wss://etk3kwziikamjm-8000.proxy.runpod.net/ws/stt"
SAMPLE_RATE = 16000
CHUNK_MS = 20

SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE


async def transcribe_file(wav_path: str):
    """Stream a WAV file and collect all transcripts."""
    
    print(f"\n🎤 Transcrevendo: {wav_path}")
    
    # Read WAV file
    with wave.open(wav_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        audio_data = wf.readframes(wf.getnframes())
        duration_s = len(audio_data) / (sample_rate * 2)
        print(f"📁 Duração: {duration_s:.2f}s, Sample rate: {sample_rate}Hz")
    
    if sample_rate != SAMPLE_RATE:
        print(f"⚠️  Áudio precisa ser {SAMPLE_RATE}Hz. Use ffmpeg para converter.")
        return
    
    chunk_bytes = int(SAMPLE_RATE * CHUNK_MS / 1000) * 2
    
    all_finals = []
    
    async with websockets.connect(RUNPOD_URL, ssl=SSL_CONTEXT) as ws:
        # Start session
        await ws.send(json.dumps({
            "type": "start_session",
            "session_id": f"transcribe-{int(time.time())}",
            "config": {
                "lang_code": "pt",
                "sample_rate": SAMPLE_RATE,
                "vad_enabled": True,
                "vad_threshold": 0.3,  # mais sensível ao som
                "partial_results": True,
            }
        }))
        
        resp = await ws.recv()
        data = json.loads(resp)
        if data.get("type") != "session_started":
            print(f"❌ Erro ao iniciar sessão: {data}")
            return
        
        print(f"\n📡 Sessão iniciada. Streaming áudio...\n")
        
        # Receiver task
        async def receive():
            try:
                async for msg in ws:
                    if isinstance(msg, bytes):
                        continue
                    data = json.loads(msg)
                    msg_type = data.get("type")
                    
                    if msg_type == "vad_event":
                        state = data.get("state")
                        if state == "speech_start":
                            print("  🎙️  [fala detectada]")
                        elif state == "speech_end":
                            print("  🔇 [fim da fala]")
                    
                    elif msg_type == "partial_transcript":
                        text = data.get("text", "")
                        timing = data.get("timing", {})
                        server_ms = timing.get("server_total_ms", 0)
                        print(f"  📝 Partial: \"{text}\" ({server_ms:.0f}ms)")
                    
                    elif msg_type == "final_transcript":
                        text = data.get("text", "")
                        timing = data.get("timing", {})
                        server_ms = timing.get("server_total_ms", 0)
                        audio_ms = data.get("audio_duration_ms", 0)
                        lang = data.get("language", "?")
                        lang_prob = data.get("language_probability", 0) * 100
                        all_finals.append(text)
                        print(f"  ✅ FINAL: \"{text}\" ({server_ms:.0f}ms, {audio_ms:.0f}ms áudio, {lang}:{lang_prob:.0f}%)")
                    
                    elif msg_type == "session_ended":
                        break
                    
                    elif msg_type == "error":
                        print(f"  ❌ Erro: {data.get('message', '')}")
                        
            except websockets.exceptions.ConnectionClosed:
                pass
        
        recv_task = asyncio.create_task(receive())
        
        # Stream audio in real-time
        start = time.monotonic()
        for i in range(0, len(audio_data), chunk_bytes):
            chunk = audio_data[i:i + chunk_bytes]
            if len(chunk) < chunk_bytes:
                chunk = chunk + b'\x00' * (chunk_bytes - len(chunk))
            
            await ws.send(chunk)
            await asyncio.sleep(CHUNK_MS / 1000)
        
        elapsed = time.monotonic() - start
        print(f"\n📤 Streaming completo em {elapsed:.1f}s")
        
        # Flush and wait for finals
        await ws.send(json.dumps({"type": "flush"}))
        await asyncio.sleep(2.0)
        
        # End session
        await ws.send(json.dumps({"type": "end_session"}))
        await asyncio.sleep(0.5)
        
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass
    
    # Print full transcript
    print("\n" + "="*60)
    print("📄 TRANSCRIÇÃO COMPLETA")
    print("="*60)
    
    if all_finals:
        full_text = " ".join(all_finals)
        print(f"\n{full_text}\n")
    else:
        print("\n(Nenhuma transcrição final recebida)\n")
    
    print("="*60)


async def main():
    if len(sys.argv) < 2:
        print("Uso: python test_transcribe.py <arquivo.wav>")
        print("\nExemplo:")
        print("  python test_transcribe.py whatsapp_test.wav")
        return
    
    wav_path = sys.argv[1]
    if not Path(wav_path).exists():
        print(f"❌ Arquivo não encontrado: {wav_path}")
        return
    
    await transcribe_file(wav_path)


if __name__ == "__main__":
    asyncio.run(main())
