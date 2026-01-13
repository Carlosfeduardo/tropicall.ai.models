"""Test script for Supertonic TTS on RunPod."""
import asyncio
import json
import ssl
import wave
import websockets
from uuid import uuid4

RUNPOD_URL = "wss://rc1nh31sh1uoax-8000.proxy.runpod.net/ws/tts"

async def test():
    text = "Olá! Tudo bem? Como posso ajudar você hoje?"
    audio_data = bytearray()
    session_id = f"test-{uuid4().hex[:8]}"
    
    print(f"Texto: {text}")
    print(f"Session ID: {session_id}")
    print(f"Conectando a {RUNPOD_URL}...")
    
    # SSL context para evitar erros de certificado no macOS
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    async with websockets.connect(RUNPOD_URL, ssl=ssl_context) as ws:
        # Start session com ID único
        # Supertonic voices: F1-F5 (female), M1-M5 (male)
        await ws.send(json.dumps({
            "type": "start_session",
            "session_id": session_id,
            "config": {"voice": "F1"}
        }))
        
        resp = json.loads(await ws.recv())
        print(f"Session: {resp}")
        
        # Send text
        await ws.send(json.dumps({"type": "send_text", "text": text}))
        await ws.send(json.dumps({"type": "flush"}))
        await ws.send(json.dumps({"type": "end_session"}))
        
        # Receive audio
        async for msg in ws:
            if isinstance(msg, bytes):
                audio_data.extend(msg)
                print(f"Audio: {len(msg)} bytes")
            else:
                data = json.loads(msg)
                print(f"Control: {data}")
                if data["type"] == "segment_done":
                    timing = data.get("timing", {})
                    ttfa = timing.get("server_total_ms", data.get("ttfa_ms", 0))
                    rtf = data.get("rtf", 0)
                    audio_dur = data.get("audio_duration_ms", 0)
                    print(f"  Server Total: {ttfa:.1f}ms | RTF: {rtf:.3f} | Audio: {audio_dur:.1f}ms")
                if data["type"] == "session_ended":
                    break
    
    # Save WAV
    with wave.open("runpod_test.wav", "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(bytes(audio_data))
    
    print(f"\nSalvo: runpod_test.wav ({len(audio_data)/2/24000:.2f}s)")

if __name__ == "__main__":
    asyncio.run(test())
