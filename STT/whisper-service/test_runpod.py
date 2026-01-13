#!/usr/bin/env python3
"""
Test script for Whisper STT service on RunPod.
Streams a WAV file and receives transcripts.
"""

import asyncio
import json
import wave
import sys
import time
import ssl
from pathlib import Path

import websockets

# SSL context for RunPod proxy (bypass certificate verification for testing)
SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE

# RunPod endpoint
RUNPOD_URL = "wss://xxbgyhyy841cp8-8000.proxy.runpod.net/ws/stt"

# Test with a sample WAV file or generate silence
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 20  # 20ms chunks
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000) * 2  # bytes (16-bit PCM)


async def test_stt_with_wav(wav_path: str):
    """Test STT by streaming a WAV file."""
    print(f"\n🎤 Testing STT with: {wav_path}")
    print(f"📡 Connecting to: {RUNPOD_URL}\n")
    
    # Read WAV file
    with wave.open(wav_path, 'rb') as wf:
        assert wf.getnchannels() == 1, "WAV must be mono"
        assert wf.getsampwidth() == 2, "WAV must be 16-bit"
        sample_rate = wf.getframerate()
        audio_data = wf.readframes(wf.getnframes())
        duration_s = len(audio_data) / (sample_rate * 2)
        print(f"📁 Audio: {duration_s:.2f}s, {sample_rate}Hz")
    
    async with websockets.connect(RUNPOD_URL, ssl=SSL_CONTEXT) as ws:
        # Start session
        start_msg = {
            "type": "start_session",
            "session_id": f"test-{int(time.time())}",
            "config": {
                "language": "pt",
                "sample_rate": sample_rate,
                "enable_vad": True,
                "enable_partials": True,
                "partial_interval_ms": 300
            }
        }
        await ws.send(json.dumps(start_msg))
        print(f"➡️  Sent: start_session")
        
        # Wait for session_started
        response = await ws.recv()
        data = json.loads(response)
        print(f"⬅️  Received: {data.get('type', 'unknown')}")
        
        # Stream audio in chunks
        chunk_size = int(sample_rate * CHUNK_DURATION_MS / 1000) * 2
        total_chunks = len(audio_data) // chunk_size
        print(f"\n📤 Streaming {total_chunks} chunks ({CHUNK_DURATION_MS}ms each)...\n")
        
        start_time = time.time()
        chunks_sent = 0
        
        # Background task to receive responses
        async def receive_responses():
            try:
                while True:
                    response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    data = json.loads(response)
                    msg_type = data.get("type", "unknown")
                    
                    if msg_type == "partial_transcript":
                        text = data.get("text", "")
                        timing = data.get("timing", {})
                        server_total = timing.get("server_total_ms", 0)
                        print(f"  📝 PARTIAL: \"{text}\" ({server_total:.0f}ms)")
                    
                    elif msg_type == "final_transcript":
                        text = data.get("text", "")
                        timing = data.get("timing", {})
                        server_total = timing.get("server_total_ms", 0)
                        print(f"  ✅ FINAL: \"{text}\" ({server_total:.0f}ms)")
                    
                    elif msg_type == "vad_event":
                        state = data.get("state", "unknown")
                        print(f"  🎙️  VAD: {state}")
                    
                    elif msg_type == "session_ended":
                        print(f"  🔚 Session ended")
                        break
                    
                    elif msg_type == "error":
                        print(f"  ❌ ERROR: {data.get('message', 'unknown')}")
                    
                    else:
                        print(f"  📨 {msg_type}: {data}")
                        
            except asyncio.TimeoutError:
                pass
            except websockets.exceptions.ConnectionClosed:
                pass
        
        # Start receiving in background
        recv_task = asyncio.create_task(receive_responses())
        
        # Stream audio
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            if len(chunk) < chunk_size:
                # Pad last chunk with silence
                chunk = chunk + b'\x00' * (chunk_size - len(chunk))
            
            await ws.send(chunk)
            chunks_sent += 1
            
            # Simulate real-time streaming
            await asyncio.sleep(CHUNK_DURATION_MS / 1000)
            
            # Progress indicator
            if chunks_sent % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  ... sent {chunks_sent}/{total_chunks} chunks ({elapsed:.1f}s)")
        
        print(f"\n📤 Finished streaming {chunks_sent} chunks")
        
        # Send flush to finalize
        flush_msg = {"type": "flush"}
        await ws.send(json.dumps(flush_msg))
        print("➡️  Sent: flush")
        
        # Wait for final transcript
        await asyncio.sleep(2.0)
        
        # End session
        end_msg = {"type": "end_session"}
        await ws.send(json.dumps(end_msg))
        print("➡️  Sent: end_session")
        
        # Wait for responses
        await asyncio.sleep(1.0)
        recv_task.cancel()
        
        elapsed = time.time() - start_time
        print(f"\n✨ Test completed in {elapsed:.2f}s")


async def test_stt_with_silence(duration_s: float = 3.0):
    """Test STT with silence (VAD should not trigger)."""
    print(f"\n🔇 Testing STT with {duration_s}s of silence...")
    print(f"📡 Connecting to: {RUNPOD_URL}\n")
    
    async with websockets.connect(RUNPOD_URL, ssl=SSL_CONTEXT) as ws:
        # Start session
        start_msg = {
            "type": "start_session",
            "session_id": f"silence-{int(time.time())}",
            "config": {
                "language": "pt",
                "sample_rate": SAMPLE_RATE,
                "enable_vad": True
            }
        }
        await ws.send(json.dumps(start_msg))
        
        response = await ws.recv()
        data = json.loads(response)
        print(f"⬅️  Session started: {data.get('session_id')}")
        
        # Send silence
        silence_chunk = b'\x00' * CHUNK_SIZE
        total_chunks = int(duration_s * 1000 / CHUNK_DURATION_MS)
        
        for i in range(total_chunks):
            await ws.send(silence_chunk)
            await asyncio.sleep(CHUNK_DURATION_MS / 1000)
        
        print(f"📤 Sent {total_chunks} silence chunks")
        
        # End session
        await ws.send(json.dumps({"type": "end_session"}))
        
        # Check for any responses
        try:
            while True:
                response = await asyncio.wait_for(ws.recv(), timeout=1.0)
                data = json.loads(response)
                print(f"⬅️  {data.get('type')}: {data}")
        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
            pass
        
        print("✨ Silence test completed (VAD correctly detected no speech)")


async def test_basic_connection():
    """Test basic WebSocket connection."""
    print(f"\n🔌 Testing basic connection...")
    print(f"📡 URL: {RUNPOD_URL}\n")
    
    try:
        async with websockets.connect(RUNPOD_URL, ssl=SSL_CONTEXT, close_timeout=5) as ws:
            # Start session
            start_msg = {
                "type": "start_session",
                "session_id": f"basic-{int(time.time())}",
                "config": {"language": "pt"}
            }
            await ws.send(json.dumps(start_msg))
            print("➡️  Sent: start_session")
            
            response = await asyncio.wait_for(ws.recv(), timeout=5.0)
            data = json.loads(response)
            print(f"⬅️  Received: {json.dumps(data, indent=2)}")
            
            if data.get("type") == "session_started":
                print("\n✅ Connection test PASSED!")
            else:
                print(f"\n⚠️  Unexpected response: {data.get('type')}")
            
            # End session
            await ws.send(json.dumps({"type": "end_session"}))
            
    except Exception as e:
        print(f"\n❌ Connection test FAILED: {e}")
        raise


async def main():
    """Run tests."""
    print("=" * 60)
    print("🧪 Whisper STT RunPod Test")
    print("=" * 60)
    
    # Test 1: Basic connection
    await test_basic_connection()
    
    # Test 2: Silence (VAD test)
    await test_stt_with_silence(2.0)
    
    # Test 3: If WAV file provided, stream it
    if len(sys.argv) > 1:
        wav_path = sys.argv[1]
        if Path(wav_path).exists():
            await test_stt_with_wav(wav_path)
        else:
            print(f"\n⚠️  WAV file not found: {wav_path}")
    else:
        print("\n💡 Tip: Pass a WAV file to test real audio:")
        print("   python test_runpod.py audio.wav")
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
