"""
Minimal Python client for testing the TTS service.

Usage:
    python clients/python_client.py "Olá, este é um teste."
    python clients/python_client.py --url ws://localhost:8000/ws/tts "Texto"
"""

import argparse
import asyncio
import json
import wave

import websockets


async def test_tts(
    text: str,
    output_file: str = "output.wav",
    url: str = "ws://localhost:8000/ws/tts",
    voice: str = "pf_dora",
) -> None:
    """
    Test the TTS service with a text input.
    
    Args:
        text: Text to synthesize
        output_file: Output WAV file path
        url: WebSocket URL of the TTS service
        voice: Voice to use
    """
    audio_data = bytearray()
    segment_metrics = []

    print(f"Connecting to {url}...")

    async with websockets.connect(url) as ws:
        # Start session
        await ws.send(
            json.dumps(
                {
                    "type": "start_session",
                    "session_id": "test-001",
                    "config": {"voice": voice, "lang_code": "p"},
                }
            )
        )

        # Wait for session_started
        resp = json.loads(await ws.recv())
        print(f"Session started: {resp}")

        # Send text
        await ws.send(json.dumps({"type": "send_text", "text": text}))
        await ws.send(json.dumps({"type": "flush"}))
        await ws.send(json.dumps({"type": "end_session"}))

        # Receive audio
        async for msg in ws:
            if isinstance(msg, bytes):
                audio_data.extend(msg)
                print(f"Received {len(msg)} bytes of audio")
            else:
                data = json.loads(msg)
                print(f"Control: {data}")
                
                if data["type"] == "segment_done":
                    segment_metrics.append(data)
                elif data["type"] == "session_ended":
                    break

    # Save as WAV
    with wave.open(output_file, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(24000)
        wav.writeframes(bytes(audio_data))

    print(f"\nAudio saved to {output_file}")
    print(f"Total bytes: {len(audio_data)}")
    print(f"Duration: {len(audio_data) / 2 / 24000:.2f} seconds")
    
    if segment_metrics:
        print("\nSegment metrics:")
        for m in segment_metrics:
            print(f"  - TTFA: {m['ttfa_ms']:.1f}ms, RTF: {m['rtf']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Test Kokoro TTS service")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("--url", default="ws://localhost:8000/ws/tts", help="WebSocket URL")
    parser.add_argument("--output", default="output.wav", help="Output WAV file")
    parser.add_argument("--voice", default="pf_dora", help="Voice to use")
    
    args = parser.parse_args()
    
    asyncio.run(test_tts(args.text, args.output, args.url, args.voice))


if __name__ == "__main__":
    main()
