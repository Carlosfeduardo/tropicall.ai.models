"""
Minimal Python client for testing the Supertonic TTS service.

Usage:
    python clients/python_client.py "Olá, este é um teste."
    python clients/python_client.py --url ws://localhost:8000/ws/tts "Texto"
    python clients/python_client.py --lang en --voice M1 "Hello, this is a test."
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
    voice: str = "F1",
    lang_code: str = "pt",
) -> None:
    """
    Test the TTS service with a text input.
    
    Args:
        text: Text to synthesize
        output_file: Output WAV file path
        url: WebSocket URL of the TTS service
        voice: Voice to use (F1-F5 female, M1-M5 male)
        lang_code: Language code (en, ko, es, pt, fr)
    """
    audio_data = bytearray()
    segment_metrics = []

    print(f"Connecting to {url}...")
    print(f"Voice: {voice}, Language: {lang_code}")

    async with websockets.connect(url) as ws:
        # Start session
        await ws.send(
            json.dumps(
                {
                    "type": "start_session",
                    "session_id": "test-001",
                    "config": {"voice": voice, "lang_code": lang_code},
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
    parser = argparse.ArgumentParser(description="Test Supertonic TTS service")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("--url", default="ws://localhost:8000/ws/tts", help="WebSocket URL")
    parser.add_argument("--output", default="output.wav", help="Output WAV file")
    parser.add_argument(
        "--voice",
        default="F1",
        choices=["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"],
        help="Voice to use (F1-F5 female, M1-M5 male)",
    )
    parser.add_argument(
        "--lang",
        default="pt",
        choices=["en", "ko", "es", "pt", "fr"],
        help="Language code",
    )
    
    args = parser.parse_args()
    
    asyncio.run(test_tts(args.text, args.output, args.url, args.voice, args.lang))


if __name__ == "__main__":
    main()
