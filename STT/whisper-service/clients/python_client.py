"""
Example Python client for Whisper STT Microservice.

Demonstrates:
- WebSocket connection with session management
- Streaming PCM16 audio from file or microphone
- Handling partial and final transcripts
"""

import asyncio
import json
import struct
import wave
from pathlib import Path
from typing import Callable
from uuid import uuid4

import websockets


class WhisperSTTClient:
    """
    WebSocket client for Whisper STT microservice.
    
    Usage:
        client = WhisperSTTClient("ws://localhost:8000/ws/stt")
        
        async with client.session() as session:
            # Stream audio file
            await session.stream_wav_file("audio.wav")
            
            # Or stream chunks manually
            await session.send_audio(pcm16_bytes)
            await session.flush()
    
    Callbacks:
        on_partial: Called with partial transcript text
        on_final: Called with final transcript text and words
        on_vad: Called with VAD state ("speech_start" or "speech_end")
        on_error: Called with error code and message
    """

    def __init__(
        self,
        service_url: str = "ws://localhost:8000/ws/stt",
        language: str = "pt",
        vad_enabled: bool = True,
        partial_results: bool = True,
        word_timestamps: bool = True,
        on_partial: Callable[[str], None] | None = None,
        on_final: Callable[[str, list], None] | None = None,
        on_vad: Callable[[str, float], None] | None = None,
        on_error: Callable[[str, str], None] | None = None,
    ):
        self.service_url = service_url
        self.language = language
        self.vad_enabled = vad_enabled
        self.partial_results = partial_results
        self.word_timestamps = word_timestamps
        
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_vad = on_vad
        self.on_error = on_error
        
        self._ws = None
        self._session_id = None
        self._receive_task = None

    async def connect(self) -> None:
        """Connect to the STT service."""
        self._ws = await websockets.connect(self.service_url)
        self._session_id = f"session-{uuid4().hex[:8]}"
        
        # Start session
        start_msg = {
            "type": "start_session",
            "session_id": self._session_id,
            "config": {
                "lang_code": self.language,
                "sample_rate": 16000,
                "vad_enabled": self.vad_enabled,
                "partial_results": self.partial_results,
                "word_timestamps": self.word_timestamps,
            },
        }
        await self._ws.send(json.dumps(start_msg))
        
        # Wait for session_started
        response = json.loads(await self._ws.recv())
        if response["type"] != "session_started":
            raise RuntimeError(f"Expected session_started, got: {response}")
        
        # Start receive task
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def disconnect(self) -> None:
        """Disconnect from the STT service."""
        if self._ws:
            # Send end_session
            await self._ws.send(json.dumps({"type": "end_session"}))
            
            # Wait for session_ended
            if self._receive_task:
                try:
                    await asyncio.wait_for(self._receive_task, timeout=5.0)
                except asyncio.TimeoutError:
                    self._receive_task.cancel()
            
            await self._ws.close()
            self._ws = None

    async def _receive_loop(self) -> None:
        """Background task to receive messages from server."""
        try:
            async for message in self._ws:
                if isinstance(message, str):
                    data = json.loads(message)
                    await self._handle_message(data)
                    
                    if data["type"] == "session_ended":
                        break
        except websockets.ConnectionClosed:
            pass
        except Exception as e:
            if self.on_error:
                self.on_error("CLIENT_ERROR", str(e))

    async def _handle_message(self, data: dict) -> None:
        """Handle incoming JSON message."""
        msg_type = data.get("type")
        
        if msg_type == "partial_transcript":
            if self.on_partial:
                self.on_partial(data["text"])
                
        elif msg_type == "final_transcript":
            if self.on_final:
                words = data.get("words", [])
                self.on_final(data["text"], words)
                
        elif msg_type == "vad_event":
            if self.on_vad:
                self.on_vad(data["state"], data["t_ms"])
                
        elif msg_type == "error":
            if self.on_error:
                self.on_error(data["code"], data["message"])
                
        elif msg_type == "session_ended":
            pass  # Will be handled in _receive_loop

    async def send_audio(self, pcm16_bytes: bytes) -> None:
        """
        Send audio chunk to the server.
        
        Args:
            pcm16_bytes: PCM16 little-endian mono 16kHz audio bytes
        """
        if self._ws:
            await self._ws.send(pcm16_bytes)

    async def flush(self) -> None:
        """Force finalization of current utterance."""
        if self._ws:
            await self._ws.send(json.dumps({"type": "flush"}))

    async def cancel(self) -> None:
        """Cancel current processing and drop buffers."""
        if self._ws:
            await self._ws.send(json.dumps({"type": "cancel"}))

    async def stream_wav_file(
        self,
        filepath: str | Path,
        chunk_duration_ms: int = 20,
        realtime: bool = True,
    ) -> None:
        """
        Stream a WAV file to the server.
        
        Args:
            filepath: Path to WAV file (must be 16kHz mono)
            chunk_duration_ms: Chunk duration in milliseconds
            realtime: If True, send at real-time rate
        """
        filepath = Path(filepath)
        
        with wave.open(str(filepath), "rb") as wf:
            # Validate format
            if wf.getnchannels() != 1:
                raise ValueError("WAV must be mono")
            if wf.getframerate() != 16000:
                raise ValueError("WAV must be 16kHz")
            if wf.getsampwidth() != 2:
                raise ValueError("WAV must be 16-bit")
            
            # Calculate chunk size
            chunk_samples = int(16000 * chunk_duration_ms / 1000)
            chunk_bytes = chunk_samples * 2
            
            # Stream chunks
            while True:
                data = wf.readframes(chunk_samples)
                if not data:
                    break
                
                await self.send_audio(data)
                
                if realtime:
                    await asyncio.sleep(chunk_duration_ms / 1000)
        
        # Flush at end
        await self.flush()

    async def stream_pcm_samples(
        self,
        samples: list[int] | bytes,
        chunk_duration_ms: int = 20,
        realtime: bool = True,
    ) -> None:
        """
        Stream PCM16 samples to the server.
        
        Args:
            samples: List of int16 samples or bytes
            chunk_duration_ms: Chunk duration in milliseconds
            realtime: If True, send at real-time rate
        """
        if isinstance(samples, list):
            # Convert list to bytes
            samples = struct.pack(f"<{len(samples)}h", *samples)
        
        chunk_samples = int(16000 * chunk_duration_ms / 1000)
        chunk_bytes = chunk_samples * 2
        
        for i in range(0, len(samples), chunk_bytes):
            chunk = samples[i:i + chunk_bytes]
            await self.send_audio(chunk)
            
            if realtime:
                await asyncio.sleep(chunk_duration_ms / 1000)


# Context manager support
class STTSession:
    """Context manager for STT session."""

    def __init__(self, client: WhisperSTTClient):
        self.client = client

    async def __aenter__(self) -> WhisperSTTClient:
        await self.client.connect()
        return self.client

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.client.disconnect()


# =============================================================================
# Example usage
# =============================================================================

async def main():
    """Example: Stream a WAV file and print transcripts."""
    
    def on_partial(text: str):
        print(f"[PARTIAL] {text}")
    
    def on_final(text: str, words: list):
        print(f"[FINAL] {text}")
        if words:
            for w in words:
                print(f"  - {w['word']} ({w['start']:.2f}s - {w['end']:.2f}s)")
    
    def on_vad(state: str, t_ms: float):
        print(f"[VAD] {state} at {t_ms:.0f}ms")
    
    def on_error(code: str, message: str):
        print(f"[ERROR] {code}: {message}")
    
    client = WhisperSTTClient(
        service_url="ws://localhost:8000/ws/stt",
        language="pt",
        on_partial=on_partial,
        on_final=on_final,
        on_vad=on_vad,
        on_error=on_error,
    )
    
    async with STTSession(client) as session:
        # Stream a WAV file (you'll need to provide your own)
        # await session.stream_wav_file("test_audio.wav")
        
        # Or generate test audio (silence + sine wave)
        import math
        
        # Generate 3 seconds of test audio
        sample_rate = 16000
        duration_s = 3
        samples = []
        
        # 1 second silence
        samples.extend([0] * sample_rate)
        
        # 1 second tone (440 Hz)
        for i in range(sample_rate):
            t = i / sample_rate
            value = int(16000 * math.sin(2 * math.pi * 440 * t))
            samples.append(value)
        
        # 1 second silence
        samples.extend([0] * sample_rate)
        
        print("Streaming test audio...")
        await session.stream_pcm_samples(samples, realtime=True)
        
        # Wait a bit for final transcript
        await asyncio.sleep(2)
    
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
