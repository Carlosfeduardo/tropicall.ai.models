"""
LiveKit plugin for Whisper STT Microservice.

Provides WhisperSTT class that implements livekit.agents.stt.STT
for integration with LiveKit voice agents.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, replace
from typing import Literal
from uuid import uuid4

import websockets
from livekit.agents import stt, utils
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
NUM_CHANNELS = 1


@dataclass
class _WhisperOptions:
    """Internal options for Whisper STT."""

    service_url: str
    language: str
    vad_enabled: bool
    partial_results: bool
    word_timestamps: bool


class WhisperSTT(stt.STT):
    """
    LiveKit STT plugin for Whisper via WebSocket microservice.
    
    Usage:
        stt = WhisperSTT(
            service_url="ws://whisper-stt:8000/ws/stt",
            language="pt",
        )
        
        # In agent:
        agent = Agent(
            stt=stt,
            ...
        )
    """

    def __init__(
        self,
        *,
        service_url: str = "ws://localhost:8000/ws/stt",
        language: str = "pt",
        vad_enabled: bool = True,
        partial_results: bool = True,
        word_timestamps: bool = True,
    ):
        """
        Initialize Whisper STT plugin.
        
        Args:
            service_url: WebSocket URL of the STT service
            language: Language code (pt for Portuguese)
            vad_enabled: Enable VAD for automatic endpointing
            partial_results: Enable partial transcript streaming
            word_timestamps: Include word-level timestamps
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=partial_results,
            ),
        )
        self._opts = _WhisperOptions(
            service_url=service_url,
            language=language,
            vad_enabled=vad_enabled,
            partial_results=partial_results,
            word_timestamps=word_timestamps,
        )

    @property
    def model(self) -> str:
        """Model identifier."""
        return "whisper"

    @property
    def provider(self) -> str:
        """Provider identifier."""
        return "whisper"

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechStream:
        """
        Create a streaming STT session.
        
        Args:
            conn_options: Connection options
            
        Returns:
            SpeechStream for streaming audio input
        """
        return WhisperSpeechStream(
            stt=self,
            conn_options=conn_options,
        )

    def update_options(
        self,
        *,
        language: str | None = None,
        vad_enabled: bool | None = None,
        partial_results: bool | None = None,
    ) -> None:
        """
        Update options at runtime.
        
        Args:
            language: New language code
            vad_enabled: Enable/disable VAD
            partial_results: Enable/disable partial results
        """
        if language is not None:
            self._opts.language = language
        if vad_enabled is not None:
            self._opts.vad_enabled = vad_enabled
        if partial_results is not None:
            self._opts.partial_results = partial_results


class WhisperSpeechStream(stt.SpeechStream):
    """Streaming STT session."""

    def __init__(
        self,
        *,
        stt: WhisperSTT,
        conn_options: APIConnectOptions,
    ):
        super().__init__(stt=stt, conn_options=conn_options)
        self._stt = stt
        self._opts = replace(stt._opts)
        self._ws = None
        self._session_id = None
        self._closed = False

    async def _run(self) -> None:
        """Main run loop for the speech stream."""
        self._session_id = f"lk-{uuid4().hex[:8]}"
        
        try:
            async with websockets.connect(self._opts.service_url) as ws:
                self._ws = ws
                
                # Start session
                start_msg = {
                    "type": "start_session",
                    "session_id": self._session_id,
                    "config": {
                        "lang_code": self._opts.language,
                        "sample_rate": SAMPLE_RATE,
                        "vad_enabled": self._opts.vad_enabled,
                        "partial_results": self._opts.partial_results,
                        "word_timestamps": self._opts.word_timestamps,
                    },
                }
                await ws.send(json.dumps(start_msg))
                
                # Wait for session_started
                response = json.loads(await ws.recv())
                if response["type"] != "session_started":
                    raise RuntimeError(f"Expected session_started, got: {response}")
                
                # Run send and receive tasks concurrently
                send_task = asyncio.create_task(self._send_audio_loop())
                recv_task = asyncio.create_task(self._receive_loop())
                
                try:
                    await asyncio.gather(send_task, recv_task)
                finally:
                    send_task.cancel()
                    recv_task.cancel()
                    
        except websockets.ConnectionClosed:
            logger.debug(f"WebSocket connection closed for session {self._session_id}")
        except Exception as e:
            logger.exception(f"Error in Whisper STT stream: {e}")
            raise
        finally:
            self._closed = True

    async def _send_audio_loop(self) -> None:
        """Send audio frames to the server."""
        try:
            async for frame in self._input_ch:
                if self._closed:
                    break
                
                if isinstance(frame, self._FlushSentinel):
                    # Flush requested
                    await self._ws.send(json.dumps({"type": "flush"}))
                else:
                    # Audio frame - convert to PCM16 bytes
                    pcm_bytes = frame.data.tobytes()
                    await self._ws.send(pcm_bytes)
            
            # Send end_session when input channel closes
            await self._ws.send(json.dumps({"type": "end_session"}))
            
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception(f"Error sending audio: {e}")

    async def _receive_loop(self) -> None:
        """Receive transcripts from the server."""
        try:
            async for message in self._ws:
                if self._closed:
                    break
                
                if isinstance(message, str):
                    data = json.loads(message)
                    await self._handle_message(data)
                    
                    if data["type"] == "session_ended":
                        break
                        
        except asyncio.CancelledError:
            pass
        except websockets.ConnectionClosed:
            pass
        except Exception as e:
            logger.exception(f"Error receiving transcripts: {e}")

    async def _handle_message(self, data: dict) -> None:
        """Handle incoming JSON message."""
        msg_type = data.get("type")
        
        if msg_type == "partial_transcript":
            # Emit interim result
            event = stt.SpeechEvent(
                type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        language=self._opts.language,
                        text=data["text"],
                        confidence=data.get("confidence", 1.0),
                    )
                ],
            )
            self._event_ch.send_nowait(event)
            
        elif msg_type == "final_transcript":
            # Emit final result
            event = stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        language=self._opts.language,
                        text=data["text"],
                        confidence=data.get("confidence", 1.0),
                    )
                ],
            )
            self._event_ch.send_nowait(event)
            
        elif msg_type == "vad_event":
            state = data["state"]
            if state == "speech_start":
                event = stt.SpeechEvent(
                    type=stt.SpeechEventType.START_OF_SPEECH,
                )
                self._event_ch.send_nowait(event)
            elif state == "speech_end":
                event = stt.SpeechEvent(
                    type=stt.SpeechEventType.END_OF_SPEECH,
                )
                self._event_ch.send_nowait(event)
                
        elif msg_type == "error":
            logger.error(f"STT error: {data.get('code')}: {data.get('message')}")
            
        elif msg_type == "session_ended":
            logger.debug(f"Session {self._session_id} ended: {data.get('reason')}")
