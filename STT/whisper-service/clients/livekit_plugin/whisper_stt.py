"""
LiveKit plugin for Whisper STT Microservice.

Provides WhisperSTT class that implements livekit.agents.stt.STT
for integration with LiveKit voice agents.

Features:
- Automatic audio resampling to 16kHz
- Streaming transcription with interim results
- VAD-based speech segmentation
- Word-level timestamps
- Recognition usage metrics
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
from livekit.agents._exceptions import APIConnectionError, APIStatusError
from livekit.agents.types import (
    APIConnectOptions,
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
    TimedString,
)
from livekit.agents.utils import is_given

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
    
    Features:
    - Automatic audio resampling to 16kHz (input can be any sample rate)
    - Streaming transcription with interim results
    - VAD-based speech segmentation
    - Word-level timestamps
    - Recognition usage metrics for billing
    
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
                aligned_transcript="word" if word_timestamps else False,
                offline_recognize=False,  # Streaming-only service
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
        return "tropicall"

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechStream:
        """
        Create a streaming STT session.
        
        Args:
            language: Override language for this stream
            conn_options: Connection options
            
        Returns:
            SpeechStream for streaming audio input
        """
        opts = replace(self._opts)
        if is_given(language):
            opts.language = language
            
        return WhisperSpeechStream(
            stt=self,
            opts=opts,
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

    async def aclose(self) -> None:
        """Close the STT and cleanup resources."""
        await super().aclose()


class WhisperSpeechStream(stt.SpeechStream):
    """Streaming STT session with automatic resampling."""

    def __init__(
        self,
        *,
        stt: WhisperSTT,
        opts: _WhisperOptions,
        conn_options: APIConnectOptions,
    ):
        # Pass sample_rate to enable automatic resampling to 16kHz
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=SAMPLE_RATE)
        self._stt = stt
        self._opts = opts
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._session_id: str | None = None
        self._request_id: str = ""
        self._closed = False
        
        # Track audio duration for metrics
        self._audio_duration: float = 0.0
        self._speaking = False

    async def _run(self) -> None:
        """Main run loop for the speech stream."""
        self._session_id = f"lk-{uuid4().hex[:8]}"
        self._request_id = f"whisper-{uuid4().hex[:12]}"
        
        try:
            async with websockets.connect(
                self._opts.service_url,
                close_timeout=5,
            ) as ws:
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
                    raise APIStatusError(
                        f"Expected session_started, got: {response.get('type')}",
                        status_code=400,
                    )
                
                # Run send and receive tasks concurrently
                send_task = asyncio.create_task(
                    self._send_audio_loop(), 
                    name="whisper-stt-send"
                )
                recv_task = asyncio.create_task(
                    self._receive_loop(),
                    name="whisper-stt-recv"
                )
                
                try:
                    await asyncio.gather(send_task, recv_task)
                finally:
                    # Use LiveKit's graceful cancel pattern
                    await utils.aio.gracefully_cancel(send_task, recv_task)
                    
        except websockets.exceptions.InvalidURI as e:
            raise APIConnectionError(f"Invalid STT service URL: {e}") from e
        except websockets.exceptions.WebSocketException as e:
            raise APIConnectionError(f"WebSocket error: {e}") from e
        except OSError as e:
            raise APIConnectionError(f"Failed to connect to Whisper STT service: {e}") from e
        finally:
            self._closed = True
            self._ws = None

    async def _send_audio_loop(self) -> None:
        """Send audio frames to the server."""
        try:
            async for frame in self._input_ch:
                if self._closed or self._ws is None:
                    break
                
                if isinstance(frame, self._FlushSentinel):
                    # Flush requested
                    await self._ws.send(json.dumps({"type": "flush"}))
                else:
                    # Audio frame - convert to PCM16 bytes
                    # Track duration for metrics (frame is already resampled to 16kHz)
                    self._audio_duration += frame.duration
                    pcm_bytes = frame.data.tobytes()
                    await self._ws.send(pcm_bytes)
            
            # Send end_session when input channel closes
            if self._ws is not None and not self._closed:
                await self._ws.send(json.dumps({"type": "end_session"}))
            
        except asyncio.CancelledError:
            pass
        except websockets.ConnectionClosed:
            logger.debug(f"WebSocket closed while sending for session {self._session_id}")
        except Exception as e:
            logger.exception(f"Error sending audio: {e}")

    async def _receive_loop(self) -> None:
        """Receive transcripts from the server."""
        if self._ws is None:
            return
            
        try:
            async for message in self._ws:
                if self._closed:
                    break
                
                if isinstance(message, str):
                    data = json.loads(message)
                    self._handle_message(data)
                    
                    if data["type"] == "session_ended":
                        break
                        
        except asyncio.CancelledError:
            pass
        except websockets.ConnectionClosed:
            logger.debug(f"WebSocket closed while receiving for session {self._session_id}")
        except Exception as e:
            logger.exception(f"Error receiving transcripts: {e}")

    def _handle_message(self, data: dict) -> None:
        """Handle incoming JSON message."""
        msg_type = data.get("type")
        
        if msg_type == "partial_transcript":
            self._handle_transcript(data, is_final=False)
            
        elif msg_type == "final_transcript":
            self._handle_transcript(data, is_final=True)
            
        elif msg_type == "vad_event":
            state = data["state"]
            if state == "speech_start":
                self._speaking = True
                # Generate new request_id for each utterance
                self._request_id = f"whisper-{uuid4().hex[:12]}"
                event = stt.SpeechEvent(
                    type=stt.SpeechEventType.START_OF_SPEECH,
                    request_id=self._request_id,
                )
                self._event_ch.send_nowait(event)
            elif state == "speech_end":
                self._speaking = False
                event = stt.SpeechEvent(
                    type=stt.SpeechEventType.END_OF_SPEECH,
                    request_id=self._request_id,
                )
                self._event_ch.send_nowait(event)
                
        elif msg_type == "error":
            error_code = data.get("code", "UNKNOWN")
            error_msg = data.get("message", "Unknown error")
            logger.error(f"STT error [{error_code}]: {error_msg}")
            
        elif msg_type == "session_ended":
            logger.debug(f"Session {self._session_id} ended: {data.get('reason')}")

    def _handle_transcript(self, data: dict, is_final: bool) -> None:
        """Handle partial or final transcript message."""
        text = data.get("text", "")
        if not text and not is_final:
            return
            
        # Emit START_OF_SPEECH if we haven't already
        if not self._speaking and text:
            self._speaking = True
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.START_OF_SPEECH,
                    request_id=self._request_id,
                )
            )
        
        # Parse word timestamps if available
        words = self._parse_words(data.get("words", []))
        
        # Calculate start/end times
        start_time = self.start_time_offset + data.get("start", 0)
        end_time = self.start_time_offset + data.get("end", 0)
        
        # If we have words, use their timing
        if words:
            start_time = words[0].start_time
            end_time = words[-1].end_time
        
        speech_data = stt.SpeechData(
            language=self._opts.language,
            text=text,
            start_time=start_time,
            end_time=end_time,
            confidence=data.get("confidence", 1.0),
            words=words if words else None,
        )
        
        if is_final:
            # Emit final transcript
            event = stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=self._request_id,
                alternatives=[speech_data],
            )
            self._event_ch.send_nowait(event)
            
            # Emit recognition usage for metrics/billing
            if self._audio_duration > 0:
                usage_event = stt.SpeechEvent(
                    type=stt.SpeechEventType.RECOGNITION_USAGE,
                    request_id=self._request_id,
                    recognition_usage=stt.RecognitionUsage(
                        audio_duration=self._audio_duration,
                    ),
                )
                self._event_ch.send_nowait(usage_event)
                self._audio_duration = 0
            
            # Reset speaking state after final
            if self._speaking:
                self._speaking = False
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.END_OF_SPEECH,
                        request_id=self._request_id,
                    )
                )
                
            # Generate new request_id for next utterance
            self._request_id = f"whisper-{uuid4().hex[:12]}"
        else:
            # Emit interim transcript
            event = stt.SpeechEvent(
                type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                request_id=self._request_id,
                alternatives=[speech_data],
            )
            self._event_ch.send_nowait(event)

    def _parse_words(self, words_data: list) -> list[TimedString]:
        """Parse word timestamps from server response."""
        if not words_data:
            return []
            
        return [
            TimedString(
                text=word.get("word", ""),
                start_time=self.start_time_offset + word.get("start", 0),
                end_time=self.start_time_offset + word.get("end", 0),
            )
            for word in words_data
            if word.get("word")
        ]
