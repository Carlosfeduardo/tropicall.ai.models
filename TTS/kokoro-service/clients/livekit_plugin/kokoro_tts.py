"""
LiveKit plugin for Kokoro TTS Microservice.

Provides KokoroTTS class that implements livekit.agents.tts.TTS
for integration with LiveKit voice agents.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, replace
from typing import Literal

import websockets
from livekit.agents import tts, utils
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000
NUM_CHANNELS = 1

# Available voices for pt-BR
KokoroVoice = Literal["pf_dora", "pm_alex", "pm_santa"]


@dataclass
class _KokoroOptions:
    """Internal options for Kokoro TTS."""
    service_url: str
    voice: KokoroVoice
    speed: float
    lang_code: str


class KokoroTTS(tts.TTS):
    """
    LiveKit TTS plugin for Kokoro-82M via WebSocket microservice.
    
    Usage:
        tts = KokoroTTS(
            service_url="ws://kokoro-tts:8000/ws/tts",
            voice="pf_dora",
            lang_code="p",
        )
        
        # In agent:
        agent = Agent(
            tts=tts,
            ...
        )
    """

    def __init__(
        self,
        *,
        service_url: str = "ws://localhost:8000/ws/tts",
        voice: KokoroVoice = "pf_dora",
        speed: float = 1.0,
        lang_code: str = "p",
    ):
        """
        Initialize Kokoro TTS plugin.
        
        Args:
            service_url: WebSocket URL of the TTS service
            voice: Voice to use (pf_dora, pm_alex, pm_santa)
            speed: Speech speed multiplier (0.5-2.0)
            lang_code: Language code ('p' for pt-BR)
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        self._opts = _KokoroOptions(
            service_url=service_url,
            voice=voice,
            speed=speed,
            lang_code=lang_code,
        )

    @property
    def model(self) -> str:
        """Model identifier."""
        return "kokoro-82m"

    @property
    def provider(self) -> str:
        """Provider identifier."""
        return "kokoro"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        """
        Synthesize complete text (non-streaming input).
        
        Args:
            text: Text to synthesize
            conn_options: Connection options
            
        Returns:
            ChunkedStream for audio output
        """
        return KokoroChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.SynthesizeStream:
        """
        Create a streaming TTS session.
        
        Args:
            conn_options: Connection options
            
        Returns:
            SynthesizeStream for streaming text input
        """
        return KokoroSynthesizeStream(
            tts=self,
            conn_options=conn_options,
        )

    def update_options(
        self,
        *,
        voice: KokoroVoice | None = None,
        speed: float | None = None,
    ) -> None:
        """
        Update options at runtime.
        
        Args:
            voice: New voice to use
            speed: New speech speed
        """
        if voice is not None:
            self._opts.voice = voice
        if speed is not None:
            self._opts.speed = speed


class KokoroChunkedStream(tts.ChunkedStream):
    """Synthesize complete text (non-streaming input)."""

    def __init__(
        self,
        *,
        tts: KokoroTTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Run the synthesis."""
        session_id = utils.shortuuid()

        async with websockets.connect(self._opts.service_url) as ws:
            # Start session
            await ws.send(
                json.dumps(
                    {
                        "type": "start_session",
                        "session_id": session_id,
                        "config": {
                            "voice": self._opts.voice,
                            "speed": self._opts.speed,
                            "lang_code": self._opts.lang_code,
                        },
                    }
                )
            )

            # Wait for confirmation
            resp = json.loads(await ws.recv())
            if resp["type"] != "session_started":
                raise RuntimeError(f"Unexpected response: {resp}")

            output_emitter.initialize(
                request_id=session_id,
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                mime_type="audio/pcm",
            )

            # Send text + flush + end
            await ws.send(json.dumps({"type": "send_text", "text": self._input_text}))
            await ws.send(json.dumps({"type": "flush"}))
            await ws.send(json.dumps({"type": "end_session"}))

            # Receive: binary = audio, text = control
            async for msg in ws:
                if isinstance(msg, bytes):
                    output_emitter.push(msg)
                else:
                    data = json.loads(msg)
                    if data["type"] == "session_ended":
                        break
                    elif data["type"] == "error":
                        raise RuntimeError(data.get("message", "Unknown error"))

            output_emitter.flush()


class KokoroSynthesizeStream(tts.SynthesizeStream):
    """Streaming TTS - receives text incrementally."""

    def __init__(
        self,
        *,
        tts: KokoroTTS,
        conn_options: APIConnectOptions,
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Run the streaming synthesis."""
        session_id = utils.shortuuid()

        async with websockets.connect(self._opts.service_url) as ws:
            # Start session
            await ws.send(
                json.dumps(
                    {
                        "type": "start_session",
                        "session_id": session_id,
                        "config": {
                            "voice": self._opts.voice,
                            "speed": self._opts.speed,
                            "lang_code": self._opts.lang_code,
                        },
                    }
                )
            )

            resp = json.loads(await ws.recv())
            if resp["type"] != "session_started":
                raise RuntimeError(f"Unexpected response: {resp}")

            output_emitter.initialize(
                request_id=session_id,
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                mime_type="audio/pcm",
                stream=True,
            )
            output_emitter.start_segment(segment_id=session_id)

            # Task to send text
            async def send_task():
                async for data in self._input_ch:
                    if isinstance(data, str):
                        self._mark_started()
                        await ws.send(json.dumps({"type": "send_text", "text": data}))
                    elif isinstance(data, self._FlushSentinel):
                        await ws.send(json.dumps({"type": "flush"}))

                await ws.send(json.dumps({"type": "end_session"}))

            # Task to receive audio
            async def recv_task():
                async for msg in ws:
                    if isinstance(msg, bytes):
                        output_emitter.push(msg)
                    else:
                        data = json.loads(msg)
                        if data["type"] == "session_ended":
                            break
                        elif data["type"] == "error":
                            raise RuntimeError(data.get("message", "Unknown error"))

                output_emitter.end_segment()

            # Execute both tasks
            send_t = asyncio.create_task(send_task())
            recv_t = asyncio.create_task(recv_task())

            try:
                await asyncio.gather(send_t, recv_t)
            finally:
                await utils.aio.gracefully_cancel(send_t, recv_t)
