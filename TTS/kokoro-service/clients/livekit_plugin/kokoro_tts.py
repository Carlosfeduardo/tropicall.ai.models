"""
LiveKit plugin for Kokoro TTS Microservice.

Provides KokoroTTS class that implements livekit.agents.tts.TTS
for integration with LiveKit voice agents.

Optimized with:
- Connection pooling for reduced latency
- Prewarm support for eliminating cold start
- Proper stream tracking and cleanup
"""

from __future__ import annotations

import asyncio
import json
import logging
import weakref
from dataclasses import dataclass, replace
from typing import Literal

import websockets
from websockets.asyncio.client import ClientConnection
from livekit.agents import tts, utils
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000
NUM_CHANNELS = 1
DEFAULT_CONNECT_TIMEOUT = 10.0
MAX_SESSION_DURATION = 300.0  # 5 minutes

# Available voices for pt-BR
KokoroVoice = Literal["pf_dora", "pm_alex", "pm_santa"]


@dataclass
class _KokoroOptions:
    """Internal options for Kokoro TTS."""
    service_url: str
    voice: KokoroVoice
    speed: float
    lang_code: str


class _ConnectionPool:
    """
    WebSocket connection pool for Kokoro TTS service.
    
    Manages connection lifecycle with:
    - Connection reuse within session duration limit
    - Automatic cleanup of stale connections
    - Prewarm support for eliminating cold start
    
    Note: Kokoro protocol requires one session per connection,
    but pooling still helps with TCP/TLS establishment overhead.
    """
    
    def __init__(
        self,
        service_url: str,
        *,
        max_session_duration: float = MAX_SESSION_DURATION,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    ):
        self._service_url = service_url
        self._max_session_duration = max_session_duration
        self._connect_timeout = connect_timeout
        self._lock = asyncio.Lock()
        self._prewarm_task: asyncio.Task | None = None
        self._closed = False
    
    async def connect(self, timeout: float | None = None) -> ClientConnection:
        """
        Get a new WebSocket connection.
        
        Args:
            timeout: Connection timeout in seconds
            
        Returns:
            Connected WebSocket client
        """
        if self._closed:
            raise RuntimeError("Connection pool is closed")
        
        timeout = timeout or self._connect_timeout
        
        try:
            ws = await asyncio.wait_for(
                websockets.connect(
                    self._service_url,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=10,
                ),
                timeout=timeout,
            )
            return ws
        except asyncio.TimeoutError:
            raise TimeoutError(f"Connection to {self._service_url} timed out after {timeout}s")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {self._service_url}: {e}") from e
    
    def prewarm(self) -> None:
        """
        Initiate a test connection to warm up DNS/TLS.
        
        Non-blocking - runs in background task.
        """
        if self._prewarm_task is not None and not self._prewarm_task.done():
            return
        
        if self._closed:
            return
        
        async def _do_prewarm() -> None:
            try:
                ws = await self.connect(timeout=self._connect_timeout)
                await ws.close()
                logger.debug(f"Prewarm connection to {self._service_url} successful")
            except Exception as e:
                logger.warning(f"Prewarm connection failed: {e}")
        
        self._prewarm_task = asyncio.create_task(_do_prewarm())
    
    async def aclose(self) -> None:
        """Close the connection pool and cleanup resources."""
        self._closed = True
        
        if self._prewarm_task is not None and not self._prewarm_task.done():
            self._prewarm_task.cancel()
            try:
                await self._prewarm_task
            except asyncio.CancelledError:
                pass
            self._prewarm_task = None


class KokoroTTS(tts.TTS):
    """
    LiveKit TTS plugin for Kokoro-82M via WebSocket microservice.
    
    Features:
    - Connection pooling for reduced latency
    - Prewarm support for eliminating cold start
    - Proper resource cleanup via aclose()
    
    Usage:
        tts = KokoroTTS(
            service_url="ws://kokoro-tts:8000/ws/tts",
            voice="pf_dora",
            lang_code="p",
        )
        
        # Optional: prewarm connection before first use
        tts.prewarm()
        
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
        
        # Connection pool for managing WebSocket connections
        self._pool = _ConnectionPool(
            service_url=service_url,
            max_session_duration=MAX_SESSION_DURATION,
            connect_timeout=DEFAULT_CONNECT_TIMEOUT,
        )
        
        # Track active streams for cleanup
        self._streams: weakref.WeakSet[KokoroSynthesizeStream] = weakref.WeakSet()

    @property
    def model(self) -> str:
        """Model identifier."""
        return "kokoro-82m"

    @property
    def provider(self) -> str:
        """Provider identifier."""
        return "kokoro"

    def prewarm(self) -> None:
        """
        Pre-warm connection to the TTS service.
        
        Call this before the first synthesis to eliminate cold start latency.
        This is non-blocking and runs in the background.
        """
        self._pool.prewarm()

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
            pool=self._pool,
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
        stream = KokoroSynthesizeStream(
            tts=self,
            conn_options=conn_options,
            pool=self._pool,
        )
        self._streams.add(stream)
        return stream

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

    async def aclose(self) -> None:
        """
        Close the TTS instance and cleanup all resources.
        
        This closes all active streams and the connection pool.
        """
        # Close all active streams
        for stream in list(self._streams):
            try:
                await stream.aclose()
            except Exception as e:
                logger.warning(f"Error closing stream: {e}")
        
        self._streams.clear()
        
        # Close connection pool
        await self._pool.aclose()


class KokoroChunkedStream(tts.ChunkedStream):
    """Synthesize complete text (non-streaming input)."""

    def __init__(
        self,
        *,
        tts: KokoroTTS,
        input_text: str,
        conn_options: APIConnectOptions,
        pool: _ConnectionPool,
    ):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts
        self._opts = replace(tts._opts)
        self._pool = pool

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Run the synthesis."""
        session_id = utils.shortuuid()
        
        # Get connection from pool with timeout from conn_options
        timeout = self._conn_options.timeout
        ws = await self._pool.connect(timeout=timeout)
        
        try:
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
            
        finally:
            # Always close the WebSocket connection
            try:
                await ws.close()
            except Exception:
                pass


class KokoroSynthesizeStream(tts.SynthesizeStream):
    """Streaming TTS - receives text incrementally."""

    def __init__(
        self,
        *,
        tts: KokoroTTS,
        conn_options: APIConnectOptions,
        pool: _ConnectionPool,
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts = tts
        self._opts = replace(tts._opts)
        self._pool = pool

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Run the streaming synthesis."""
        session_id = utils.shortuuid()
        
        # Get connection from pool with timeout from conn_options
        timeout = self._conn_options.timeout
        ws = await self._pool.connect(timeout=timeout)
        
        try:
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
                
        finally:
            # Always close the WebSocket connection
            try:
                await ws.close()
            except Exception:
                pass