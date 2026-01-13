"""
WebSocket endpoint for STT streaming.

Handles the WebSocket protocol:
- JSON messages for control (start_session, flush, end_session, cancel)
- Binary frames for audio input (PCM16 LE mono 16kHz)
"""

import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from ..core.session import session_manager
from ..protocol.errors import ErrorCode, STTError
from ..protocol.messages import (
    CancelMessage,
    EndSessionMessage,
    ErrorMessage,
    FlushMessage,
    MessageType,
    SessionEndedMessage,
    SessionEndReason,
    SessionStartedMessage,
    StartSessionMessage,
)

logger = logging.getLogger(__name__)

router = APIRouter()


async def send_error(
    websocket: WebSocket,
    code: ErrorCode,
    message: str,
    session_id: str | None = None,
    segment_id: str | None = None,
) -> None:
    """Send error message to client."""
    error = ErrorMessage(
        code=code.value,
        message=message,
        session_id=session_id,
        segment_id=segment_id,
    )
    await websocket.send_json(error.model_dump())


async def send_session_started(websocket: WebSocket, session_id: str) -> None:
    """Send session_started message."""
    msg = SessionStartedMessage(session_id=session_id)
    await websocket.send_json(msg.model_dump())


async def send_session_ended(
    websocket: WebSocket,
    session_id: str,
    reason: SessionEndReason,
    total_audio_ms: float = 0.0,
    total_segments: int = 0,
) -> None:
    """Send session_ended message."""
    msg = SessionEndedMessage(
        session_id=session_id,
        reason=reason,
        total_audio_ms=total_audio_ms,
        total_segments=total_segments,
    )
    await websocket.send_json(msg.model_dump())


def parse_message(data: dict[str, Any]) -> (
    StartSessionMessage
    | FlushMessage
    | EndSessionMessage
    | CancelMessage
    | None
):
    """Parse incoming JSON message into typed message object."""
    msg_type = data.get("type")

    try:
        if msg_type == MessageType.START_SESSION:
            return StartSessionMessage.model_validate(data)
        elif msg_type == MessageType.FLUSH:
            return FlushMessage.model_validate(data)
        elif msg_type == MessageType.END_SESSION:
            return EndSessionMessage.model_validate(data)
        elif msg_type == MessageType.CANCEL:
            return CancelMessage.model_validate(data)
        else:
            return None
    except ValidationError:
        return None


@router.websocket("/ws/stt")
async def websocket_stt(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for STT streaming.
    
    Protocol:
    1. Client sends start_session with session_id and config
    2. Server responds with session_started
    3. Client sends binary audio frames (PCM16 LE mono 16kHz)
    4. Server sends partial_transcript / final_transcript / vad_event JSON
    5. Client sends flush to force finalization, or end_session to finish
    6. Server sends session_ended
    
    Cancel can be sent at any time to immediately stop processing.
    
    Binary frames:
    - Format: PCM16 little-endian, mono, 16kHz
    - Recommended chunk size: 20ms = 640 samples = 1280 bytes
    """
    await websocket.accept()
    
    session = None
    session_id = None
    end_reason = SessionEndReason.COMPLETED
    total_audio_ms = 0.0
    total_segments = 0

    try:
        # Wait for start_session message
        raw_data = await websocket.receive_text()
        data = json.loads(raw_data)
        
        msg = parse_message(data)
        if not isinstance(msg, StartSessionMessage):
            await send_error(
                websocket,
                ErrorCode.INVALID_MESSAGE,
                "First message must be start_session",
            )
            return

        session_id = msg.session_id
        
        try:
            # Create session
            session = await session_manager.create_session(
                session_id=session_id,
                websocket=websocket,
                language=msg.config.lang_code,
                vad_enabled=msg.config.vad_enabled,
                vad_threshold=msg.config.vad_threshold,
                partial_enabled=msg.config.partial_results,
                word_timestamps=msg.config.word_timestamps,
            )
        except STTError as e:
            await send_error(websocket, e.code, e.message)
            return

        # Confirm session started
        await send_session_started(websocket, session_id)
        logger.info(f"Session {session_id} started")

        # Main message loop
        while True:
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                end_reason = SessionEndReason.ERROR
                break

            # Handle binary (audio) messages
            if message["type"] == "websocket.receive":
                if "bytes" in message and message["bytes"]:
                    # Binary audio data
                    await session.handle_audio_chunk(message["bytes"])
                    continue
                
                if "text" in message and message["text"]:
                    # JSON control message
                    try:
                        data = json.loads(message["text"])
                    except json.JSONDecodeError:
                        await send_error(
                            websocket,
                            ErrorCode.INVALID_MESSAGE,
                            "Invalid JSON",
                            session_id=session_id,
                        )
                        continue

                    msg = parse_message(data)
                    
                    if msg is None:
                        await send_error(
                            websocket,
                            ErrorCode.INVALID_MESSAGE,
                            f"Unknown message type: {data.get('type')}",
                            session_id=session_id,
                        )
                        continue

                    try:
                        if isinstance(msg, FlushMessage):
                            await session.handle_flush()
                            
                        elif isinstance(msg, EndSessionMessage):
                            await session.handle_end_session()
                            end_reason = SessionEndReason.COMPLETED
                            break
                            
                        elif isinstance(msg, CancelMessage):
                            await session.handle_cancel()
                            end_reason = SessionEndReason.CANCELLED
                            break
                            
                        elif isinstance(msg, StartSessionMessage):
                            await send_error(
                                websocket,
                                ErrorCode.SESSION_EXISTS,
                                "Session already started",
                                session_id=session_id,
                            )
                            
                    except STTError as e:
                        await send_error(
                            websocket,
                            e.code,
                            e.message,
                            session_id=session_id,
                        )
                        
                        # Admission control errors: continue session
                        if e.code in (ErrorCode.QUEUE_FULL, ErrorCode.QUEUE_CONGESTION):
                            logger.debug(
                                f"Admission control rejected in {session_id}: {e.message}"
                            )
                            continue
                        
                        # Other errors: close session
                        end_reason = SessionEndReason.ERROR
                        break
                        
                    except Exception as e:
                        logger.exception(f"Error handling message in session {session_id}")
                        await send_error(
                            websocket,
                            ErrorCode.INTERNAL_ERROR,
                            str(e),
                            session_id=session_id,
                        )
                        end_reason = SessionEndReason.ERROR
                        break

    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {session_id or 'unknown'}")
        end_reason = SessionEndReason.ERROR
    except Exception as e:
        logger.exception(f"Unexpected error in session {session_id or 'unknown'}")
        end_reason = SessionEndReason.ERROR
    finally:
        # Get final metrics if session was created
        if session is not None:
            metrics = session.get_metrics()
            total_audio_ms = metrics.get("total_audio_ms", 0.0)
            total_segments = metrics.get("total_segments", 0)
            
            try:
                # Send session ended
                await send_session_ended(
                    websocket,
                    session_id,
                    end_reason,
                    total_audio_ms=total_audio_ms,
                    total_segments=total_segments,
                )
            except Exception:
                pass  # Client might already be disconnected

            # Remove session
            await session_manager.remove_session(session_id)
            logger.info(
                f"Session {session_id} ended: {end_reason.value} "
                f"(audio={total_audio_ms:.0f}ms, segments={total_segments})"
            )

        # Close WebSocket
        try:
            await websocket.close()
        except Exception:
            pass
