"""
WebSocket endpoint for TTS streaming.

Handles the WebSocket protocol:
- JSON messages for control (start_session, send_text, flush, end_session, cancel)
- Binary frames for audio output
"""

import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from ..core.session import session_manager
from ..protocol.errors import TTSError
from ..protocol.messages import (
    CancelMessage,
    EndSessionMessage,
    ErrorCode,
    ErrorMessage,
    FlushMessage,
    MessageType,
    MetricsMessage,
    SendTextMessage,
    SessionEndedMessage,
    SessionEndReason,
    SessionStartedMessage,
    StartSessionMessage,
)

logger = logging.getLogger(__name__)

router = APIRouter()


async def send_error(websocket: WebSocket, code: ErrorCode, message: str) -> None:
    """Send error message to client."""
    error = ErrorMessage(code=code, message=message)
    await websocket.send_json(error.model_dump())


async def send_session_started(websocket: WebSocket, session_id: str) -> None:
    """Send session_started message."""
    msg = SessionStartedMessage(session_id=session_id)
    await websocket.send_json(msg.model_dump())


async def send_session_ended(
    websocket: WebSocket,
    reason: SessionEndReason,
) -> None:
    """Send session_ended message."""
    msg = SessionEndedMessage(reason=reason)
    await websocket.send_json(msg.model_dump())


async def send_metrics(websocket: WebSocket, metrics: dict) -> None:
    """Send metrics message."""
    msg = MetricsMessage(
        rtf=metrics.get("rtf", 0),
        ttfa_ms=metrics.get("ttfa_ms", 0),
        segments_processed=metrics.get("segments_processed", 0),
        audio_duration_ms=metrics.get("audio_duration_ms", 0),
    )
    await websocket.send_json(msg.model_dump())


def parse_message(data: dict[str, Any]) -> (
    StartSessionMessage
    | SendTextMessage
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
        elif msg_type == MessageType.SEND_TEXT:
            return SendTextMessage.model_validate(data)
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


@router.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for TTS streaming.
    
    Protocol:
    1. Client sends start_session with session_id and config
    2. Server responds with session_started
    3. Client sends send_text messages (can send multiple)
    4. Server sends binary audio frames + segment_done JSON
    5. Client sends flush to force processing, or end_session to finish
    6. Server sends session_ended + metrics
    
    Cancel can be sent at any time to immediately stop processing.
    """
    await websocket.accept()
    
    session = None
    session_id = None
    end_reason = SessionEndReason.COMPLETED

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
                voice=msg.config.voice,
                speed=msg.config.speed,
                websocket=websocket,
            )
        except TTSError as e:
            await send_error(websocket, e.code, e.message)
            return

        # Confirm session started
        await send_session_started(websocket, session_id)
        logger.info(f"Session {session_id} started")

        # Main message loop
        while True:
            try:
                raw_data = await websocket.receive_text()
            except WebSocketDisconnect:
                end_reason = SessionEndReason.ERROR
                break

            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError:
                await send_error(
                    websocket,
                    ErrorCode.INVALID_MESSAGE,
                    "Invalid JSON",
                )
                continue

            msg = parse_message(data)
            
            if msg is None:
                await send_error(
                    websocket,
                    ErrorCode.INVALID_MESSAGE,
                    f"Unknown message type: {data.get('type')}",
                )
                continue

            try:
                if isinstance(msg, SendTextMessage):
                    await session.handle_send_text(msg.text)
                    
                elif isinstance(msg, FlushMessage):
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
                    )
                    
            except TTSError as e:
                # Send error to client
                await send_error(websocket, e.code, e.message)
                
                # Admission control errors: continue session (don't close)
                # This allows the client to retry or wait
                if e.code in (ErrorCode.QUEUE_FULL, ErrorCode.QUEUE_CONGESTION):
                    logger.debug(
                        f"Admission control rejected segment in {session_id}: {e.message}"
                    )
                    continue  # Don't close session, just continue
                
                # Other errors: close session
                end_reason = SessionEndReason.ERROR
                break
                
            except Exception as e:
                logger.exception(f"Error handling message in session {session_id}")
                await send_error(
                    websocket,
                    ErrorCode.INTERNAL_ERROR,
                    str(e),
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
        # Send final messages if session was created
        if session is not None:
            try:
                # Send session ended
                await send_session_ended(websocket, end_reason)
                
                # Send final metrics
                metrics = session.get_metrics()
                await send_metrics(websocket, metrics)
            except Exception:
                pass  # Client might already be disconnected

            # Remove session
            await session_manager.remove_session(session_id)
            logger.info(f"Session {session_id} ended: {end_reason.value}")

        # Close WebSocket
        try:
            await websocket.close()
        except Exception:
            pass
