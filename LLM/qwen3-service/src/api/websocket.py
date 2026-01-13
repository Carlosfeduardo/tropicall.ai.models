"""
WebSocket endpoint for chat streaming.

Handles the WebSocket protocol:
- JSON messages for control (start_session, send_message, cancel, end_session)
- JSON messages for streaming tokens and completion
"""

import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from ..core.session import session_manager
from ..core.token_streamer import stream_tokens_to_client
from ..protocol.errors import LLMError
from ..protocol.messages import (
    CancelMessage,
    EndSessionMessage,
    ErrorCode,
    ErrorMessage,
    MessageType,
    SendMessageMessage,
    SessionEndedMessage,
    SessionEndReason,
    SessionMetrics,
    SessionStartedMessage,
    StartSessionMessage,
)

logger = logging.getLogger(__name__)

router = APIRouter()


async def send_error(
    websocket: WebSocket,
    code: ErrorCode,
    message: str,
    request_id: str | None = None,
) -> None:
    """Send error message to client."""
    error = ErrorMessage(code=code, message=message, request_id=request_id)
    await websocket.send_json(error.model_dump())


async def send_session_started(websocket: WebSocket, session_id: str) -> None:
    """Send session_started message."""
    msg = SessionStartedMessage(session_id=session_id)
    await websocket.send_json(msg.model_dump())


async def send_session_ended(
    websocket: WebSocket,
    reason: SessionEndReason,
    metrics: dict | None = None,
) -> None:
    """Send session_ended message."""
    session_metrics = None
    if metrics:
        session_metrics = SessionMetrics(
            total_requests=metrics.get("total_requests", 0),
            total_prompt_tokens=metrics.get("total_prompt_tokens", 0),
            total_completion_tokens=metrics.get("total_completion_tokens", 0),
            avg_ttft_ms=metrics.get("avg_ttft_ms", 0),
            avg_tokens_per_second=metrics.get("avg_tokens_per_second", 0),
            session_duration_seconds=metrics.get("session_duration_seconds", 0),
        )
    
    msg = SessionEndedMessage(reason=reason, metrics=session_metrics)
    await websocket.send_json(msg.model_dump())


def parse_message(data: dict[str, Any]) -> (
    StartSessionMessage
    | SendMessageMessage
    | CancelMessage
    | EndSessionMessage
    | None
):
    """Parse incoming JSON message into typed message object."""
    msg_type = data.get("type")

    try:
        if msg_type == MessageType.START_SESSION:
            return StartSessionMessage.model_validate(data)
        elif msg_type == MessageType.SEND_MESSAGE:
            return SendMessageMessage.model_validate(data)
        elif msg_type == MessageType.CANCEL:
            return CancelMessage.model_validate(data)
        elif msg_type == MessageType.END_SESSION:
            return EndSessionMessage.model_validate(data)
        else:
            return None
    except ValidationError as e:
        logger.debug(f"Message validation error: {e}")
        return None


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for chat streaming.
    
    Protocol:
    1. Client sends start_session with session_id and config
    2. Server responds with session_started
    3. Client sends send_message with user content
    4. Server streams token messages + message_done with metrics
    5. Repeat 3-4 for conversation
    6. Client sends end_session to finish (or cancel to abort current generation)
    7. Server sends session_ended + session metrics
    
    Cancel can be sent at any time to stop current generation.
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
                websocket=websocket,
                config=msg.config,
            )
        except LLMError as e:
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
                if isinstance(msg, SendMessageMessage):
                    # Check if already generating
                    if session.generating:
                        await send_error(
                            websocket,
                            ErrorCode.INVALID_MESSAGE,
                            "Generation already in progress. Send 'cancel' first.",
                        )
                        continue
                    
                    # Add user message and get request ID
                    request_id = session.add_user_message(msg.content)
                    
                    # Stream response tokens
                    await stream_tokens_to_client(session, request_id)
                    
                elif isinstance(msg, CancelMessage):
                    cancelled_request_id = session.cancel()
                    if cancelled_request_id:
                        logger.info(
                            f"Session {session_id}: Generation cancelled"
                        )
                    
                elif isinstance(msg, EndSessionMessage):
                    # Cancel any ongoing generation
                    session.cancel()
                    end_reason = SessionEndReason.COMPLETED
                    break
                    
                elif isinstance(msg, StartSessionMessage):
                    await send_error(
                        websocket,
                        ErrorCode.SESSION_EXISTS,
                        "Session already started",
                    )
                    
            except LLMError as e:
                # Send error to client
                await send_error(websocket, e.code, e.message)
                
                # Queue errors: continue session
                if e.code in (ErrorCode.QUEUE_FULL, ErrorCode.QUEUE_CONGESTION):
                    logger.debug(
                        f"Admission control rejected request in {session_id}: {e.message}"
                    )
                    session.generating = False
                    continue
                
                # Other errors: continue session but log
                logger.warning(f"Error in session {session_id}: {e.message}")
                session.generating = False
                continue
                
            except Exception as e:
                logger.exception(f"Unexpected error in session {session_id}")
                await send_error(
                    websocket,
                    ErrorCode.INTERNAL_ERROR,
                    str(e),
                )
                session.generating = False
                continue

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
                # Get final metrics
                metrics = session.get_metrics()
                
                # Send session ended
                await send_session_ended(websocket, end_reason, metrics)
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
