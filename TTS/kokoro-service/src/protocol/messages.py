"""Pydantic models for WebSocket protocol messages."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================


class MessageType(str, Enum):
    """Message types for the WebSocket protocol."""

    # Client -> Server
    START_SESSION = "start_session"
    SEND_TEXT = "send_text"
    FLUSH = "flush"
    END_SESSION = "end_session"
    CANCEL = "cancel"

    # Server -> Client
    SESSION_STARTED = "session_started"
    SEGMENT_DONE = "segment_done"
    ERROR = "error"
    METRICS = "metrics"
    SESSION_ENDED = "session_ended"


class ErrorCode(str, Enum):
    """Error codes for the error message."""

    INVALID_MESSAGE = "INVALID_MESSAGE"
    INVALID_VOICE = "INVALID_VOICE"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    SESSION_EXISTS = "SESSION_EXISTS"
    QUEUE_OVERFLOW = "QUEUE_OVERFLOW"
    CLIENT_SLOW = "CLIENT_SLOW"
    INFERENCE_ERROR = "INFERENCE_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    MAX_SESSIONS_REACHED = "MAX_SESSIONS_REACHED"


class SessionEndReason(str, Enum):
    """Reasons for session ending."""

    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"
    TIMEOUT = "timeout"


# =============================================================================
# Client -> Server Messages
# =============================================================================


class SessionConfig(BaseModel):
    """Configuration for a TTS session."""

    voice: Literal["pf_dora", "pm_alex", "pm_santa"] = Field(
        default="pf_dora",
        description="Voice to use for TTS",
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speech speed multiplier",
    )
    lang_code: str = Field(
        default="p",
        description="Language code (p for pt-BR)",
    )


class StartSessionMessage(BaseModel):
    """Message to start a new TTS session."""

    type: Literal[MessageType.START_SESSION] = MessageType.START_SESSION
    session_id: str = Field(..., description="Unique session identifier")
    config: SessionConfig = Field(default_factory=SessionConfig)


class SendTextMessage(BaseModel):
    """Message to send text for synthesis."""

    type: Literal[MessageType.SEND_TEXT] = MessageType.SEND_TEXT
    text: str = Field(..., description="Text to synthesize")


class FlushMessage(BaseModel):
    """Message to flush pending text."""

    type: Literal[MessageType.FLUSH] = MessageType.FLUSH


class EndSessionMessage(BaseModel):
    """Message to end session gracefully."""

    type: Literal[MessageType.END_SESSION] = MessageType.END_SESSION


class CancelMessage(BaseModel):
    """Message to cancel session immediately (barge-in)."""

    type: Literal[MessageType.CANCEL] = MessageType.CANCEL


# =============================================================================
# Server -> Client Messages
# =============================================================================


class SessionStartedMessage(BaseModel):
    """Message confirming session started."""

    type: Literal[MessageType.SESSION_STARTED] = MessageType.SESSION_STARTED
    session_id: str = Field(..., description="Session identifier")


class SegmentDoneMessage(BaseModel):
    """
    Message indicating a segment was completely sent.
    
    Includes telemetry metrics for observability.
    """

    type: Literal[MessageType.SEGMENT_DONE] = MessageType.SEGMENT_DONE
    segment_id: str = Field(..., description="Unique segment identifier")
    request_id: str = Field(..., description="Request ID for correlation")
    ttfa_ms: float = Field(..., description="Time to first audio in milliseconds")
    rtf: float = Field(..., description="Real-time factor (< 1.0 = faster than realtime)")
    audio_duration_ms: float = Field(..., description="Duration of audio generated")
    total_samples: int = Field(..., description="Total audio samples in segment")


class ErrorMessage(BaseModel):
    """Error message."""

    type: Literal[MessageType.ERROR] = MessageType.ERROR
    code: ErrorCode = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")


class MetricsMessage(BaseModel):
    """Metrics message sent periodically or at session end."""

    type: Literal[MessageType.METRICS] = MessageType.METRICS
    rtf: float = Field(..., description="Average real-time factor")
    ttfa_ms: float = Field(..., description="Average time to first audio")
    segments_processed: int = Field(..., description="Number of segments processed")
    audio_duration_ms: float = Field(..., description="Total audio duration generated")


class SessionEndedMessage(BaseModel):
    """Message indicating session ended."""

    type: Literal[MessageType.SESSION_ENDED] = MessageType.SESSION_ENDED
    reason: SessionEndReason = Field(..., description="Reason for session ending")


# =============================================================================
# Union types for parsing
# =============================================================================

ClientMessage = (
    StartSessionMessage
    | SendTextMessage
    | FlushMessage
    | EndSessionMessage
    | CancelMessage
)

ServerMessage = (
    SessionStartedMessage
    | SegmentDoneMessage
    | ErrorMessage
    | MetricsMessage
    | SessionEndedMessage
)
