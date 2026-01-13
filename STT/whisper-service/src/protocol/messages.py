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
    FLUSH = "flush"
    END_SESSION = "end_session"
    CANCEL = "cancel"

    # Server -> Client
    SESSION_STARTED = "session_started"
    PARTIAL_TRANSCRIPT = "partial_transcript"
    FINAL_TRANSCRIPT = "final_transcript"
    VAD_EVENT = "vad_event"
    ERROR = "error"
    SESSION_ENDED = "session_ended"


class VADState(str, Enum):
    """VAD state events."""

    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


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
    """Configuration for an STT session."""

    lang_code: str = Field(
        default="pt",
        description="Language code for transcription",
    )
    sample_rate: int = Field(
        default=16000,
        description="Audio sample rate in Hz",
    )
    vad_enabled: bool = Field(
        default=True,
        description="Enable VAD for automatic endpointing",
    )
    vad_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="VAD threshold (0.0-1.0)",
    )
    partial_results: bool = Field(
        default=True,
        description="Enable partial transcript results",
    )
    word_timestamps: bool = Field(
        default=True,
        description="Include word-level timestamps in final transcripts",
    )


class StartSessionMessage(BaseModel):
    """Message to start a new STT session."""

    type: Literal[MessageType.START_SESSION] = MessageType.START_SESSION
    session_id: str = Field(..., description="Unique session identifier")
    config: SessionConfig = Field(default_factory=SessionConfig)


class FlushMessage(BaseModel):
    """Message to force finalization of current utterance."""

    type: Literal[MessageType.FLUSH] = MessageType.FLUSH


class EndSessionMessage(BaseModel):
    """Message to end session gracefully."""

    type: Literal[MessageType.END_SESSION] = MessageType.END_SESSION


class CancelMessage(BaseModel):
    """Message to cancel session immediately (drop buffers)."""

    type: Literal[MessageType.CANCEL] = MessageType.CANCEL


# =============================================================================
# Server -> Client Messages
# =============================================================================


class TimingInfo(BaseModel):
    """Timing breakdown for transcript events."""

    preprocess_ms: float = Field(
        default=0.0,
        description="Time for audio preprocessing (decode, resample, VAD)",
    )
    queue_wait_ms: float = Field(
        default=0.0,
        description="Time spent waiting in inference queue",
    )
    inference_ms: float = Field(
        default=0.0,
        description="Time for Whisper inference",
    )
    postprocess_ms: float = Field(
        default=0.0,
        description="Time for text postprocessing",
    )
    server_total_ms: float = Field(
        default=0.0,
        description="Total server-side processing time",
    )


class WordInfo(BaseModel):
    """Word-level information for final transcripts."""

    word: str = Field(..., description="The word text")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    probability: float = Field(
        default=1.0,
        description="Confidence probability (0.0-1.0)",
    )


class SessionStartedMessage(BaseModel):
    """Message confirming session started."""

    type: Literal[MessageType.SESSION_STARTED] = MessageType.SESSION_STARTED
    session_id: str = Field(..., description="Session identifier")


class PartialTranscriptMessage(BaseModel):
    """Partial transcript update (non-final)."""

    type: Literal[MessageType.PARTIAL_TRANSCRIPT] = MessageType.PARTIAL_TRANSCRIPT
    session_id: str = Field(..., description="Session identifier")
    segment_id: str = Field(..., description="Segment/utterance identifier")
    text: str = Field(..., description="Transcribed text")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Overall confidence (0.0-1.0)",
    )
    is_final: Literal[False] = False
    timing: TimingInfo = Field(default_factory=TimingInfo)


class FinalTranscriptMessage(BaseModel):
    """Final transcript for a completed utterance."""

    type: Literal[MessageType.FINAL_TRANSCRIPT] = MessageType.FINAL_TRANSCRIPT
    session_id: str = Field(..., description="Session identifier")
    segment_id: str = Field(..., description="Segment/utterance identifier")
    text: str = Field(..., description="Transcribed text")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Overall confidence (0.0-1.0)",
    )
    language: str = Field(default="pt", description="Detected language code")
    language_probability: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Language detection probability",
    )
    is_final: Literal[True] = True
    words: list[WordInfo] = Field(
        default_factory=list,
        description="Word-level timestamps (if enabled)",
    )
    audio_duration_ms: float = Field(
        default=0.0,
        description="Duration of audio transcribed",
    )
    timing: TimingInfo = Field(default_factory=TimingInfo)


class VADEventMessage(BaseModel):
    """VAD state change event."""

    type: Literal[MessageType.VAD_EVENT] = MessageType.VAD_EVENT
    session_id: str = Field(..., description="Session identifier")
    state: VADState = Field(..., description="VAD state")
    t_ms: float = Field(..., description="Timestamp in milliseconds from session start")


class ErrorMessage(BaseModel):
    """Error message."""

    type: Literal[MessageType.ERROR] = MessageType.ERROR
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    session_id: str | None = Field(default=None, description="Session identifier")
    segment_id: str | None = Field(default=None, description="Segment identifier")


class SessionEndedMessage(BaseModel):
    """Message indicating session ended."""

    type: Literal[MessageType.SESSION_ENDED] = MessageType.SESSION_ENDED
    session_id: str = Field(..., description="Session identifier")
    reason: SessionEndReason = Field(..., description="Reason for session ending")
    total_audio_ms: float = Field(
        default=0.0,
        description="Total audio duration processed",
    )
    total_segments: int = Field(
        default=0,
        description="Total utterances/segments processed",
    )


# =============================================================================
# Union types for parsing
# =============================================================================

ClientMessage = StartSessionMessage | FlushMessage | EndSessionMessage | CancelMessage

ServerMessage = (
    SessionStartedMessage
    | PartialTranscriptMessage
    | FinalTranscriptMessage
    | VADEventMessage
    | ErrorMessage
    | SessionEndedMessage
)
