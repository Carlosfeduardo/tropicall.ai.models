"""Pydantic models for WebSocket and API protocol messages."""

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
    SEND_MESSAGE = "send_message"
    CANCEL = "cancel"
    END_SESSION = "end_session"

    # Server -> Client
    SESSION_STARTED = "session_started"
    TOKEN = "token"
    MESSAGE_DONE = "message_done"
    ERROR = "error"
    SESSION_ENDED = "session_ended"


class ErrorCode(str, Enum):
    """Error codes for the error message."""

    INVALID_MESSAGE = "INVALID_MESSAGE"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    SESSION_EXISTS = "SESSION_EXISTS"
    MAX_SESSIONS_REACHED = "MAX_SESSIONS_REACHED"
    QUEUE_FULL = "QUEUE_FULL"
    QUEUE_CONGESTION = "QUEUE_CONGESTION"
    CONTEXT_LENGTH_EXCEEDED = "CONTEXT_LENGTH_EXCEEDED"
    GENERATION_CANCELLED = "GENERATION_CANCELLED"
    INFERENCE_ERROR = "INFERENCE_ERROR"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    RATE_LIMITED = "RATE_LIMITED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    TIMEOUT = "TIMEOUT"


class SessionEndReason(str, Enum):
    """Reasons for session ending."""

    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"
    TIMEOUT = "timeout"


class Role(str, Enum):
    """Message role in conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


# =============================================================================
# Chat Message Models
# =============================================================================


class ChatMessage(BaseModel):
    """A single message in the conversation."""

    role: Role = Field(..., description="Message role")
    content: str = Field(..., description="Message content")


# =============================================================================
# Client -> Server Messages
# =============================================================================


class GenerationConfig(BaseModel):
    """Configuration for text generation."""

    max_tokens: int | None = Field(
        default=None,
        ge=1,
        le=8192,
        description="Maximum tokens to generate",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Top-p (nucleus) sampling",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        description="Top-k sampling",
    )
    repetition_penalty: float | None = Field(
        default=None,
        ge=1.0,
        le=2.0,
        description="Repetition penalty",
    )
    stop: list[str] | None = Field(
        default=None,
        description="Stop sequences",
    )


class SessionConfig(BaseModel):
    """Configuration for a chat session."""

    system_prompt: str | None = Field(
        default=None,
        description="System prompt for the session",
    )
    generation: GenerationConfig = Field(
        default_factory=GenerationConfig,
        description="Generation configuration",
    )


class StartSessionMessage(BaseModel):
    """Message to start a new chat session."""

    type: Literal[MessageType.START_SESSION] = MessageType.START_SESSION
    session_id: str = Field(..., description="Unique session identifier")
    config: SessionConfig = Field(default_factory=SessionConfig)


class SendMessageMessage(BaseModel):
    """Message to send user input for completion."""

    type: Literal[MessageType.SEND_MESSAGE] = MessageType.SEND_MESSAGE
    content: str = Field(..., min_length=1, description="User message content")
    request_id: str | None = Field(
        default=None,
        description="Optional request ID for correlation",
    )


class CancelMessage(BaseModel):
    """Message to cancel current generation."""

    type: Literal[MessageType.CANCEL] = MessageType.CANCEL


class EndSessionMessage(BaseModel):
    """Message to end session gracefully."""

    type: Literal[MessageType.END_SESSION] = MessageType.END_SESSION


# =============================================================================
# Server -> Client Messages
# =============================================================================


class SessionStartedMessage(BaseModel):
    """Message confirming session started."""

    type: Literal[MessageType.SESSION_STARTED] = MessageType.SESSION_STARTED
    session_id: str = Field(..., description="Session identifier")


class TokenMessage(BaseModel):
    """Message with a streamed token."""

    type: Literal[MessageType.TOKEN] = MessageType.TOKEN
    token: str = Field(..., description="Generated token")
    request_id: str = Field(..., description="Request ID for correlation")


class MessageDoneMessage(BaseModel):
    """Message indicating generation completed."""

    type: Literal[MessageType.MESSAGE_DONE] = MessageType.MESSAGE_DONE
    request_id: str = Field(..., description="Request ID for correlation")
    content: str = Field(..., description="Full generated content")
    finish_reason: str = Field(..., description="Reason generation stopped")
    usage: "UsageInfo" = Field(..., description="Token usage statistics")
    metrics: "GenerationMetrics" = Field(..., description="Performance metrics")


class UsageInfo(BaseModel):
    """Token usage information."""

    prompt_tokens: int = Field(..., description="Tokens in prompt")
    completion_tokens: int = Field(..., description="Tokens generated")
    total_tokens: int = Field(..., description="Total tokens")


class GenerationMetrics(BaseModel):
    """Performance metrics for generation."""

    ttft_ms: float = Field(..., description="Time to first token in ms")
    total_time_ms: float = Field(..., description="Total generation time in ms")
    tokens_per_second: float = Field(..., description="Token generation rate")


class ErrorMessage(BaseModel):
    """Error message."""

    type: Literal[MessageType.ERROR] = MessageType.ERROR
    code: ErrorCode = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    request_id: str | None = Field(
        default=None,
        description="Request ID if applicable",
    )


class SessionEndedMessage(BaseModel):
    """Message indicating session ended."""

    type: Literal[MessageType.SESSION_ENDED] = MessageType.SESSION_ENDED
    reason: SessionEndReason = Field(..., description="Reason for session ending")
    metrics: "SessionMetrics | None" = Field(
        default=None,
        description="Session-level metrics",
    )


class SessionMetrics(BaseModel):
    """Metrics for the entire session."""

    total_requests: int = Field(..., description="Total requests in session")
    total_prompt_tokens: int = Field(..., description="Total prompt tokens")
    total_completion_tokens: int = Field(..., description="Total completion tokens")
    avg_ttft_ms: float = Field(..., description="Average TTFT in ms")
    avg_tokens_per_second: float = Field(..., description="Average tokens/second")
    session_duration_seconds: float = Field(..., description="Session duration")


# =============================================================================
# OpenAI-Compatible API Models
# =============================================================================


class OpenAIChatMessage(BaseModel):
    """OpenAI-compatible chat message."""

    role: str = Field(..., description="Message role")
    content: str = Field(..., description="Message content")


class OpenAIChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = Field(default="qwen3-32b", description="Model name (ignored)")
    messages: list[OpenAIChatMessage] = Field(
        ...,
        min_length=1,
        description="Conversation messages",
    )
    max_tokens: int | None = Field(default=None, description="Max tokens")
    temperature: float | None = Field(default=None, description="Temperature")
    top_p: float | None = Field(default=None, description="Top-p sampling")
    n: int = Field(default=1, description="Number of completions")
    stream: bool = Field(default=False, description="Stream responses")
    stop: list[str] | str | None = Field(default=None, description="Stop sequences")
    presence_penalty: float = Field(default=0.0, description="Presence penalty")
    frequency_penalty: float = Field(default=0.0, description="Frequency penalty")
    user: str | None = Field(default=None, description="User identifier")


class OpenAIChatCompletionChoice(BaseModel):
    """OpenAI-compatible completion choice."""

    index: int
    message: OpenAIChatMessage
    finish_reason: str | None


class OpenAIChatCompletionUsage(BaseModel):
    """OpenAI-compatible usage info."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChatCompletionChoice]
    usage: OpenAIChatCompletionUsage


class OpenAIChatCompletionChunkDelta(BaseModel):
    """Delta for streaming chunks."""

    role: str | None = None
    content: str | None = None


class OpenAIChatCompletionChunkChoice(BaseModel):
    """Choice in streaming chunk."""

    index: int
    delta: OpenAIChatCompletionChunkDelta
    finish_reason: str | None = None


class OpenAIChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[OpenAIChatCompletionChunkChoice]


# =============================================================================
# Union types for parsing
# =============================================================================

ClientMessage = (
    StartSessionMessage
    | SendMessageMessage
    | CancelMessage
    | EndSessionMessage
)

ServerMessage = (
    SessionStartedMessage
    | TokenMessage
    | MessageDoneMessage
    | ErrorMessage
    | SessionEndedMessage
)
