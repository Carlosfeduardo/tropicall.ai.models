"""Error types for the STT service."""

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """Error codes for the STT service."""

    # Protocol errors
    INVALID_MESSAGE = "INVALID_MESSAGE"
    INVALID_CONFIG = "INVALID_CONFIG"
    INVALID_AUDIO = "INVALID_AUDIO"

    # Session errors
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    SESSION_EXISTS = "SESSION_EXISTS"
    MAX_SESSIONS_REACHED = "MAX_SESSIONS_REACHED"

    # Admission control errors
    QUEUE_FULL = "QUEUE_FULL"
    QUEUE_CONGESTION = "QUEUE_CONGESTION"

    # Buffer errors
    BUFFER_OVERFLOW = "BUFFER_OVERFLOW"

    # Inference errors
    INFERENCE_ERROR = "INFERENCE_ERROR"
    MODEL_NOT_READY = "MODEL_NOT_READY"

    # General errors
    INTERNAL_ERROR = "INTERNAL_ERROR"
    TIMEOUT = "TIMEOUT"


class STTError(Exception):
    """Base exception for STT service errors."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        details: dict[str, Any] | None = None,
    ):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "code": self.code.value,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        return result


class SessionExistsError(STTError):
    """Raised when trying to create a session that already exists."""

    def __init__(self, session_id: str):
        super().__init__(
            code=ErrorCode.SESSION_EXISTS,
            message=f"Session already exists: {session_id}",
            details={"session_id": session_id},
        )


class SessionNotFoundError(STTError):
    """Raised when session is not found."""

    def __init__(self, session_id: str):
        super().__init__(
            code=ErrorCode.SESSION_NOT_FOUND,
            message=f"Session not found: {session_id}",
            details={"session_id": session_id},
        )


class MaxSessionsReachedError(STTError):
    """Raised when maximum number of sessions is reached."""

    def __init__(self, max_sessions: int):
        super().__init__(
            code=ErrorCode.MAX_SESSIONS_REACHED,
            message=f"Maximum sessions reached: {max_sessions}",
            details={"max_sessions": max_sessions},
        )


class QueueFullError(STTError):
    """Raised when inference queue is full."""

    def __init__(self, current_depth: int, max_depth: int):
        super().__init__(
            code=ErrorCode.QUEUE_FULL,
            message=f"Queue full: {current_depth}/{max_depth}",
            details={"current_depth": current_depth, "max_depth": max_depth},
        )


class QueueCongestionError(STTError):
    """Raised when estimated wait time exceeds threshold."""

    def __init__(self, estimated_wait_ms: float, max_wait_ms: float):
        super().__init__(
            code=ErrorCode.QUEUE_CONGESTION,
            message=f"Queue congested: estimated wait {estimated_wait_ms:.1f}ms > {max_wait_ms:.1f}ms",
            details={
                "estimated_wait_ms": round(estimated_wait_ms, 2),
                "max_wait_ms": round(max_wait_ms, 2),
            },
        )


class BufferOverflowError(STTError):
    """Raised when audio buffer overflows."""

    def __init__(self, buffer_ms: float, max_buffer_ms: int, trimmed_ms: float):
        super().__init__(
            code=ErrorCode.BUFFER_OVERFLOW,
            message=f"Buffer overflow: {buffer_ms:.0f}ms > {max_buffer_ms}ms, trimmed {trimmed_ms:.0f}ms",
            details={
                "buffer_ms": round(buffer_ms, 2),
                "max_buffer_ms": max_buffer_ms,
                "trimmed_ms": round(trimmed_ms, 2),
            },
        )


class InferenceError(STTError):
    """Raised when inference fails."""

    def __init__(self, message: str, original_error: Exception | None = None):
        details = {}
        if original_error:
            details["original_error"] = str(original_error)
        super().__init__(
            code=ErrorCode.INFERENCE_ERROR,
            message=message,
            details=details,
        )


class ModelNotReadyError(STTError):
    """Raised when model is not ready for inference."""

    def __init__(self):
        super().__init__(
            code=ErrorCode.MODEL_NOT_READY,
            message="Model not ready for inference",
        )


class InvalidAudioError(STTError):
    """Raised when audio data is invalid."""

    def __init__(self, message: str):
        super().__init__(
            code=ErrorCode.INVALID_AUDIO,
            message=message,
        )
