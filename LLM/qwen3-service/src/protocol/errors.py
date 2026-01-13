"""Custom exceptions for the LLM service."""

from .messages import ErrorCode


class LLMError(Exception):
    """Base exception for LLM service errors."""

    def __init__(self, code: ErrorCode, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class SessionExistsError(LLMError):
    """Raised when trying to create a session that already exists."""

    def __init__(self, session_id: str):
        super().__init__(
            ErrorCode.SESSION_EXISTS,
            f"Session '{session_id}' already exists",
        )


class SessionNotFoundError(LLMError):
    """Raised when session is not found."""

    def __init__(self, session_id: str):
        super().__init__(
            ErrorCode.SESSION_NOT_FOUND,
            f"Session '{session_id}' not found",
        )


class MaxSessionsReachedError(LLMError):
    """Raised when maximum session limit is reached."""

    def __init__(self, max_sessions: int):
        super().__init__(
            ErrorCode.MAX_SESSIONS_REACHED,
            f"Maximum sessions ({max_sessions}) reached",
        )


class QueueFullError(LLMError):
    """Raised when inference queue is full."""

    def __init__(self, current_depth: int, max_depth: int):
        super().__init__(
            ErrorCode.QUEUE_FULL,
            f"Queue full: {current_depth}/{max_depth}",
        )


class QueueCongestionError(LLMError):
    """Raised when queue is congested (high estimated wait)."""

    def __init__(self, estimated_wait_ms: float, max_wait_ms: float):
        super().__init__(
            ErrorCode.QUEUE_CONGESTION,
            f"Queue congested: estimated {estimated_wait_ms:.0f}ms > max {max_wait_ms:.0f}ms",
        )


class ContextLengthExceededError(LLMError):
    """Raised when context length exceeds model limit."""

    def __init__(self, current_tokens: int, max_tokens: int):
        super().__init__(
            ErrorCode.CONTEXT_LENGTH_EXCEEDED,
            f"Context length exceeded: {current_tokens}/{max_tokens} tokens",
        )


class GenerationCancelledError(LLMError):
    """Raised when generation is cancelled."""

    def __init__(self, request_id: str):
        super().__init__(
            ErrorCode.GENERATION_CANCELLED,
            f"Generation cancelled for request '{request_id}'",
        )


class InferenceError(LLMError):
    """Raised when inference fails."""

    def __init__(self, message: str):
        super().__init__(
            ErrorCode.INFERENCE_ERROR,
            f"Inference error: {message}",
        )


class AuthenticationError(LLMError):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Invalid or missing API key"):
        super().__init__(
            ErrorCode.AUTHENTICATION_ERROR,
            message,
        )


class RateLimitError(LLMError):
    """Raised when rate limit is exceeded."""

    def __init__(self, retry_after_seconds: float | None = None):
        message = "Rate limit exceeded"
        if retry_after_seconds:
            message += f", retry after {retry_after_seconds:.1f}s"
        super().__init__(
            ErrorCode.RATE_LIMITED,
            message,
        )
        self.retry_after_seconds = retry_after_seconds
