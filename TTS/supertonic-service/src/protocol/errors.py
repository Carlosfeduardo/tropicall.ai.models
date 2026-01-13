"""Error definitions for the TTS service."""

from .messages import ErrorCode


class TTSError(Exception):
    """Base exception for TTS service errors."""

    def __init__(self, code: ErrorCode, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class SessionNotFoundError(TTSError):
    """Raised when session is not found."""

    def __init__(self, session_id: str):
        super().__init__(
            ErrorCode.SESSION_NOT_FOUND,
            f"Session not found: {session_id}",
        )


class SessionExistsError(TTSError):
    """Raised when session already exists."""

    def __init__(self, session_id: str):
        super().__init__(
            ErrorCode.SESSION_EXISTS,
            f"Session already exists: {session_id}",
        )


class InvalidVoiceError(TTSError):
    """Raised when voice is invalid."""

    def __init__(self, voice: str):
        super().__init__(
            ErrorCode.INVALID_VOICE,
            f"Invalid voice: {voice}. Valid voices: pf_dora, pm_alex, pm_santa",
        )


class QueueOverflowError(TTSError):
    """Raised when inference queue is full."""

    def __init__(self):
        super().__init__(
            ErrorCode.QUEUE_OVERFLOW,
            "Inference queue is full",
        )


class QueueFullError(TTSError):
    """Raised when queue depth exceeds maximum limit."""

    def __init__(self, current_depth: int, max_depth: int):
        super().__init__(
            ErrorCode.QUEUE_FULL,
            f"Queue full: {current_depth}/{max_depth} requests pending",
        )
        self.current_depth = current_depth
        self.max_depth = max_depth


class QueueCongestionError(TTSError):
    """Raised when estimated wait time exceeds maximum limit."""

    def __init__(self, estimated_wait_ms: float, max_wait_ms: float):
        super().__init__(
            ErrorCode.QUEUE_CONGESTION,
            f"Queue congested: estimated wait {estimated_wait_ms:.0f}ms > {max_wait_ms:.0f}ms limit",
        )
        self.estimated_wait_ms = estimated_wait_ms
        self.max_wait_ms = max_wait_ms


class ClientSlowError(TTSError):
    """Raised when client is not consuming audio fast enough."""

    def __init__(self, inflight_segments: int):
        super().__init__(
            ErrorCode.CLIENT_SLOW,
            f"Client too slow: {inflight_segments} segments in flight",
        )


class InferenceError(TTSError):
    """Raised when inference fails."""

    def __init__(self, message: str):
        super().__init__(
            ErrorCode.INFERENCE_ERROR,
            f"Inference error: {message}",
        )


class MaxSessionsReachedError(TTSError):
    """Raised when maximum sessions limit is reached."""

    def __init__(self, max_sessions: int):
        super().__init__(
            ErrorCode.MAX_SESSIONS_REACHED,
            f"Maximum sessions reached: {max_sessions}",
        )
