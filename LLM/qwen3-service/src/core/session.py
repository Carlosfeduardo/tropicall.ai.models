"""
Session management for the LLM service.

Implements:
- Chat session with conversation history
- Cancellation support for barge-in
- Token streaming coordination
- Session-level metrics tracking
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import uuid4

from fastapi import WebSocket

from ..config import settings
from ..protocol.errors import (
    ContextLengthExceededError,
    MaxSessionsReachedError,
    SessionExistsError,
)
from ..protocol.messages import (
    ChatMessage,
    GenerationConfig,
    Role,
    SessionConfig,
)

if TYPE_CHECKING:
    from ..inference.vllm_engine import GenerationOutput

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """
    Chat session with conversation history and generation state.
    
    Features:
    - Maintains conversation history
    - Tracks active generation for cancellation
    - Collects session-level metrics
    """

    session_id: str
    websocket: WebSocket
    config: SessionConfig

    # Conversation history
    messages: list[ChatMessage] = field(default_factory=list)

    # Generation state
    cancelled: bool = field(default=False, init=False)
    current_request_id: str | None = field(default=None, init=False)
    generating: bool = field(default=False, init=False)

    # Session timing
    created_at: float = field(default_factory=time.monotonic, init=False)
    last_activity_at: float = field(default_factory=time.monotonic, init=False)

    # Metrics
    request_count: int = field(default=0, init=False)
    total_prompt_tokens: int = field(default=0, init=False)
    total_completion_tokens: int = field(default=0, init=False)
    total_ttft_ms: float = field(default=0.0, init=False)
    total_generation_time_ms: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        """Initialize session with system prompt if provided."""
        if self.config.system_prompt:
            self.messages.append(
                ChatMessage(role=Role.SYSTEM, content=self.config.system_prompt)
            )

    def add_user_message(self, content: str) -> str:
        """
        Add a user message and prepare for generation.
        
        Args:
            content: User message content
            
        Returns:
            Request ID for this generation
        """
        self.messages.append(ChatMessage(role=Role.USER, content=content))
        self.last_activity_at = time.monotonic()
        self.request_count += 1
        
        # Generate request ID
        self.current_request_id = f"{self.session_id}-{uuid4().hex[:8]}"
        self.generating = True
        self.cancelled = False
        
        return self.current_request_id

    def add_assistant_message(self, content: str) -> None:
        """
        Add assistant response to history.
        
        Args:
            content: Generated response content
        """
        self.messages.append(ChatMessage(role=Role.ASSISTANT, content=content))
        self.generating = False
        self.last_activity_at = time.monotonic()
        
        # Trim history if too long
        self._trim_history()

    def _trim_history(self) -> None:
        """Trim conversation history to max turns."""
        max_messages = settings.max_history_turns * 2  # User + assistant pairs
        
        # Keep system message if present
        has_system = (
            len(self.messages) > 0 and self.messages[0].role == Role.SYSTEM
        )
        
        if has_system:
            max_messages += 1
            start_idx = 1
        else:
            start_idx = 0
        
        if len(self.messages) > max_messages:
            system_msg = self.messages[0] if has_system else None
            # Keep most recent messages
            self.messages = self.messages[-(max_messages - (1 if has_system else 0)):]
            if system_msg:
                self.messages.insert(0, system_msg)
            
            logger.debug(
                f"Session {self.session_id}: Trimmed history to {len(self.messages)} messages"
            )

    def get_messages_for_generation(self) -> list[dict[str, str]]:
        """
        Get messages formatted for vLLM generation.
        
        Returns:
            List of message dicts with role and content
        """
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in self.messages
        ]

    def cancel(self) -> str | None:
        """
        Cancel current generation.
        
        Returns:
            Request ID that was cancelled, or None if not generating
        """
        if not self.generating:
            return None
        
        self.cancelled = True
        self.generating = False
        request_id = self.current_request_id
        logger.info(f"Session {self.session_id}: Cancelled generation {request_id}")
        return request_id

    def is_request_valid(self, request_id: str) -> bool:
        """Check if request is still valid (not cancelled or superseded)."""
        return (
            not self.cancelled
            and self.generating
            and self.current_request_id == request_id
        )

    def record_generation_metrics(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        ttft_ms: float,
        total_time_ms: float,
    ) -> None:
        """Record metrics from completed generation."""
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_ttft_ms += ttft_ms
        self.total_generation_time_ms += total_time_ms

    def get_generation_config(self) -> dict:
        """Get generation parameters, using session config or defaults."""
        config = self.config.generation
        return {
            "max_tokens": config.max_tokens or settings.max_tokens,
            "temperature": config.temperature if config.temperature is not None else settings.temperature,
            "top_p": config.top_p if config.top_p is not None else settings.top_p,
            "top_k": config.top_k if config.top_k is not None else settings.top_k,
            "repetition_penalty": config.repetition_penalty or settings.repetition_penalty,
            "stop": config.stop,
        }

    def get_metrics(self) -> dict:
        """Get session-level metrics."""
        session_duration = time.monotonic() - self.created_at
        
        avg_ttft = (
            self.total_ttft_ms / self.request_count
            if self.request_count > 0 else 0
        )
        
        avg_tokens_per_second = (
            self.total_completion_tokens / (self.total_generation_time_ms / 1000)
            if self.total_generation_time_ms > 0 else 0
        )
        
        return {
            "total_requests": self.request_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "avg_ttft_ms": round(avg_ttft, 2),
            "avg_tokens_per_second": round(avg_tokens_per_second, 2),
            "session_duration_seconds": round(session_duration, 2),
        }

    @property
    def is_expired(self) -> bool:
        """Check if session has timed out."""
        idle_time = time.monotonic() - self.last_activity_at
        return idle_time > settings.session_timeout_seconds


class SessionManager:
    """
    Manages active chat sessions.
    
    Thread-safe session tracking with limits and cleanup.
    """

    def __init__(self, max_sessions: int | None = None):
        """
        Initialize session manager.
        
        Args:
            max_sessions: Maximum concurrent sessions
        """
        self.max_sessions = max_sessions or settings.max_sessions
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None

    def start(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Session cleanup task started")

    async def stop(self) -> None:
        """Stop background cleanup task."""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("Session cleanup task stopped")

    async def _cleanup_loop(self) -> None:
        """Periodically clean up expired sessions."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in session cleanup: {e}")

    async def _cleanup_expired_sessions(self) -> None:
        """Remove expired sessions."""
        async with self._lock:
            expired = [
                session_id
                for session_id, session in self._sessions.items()
                if session.is_expired
            ]
            
            for session_id in expired:
                del self._sessions[session_id]
                logger.info(f"Session {session_id} expired and removed")
            
            if expired:
                logger.info(f"Cleaned up {len(expired)} expired sessions")

    async def create_session(
        self,
        session_id: str,
        websocket: WebSocket,
        config: SessionConfig,
    ) -> Session:
        """
        Create a new chat session.
        
        Args:
            session_id: Unique session identifier
            websocket: WebSocket connection
            config: Session configuration
            
        Returns:
            New Session instance
            
        Raises:
            SessionExistsError: If session already exists
            MaxSessionsReachedError: If at capacity
        """
        async with self._lock:
            if session_id in self._sessions:
                raise SessionExistsError(session_id)

            if len(self._sessions) >= self.max_sessions:
                raise MaxSessionsReachedError(self.max_sessions)

            session = Session(
                session_id=session_id,
                websocket=websocket,
                config=config,
            )
            self._sessions[session_id] = session
            
            logger.info(
                f"Session {session_id} created (total: {len(self._sessions)})"
            )
            return session

    async def get_session(self, session_id: str) -> Session | None:
        """Get session by ID."""
        return self._sessions.get(session_id)

    async def remove_session(self, session_id: str) -> Session | None:
        """
        Remove and return session.
        
        Args:
            session_id: Session to remove
            
        Returns:
            Removed session or None if not found
        """
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if session:
                logger.info(
                    f"Session {session_id} removed (total: {len(self._sessions)})"
                )
            return session

    @property
    def active_sessions(self) -> int:
        """Number of active sessions."""
        return len(self._sessions)

    def get_all_sessions(self) -> list[Session]:
        """Get all active sessions."""
        return list(self._sessions.values())


# Global instance
session_manager = SessionManager()
