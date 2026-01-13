"""
Logging configuration for the LLM service.

Uses structlog for structured JSON logging in production.
"""

import logging
import sys
from typing import Any

import structlog

from ..config import settings


def configure_logging() -> None:
    """
    Configure structured logging for the application.
    
    Uses structlog with JSON output for production, pretty output for development.
    """
    # Determine if we're in development mode
    is_dev = settings.log_level == "DEBUG"
    
    # Configure structlog processors
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.ExtraAdder(),
    ]
    
    if is_dev:
        # Development: pretty console output
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        # Production: JSON output
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level),
    )
    
    # Set log levels for noisy libraries
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    # vLLM can be quite verbose
    logging.getLogger("vllm").setLevel(logging.INFO)
    
    logger = structlog.get_logger()
    logger.info(
        "Logging configured",
        level=settings.log_level,
        format="json" if not is_dev else "console",
    )
