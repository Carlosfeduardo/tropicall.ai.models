"""
Structured logging configuration using structlog.
"""

import logging
import sys
from typing import Any

import structlog

from ..config import settings


def configure_logging() -> None:
    """
    Configure structured logging for the application.
    
    Uses structlog for structured, JSON-formatted logs in production
    and human-readable logs in development.
    """
    # Determine if we're in development mode
    is_dev = settings.log_level == "DEBUG"
    
    # Configure timestamper
    timestamper = structlog.processors.TimeStamper(fmt="iso")
    
    # Shared processors
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    
    if is_dev:
        # Development: human-readable output
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        # Production: JSON output
        processors = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    log_level = getattr(logging, settings.log_level)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add stream handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    
    if is_dev:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    else:
        formatter = logging.Formatter("%(message)s")
    
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    
    # Set levels for noisy loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)
