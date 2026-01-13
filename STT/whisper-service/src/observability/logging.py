"""
Logging configuration for the STT service.

Configures structured logging with appropriate levels and formats.
"""

import logging
import sys

from ..config import settings


def configure_logging() -> None:
    """
    Configure logging for the application.
    
    Sets up structured logging with appropriate format for
    development and production environments.
    """
    # Determine log level
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    # Reduce noise from external libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    
    # Set our logger to configured level
    logging.getLogger("src").setLevel(log_level)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured at level: {settings.log_level}")
