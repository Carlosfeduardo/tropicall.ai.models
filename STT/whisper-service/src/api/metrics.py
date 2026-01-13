"""
Prometheus metrics endpoint for the STT service.

Provides /metrics endpoint for Prometheus scraping.
"""

import logging

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["metrics"])


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> PlainTextResponse:
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus text format.
    """
    if not settings.prometheus_enabled:
        return PlainTextResponse(
            content="# Prometheus metrics disabled",
            media_type="text/plain",
        )
    
    return PlainTextResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
