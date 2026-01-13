"""
Prometheus metrics endpoint for the LLM service.

Exposes /metrics endpoint for Prometheus scraping.
"""

import logging

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def metrics() -> PlainTextResponse:
    """
    Prometheus metrics endpoint.
    
    Returns all registered Prometheus metrics in exposition format.
    """
    return PlainTextResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
