"""
Health check endpoints for the LLM service.

Provides:
- /health/ready - Readiness probe (model loaded and ready)
- /health/live - Liveness probe (service is running)
- /health/startup - Startup probe (service is starting)
- /health/status - Detailed status for monitoring
"""

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, status

from ..config import settings
from ..core.session import session_manager
from ..inference.request_handler import get_request_handler
from ..inference.vllm_engine import get_vllm_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])

# Startup state
_startup_complete = False
_startup_time: float | None = None


def mark_startup_complete() -> None:
    """Mark startup as complete (called after warmup)."""
    global _startup_complete, _startup_time
    _startup_complete = True
    _startup_time = time.time()
    logger.info("Startup marked as complete")


def is_startup_complete() -> bool:
    """Check if startup is complete."""
    return _startup_complete


@router.get("/ready")
async def readiness() -> dict[str, Any]:
    """
    Readiness probe.
    
    Returns 200 if the service is ready to accept requests:
    - vLLM engine is loaded and ready
    - Request handler is initialized
    - Startup is complete
    
    Returns 503 if not ready.
    """
    try:
        engine = get_vllm_engine()
        
        if not engine.is_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="vLLM engine not ready",
            )
        
        if not _startup_complete:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Startup not complete",
            )
        
        handler = get_request_handler()
        
        return {
            "status": "ready",
            "active_requests": engine.active_requests,
            "pending_requests": handler.pending_count,
            "active_sessions": session_manager.active_sessions,
            "uptime_seconds": round(time.time() - _startup_time, 1) if _startup_time else 0,
        }
        
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )


@router.get("/live")
async def liveness() -> dict[str, str]:
    """
    Liveness probe.
    
    Returns 200 if the service is alive.
    This is a simple check that the process is running.
    """
    return {"status": "alive"}


@router.get("/startup")
async def startup() -> dict[str, Any]:
    """
    Startup probe.
    
    Returns 200 if startup is complete, 503 otherwise.
    Used by Kubernetes to know when the container is ready.
    """
    if not _startup_complete:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Startup in progress",
        )
    
    return {
        "status": "started",
        "startup_time": _startup_time,
    }


@router.get("/")
async def health_root() -> dict[str, Any]:
    """
    Combined health check.
    
    Returns full health status including all checks.
    """
    ready_status = "ready" if _startup_complete else "not_ready"
    
    engine_status = {
        "ready": False,
        "active_requests": 0,
    }
    
    handler_status = {
        "accepting_requests": False,
        "pending_count": 0,
        "active_count": 0,
    }
    
    try:
        engine = get_vllm_engine()
        engine_status = {
            "ready": engine.is_ready,
            "active_requests": engine.active_requests,
        }
    except RuntimeError:
        pass
    
    try:
        handler = get_request_handler()
        handler_status = {
            "accepting_requests": handler.accepting_requests,
            "pending_count": handler.pending_count,
            "active_count": handler.active_count,
        }
    except RuntimeError:
        pass
    
    return {
        "status": ready_status,
        "startup_complete": _startup_complete,
        "engine": engine_status,
        "request_handler": handler_status,
        "active_sessions": session_manager.active_sessions,
        "uptime_seconds": round(time.time() - _startup_time, 1) if _startup_time else 0,
    }


@router.get("/status")
async def detailed_status() -> dict[str, Any]:
    """
    Detailed status endpoint for capacity monitoring.
    
    Returns engine stats, request handler stats, and capacity information.
    Used by load balancers and monitoring systems.
    """
    engine_stats = {
        "model_id": settings.model_id,
        "ready": False,
        "active_requests": 0,
        "max_model_len": settings.max_model_len,
    }
    
    handler_stats = {
        "pending_count": 0,
        "active_count": 0,
        "max_queue_depth": settings.max_queue_depth,
        "max_concurrent": settings.max_concurrent_requests,
        "accepting_requests": False,
    }
    
    try:
        engine = get_vllm_engine()
        engine_stats = engine.get_stats()
    except RuntimeError:
        pass
    
    try:
        handler = get_request_handler()
        handler_stats = handler.get_stats()
    except RuntimeError:
        pass
    
    return {
        "status": "ready" if _startup_complete else "not_ready",
        "accepting_requests": handler_stats.get("accepting_requests", False),
        "engine": engine_stats,
        "request_handler": handler_stats,
        "sessions": {
            "active": session_manager.active_sessions,
            "max": settings.max_sessions,
        },
        "slo": {
            "ttft_p95_target_ms": settings.slo_ttft_p95_ms,
            "tokens_per_second_min": settings.slo_tokens_per_second_min,
        },
        "uptime_seconds": round(time.time() - _startup_time, 1) if _startup_time else 0,
    }
