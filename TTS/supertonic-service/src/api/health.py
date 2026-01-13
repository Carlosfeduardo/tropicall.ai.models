"""
Health check endpoints for the TTS service.

Provides:
- /health/ready - Readiness probe (model loaded and ready)
- /health/live - Liveness probe (service is running)
- /health/startup - Startup probe (service is starting)
"""

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, status

from ..core.session import session_manager
from ..inference.inference_queue import get_inference_queue

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
    - Model is loaded
    - Inference queue is running
    - Startup is complete
    
    Returns 503 if not ready.
    """
    try:
        queue = get_inference_queue()
        
        if not queue.is_running:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference queue not running",
            )
        
        if not _startup_complete:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Startup not complete",
            )
        
        return {
            "status": "ready",
            "queue_depth": queue.depth,
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
    
    queue_status = {
        "running": False,
        "depth": 0,
    }
    
    try:
        queue = get_inference_queue()
        queue_status = {
            "running": queue.is_running,
            "depth": queue.depth,
        }
    except RuntimeError:
        pass
    
    return {
        "status": ready_status,
        "startup_complete": _startup_complete,
        "inference_queue": queue_status,
        "active_sessions": session_manager.active_sessions,
        "uptime_seconds": round(time.time() - _startup_time, 1) if _startup_time else 0,
    }


@router.get("/status")
async def status() -> dict[str, Any]:
    """
    Detailed status endpoint for capacity monitoring.
    
    Returns queue metrics, SLO status, and capacity information.
    Used by load balancers and monitoring systems.
    """
    from ..config import settings
    
    queue_stats = {
        "depth": 0,
        "avg_inference_ms": 0.0,
        "estimated_wait_ms": 0.0,
        "max_queue_depth": settings.max_queue_depth,
        "max_estimated_wait_ms": settings.max_estimated_wait_ms,
        "accepting_requests": False,
        "total_requests": 0,
        "rejected_requests": 0,
    }
    
    try:
        queue = get_inference_queue()
        queue_stats = queue.get_stats()
    except RuntimeError:
        pass
    
    # Calculate estimated capacity based on SLO
    # max_talkers ≈ SLO_TTFA_P95 / avg_inference_time
    avg_inference_ms = queue_stats.get("avg_inference_ms", 80.0)
    estimated_capacity = int(settings.slo_ttfa_p95_ms / avg_inference_ms) if avg_inference_ms > 0 else 0
    
    # Current load as percentage of capacity
    current_depth = queue_stats.get("depth", 0)
    load_percent = (current_depth / estimated_capacity * 100) if estimated_capacity > 0 else 0
    
    return {
        "status": "ready" if _startup_complete else "not_ready",
        "accepting_requests": queue_stats.get("accepting_requests", False),
        "queue": queue_stats,
        "sessions": {
            "active": session_manager.active_sessions,
            "max": settings.max_sessions,
        },
        "slo": {
            "ttfa_p95_target_ms": settings.slo_ttfa_p95_ms,
            "ttfa_p99_target_ms": settings.slo_ttfa_p99_ms,
            "rtf_max_target": settings.slo_rtf_max,
        },
        "capacity": {
            "estimated_max_talkers": estimated_capacity,
            "current_load_percent": round(load_percent, 1),
        },
        "uptime_seconds": round(time.time() - _startup_time, 1) if _startup_time else 0,
    }
