"""
Main application entry point for Qwen3 LLM Microservice.

Initializes:
- FastAPI application
- vLLM engine with warmup
- Request handler
- Session manager
- Health and metrics endpoints
"""

import asyncio
import importlib
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import health, metrics, openai_compat, websocket
from .config import settings
from .core.session import session_manager
from .inference.request_handler import RequestHandler
from .inference.vllm_engine import VLLMEngine
from .observability.logging import configure_logging
from .observability.metrics import (
    update_gpu_metrics,
    update_queue_metrics,
    update_session_metrics,
)

# Import modules (not variables) to allow setting global instances
engine_module = importlib.import_module('.inference.vllm_engine', __package__)
rh_module = importlib.import_module('.inference.request_handler', __package__)

logger = logging.getLogger(__name__)


async def update_metrics_loop() -> None:
    """Background task to update metrics periodically."""
    while True:
        try:
            # Update request handler metrics
            if rh_module.request_handler is not None:
                handler = rh_module.request_handler
                update_queue_metrics(
                    pending_count=handler.pending_count,
                    active_count=handler.active_count,
                    estimated_wait_ms=handler.estimated_wait_ms,
                    avg_ttft_ms=handler._avg_ttft_ms,
                )
            
            # Update session metrics
            update_session_metrics(session_manager.active_sessions)
            
            # Update GPU metrics
            update_gpu_metrics()
            
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.exception(f"Error updating metrics: {e}")
            await asyncio.sleep(10)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Startup:
    1. Configure logging
    2. Create and warm up vLLM engine
    3. Create request handler
    4. Start session manager
    5. Start background metrics task
    
    Shutdown:
    1. Stop metrics task
    2. Stop session manager
    3. Stop vLLM engine
    """
    # Configure logging
    configure_logging()
    logger.info("Starting Qwen3 LLM Microservice...")
    
    # 1. Create vLLM engine and warm up
    logger.info(f"Creating vLLM engine with model: {settings.model_id}")
    engine_module.vllm_engine = VLLMEngine()
    
    logger.info("Starting vLLM engine (this may take several minutes)...")
    await engine_module.vllm_engine.start()
    logger.info("vLLM engine ready")
    
    # 2. Create request handler
    logger.info("Creating request handler...")
    rh_module.request_handler = RequestHandler(
        max_queue_depth=settings.max_queue_depth,
        max_concurrent=settings.max_concurrent_requests,
    )
    logger.info(
        f"Request handler ready: max_queue={settings.max_queue_depth}, "
        f"max_concurrent={settings.max_concurrent_requests}"
    )
    
    # 3. Start session manager
    session_manager.start()
    
    # 4. Mark startup complete
    health.mark_startup_complete()
    
    # 5. Start background metrics task
    metrics_task = asyncio.create_task(update_metrics_loop())
    
    logger.info(
        f"Qwen3 LLM ready - listening on {settings.host}:{settings.port}"
    )
    
    yield
    
    # Shutdown
    logger.info("Shutting down Qwen3 LLM Microservice...")
    
    # Stop metrics task
    metrics_task.cancel()
    try:
        await metrics_task
    except asyncio.CancelledError:
        pass
    
    # Stop session manager
    await session_manager.stop()
    
    # Stop vLLM engine
    if engine_module.vllm_engine is not None:
        await engine_module.vllm_engine.stop()
    
    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Qwen3 LLM Microservice",
    description="Real-time LLM inference with Qwen3-32B-Instruct via vLLM",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(websocket.router)
app.include_router(openai_compat.router)
app.include_router(health.router)
app.include_router(metrics.router)


@app.get("/")
async def root() -> dict:
    """Root endpoint with service info."""
    return {
        "service": "qwen3-llm",
        "version": "0.1.0",
        "model": settings.model_id,
        "status": "running",
        "docs": "/docs",
        "health": "/health/ready",
        "websocket": "/ws/chat",
        "openai_api": "/v1/chat/completions",
        "metrics": "/metrics",
    }


def run() -> None:
    """Run the application with uvicorn."""
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        workers=1,  # Single worker - GPU not shareable
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
