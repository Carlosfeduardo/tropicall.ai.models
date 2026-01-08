"""
Main application entry point for Kokoro TTS Microservice.

Initializes:
- FastAPI application
- Kokoro worker with warmup
- Inference queue with single consumer
- Health and metrics endpoints
"""

import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from .api import health, metrics, websocket
from .config import settings
from .inference import inference_queue as iq_module
from .inference.kokoro_worker import KokoroWorker
from .inference.inference_queue import InferenceQueue
from .observability.logging import configure_logging
from .observability.metrics import (
    tts_active_sessions,
    tts_queue_depth,
    update_gpu_metrics,
)
from .core.session import session_manager

logger = logging.getLogger(__name__)


async def update_metrics_loop() -> None:
    """Background task to update metrics periodically."""
    while True:
        try:
            # Update queue depth
            if iq_module.inference_queue is not None:
                tts_queue_depth.set(iq_module.inference_queue.depth)
            
            # Update active sessions
            tts_active_sessions.set(session_manager.active_sessions)
            
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
    2. Create and warm up Kokoro worker
    3. Create and start inference queue
    4. Start background metrics task
    
    Shutdown:
    1. Stop metrics task
    2. Stop inference queue
    """
    # Configure logging
    configure_logging()
    logger.info("Starting Kokoro TTS Microservice...")
    
    # 1. Create worker and warm up
    logger.info("Creating Kokoro worker...")
    worker = KokoroWorker(
        lang_code=settings.lang_code,
        repo_id=settings.model_repo,
    )
    
    logger.info("Warming up Kokoro worker (this may take a minute)...")
    await asyncio.to_thread(worker.warmup)
    logger.info("Kokoro worker ready")
    
    # 2. Create and start inference queue
    logger.info("Starting inference queue...")
    iq_module.inference_queue = InferenceQueue(worker)
    iq_module.inference_queue.start()
    
    # 3. Mark startup complete
    health.mark_startup_complete()
    
    # 4. Start background metrics task
    metrics_task = asyncio.create_task(update_metrics_loop())
    
    logger.info(
        f"Kokoro TTS ready - listening on {settings.host}:{settings.port}"
    )
    
    yield
    
    # Shutdown
    logger.info("Shutting down Kokoro TTS Microservice...")
    
    # Stop metrics task
    metrics_task.cancel()
    try:
        await metrics_task
    except asyncio.CancelledError:
        pass
    
    # Stop inference queue
    if iq_module.inference_queue is not None:
        await iq_module.inference_queue.stop()
    
    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Kokoro TTS Microservice",
    description="Real-time TTS with Kokoro-82M for Brazilian Portuguese",
    version="0.1.0",
    lifespan=lifespan,
)

# Include routers
app.include_router(websocket.router)
app.include_router(health.router)
app.include_router(metrics.router)


@app.get("/")
async def root() -> dict:
    """Root endpoint with service info."""
    return {
        "service": "kokoro-tts",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health/ready",
        "websocket": "/ws/tts",
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
