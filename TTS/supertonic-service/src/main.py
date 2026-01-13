"""
Main application entry point for Supertonic TTS Microservice.

Initializes:
- FastAPI application
- Supertonic worker with warmup
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
from .inference.supertonic_worker import SupertonicWorker
from .inference.inference_queue import InferenceQueue
from .observability.logging import configure_logging
from .observability.metrics import (
    tts_active_sessions,
    tts_queue_depth,
    update_gpu_metrics,
    update_queue_metrics,
)
from .core.session import session_manager

logger = logging.getLogger(__name__)


async def update_metrics_loop() -> None:
    """Background task to update metrics periodically."""
    while True:
        try:
            # Update queue metrics
            if iq_module.inference_queue is not None:
                queue = iq_module.inference_queue
                tts_queue_depth.set(queue.depth)
                update_queue_metrics(
                    queue_depth=queue.depth,
                    estimated_server_total_ms=queue.estimated_server_total_ms,
                    avg_processing_ms=queue.avg_processing_ms,
                )
            
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
    2. Create and warm up Supertonic worker
    3. Create and start inference queue
    4. Start background metrics task
    
    Shutdown:
    1. Stop metrics task
    2. Stop inference queue
    """
    # Configure logging
    configure_logging()
    logger.info("Starting Supertonic TTS Microservice...")
    
    # 1. Create worker and warm up
    logger.info("Creating Supertonic worker...")
    worker = SupertonicWorker(
        default_voice=settings.default_voice,
        model_dir=settings.model_path,
        num_inference_steps=settings.num_inference_steps,
    )
    
    logger.info("Warming up Supertonic worker (this may take a minute)...")
    await asyncio.to_thread(worker.warmup)
    logger.info("Supertonic worker ready")
    
    # 2. Create and start inference queue with admission control settings
    logger.info("Starting inference queue...")
    iq_module.inference_queue = InferenceQueue(
        worker=worker,
        max_queue_depth=settings.max_queue_depth,
        max_estimated_server_total_ms=settings.max_estimated_server_total_ms,
    )
    iq_module.inference_queue.start()
    logger.info(
        f"Admission control (server-only SLO): "
        f"max_queue_depth={settings.max_queue_depth}, "
        f"max_estimated_server_total_ms={settings.max_estimated_server_total_ms}"
    )
    
    # 3. Mark startup complete
    health.mark_startup_complete()
    
    # 4. Start background metrics task
    metrics_task = asyncio.create_task(update_metrics_loop())
    
    logger.info(
        f"Supertonic TTS ready - listening on {settings.host}:{settings.port}"
    )
    
    yield
    
    # Shutdown
    logger.info("Shutting down Supertonic TTS Microservice...")
    
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
    title="Supertonic TTS Microservice",
    description="Real-time TTS with Supertonic 2 - Lightning Fast Multilingual TTS",
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
        "service": "supertonic-tts",
        "model": "supertonic-2",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health/ready",
        "websocket": "/ws/tts",
        "languages": ["en", "ko", "es", "pt", "fr"],
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
