# ============================================
# Whisper STT Microservice Dockerfile
# Multi-stage build optimized for GPU inference
# ============================================

# ============================================
# Stage 1: Builder - Download model + deps
# ============================================
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder

WORKDIR /build

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    git curl wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install Python dependencies
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -U pip wheel setuptools && \
    python -m pip install --no-cache-dir -r requirements.txt

# Download Whisper model at build time (cached in layer)
# Models: large-v3-turbo (fast multilingual), large-v3 (multilingual), distil-large-v3 (English), etc.
ARG WHISPER_MODEL=large-v3-turbo
ENV WHISPER_MODEL=${WHISPER_MODEL}

# Download model using huggingface_hub (no memory-intensive loading)
RUN python -c "\
from huggingface_hub import snapshot_download; \
model_map = { \
    'large-v3-turbo': 'deepdml/faster-whisper-large-v3-turbo-ct2', \
    'large-v3': 'Systran/faster-whisper-large-v3', \
    'large-v2': 'guillaumekln/faster-whisper-large-v2', \
    'medium': 'guillaumekln/faster-whisper-medium', \
    'small': 'guillaumekln/faster-whisper-small', \
    'base': 'guillaumekln/faster-whisper-base', \
    'tiny': 'guillaumekln/faster-whisper-tiny', \
    'distil-large-v3': 'Systran/faster-distil-whisper-large-v3', \
}; \
model_id = model_map.get('${WHISPER_MODEL}', 'deepdml/faster-whisper-large-v3-turbo-ct2'); \
print(f'Downloading model: ${WHISPER_MODEL} ({model_id})'); \
snapshot_download(model_id, local_files_only=False); \
print('Model downloaded successfully')"

# Download Silero VAD model
RUN python -c "\
import torch; \
print('Downloading Silero VAD model...'); \
torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False, onnx=False, trust_repo=True); \
print('Silero VAD downloaded successfully')"

# ============================================
# Stage 2: Runtime
# ============================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS runtime

WORKDIR /app

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Copy Python environment from builder
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/lib/python3.11 /usr/lib/python3.11

# Copy cached models
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface
COPY --from=builder /root/.cache/torch /root/.cache/torch

# Copy application
COPY src/ ./src/
COPY pyproject.toml .

# Environment defaults
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0

# STT service configuration
ENV STT_HOST=0.0.0.0
ENV STT_PORT=8000
ENV STT_MAX_SESSIONS=100
ENV STT_WHISPER_MODEL=large-v3-turbo
ENV STT_COMPUTE_TYPE=float16
ENV STT_DEVICE=cuda
ENV STT_LANGUAGE=pt
ENV STT_VAD_ENABLED=true
ENV STT_VAD_THRESHOLD=0.5
ENV STT_PARTIAL_INTERVAL_MS=300
ENV STT_MAX_BUFFER_MS=5000
ENV STT_MAX_QUEUE_DEPTH=20
ENV STT_LOG_LEVEL=INFO

# Force offline mode for HuggingFace
ENV HF_HOME="/root/.cache/huggingface"
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Health check - verify model is loaded
HEALTHCHECK --interval=10s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health/ready || exit 1

EXPOSE 8000

# Startup: uvicorn with single worker (GPU not shareable)
CMD ["python", "-m", "src.main"]
