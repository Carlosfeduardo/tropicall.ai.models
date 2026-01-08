# ============================================
# Kokoro TTS Microservice Dockerfile
# Multi-stage build optimized for GPU inference
# ============================================

# ============================================
# Stage 1: Builder - Download model + deps
# ============================================
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel AS builder

WORKDIR /build

# System deps for espeak-ng (G2P)
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng libsndfile1-dev git curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip wheel && \
    pip install --no-cache-dir -r requirements.txt

# Download model weights at build time (cached in layer)
# IMPORTANTE: Use SHA/commit (NOT "main") for reproducibility
# Get SHA from: https://huggingface.co/hexgrad/Kokoro-82M/commits/main
ARG HF_MODEL_REVISION="main"
ENV HF_HOME="/root/.cache/huggingface"

RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('hexgrad/Kokoro-82M', revision='${HF_MODEL_REVISION}')"

# Pre-compile model to reduce TTFA on first request
RUN python -c "\
from kokoro import KPipeline; \
p = KPipeline(lang_code='p', repo_id='hexgrad/Kokoro-82M'); \
_ = list(p('Teste de inicialização.', voice='pf_dora'))"

# ============================================
# Stage 2: Runtime
# ============================================
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS runtime

WORKDIR /app

# Runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

# IMPORTANTE: Copy /opt/conda INTEIRO (not just site-packages)
# Copying only site-packages can fail due to native libs/paths
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Force offline mode for HuggingFace and Transformers
ENV HF_HOME="/root/.cache/huggingface"
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Copy application
COPY src/ ./src/
COPY pyproject.toml .

# Environment defaults
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV TTS_HOST=0.0.0.0
ENV TTS_PORT=8000
ENV TTS_MAX_SESSIONS=100
ENV TTS_MAX_INFLIGHT_SEGMENTS=2
ENV TTS_DEFAULT_VOICE=pf_dora
ENV TTS_LANG_CODE=p
ENV TTS_CHUNK_SIZE_MS=20
ENV TTS_DEBOUNCE_MS=150
ENV TTS_LOG_LEVEL=INFO

# Health check - verify model is loaded
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health/ready || exit 1

EXPOSE 8000

# Startup: warmup + uvicorn
# Single worker - GPU not shareable between processes
CMD ["python", "-m", "src.main"]
