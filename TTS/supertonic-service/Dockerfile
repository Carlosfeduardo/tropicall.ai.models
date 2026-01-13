# ============================================
# Supertonic 2 TTS Microservice Dockerfile
# Multi-stage build optimized for GPU inference
# ============================================

# ============================================
# Stage 1: Builder - Download model + deps
# ============================================
# Using CUDA 12.6 with cuDNN 9 (compatible with RunPod drivers)
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04 AS builder

WORKDIR /build

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    libsndfile1-dev git curl git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Setup Python
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
# CRITICAL: Install onnxruntime-gpu FIRST, then supertonic without its onnxruntime dep
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip wheel && \
    # Install onnxruntime-gpu first (provides CUDA support)
    pip install --no-cache-dir onnxruntime-gpu>=1.17.0 && \
    # Install supertonic without its onnxruntime dependency (we already have GPU version)
    pip install --no-cache-dir --no-deps supertonic>=1.0.0 && \
    # Install remaining dependencies from requirements (excluding onnxruntime packages)
    pip install --no-cache-dir \
        "fastapi>=0.109.0,<0.129.0" \
        "uvicorn[standard]>=0.27.0" \
        "websockets>=12.0" \
        "pydantic>=2.5.0,<3.0.0" \
        "pydantic-settings>=2.1.0,<3.0.0" \
        "huggingface-hub>=0.20.0,<0.30.0" \
        "numpy>=1.24.0,<2.0.0" \
        "soundfile>=0.12.0" \
        "prometheus-client>=0.19.0" \
        "structlog>=24.1.0" \
        "httpx>=0.26.0" && \
    # Verify onnxruntime is properly installed
    python -c "import onnxruntime as ort; print('ONNX Runtime version:', ort.__version__); print('Available providers:', ort.get_available_providers())"

# Download model weights at build time (cached in layer)
# Supertonic uses its own cache at /root/.cache/supertonic
# The package downloads from Supertone/supertonic (not supertonic-2)

# Pre-run warmup to download model and optimize ONNX Runtime
# This will download the model to /root/.cache/supertonic
RUN python -c "\
from supertonic import TTS; \
tts = TTS(auto_download=True); \
style = tts.get_voice_style('F1'); \
_ = tts.synthesize('Initialization test.', voice_style=style, total_steps=5)"

# ============================================
# Stage 2: Runtime
# ============================================
# Using CUDA 12.6 with cuDNN 9 (compatible with RunPod drivers)
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04 AS runtime

WORKDIR /app

# Runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv \
    libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python environment and Supertonic model cache
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /root/.cache/supertonic /root/.cache/supertonic

# Set Python path
ENV PATH="/opt/venv/bin:$PATH"

# Force offline mode for HuggingFace (model is pre-cached)
ENV HF_HUB_OFFLINE=1

# Copy application
COPY src/ ./src/
COPY pyproject.toml .

# Environment defaults for Supertonic 2
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV TTS_HOST=0.0.0.0
ENV TTS_PORT=8000
ENV TTS_MAX_SESSIONS=100
ENV TTS_MAX_INFLIGHT_SEGMENTS=2
# Supertonic 2 default voice (F1-F5 female, M1-M5 male)
ENV TTS_DEFAULT_VOICE=F1
ENV TTS_NUM_INFERENCE_STEPS=5
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
