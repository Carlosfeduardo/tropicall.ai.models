# ============================================
# Qwen3 LLM Microservice Dockerfile
# Multi-stage build optimized for vLLM on H100
# Model is downloaded at runtime (not build-time)
# ============================================

# ============================================
# Stage 1: Builder - Install dependencies
# ============================================
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

WORKDIR /build

# Install Python and system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --no-cache-dir -U pip wheel setuptools

# Install PyTorch first (CUDA 12.4) - version compatible with vLLM 0.8.x
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Install vLLM 0.8.x with Qwen3 support
RUN pip install --no-cache-dir vllm>=0.8.0

# Install LMCache for KV cache CPU offloading
# Allows offloading KV cache to CPU RAM when GPU memory is insufficient
RUN pip install --no-cache-dir lmcache

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================================
# Stage 2: Runtime
# ============================================
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS runtime

WORKDIR /app

# Install Python runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.11 /usr/bin/python

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY src/ ./src/
COPY pyproject.toml .

# Environment defaults
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# HuggingFace settings
# Model will be downloaded at runtime to this cache dir
ENV HF_HOME="/root/.cache/huggingface"
# Set HF_TOKEN at runtime for gated models like Qwen3
# ENV HF_TOKEN=your_token_here

# LLM service configuration
ENV LLM_HOST=0.0.0.0
ENV LLM_PORT=8001
ENV LLM_MODEL_ID=openai/gpt-oss-120b
ENV LLM_MAX_MODEL_LEN=8192
ENV LLM_GPU_MEMORY_UTILIZATION=0.9
ENV LLM_MAX_SESSIONS=50
ENV LLM_MAX_TOKENS=2048
ENV LLM_ENABLE_THINKING=false
ENV LLM_LOG_LEVEL=INFO

# Health check - model loading takes time, so longer start period
# First startup will download model (~60GB), subsequent starts use cache
HEALTHCHECK --interval=15s --timeout=10s --start-period=600s --retries=5 \
    CMD curl -f http://localhost:8001/health/ready || exit 1

EXPOSE 8001

# Startup: run the service
# Single worker - GPU not shareable between processes
CMD ["python", "-m", "src.main"]
