# Qwen3-Reranker Multi-Backend Service - CUDA Dockerfile
# Optimized for Lambda Cloud and NVIDIA GPU deployment

# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Install PyTorch with CUDA support first (for better caching)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Install Flash Attention for faster inference
RUN pip install --no-cache-dir flash-attn==2.7.2.post1 --no-build-isolation

# Copy project files
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY config/ ./config/

# Install the package with CUDA dependencies
RUN pip install --no-cache-dir -e ".[cuda]"

# Production image
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install Python runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application
WORKDIR /app
COPY --from=builder /app/src ./src
COPY --from=builder /app/config ./config

# Set environment variables for Qwen3 Reranker
ENV QWEN_RERANK_HOST=0.0.0.0
ENV QWEN_RERANK_PORT=9003
ENV QWEN_RERANK_BACKEND=pytorch
ENV QWEN_RERANK_PROFILE=qwen3_4b_cuda
ENV QWEN_RERANK_LOG_LEVEL=INFO
ENV QWEN_RERANK_LOG_FORMAT=json

# CUDA memory configuration
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Trust remote code for Qwen models
ENV TRANSFORMERS_TRUST_REMOTE_CODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:9003/health || exit 1

# Expose port
EXPOSE 9003

# Run the service
CMD ["python", "-m", "qwen3_reranker.api.app"]
