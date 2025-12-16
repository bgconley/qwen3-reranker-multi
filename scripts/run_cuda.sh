#!/bin/bash
# Run Qwen3-Reranker with PyTorch CUDA backend
# Recommended for Lambda Cloud and NVIDIA GPU deployment

set -e

# Set environment
export QWEN_RERANK_BACKEND=pytorch
export QWEN_RERANK_PROFILE=qwen3_4b_cuda
export QWEN_RERANK_HOST=0.0.0.0
export QWEN_RERANK_PORT=${QWEN_RERANK_PORT:-9003}
export QWEN_RERANK_LOG_LEVEL=${QWEN_RERANK_LOG_LEVEL:-INFO}

# CUDA memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting Qwen3-Reranker with PyTorch CUDA backend..."
echo "Profile: $QWEN_RERANK_PROFILE"
echo "Port: $QWEN_RERANK_PORT"

python -m qwen3_reranker.api.app
