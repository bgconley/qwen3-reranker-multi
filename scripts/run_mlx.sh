#!/bin/bash
# Run Qwen3-Reranker with MLX backend
# For Apple Silicon local development

set -e

# Set environment
export QWEN_RERANK_BACKEND=mlx
export QWEN_RERANK_PROFILE=qwen3_4b_mlx_fp16
export QWEN_RERANK_HOST=127.0.0.1
export QWEN_RERANK_PORT=${QWEN_RERANK_PORT:-9003}
export QWEN_RERANK_LOG_LEVEL=${QWEN_RERANK_LOG_LEVEL:-INFO}

echo "Starting Qwen3-Reranker with MLX backend..."
echo "Profile: $QWEN_RERANK_PROFILE"
echo "Port: $QWEN_RERANK_PORT"

python -m qwen3_reranker.api.app
