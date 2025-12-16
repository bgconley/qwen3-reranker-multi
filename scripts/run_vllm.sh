#!/bin/bash
# Run Qwen3-Reranker with vLLM backend
# For high-throughput CUDA deployment

set -e

# Set environment
export QWEN_RERANK_BACKEND=vllm
export QWEN_RERANK_PROFILE=qwen3_4b_vllm
export QWEN_RERANK_HOST=0.0.0.0
export QWEN_RERANK_PORT=${QWEN_RERANK_PORT:-9003}
export QWEN_RERANK_LOG_LEVEL=${QWEN_RERANK_LOG_LEVEL:-INFO}

# vLLM configuration
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "Starting Qwen3-Reranker with vLLM backend..."
echo "Profile: $QWEN_RERANK_PROFILE"
echo "Port: $QWEN_RERANK_PORT"

python -m qwen3_reranker.api.app
