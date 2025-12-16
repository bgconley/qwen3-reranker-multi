"""Qwen3-Reranker Multi-Backend Service.

A high-performance reranking service using Qwen3-Reranker-4B with support
for multiple inference backends:
- PyTorch (PRIMARY): Cross-platform CUDA/MPS/CPU support
- vLLM (SECONDARY): High-throughput CUDA deployment
- MLX (TERTIARY): Apple Silicon optimization
"""

from qwen3_reranker.version import __version__

__all__ = ["__version__"]
