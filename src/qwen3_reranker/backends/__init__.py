"""Backend implementations for Qwen3-Reranker.

This module provides the backend abstraction layer and implementations:
- PyTorchBackend: PRIMARY - Cross-platform CUDA/MPS/CPU
- VLLMBackend: SECONDARY - High-throughput CUDA
- MLXBackend: TERTIARY - Apple Silicon optimization
"""

from qwen3_reranker.backends.base import BackendCapabilities, RerankerBackend
from qwen3_reranker.backends.registry import (
    BACKEND_PRIORITY,
    detect_available_backends,
    get_backend,
    get_model_id_for_backend,
)

__all__ = [
    "RerankerBackend",
    "BackendCapabilities",
    "BACKEND_PRIORITY",
    "detect_available_backends",
    "get_backend",
    "get_model_id_for_backend",
]
