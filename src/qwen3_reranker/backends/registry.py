"""Backend Registry - Auto-detection with PyTorch-first priority.

Priority order:
1. PyTorch (CUDA > MPS > CPU) - Cross-platform, mature ecosystem
2. vLLM (CUDA only) - High-throughput production workloads
3. MLX (Apple Silicon only) - Optional 2-3x speedup on Mac
"""

import logging
import platform
from typing import Any

from qwen3_reranker.backends.base import RerankerBackend

logger = logging.getLogger(__name__)

# PyTorch is primary for cross-platform compatibility
BACKEND_PRIORITY = ["pytorch", "vllm", "mlx"]


def detect_platform() -> dict[str, Any]:
    """Detect current platform capabilities."""
    return {
        "os": platform.system(),
        "arch": platform.machine(),
        "processor": platform.processor(),
        "is_apple_silicon": (
            platform.system() == "Darwin" and platform.machine() == "arm64"
        ),
        "is_linux": platform.system() == "Linux",
        "is_windows": platform.system() == "Windows",
    }


def detect_available_backends() -> list[str]:
    """Detect which backends are available on this system.

    Returns backends in priority order (PyTorch first for cross-platform).
    """
    available: list[str] = []
    plat = detect_platform()

    # Check PyTorch first (PRIMARY - cross-platform)
    try:
        import torch

        available.append("pytorch")

        if torch.cuda.is_available():
            device = f"CUDA: {torch.cuda.get_device_name(0)}"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "MPS (Apple Silicon)"
        else:
            device = "CPU"
        logger.info(f"✓ PyTorch backend available ({device}) - PRIMARY")
    except ImportError:
        logger.warning("PyTorch not installed - primary backend unavailable")

    # Check vLLM (SECONDARY - CUDA high-throughput)
    try:
        import torch

        if torch.cuda.is_available():
            try:
                import vllm  # noqa: F401

                available.append("vllm")
                logger.info(
                    f"✓ vLLM backend available (CUDA: {torch.cuda.get_device_name(0)}) - SECONDARY"
                )
            except ImportError:
                logger.debug("vLLM not installed")
    except ImportError:
        pass

    # Check MLX (TERTIARY - Apple Silicon optimization)
    if plat["is_apple_silicon"]:
        try:
            import mlx.core  # noqa: F401

            available.append("mlx")
            logger.info("✓ MLX backend available (Apple Silicon) - TERTIARY")
        except ImportError:
            logger.debug("MLX not installed")

    if not available:
        logger.error(
            "No backends available! Install one of:\n"
            "  - torch (pip install torch) for PyTorch\n"
            "  - vllm (pip install vllm) for vLLM (CUDA only)\n"
            "  - mlx-lm (pip install mlx-lm) for MLX (Apple Silicon only)"
        )

    return available


def get_backend(
    backend_name: str | None = None,
    **kwargs: Any,
) -> RerankerBackend:
    """Get a backend instance.

    Args:
        backend_name: Explicit backend name (pytorch, vllm, mlx) or None for auto
        **kwargs: Backend-specific configuration

    Returns:
        Initialized (but not loaded) backend instance

    Raises:
        RuntimeError: If no backends are available
        ValueError: If requested backend is not available
    """
    available = detect_available_backends()

    if not available:
        raise RuntimeError(
            "No backends available. Install one of:\n"
            "  - torch (any platform)\n"
            "  - vllm (CUDA only)\n"
            "  - mlx-lm (Apple Silicon only)"
        )

    # Normalize backend name
    if backend_name:
        backend_name = backend_name.lower()
        if backend_name == "auto":
            backend_name = None

    if backend_name:
        # Explicit backend requested
        if backend_name not in available:
            raise ValueError(
                f"Backend '{backend_name}' not available.\n"
                f"Available backends: {available}\n"
                f"Install the required dependencies or choose a different backend."
            )
        selected = backend_name
        logger.info(f"Using explicitly requested backend: {selected}")
    else:
        # Auto-select based on priority
        selected = None
        for backend in BACKEND_PRIORITY:
            if backend in available:
                selected = backend
                break
        if selected is None:
            raise RuntimeError("No backends available after priority selection")
        logger.info(f"Auto-selected backend: {selected}")

    # Import and instantiate the selected backend
    if selected == "pytorch":
        from qwen3_reranker.backends.pytorch_backend import PyTorchBackend

        return PyTorchBackend(device=kwargs.get("device"))

    elif selected == "vllm":
        from qwen3_reranker.backends.vllm_backend import VLLMBackend

        return VLLMBackend(
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.8),
            max_model_len=kwargs.get("max_model_len"),
        )

    elif selected == "mlx":
        from qwen3_reranker.backends.mlx_backend import MLXBackend

        return MLXBackend()

    else:
        raise ValueError(f"Unknown backend: {selected}")


def get_model_id_for_backend(
    backend_name: str,
    base_model: str = "Qwen/Qwen3-Reranker-4B",
    quantization: str | None = None,
) -> str:
    """Get the appropriate model ID for a given backend.

    MLX uses converted models (Lipdog/), others use HF originals.

    Args:
        backend_name: Backend name (pytorch, vllm, mlx)
        base_model: Base HuggingFace model ID
        quantization: Optional quantization type (fp16, 8bit, 4bit)

    Returns:
        Model ID appropriate for the backend
    """
    if backend_name == "mlx":
        quant_suffix = {
            None: "fp16",
            "fp16": "fp16",
            "8bit": "8bit",
            "4bit": "4bit",
        }.get(quantization, "fp16")

        # Map Qwen/ to Lipdog/ for MLX
        if base_model.startswith("Qwen/"):
            model_name = base_model.replace("Qwen/", "")
            return f"Lipdog/{model_name}-mlx-{quant_suffix}"
        return base_model
    else:
        # PyTorch/vLLM use original HF models
        return base_model


def get_pytorch_device_priority() -> str:
    """Get the best available PyTorch device.

    Returns:
        Device string: "cuda", "mps", or "cpu"
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    except ImportError:
        return "cpu"
