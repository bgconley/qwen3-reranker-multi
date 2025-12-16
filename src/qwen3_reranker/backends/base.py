"""Backend abstraction layer for Qwen3-Reranker.

Defines the Protocol interface that all backends must implement.
PyTorch is the reference implementation; vLLM and MLX must produce
numerically equivalent scores (within tolerance).
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class RerankerBackend(Protocol):
    """Protocol defining the interface all backends must implement.

    Design Notes:
    - All backends must produce numerically equivalent scores (within tolerance)
    - PyTorch is the reference implementation; others are validated against it
    - Forward pass returns logits at final position, not scores
    - Scoring computation is backend-agnostic (see core/scoring.py)
    """

    @abstractmethod
    def load_model(self, model_id: str, **kwargs: Any) -> None:
        """Load model and tokenizer. Called once at startup.

        Args:
            model_id: HuggingFace model ID or local path
                     PyTorch/vLLM: "Qwen/Qwen3-Reranker-4B"
                     MLX: "Lipdog/Qwen3-Reranker-4B-mlx-fp16"
            **kwargs: Backend-specific options
        """
        ...

    @abstractmethod
    def get_tokenizer(self) -> Any:
        """Return the tokenizer (HF-compatible interface expected).

        Note: All backends use HF tokenizer for cross-backend consistency.
        """
        ...

    @abstractmethod
    def forward(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
    ) -> np.ndarray:
        """Run forward pass and return logits at final position.

        Args:
            input_ids: Shape [batch, seq_len], int64
            attention_mask: Shape [batch, seq_len], int64

        Returns:
            np.ndarray of shape [batch, vocab_size], float32

        Note: With left-padding, logits[:, -1, :] is the next-token distribution.
        """
        ...

    @abstractmethod
    def device_info(self) -> dict[str, Any]:
        """Return device information for health checks and logging."""
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if model is loaded and ready."""
        ...

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return backend identifier (pytorch, vllm, mlx)."""
        ...

    def warmup(self, batch_size: int = 1, seq_len: int = 128) -> float:
        """Run warmup pass to compile kernels and allocate memory.

        Args:
            batch_size: Warmup batch size
            seq_len: Warmup sequence length

        Returns:
            Warmup time in milliseconds
        """
        import time

        dummy_ids = np.ones((batch_size, seq_len), dtype=np.int64)
        dummy_mask = np.ones((batch_size, seq_len), dtype=np.int64)

        start = time.perf_counter()
        _ = self.forward(dummy_ids, dummy_mask)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return elapsed_ms


@dataclass
class BackendCapabilities:
    """Declare what each backend supports."""

    platforms: list[str] = field(default_factory=list)
    devices: list[str] = field(default_factory=list)
    quantization: list[str] = field(default_factory=list)
    flash_attention: bool = False
    batch_inference: bool = True
    continuous_batching: bool = False
    tensor_parallelism: bool = False
    memory_efficient: bool = False


# Predefined capability sets for each backend
PYTORCH_CAPABILITIES = BackendCapabilities(
    platforms=["darwin_arm64", "darwin_x86_64", "linux_x86_64", "win32"],
    devices=["cuda", "mps", "cpu"],
    quantization=["fp16", "bf16", "fp32"],
    flash_attention=True,  # CUDA only, SM 8.0+
    batch_inference=True,
)

VLLM_CAPABILITIES = BackendCapabilities(
    platforms=["linux_x86_64"],
    devices=["cuda"],
    quantization=["fp16", "awq", "gptq"],
    flash_attention=True,
    batch_inference=True,
    continuous_batching=True,
    tensor_parallelism=True,
)

MLX_CAPABILITIES = BackendCapabilities(
    platforms=["darwin_arm64"],
    devices=["apple_silicon"],
    quantization=["fp16", "8bit", "4bit"],
    flash_attention=True,  # Native Metal optimization
    batch_inference=True,
    memory_efficient=True,  # Lazy evaluation
)


@dataclass
class ModelState:
    """Holds common model state across backends."""

    model: Any = None
    tokenizer: Any = None
    model_id: str = ""
    loaded: bool = False
    warmup_complete: bool = False
    load_time_ms: float = 0.0
    error: str | None = None
    yes_token_id: int = 0
    no_token_id: int = 0
