"""MLX Backend - TERTIARY implementation for Apple Silicon optimization.

Use cases:
- Apple Silicon local development (2-3x faster than PyTorch MPS)
- macOS development machines with M1/M2/M3 chips
- Optional optimization path when MLX is available

Performance characteristics:
- ~50-100ms/batch (2-3x faster than PyTorch MPS)
- Native unified memory (no CPU/GPU transfer overhead)
- Lazy evaluation with JIT compilation via mx.compile()
- Optimized Metal shaders for attention

Memory usage (Qwen3-Reranker-4B fp16):
- Model weights: ~8GB
- Working set: ~9-10GB total
- Recommended: 16GB+ unified memory

Note: MLX is Apple Silicon only (darwin_arm64).
"""

import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class MLXBackend:
    """MLX backend for Apple Silicon Macs - TERTIARY BACKEND (optional optimization)."""

    def __init__(self) -> None:
        """Initialize the MLX backend."""
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded: bool = False
        self._compiled_forward: Any = None  # JIT-compiled forward pass
        self._model_id: str = ""
        self._warmup_complete: bool = False
        self._yes_token_id: int = 0
        self._no_token_id: int = 0

    def load_model(self, model_id: str, **kwargs: Any) -> None:
        """Load MLX model and HF tokenizer.

        Args:
            model_id: MLX model path (e.g., "Lipdog/Qwen3-Reranker-4B-mlx-fp16")
            **kwargs:
                hf_tokenizer_id: Override tokenizer source
                compile: Enable mx.compile() for forward pass (default: True)
        """
        import mlx.core as mx
        from mlx_lm import load as mlx_load
        from transformers import AutoTokenizer

        self._model_id = model_id
        logger.info(f"Loading MLX model: {model_id}")

        start_time = time.perf_counter()

        # Load MLX model (returns model, tokenizer tuple)
        # We discard MLX tokenizer and use HF for consistency
        self._model, _ = mlx_load(model_id)

        # Load HF tokenizer for cross-backend consistency
        hf_tokenizer_id = kwargs.get("hf_tokenizer_id")
        if not hf_tokenizer_id:
            # Map Lipdog MLX models back to Qwen originals
            hf_tokenizer_id = (
                model_id.replace("-mlx-fp16", "")
                .replace("-mlx-8bit", "")
                .replace("-mlx-4bit", "")
                .replace("Lipdog/", "Qwen/")
            )
        logger.info(f"Loading HF tokenizer: {hf_tokenizer_id}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            hf_tokenizer_id,
            padding_side="left",  # Critical for causal LM reranking
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Get yes/no token IDs
        yes_tokens = self._tokenizer.encode("yes", add_special_tokens=False)
        no_tokens = self._tokenizer.encode("no", add_special_tokens=False)
        if len(yes_tokens) != 1 or len(no_tokens) != 1:
            raise ValueError(
                f"yes/no must be single tokens: yes={yes_tokens}, no={no_tokens}"
            )
        self._yes_token_id = yes_tokens[0]
        self._no_token_id = no_tokens[0]

        # Optionally compile forward pass for better performance
        if kwargs.get("compile", True):
            self._compiled_forward = mx.compile(self._forward_impl)
            logger.info("Compiled forward pass with mx.compile()")

        self._loaded = True
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"MLX backend loaded on {mx.default_device()} in {elapsed_ms:.1f}ms"
        )

    def _forward_impl(self, tokens: "mx.array") -> "mx.array":
        """Internal forward pass - may be compiled."""
        return self._model(tokens)

    def get_tokenizer(self) -> Any:
        """Return the HuggingFace tokenizer."""
        return self._tokenizer

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

        Note: MLX models typically don't use attention_mask in the same way
        as PyTorch. With left-padding, we rely on position to find the last token.
        """
        import mlx.core as mx

        # Convert to MLX array
        tokens = mx.array(input_ids)

        # Forward pass (use compiled version if available)
        if self._compiled_forward is not None:
            logits = self._compiled_forward(tokens)
        else:
            logits = self._model(tokens)

        # Extract last position logits
        # With left-padding, -1 is always the actual last token
        last_logits = logits[:, -1, :]

        # Ensure computation is complete before converting
        mx.eval(last_logits)

        # Convert back to numpy for backend-agnostic scoring
        return np.array(last_logits, dtype=np.float32)

    def warmup(self, batch_size: int = 1, seq_len: int = 128) -> float:
        """Run warmup pass to compile kernels and allocate memory.

        Args:
            batch_size: Warmup batch size
            seq_len: Warmup sequence length

        Returns:
            Warmup time in milliseconds
        """
        import mlx.core as mx

        logger.info(f"Running MLX warmup (batch={batch_size}, seq={seq_len})")

        # Create dummy input
        dummy_ids = np.ones((batch_size, seq_len), dtype=np.int64)
        dummy_mask = np.ones((batch_size, seq_len), dtype=np.int64)

        start = time.perf_counter()
        _ = self.forward(dummy_ids, dummy_mask)
        mx.synchronize()  # Ensure GPU work is complete
        elapsed_ms = (time.perf_counter() - start) * 1000

        self._warmup_complete = True
        logger.info(f"MLX warmup complete: {elapsed_ms:.1f}ms")
        return elapsed_ms

    def device_info(self) -> dict[str, Any]:
        """Return device information for health checks and logging."""
        import platform

        import mlx.core as mx

        info: dict[str, Any] = {
            "backend": "mlx",
            "model_id": self._model_id,
            "loaded": self._loaded,
            "warmup_complete": self._warmup_complete,
            "device": "apple_silicon",
            "device_name": platform.processor(),
            "default_device": str(mx.default_device()),
            "metal_available": True,
            "unified_memory": True,
        }

        try:
            import mlx

            info["mlx_version"] = mlx.__version__
        except (ImportError, AttributeError):
            info["mlx_version"] = "unknown"

        return info

    @property
    def is_loaded(self) -> bool:
        """Return True if model is loaded and ready."""
        return self._loaded

    @property
    def backend_name(self) -> str:
        """Return backend identifier."""
        return "mlx"
