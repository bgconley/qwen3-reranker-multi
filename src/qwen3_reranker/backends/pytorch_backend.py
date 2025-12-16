"""PyTorch Backend - PRIMARY implementation for cross-platform support.

Use cases:
- CUDA deployment (Linux servers, Lambda Cloud, cloud GPUs) - RECOMMENDED
- MPS on Apple Silicon (good debugging, familiar API)
- CPU fallback (any platform)
- Reference implementation for score parity validation

Performance characteristics:
- CUDA with Flash Attention: ~30-50ms/batch (fastest for NVIDIA)
- MPS: ~150-300ms/batch (2-3x slower than MLX on Apple Silicon)
- CPU: ~500ms+/batch (fallback only)

Memory requirements (Qwen3-4B fp16):
- Model weights: ~8GB
- Working set: ~9-10GB total
- Peak (during inference): ~12GB
"""

import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PyTorchBackend:
    """PyTorch backend supporting CUDA, MPS, and CPU - PRIMARY BACKEND."""

    def __init__(self, device: str | None = None) -> None:
        """Initialize the PyTorch backend.

        Args:
            device: Override device selection (auto | cuda | cuda:N | mps | cpu)
        """
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: Any = None
        self._loaded: bool = False
        self._requested_device = device
        self._model_id: str = ""
        self._dtype: Any = None
        self._warmup_complete: bool = False

    def _select_device(self) -> "torch.device":
        """Auto-select best available device.

        Priority: CUDA > MPS > CPU
        """
        import torch

        if self._requested_device and self._requested_device != "auto":
            device = torch.device(self._requested_device)
            logger.info(f"Using explicitly requested device: {device}")
            return device

        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Auto-selected CUDA: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Auto-selected MPS (Apple Silicon)")
        else:
            device = torch.device("cpu")
            logger.warning("Using CPU - inference will be slow")

        return device

    def _supports_flash_attn(self) -> bool:
        """Check if Flash Attention 2 is available."""
        import torch

        if not torch.cuda.is_available():
            return False
        try:
            import flash_attn  # noqa: F401

            # Check compute capability (requires SM 8.0+)
            cc = torch.cuda.get_device_capability()
            return cc[0] >= 8
        except ImportError:
            return False

    def load_model(self, model_id: str, **kwargs: Any) -> None:
        """Load PyTorch model with appropriate optimizations.

        Args:
            model_id: HuggingFace model ID (e.g., "Qwen/Qwen3-Reranker-4B")
            **kwargs:
                device: Override device selection
                dtype: Override dtype (auto-selected based on device)
                use_flash_attn: Enable Flash Attention 2 (CUDA only)
                trust_remote_code: Trust remote code (default: True for Qwen)
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._model_id = model_id
        self._device = self._select_device()

        # Determine dtype and attention implementation based on device
        if self._device.type == "cuda":
            self._dtype = kwargs.get("dtype", torch.float16)
            requested_flash = kwargs.get("use_flash_attn")
            supports_flash = self._supports_flash_attn()
            use_flash = supports_flash if requested_flash is None else (requested_flash and supports_flash)
            if requested_flash and not supports_flash:
                logger.warning(
                    "Flash Attention requested but not available; falling back to eager attention."
                )
            attn_impl = "flash_attention_2" if use_flash else "eager"
            if use_flash:
                logger.info("Using Flash Attention 2 (CUDA)")
        elif self._device.type == "mps":
            self._dtype = kwargs.get("dtype", torch.float16)
            attn_impl = "eager"  # Flash Attention not supported on MPS
            logger.info("MPS does not support Flash Attention - using eager")
        else:
            self._dtype = kwargs.get("dtype", torch.float32)
            attn_impl = "eager"

        logger.info(
            f"Loading PyTorch model: {model_id} "
            f"(dtype={self._dtype}, attn={attn_impl}, device={self._device})"
        )

        start_time = time.perf_counter()

        # Load tokenizer with left padding for causal LM reranking
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            padding_side="left",
            trust_remote_code=kwargs.get("trust_remote_code", True),
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model with appropriate settings
        load_kwargs: dict[str, Any] = {
            "torch_dtype": self._dtype,
            "trust_remote_code": kwargs.get("trust_remote_code", True),
        }

        # Only set attn_implementation if not using default eager
        if attn_impl != "eager":
            load_kwargs["attn_implementation"] = attn_impl

        # Use device_map for CUDA, manual .to() for MPS/CPU
        if self._device.type == "cuda":
            load_kwargs["device_map"] = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

        if self._device.type != "cuda":
            self._model = self._model.to(self._device)

        self._model.eval()
        self._loaded = True

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"PyTorch backend loaded on {self._device} in {elapsed_ms:.1f}ms")

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
        """
        import torch

        # Convert to tensors
        input_ids_t = torch.from_numpy(input_ids).to(self._device)
        attention_mask_t = torch.from_numpy(attention_mask).to(self._device)

        # Forward pass with no gradient computation
        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids_t,
                attention_mask=attention_mask_t,
                return_dict=True,
            )

            # Extract last position logits
            logits = outputs.logits[:, -1, :]

        # Synchronize CUDA before measuring time / returning
        if self._device.type == "cuda":
            torch.cuda.synchronize()

        # Convert to numpy (always float32 for consistency)
        return logits.float().cpu().numpy()

    def warmup(self, batch_size: int = 1, seq_len: int = 128) -> float:
        """Run warmup pass to compile kernels and allocate memory.

        Args:
            batch_size: Warmup batch size
            seq_len: Warmup sequence length

        Returns:
            Warmup time in milliseconds
        """
        import torch

        logger.info(f"Running PyTorch warmup (batch={batch_size}, seq={seq_len})")

        dummy_ids = np.ones((batch_size, seq_len), dtype=np.int64)
        dummy_mask = np.ones((batch_size, seq_len), dtype=np.int64)

        start = time.perf_counter()
        _ = self.forward(dummy_ids, dummy_mask)

        if self._device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._warmup_complete = True
        logger.info(f"PyTorch warmup complete: {elapsed_ms:.1f}ms")
        return elapsed_ms

    def device_info(self) -> dict[str, Any]:
        """Return device information for health checks and logging."""
        import torch

        info: dict[str, Any] = {
            "backend": "pytorch",
            "backend_version": torch.__version__,
            "device": str(self._device) if self._device else "not_initialized",
            "cuda_available": torch.cuda.is_available(),
            "mps_available": hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available(),
            "model_id": self._model_id,
            "loaded": self._loaded,
            "warmup_complete": self._warmup_complete,
        }
        if self._device and self._device.type == "cuda":
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_compute_capability"] = torch.cuda.get_device_capability(0)
            info["cuda_memory_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2
            )
            info["flash_attention"] = self._supports_flash_attn()
        if self._dtype:
            info["dtype"] = str(self._dtype)
        return info

    @property
    def is_loaded(self) -> bool:
        """Return True if model is loaded and ready."""
        return self._loaded

    @property
    def backend_name(self) -> str:
        """Return backend identifier."""
        return "pytorch"
