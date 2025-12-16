"""vLLM Backend - SECONDARY implementation for high-throughput CUDA deployment.

Use cases:
- High-throughput production on CUDA (Lambda Cloud, cloud GPUs)
- Multi-GPU deployment with tensor parallelism
- Large batch processing with continuous batching

Performance characteristics:
- ~20-40ms/batch with continuous batching
- Higher throughput than PyTorch for large workloads
- Optimized CUDA kernels with PagedAttention

Note: vLLM is CUDA-only (Linux). For other platforms, use PyTorch backend.

Important: Qwen3-Reranker uses yes/no token probability scoring, not the standard
cross-encoder score() method. We use the LLM class with generate() to get logits.
"""

import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class VLLMBackend:
    """vLLM backend for high-throughput CUDA inference - SECONDARY BACKEND."""

    def __init__(
        self,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        max_model_len: int | None = None,
    ) -> None:
        """Initialize the vLLM backend.

        Args:
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length (auto-detect if None)
        """
        self._llm: Any = None
        self._tokenizer: Any = None
        self._loaded: bool = False
        self._model_id: str = ""
        self._warmup_complete: bool = False
        self._tensor_parallel_size = tensor_parallel_size
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._yes_token_id: int = 0
        self._no_token_id: int = 0

    def load_model(self, model_id: str, **kwargs: Any) -> None:
        """Load vLLM model with optimizations.

        Args:
            model_id: HuggingFace model ID (e.g., "Qwen/Qwen3-Reranker-4B")
            **kwargs:
                tensor_parallel_size: Override TP size
                gpu_memory_utilization: Override GPU memory fraction
                max_model_len: Override max sequence length
                enable_prefix_caching: Enable prefix caching (default: True)
                trust_remote_code: Trust remote code (default: True for Qwen)
        """
        from transformers import AutoTokenizer
        from vllm import LLM

        self._model_id = model_id

        # Get parameters with overrides
        tp_size = kwargs.get("tensor_parallel_size", self._tensor_parallel_size)
        gpu_util = kwargs.get("gpu_memory_utilization", self._gpu_memory_utilization)
        max_len = kwargs.get("max_model_len", self._max_model_len)
        enable_prefix_caching = kwargs.get("enable_prefix_caching", True)
        trust_remote_code = kwargs.get("trust_remote_code", True)

        logger.info(
            f"Loading vLLM model: {model_id} (tp={tp_size}, gpu_util={gpu_util}, max_len={max_len})"
        )

        start_time = time.perf_counter()

        # Load tokenizer separately for left-padding control
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            padding_side="left",
            trust_remote_code=trust_remote_code,
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

        # Build vLLM engine arguments
        engine_kwargs: dict[str, Any] = {
            "model": model_id,
            "tensor_parallel_size": tp_size,
            "gpu_memory_utilization": gpu_util,
            "trust_remote_code": trust_remote_code,
            "dtype": "float16",
            "enable_prefix_caching": enable_prefix_caching,
        }
        if max_len:
            engine_kwargs["max_model_len"] = max_len

        # Create LLM engine
        self._llm = LLM(**engine_kwargs)
        self._loaded = True

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"vLLM backend loaded in {elapsed_ms:.1f}ms")

    def get_tokenizer(self) -> Any:
        """Return the HuggingFace tokenizer."""
        return self._tokenizer

    def forward(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
    ) -> np.ndarray:
        """Run forward pass using vLLM and return logits at final position.

        For Qwen3-Reranker, we need to get the raw logits at the final position
        to compute yes/no probabilities. vLLM's generate() API can be configured
        to return logprobs which we can use to extract the relevant token logits.

        Args:
            input_ids: Shape [batch, seq_len], int64
            attention_mask: Shape [batch, seq_len], int64

        Returns:
            np.ndarray of shape [batch, vocab_size], float32

        Note: This implementation uses vLLM's generate() with logprobs to extract
        the logits for yes/no tokens. For a full vocab_size output, we would need
        to use the internal engine APIs.
        """
        from vllm import SamplingParams

        batch_size = input_ids.shape[0]

        # vLLM does not accept an attention mask here, so we must strip padding tokens.
        # Our tokenizer left-pads; the real tokens are the last `sum(mask)` positions.
        prompt_token_ids: list[list[int]] = []
        for i in range(batch_size):
            length = int(np.sum(attention_mask[i]))
            if length <= 0:
                raise ValueError("Empty sequence after applying attention_mask")
            prompt_token_ids.append(input_ids[i, -length:].tolist())

        # Configure sampling to get logprobs for yes/no tokens
        # NOTE: vLLM returns *top-k* logprobs; we require that both yes/no appear.
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,  # Deterministic
            logprobs=20,
            prompt_logprobs=None,
        )

        # Run generation
        outputs = self._llm.generate(
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        # Extract logits from logprobs
        # vLLM returns normalized log probabilities; for a softmax over [no, yes]
        # the shared normalization constant cancels, so logprobs work as logits.
        vocab_size = len(self._tokenizer)
        logits = np.full((batch_size, vocab_size), -1e9, dtype=np.float32)

        missing_yes = 0
        missing_no = 0
        for i, output in enumerate(outputs):
            if output.outputs and output.outputs[0].logprobs:
                # Get the logprobs dict for the first generated token position
                token_logprobs = output.outputs[0].logprobs[0]
                if self._yes_token_id in token_logprobs:
                    logits[i, self._yes_token_id] = token_logprobs[
                        self._yes_token_id
                    ].logprob
                else:
                    missing_yes += 1
                if self._no_token_id in token_logprobs:
                    logits[i, self._no_token_id] = token_logprobs[
                        self._no_token_id
                    ].logprob
                else:
                    missing_no += 1

        if missing_yes or missing_no:
            raise RuntimeError(
                "vLLM logprobs did not include required yes/no tokens; "
                "increase SamplingParams.logprobs or use the PyTorch backend. "
                f"(missing_yes={missing_yes}, missing_no={missing_no})"
            )

        return logits

    def warmup(self, batch_size: int = 1, seq_len: int = 128) -> float:
        """Run warmup pass to compile kernels.

        Args:
            batch_size: Warmup batch size
            seq_len: Warmup sequence length

        Returns:
            Warmup time in milliseconds
        """
        logger.info(f"Running vLLM warmup (batch={batch_size}, seq={seq_len})")

        dummy_ids = np.ones((batch_size, seq_len), dtype=np.int64)
        dummy_mask = np.ones((batch_size, seq_len), dtype=np.int64)

        start = time.perf_counter()
        _ = self.forward(dummy_ids, dummy_mask)
        elapsed_ms = (time.perf_counter() - start) * 1000

        self._warmup_complete = True
        logger.info(f"vLLM warmup complete: {elapsed_ms:.1f}ms")
        return elapsed_ms

    def device_info(self) -> dict[str, Any]:
        """Return device information for health checks and logging."""
        import torch

        info: dict[str, Any] = {
            "backend": "vllm",
            "model_id": self._model_id,
            "loaded": self._loaded,
            "warmup_complete": self._warmup_complete,
            "tensor_parallel_size": self._tensor_parallel_size,
            "gpu_memory_utilization": self._gpu_memory_utilization,
        }

        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_memory_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2
            )

        try:
            import vllm

            info["vllm_version"] = vllm.__version__
        except (ImportError, AttributeError):
            info["vllm_version"] = "unknown"

        return info

    @property
    def is_loaded(self) -> bool:
        """Return True if model is loaded and ready."""
        return self._loaded

    @property
    def backend_name(self) -> str:
        """Return backend identifier."""
        return "vllm"
