# Qwen3-Reranker Multi-Backend Service — Implementation Plan

**Version:** 2.0
**Date:** 2025-12-15
**Status:** Draft
**Author:** Claude (Plan Review & Revision)

---

## Executive Summary

This document provides an evaluation of the original Qwen3-Reranker implementation plans and presents a **revised plan** that uses **PyTorch as the primary backend** (cross-platform, CUDA/MPS/CPU) with **vLLM as secondary** for high-throughput production and **MLX as tertiary** for Apple Silicon optimization. The service is designed to be standalone while remaining drop-in compatible with `wekadocs-matrix`.

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **PyTorch Primary** | Cross-platform (CUDA/MPS/CPU), mature ecosystem, best debugging tools |
| **vLLM Secondary** | High-throughput CUDA path for production workloads |
| **MLX Tertiary** | 2-3x faster than PyTorch MPS on Apple Silicon (optional optimization) |
| **Backend Abstraction** | Common Protocol enables score parity testing |

### Performance Comparison (LLM Inference on Apple Silicon)

| Runtime | Throughput | vs PyTorch MPS |
|---------|------------|----------------|
| **MLX** | ~230 tok/s | **2-5x faster** |
| PyTorch MPS | ~40-50 tok/s | Baseline |

*Source: arxiv 2511.05502, TDS benchmarks*

---

## Part 1: Evaluation of Original Plans

### 1.1 Strengths (What's Correct)

The original plans are **exceptionally well-researched** and get the fundamentals right:

#### Scoring Mechanism ✅
The yes/no token probability scoring is **correct per Qwen's official documentation**:
```python
# Correct: Extract logits at final position
logit_no = logits[:, token_false_id]
logit_yes = logits[:, token_true_id]
p_yes = softmax([logit_no, logit_yes])[1]  # Score in [0, 1]
```

**Verified via Hugging Face model card** (Qwen/Qwen3-Reranker-4B): The model uses a causal LM architecture where reranking is performed by measuring the probability that the next token is "yes" vs "no" after a structured prompt.

#### Prompt Templates ✅
The prefix/suffix templates match the official Qwen documentation exactly:
- System prompt instructs "yes or no" judgment
- `<|im_start|>` / `<|im_end|>` markers are correct
- The `<think>` block in the suffix is required (Qwen3's thinking format)

#### API Contract ✅
The `POST /v1/rerank` response format is correct for wekadocs-matrix compatibility:
```json
{
  "results": [{"index": int, "score": float}, ...]
}
```

#### Memory/Performance Estimates ✅
- 4B params × 2 bytes (fp16) ≈ 8GB is accurate
- The Lipdog MLX conversion is confirmed at ~8.06GB on disk
- `batch_size=8`, `max_length=4096` are reasonable starting points

#### Operational Recommendations ✅
- Single-worker constraint to avoid model duplication: **Correct**
- Left-padding for causal LMs: **Correct**
- Warmup pass at startup: **Best practice**
- Concurrency semaphore: **Essential for stability**

### 1.2 Gaps and Issues

#### Gap 1: No CUDA/NVIDIA Support 🔴 CRITICAL

The original plan is **MLX-only**, which means:
- Cannot run on Linux servers with NVIDIA GPUs
- Cannot deploy to cloud GPU instances (most use CUDA)
- No Docker deployment option for team/production use

**Impact:** Limits the service to personal macOS development machines.

#### Gap 2: No Backend Abstraction Layer 🟠 SIGNIFICANT

The plan tightly couples to MLX-LM APIs:
```python
# Original plan assumes this exact signature
model, tokenizer = mlx_lm.load("Lipdog/Qwen3-Reranker-4B-mlx-fp16")
```

Adding CUDA later would require **significant refactoring** rather than swapping a backend.

#### Gap 3: MLX API Assumptions May Be Incorrect 🟡 MODERATE

The plan references:
```python
logits = model(tokens[None], cache=cache)
```

However, MLX-LM's `generate()` API is the primary interface. Direct model calls for logit extraction need verification. The Hugging Face Transformers example shows a different pattern:
```python
batch_scores = model(**inputs).logits[:, -1, :]  # Transformers API
```

#### Gap 4: vLLM Integration Not Considered 🟡 MODERATE

vLLM has **native Qwen3 reranker support** with optimizations:
- Continuous batching for higher throughput
- `task="score"` mode with `Qwen3ForSequenceClassification`
- Tensor parallelism for multi-GPU

For high-throughput production, vLLM may outperform a custom implementation.

### 1.3 Benchmark Claims Verification

**Verified via Hugging Face model card:**

| Model | MTEB-R | CMTEB-R | MMTEB-R | MTEB-Code | FollowIR |
|-------|--------|---------|---------|-----------|----------|
| BGE-reranker-v2-m3 (0.6B) | 57.03 | 72.16 | 58.36 | 41.38 | -0.01 |
| Qwen3-Reranker-0.6B | 65.80 | 71.31 | 66.36 | 73.42 | 5.41 |
| **Qwen3-Reranker-4B** | **69.76** | 75.94 | 72.74 | **81.20** | **14.84** |
| Qwen3-Reranker-8B | 69.02 | **77.45** | **72.94** | 81.22 | 8.05 |

**Conclusion:** Qwen3-Reranker-4B significantly outperforms BGE-reranker-v2-m3, especially on code retrieval (+40 points) and instruction-following (+15 points). The benchmark claims are **valid**.

---

## Part 2: MLX vs PyTorch MPS Performance Analysis

### 2.1 Why MLX is Faster for LLM Inference

Based on benchmarks from 2024-2025:

| Factor | MLX | PyTorch MPS |
|--------|-----|-------------|
| **Design** | Built for Apple Silicon | CUDA adaptation layer |
| **Memory** | Unified memory native | Some transfer overhead |
| **Compilation** | Lazy eval + JIT (`mx.compile`) | Eager execution |
| **Attention** | Optimized Metal shaders | No Flash Attention 2 |

### 2.2 Benchmark Data

**LLM Inference (Qwen-2.5 on M2 Ultra):**
| Runtime | Throughput |
|---------|------------|
| MLX | ~230 tok/s |
| MLC-LLM | ~190 tok/s |
| llama.cpp | ~150 tok/s |
| PyTorch MPS | ~40 tok/s |

*Source: arxiv 2511.05502*

**Operation-Level Speedup:**
| Chip | MLX vs MPS |
|------|-----------|
| M1 Pro | **2.34x faster** |
| M2 Ultra | **1.24x faster** |
| M3 Pro | ~1.0x (parity) |

*Source: TDS benchmarks*

### 2.3 Implications for Qwen3-Reranker

For batch reranking (8-16 documents per request):

| Backend | Expected Latency | Use Case |
|---------|------------------|----------|
| **MLX** | ~50-100ms/batch | Primary (macOS dev) |
| PyTorch MPS | ~150-300ms/batch | Debugging, fallback |
| PyTorch CUDA | ~30-50ms/batch | Production (cloud) |
| vLLM CUDA | ~20-40ms/batch | High-throughput |

**Recommendation:** Use MLX as primary for Apple Silicon development, with PyTorch for cross-platform compatibility and score parity validation.

---

## Part 3: Revised Implementation Plan

### 3.0 Design Principles

1. **PyTorch-First Development** — Implement and optimize for PyTorch first (primary target, cross-platform)
2. **Backend Agnostic Core** — Prompt building, tokenization, scoring are backend-independent
3. **Plugin-Style Backends** — PyTorch, vLLM, MLX share a common Protocol
4. **Score Parity Testing** — Use PyTorch as ground truth, validate vLLM and MLX produce ≈ same scores
5. **Graceful Degradation** — Auto-detect: CUDA → MPS → CPU (PyTorch), with optional vLLM/MLX

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Service                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ POST /v1/    │  │ GET /health  │  │ GET /v1/config       │  │
│  │    rerank    │  │ GET /healthz │  │ GET /ready           │  │
│  └──────┬───────┘  └──────────────┘  └──────────────────────┘  │
│         │                                                       │
│  ┌──────▼───────────────────────────────────────────────────┐  │
│  │                    Reranker Core                          │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │  │
│  │  │   Prompt    │  │ Tokenization│  │    Scoring      │   │  │
│  │  │   Builder   │  │   Manager   │  │   (yes/no)      │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘   │  │
│  └──────────────────────────┬───────────────────────────────┘  │
│                             │                                   │
│  ┌──────────────────────────▼───────────────────────────────┐  │
│  │                 Backend Abstraction Layer                 │  │
│  │                                                           │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │              RerankerBackend (Protocol)              │ │  │
│  │  │  - load_model(config) -> None                        │ │  │
│  │  │  - forward(input_ids, attention_mask) -> logits      │ │  │
│  │  │  - get_tokenizer() -> Tokenizer                      │ │  │
│  │  │  - device_info() -> dict                             │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │                             │                             │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │  │
│  │  │ PyTorch  │  │  vLLM    │  │   MLX    │  │  ONNX    │ │  │
│  │  │ Backend  │  │ Backend  │  │ Backend  │  │ (future) │ │  │
│  │  │(PRIMARY) │  │(SECOND.) │  │(TERTIARY)│  │          │ │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 File Structure

```
/Users/brennanconley/vibecode/qwen3-reranker/
├── README.md
├── pyproject.toml
├── Dockerfile                      # CUDA deployment
├── Dockerfile.cpu                  # CPU-only deployment
├── docker-compose.yml
├── config/
│   ├── reranker_profiles.yaml      # Model configurations
│   └── development.yaml            # Runtime settings
├── src/
│   └── qwen3_reranker/
│       ├── __init__.py
│       ├── main.py                 # Entry point
│       │
│       ├── api/
│       │   ├── __init__.py
│       │   ├── routes.py           # FastAPI routes
│       │   ├── models.py           # Pydantic request/response
│       │   └── middleware.py       # Correlation ID, timing
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py           # Config loading + validation
│       │   ├── prompt.py           # Prompt template builder
│       │   ├── tokenization.py     # Truncation, padding logic
│       │   ├── scoring.py          # yes/no probability extraction
│       │   └── batching.py         # Request batching + concurrency
│       │
│       ├── backends/
│       │   ├── __init__.py
│       │   ├── base.py             # RerankerBackend Protocol
│       │   ├── registry.py         # Backend auto-detection
│       │   ├── pytorch_backend.py  # PyTorch (CUDA/MPS/CPU) - PRIMARY
│       │   ├── vllm_backend.py     # vLLM high-throughput (SECONDARY)
│       │   └── mlx_backend.py      # Apple Silicon MLX (TERTIARY)
│       │
│       ├── logging/
│       │   ├── __init__.py
│       │   └── structured.py       # JSON logging setup
│       │
│       └── utils/
│           ├── __init__.py
│           ├── warmup.py           # Model warmup routines
│           └── health.py           # Health check logic
│
├── scripts/
│   ├── run_dev.sh                  # MLX dev server
│   ├── run_prod.sh                 # Production (auto-detect)
│   ├── run_cuda.sh                 # Force CUDA backend
│   └── smoke_test.sh
│
├── tests/
│   ├── test_prompt.py
│   ├── test_scoring.py
│   ├── test_api_contract.py
│   ├── test_pytorch_backend.py     # PyTorch-specific tests (PRIMARY)
│   ├── test_vllm_backend.py        # vLLM-specific tests
│   ├── test_mlx_backend.py         # MLX-specific tests
│   └── test_backend_parity.py      # PyTorch vs vLLM vs MLX score parity
│
└── eval/
    ├── README.md
    ├── build_pool.py
    └── run_eval.py
```

### 3.3 Backend Abstraction Layer

```python
# src/qwen3_reranker/backends/base.py
from typing import Protocol, Any, List, Tuple
from abc import abstractmethod
import numpy as np

class RerankerBackend(Protocol):
    """
    Protocol defining the interface all backends must implement.

    Design Notes:
    - All backends must produce numerically equivalent scores (within tolerance)
    - PyTorch is the reference implementation; others are validated against it
    - Forward pass returns logits, not scores (scoring is backend-agnostic)
    """

    @abstractmethod
    def load_model(self, model_id: str, **kwargs) -> None:
        """
        Load model and tokenizer. Called once at startup.

        Args:
            model_id: HuggingFace model ID or local path
                     MLX: "Lipdog/Qwen3-Reranker-4B-mlx-fp16"
                     PyTorch: "Qwen/Qwen3-Reranker-4B"
        """
        ...

    @abstractmethod
    def get_tokenizer(self) -> Any:
        """
        Return the tokenizer (HF-compatible interface expected).

        Note: MLX uses HF tokenizer loaded separately for consistency.
        """
        ...

    @abstractmethod
    def forward(
        self,
        input_ids: np.ndarray,      # Shape: [batch, seq_len]
        attention_mask: np.ndarray   # Shape: [batch, seq_len]
    ) -> np.ndarray:
        """
        Run forward pass and return logits at final position.

        Returns:
            np.ndarray of shape [batch, vocab_size]

        Note: With left-padding, logits[:, -1, :] is the next-token distribution.
        """
        ...

    @abstractmethod
    def device_info(self) -> dict:
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
        """Return backend identifier (mlx, pytorch, vllm)."""
        ...


class BackendCapabilities:
    """Declare what each backend supports."""

    MLX = {
        "platforms": ["darwin_arm64"],
        "quantization": ["fp16", "8bit", "4bit"],
        "flash_attention": True,  # Native Metal optimization
        "batch_inference": True,
        "memory_efficient": True,  # Lazy evaluation
    }

    PYTORCH = {
        "platforms": ["darwin_arm64", "darwin_x86_64", "linux_x86_64", "win32"],
        "devices": ["cuda", "mps", "cpu"],
        "quantization": ["fp16", "bf16", "fp32"],
        "flash_attention": True,  # CUDA only
        "batch_inference": True,
    }

    VLLM = {
        "platforms": ["linux_x86_64"],
        "devices": ["cuda"],
        "quantization": ["fp16", "awq", "gptq"],
        "flash_attention": True,
        "continuous_batching": True,
        "tensor_parallelism": True,
    }
```

### 3.4 PyTorch Backend Implementation (Primary)

```python
# src/qwen3_reranker/backends/pytorch_backend.py
"""
PyTorch Backend - Primary implementation for cross-platform support.

Use cases:
- CUDA deployment (Linux servers, cloud GPUs) - RECOMMENDED
- MPS on Apple Silicon (good debugging, familiar API)
- CPU fallback (any platform)
- Reference implementation for score parity validation

Performance characteristics:
- CUDA with Flash Attention: ~30-50ms/batch (fastest for NVIDIA)
- MPS: ~150-300ms/batch (2-3x slower than MLX on Apple Silicon)
- CPU: ~500ms+/batch (fallback only)
"""

import numpy as np
import torch
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


class PyTorchBackend:
    """PyTorch backend supporting CUDA, MPS, and CPU - PRIMARY BACKEND."""

    def __init__(self, device: Optional[str] = None):
        self._model = None
        self._tokenizer = None
        self._device = None
        self._loaded = False
        self._requested_device = device

    def _select_device(self) -> torch.device:
        """
        Auto-select best available device.

        Priority: CUDA > MPS > CPU
        """
        if self._requested_device:
            device = torch.device(self._requested_device)
            logger.info(f"Using explicitly requested device: {device}")
            return device

        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Auto-selected CUDA: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Auto-selected MPS (Apple Silicon)")
        else:
            device = torch.device("cpu")
            logger.warning("Using CPU - inference will be slow")

        return device

    def load_model(self, model_id: str, **kwargs) -> None:
        """
        Load PyTorch model with appropriate optimizations.

        Args:
            model_id: HuggingFace model ID (e.g., "Qwen/Qwen3-Reranker-4B")
            **kwargs:
                device: Override device selection
                dtype: Override dtype (auto-selected based on device)
                use_flash_attn: Enable Flash Attention 2 (CUDA only)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = self._select_device()

        # Determine dtype and attention implementation based on device
        if self._device.type == "cuda":
            dtype = kwargs.get("dtype", torch.float16)
            use_flash = kwargs.get("use_flash_attn", self._supports_flash_attn())
            attn_impl = "flash_attention_2" if use_flash else "eager"
            if use_flash:
                logger.info("Using Flash Attention 2 (CUDA)")
        elif self._device.type == "mps":
            dtype = kwargs.get("dtype", torch.float16)
            attn_impl = "eager"  # Flash Attention not supported on MPS
            logger.info("MPS does not support Flash Attention - using eager")
        else:
            dtype = kwargs.get("dtype", torch.float32)
            attn_impl = "eager"

        logger.info(f"Loading PyTorch model: {model_id} (dtype={dtype}, attn={attn_impl})")

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            padding_side="left"
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model with appropriate settings
        load_kwargs = {
            "torch_dtype": dtype,
            "attn_implementation": attn_impl,
        }

        # Use device_map for CUDA, manual .to() for MPS/CPU
        if self._device.type == "cuda":
            load_kwargs["device_map"] = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

        if self._device.type != "cuda":
            self._model = self._model.to(self._device)

        self._model.eval()
        self._loaded = True

        logger.info(f"PyTorch backend loaded on {self._device}")

    def _supports_flash_attn(self) -> bool:
        """Check if Flash Attention 2 is available."""
        if not torch.cuda.is_available():
            return False
        try:
            import flash_attn
            # Check compute capability (requires SM 8.0+)
            cc = torch.cuda.get_device_capability()
            return cc[0] >= 8
        except ImportError:
            return False

    def get_tokenizer(self) -> Any:
        return self._tokenizer

    @torch.no_grad()
    def forward(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray
    ) -> np.ndarray:
        """
        Run forward pass and return logits at final position.

        Note: Unlike MLX, PyTorch requires explicit attention_mask handling.
        """
        # Convert to tensors
        input_ids_t = torch.from_numpy(input_ids).to(self._device)
        attention_mask_t = torch.from_numpy(attention_mask).to(self._device)

        # Forward pass
        outputs = self._model(
            input_ids=input_ids_t,
            attention_mask=attention_mask_t,
            return_dict=True
        )

        # Extract last position logits
        logits = outputs.logits[:, -1, :]

        # Convert to numpy (always float32 for consistency)
        return logits.float().cpu().numpy()

    def warmup(self, batch_size: int = 1, seq_len: int = 128) -> float:
        """Run warmup pass."""
        import time

        logger.info(f"Running PyTorch warmup (batch={batch_size}, seq={seq_len})")

        dummy_ids = np.ones((batch_size, seq_len), dtype=np.int64)
        dummy_mask = np.ones((batch_size, seq_len), dtype=np.int64)

        start = time.perf_counter()
        _ = self.forward(dummy_ids, dummy_mask)

        if self._device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"PyTorch warmup complete: {elapsed_ms:.1f}ms")
        return elapsed_ms

    def device_info(self) -> dict:
        info = {
            "backend": "pytorch",
            "backend_version": torch.__version__,
            "device": str(self._device),
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
        }
        if self._device and self._device.type == "cuda":
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_compute_capability"] = torch.cuda.get_device_capability(0)
            info["cuda_memory_total"] = torch.cuda.get_device_properties(0).total_memory
            info["flash_attention"] = self._supports_flash_attn()
        return info

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def backend_name(self) -> str:
        return "pytorch"
```

### 3.5 MLX Backend Implementation (Tertiary)

```python
# src/qwen3_reranker/backends/mlx_backend.py
"""
MLX Backend - Tertiary implementation for Apple Silicon optimization.

Use cases:
- Apple Silicon development (2-3x faster than PyTorch MPS)
- Local development on Mac when maximum performance is needed
- Optional optimization path when MLX is available

Performance characteristics:
- 2-3x faster than PyTorch MPS for LLM inference
- Native unified memory (no CPU/GPU transfer overhead)
- Lazy evaluation with JIT compilation via mx.compile()
- Optimized Metal shaders for attention

Memory usage (Qwen3-Reranker-4B fp16):
- Model weights: ~8GB
- Working set: ~9-10GB total
- Recommended: 16GB+ unified memory
"""

import numpy as np
from typing import Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class MLXBackend:
    """MLX backend for Apple Silicon Macs - TERTIARY BACKEND (optional optimization)."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._compiled_forward = None  # JIT-compiled forward pass

    def load_model(self, model_id: str, **kwargs) -> None:
        """
        Load MLX model and HF tokenizer.

        Args:
            model_id: MLX model path (e.g., "Lipdog/Qwen3-Reranker-4B-mlx-fp16")
            **kwargs:
                hf_tokenizer_id: Override tokenizer source
                compile: Enable mx.compile() for forward pass (default: True)
        """
        import mlx.core as mx
        from mlx_lm import load as mlx_load
        from transformers import AutoTokenizer

        logger.info(f"Loading MLX model: {model_id}")

        # Load MLX model (returns model, tokenizer tuple)
        # We discard MLX tokenizer and use HF for consistency
        self._model, _ = mlx_load(model_id)

        # Load HF tokenizer for cross-backend consistency
        hf_tokenizer_id = kwargs.get(
            "hf_tokenizer_id",
            model_id.replace("-mlx-fp16", "").replace("-mlx-8bit", "")
                    .replace("-mlx-4bit", "").replace("Lipdog/", "Qwen/")
        )
        logger.info(f"Loading HF tokenizer: {hf_tokenizer_id}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            hf_tokenizer_id,
            padding_side="left"  # Critical for causal LM reranking
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Optionally compile forward pass for better performance
        if kwargs.get("compile", True):
            self._compiled_forward = mx.compile(self._forward_impl)
            logger.info("Compiled forward pass with mx.compile()")

        self._loaded = True
        logger.info(f"MLX backend loaded successfully on {mx.default_device()}")

    def _forward_impl(self, tokens: "mx.array") -> "mx.array":
        """Internal forward pass - may be compiled."""
        return self._model(tokens)

    def get_tokenizer(self) -> Any:
        return self._tokenizer

    def forward(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray
    ) -> np.ndarray:
        """
        Run forward pass and return logits at final position.

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
        """
        Run warmup pass to compile kernels and allocate memory.

        Returns: warmup time in milliseconds
        """
        import mlx.core as mx
        import time

        logger.info(f"Running MLX warmup (batch={batch_size}, seq={seq_len})")

        # Create dummy input
        dummy_ids = np.ones((batch_size, seq_len), dtype=np.int64)
        dummy_mask = np.ones((batch_size, seq_len), dtype=np.int64)

        start = time.perf_counter()
        _ = self.forward(dummy_ids, dummy_mask)
        mx.synchronize()  # Ensure GPU work is complete
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(f"MLX warmup complete: {elapsed_ms:.1f}ms")
        return elapsed_ms

    def device_info(self) -> dict:
        import mlx.core as mx
        import platform

        return {
            "backend": "mlx",
            "backend_version": self._get_mlx_version(),
            "device": "apple_silicon",
            "device_name": platform.processor(),
            "default_device": str(mx.default_device()),
            "metal_available": True,
            "unified_memory": True,
        }

    def _get_mlx_version(self) -> str:
        try:
            import mlx
            return mlx.__version__
        except:
            return "unknown"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def backend_name(self) -> str:
        return "mlx"
```

### 3.6 Backend Registry (PyTorch-First Priority)

```python
# src/qwen3_reranker/backends/registry.py
"""
Backend Registry - Auto-detection with PyTorch-first priority.

Priority order:
1. PyTorch (CUDA > MPS > CPU) - Cross-platform, mature ecosystem
2. vLLM (CUDA only) - High-throughput production workloads
3. MLX (Apple Silicon only) - Optional 2-3x speedup on Mac
"""

from typing import Optional, List, Dict, Any
import logging
import platform

from .base import RerankerBackend

logger = logging.getLogger(__name__)

# PyTorch is primary for cross-platform compatibility
BACKEND_PRIORITY = ["pytorch", "vllm", "mlx"]


def detect_platform() -> Dict[str, Any]:
    """Detect current platform capabilities."""
    info = {
        "os": platform.system(),
        "arch": platform.machine(),
        "processor": platform.processor(),
        "is_apple_silicon": (
            platform.system() == "Darwin" and
            platform.machine() == "arm64"
        ),
    }
    return info


def detect_available_backends() -> List[str]:
    """
    Detect which backends are available on this system.

    Returns backends in priority order (PyTorch first for cross-platform).
    """
    available = []
    plat = detect_platform()

    # Check PyTorch first (PRIMARY - cross-platform)
    try:
        import torch
        available.append("pytorch")

        if torch.cuda.is_available():
            device = f"CUDA: {torch.cuda.get_device_name(0)}"
        elif torch.backends.mps.is_available():
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
                import vllm
                available.append("vllm")
                logger.info(f"✓ vLLM backend available (CUDA: {torch.cuda.get_device_name(0)}) - SECONDARY")
            except ImportError:
                logger.debug("vLLM not installed")
    except ImportError:
        pass

    # Check MLX (TERTIARY - Apple Silicon optimization)
    if plat["is_apple_silicon"]:
        try:
            import mlx.core
            available.append("mlx")
            logger.info("✓ MLX backend available (Apple Silicon) - TERTIARY")
        except ImportError:
            logger.debug("MLX not installed")

    return available


def get_backend(
    backend_name: Optional[str] = None,
    **kwargs
) -> RerankerBackend:
    """
    Get a backend instance.

    Args:
        backend_name: Explicit backend name, or None for auto-detection
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
            "  - mlx-lm (Apple Silicon)\n"
            "  - torch (any platform)\n"
            "  - vllm (CUDA)"
        )

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
        for backend in BACKEND_PRIORITY:
            if backend in available:
                selected = backend
                break
        logger.info(f"Auto-selected backend: {selected}")

    # Import and instantiate
    if selected == "mlx":
        from .mlx_backend import MLXBackend
        return MLXBackend()
    elif selected == "vllm":
        from .vllm_backend import VLLMBackend
        return VLLMBackend(
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 1)
        )
    elif selected == "pytorch":
        from .pytorch_backend import PyTorchBackend
        return PyTorchBackend(device=kwargs.get("device"))
    else:
        raise ValueError(f"Unknown backend: {selected}")


def get_model_id_for_backend(
    backend_name: str,
    base_model: str = "Qwen/Qwen3-Reranker-4B",
    quantization: Optional[str] = None
) -> str:
    """
    Get the appropriate model ID for a given backend.

    MLX uses converted models (Lipdog/), others use HF originals.
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
```

### 3.7 Scoring Implementation (Backend-Agnostic)

```python
# src/qwen3_reranker/core/scoring.py
"""
Scoring Module - Backend-agnostic yes/no probability extraction.

This module implements the Qwen3 reranker scoring method:
1. Extract logits at final position (next-token prediction)
2. Get logits for "yes" and "no" tokens only
3. Apply softmax over [no, yes] to get p(yes) ∈ [0, 1]

The scoring is identical regardless of backend (MLX/PyTorch/vLLM).
"""

import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def extract_yes_no_scores(
    logits: np.ndarray,
    yes_token_id: int,
    no_token_id: int
) -> np.ndarray:
    """
    Extract p(yes) scores from logits using softmax over [no, yes].

    Args:
        logits: Shape [batch, vocab_size], logits at final position
        yes_token_id: Token ID for "yes"
        no_token_id: Token ID for "no"

    Returns:
        np.ndarray of shape [batch] with p(yes) scores in [0, 1]

    Note:
        Higher score = more relevant document.
        Score of 0.5 means model is uncertain.
    """
    # Extract relevant logits
    logit_no = logits[:, no_token_id]
    logit_yes = logits[:, yes_token_id]

    # Stack: [batch, 2] where [:, 0] = no, [:, 1] = yes
    stacked = np.stack([logit_no, logit_yes], axis=1)

    # Numerically stable softmax
    max_logits = np.max(stacked, axis=1, keepdims=True)
    exp_logits = np.exp(stacked - max_logits)
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Return p(yes) which is index 1
    scores = softmax_probs[:, 1]

    return scores.astype(np.float32)


def get_yes_no_token_ids(tokenizer) -> Tuple[int, int]:
    """
    Get token IDs for "yes" and "no", with validation.

    Critical: Both "yes" and "no" must be single tokens for the
    Qwen3 reranker scoring method to work correctly.

    Returns:
        Tuple of (yes_token_id, no_token_id)

    Raises:
        ValueError: If "yes" or "no" tokenize to multiple tokens
    """
    yes_tokens = tokenizer.encode("yes", add_special_tokens=False)
    no_tokens = tokenizer.encode("no", add_special_tokens=False)

    if len(yes_tokens) != 1:
        raise ValueError(
            f"'yes' tokenizes to {len(yes_tokens)} tokens: {yes_tokens}. "
            "Expected single token. This tokenizer may not be compatible "
            "with Qwen3 reranker scoring."
        )
    if len(no_tokens) != 1:
        raise ValueError(
            f"'no' tokenizes to {len(no_tokens)} tokens: {no_tokens}. "
            "Expected single token. This tokenizer may not be compatible "
            "with Qwen3 reranker scoring."
        )

    logger.debug(f"Token IDs: yes={yes_tokens[0]}, no={no_tokens[0]}")
    return yes_tokens[0], no_tokens[0]


def validate_score_distribution(scores: np.ndarray) -> dict:
    """
    Analyze score distribution for debugging/monitoring.

    Returns statistics useful for detecting scoring issues.
    """
    return {
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "std": float(np.std(scores)),
        "num_above_0.5": int(np.sum(scores > 0.5)),
        "num_below_0.5": int(np.sum(scores < 0.5)),
    }
```

### 3.8 Configuration Schema (Updated)

```yaml
# config/reranker_profiles.yaml
profiles:
  # ============================================
  # PyTorch Profiles (PRIMARY - Cross-platform)
  # ============================================
  qwen3_4b_cuda:
    description: "Qwen3 reranker 4B on CUDA (PyTorch + Flash Attention) - RECOMMENDED"
    backend: "pytorch"
    model_id: "Qwen/Qwen3-Reranker-4B"
    performance:
      expected_latency_ms: 30-50  # with Flash Attention 2
      memory_gb: 9-10
    scoring:
      method: "yes_no_next_token_prob"
      yes_token: "yes"
      no_token: "no"
    prompts:
      prefix: |
        <|im_start|>system
        Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
        <|im_start|>user
      suffix: |
        <|im_end|>
        <|im_start|>assistant
        <think>

        </think>

      query_template: "<Instruct>: {instruction}\n<Query>: {query}\n"
      document_template: "<Document>: {document}"
    limits:
      max_length: 8192  # Can go higher with CUDA
      max_docs_per_request: 500
      max_query_chars: 16000
      max_doc_chars: 40000
    batching:
      batch_size: 16
      max_concurrent_forwards: 2
    pytorch_options:
      device: "auto"  # auto | cuda | cuda:0 | mps | cpu
      dtype: "float16"
      use_flash_attn: true
    defaults:
      instruction: "Given a web search query, retrieve relevant passages that answer the query"

  qwen3_4b_mps:
    description: "Qwen3 reranker 4B on MPS (Apple Silicon via PyTorch)"
    backend: "pytorch"
    model_id: "Qwen/Qwen3-Reranker-4B"
    performance:
      expected_latency_ms: 150-300  # 2-3x slower than MLX
      memory_gb: 10-12
      note: "Consider qwen3_4b_mlx_fp16 for 2-3x better performance"
    pytorch_options:
      device: "mps"
      dtype: "float16"
      use_flash_attn: false  # Not supported on MPS
    # ... (same scoring/prompts)

  # ============================================
  # vLLM Profiles (SECONDARY - High-throughput CUDA)
  # ============================================
  qwen3_4b_vllm:
    description: "Qwen3 reranker 4B via vLLM (high-throughput CUDA) - SECONDARY"
    backend: "vllm"
    model_id: "Qwen/Qwen3-Reranker-4B"
    performance:
      expected_latency_ms: 20-40
      continuous_batching: true
    vllm_options:
      tensor_parallel_size: 1
      gpu_memory_utilization: 0.8
      enable_prefix_caching: true
      max_model_len: 8192
    limits:
      max_docs_per_request: 1000
    batching:
      batch_size: 64  # vLLM handles larger batches
      max_concurrent_forwards: 4
    # ... (same scoring/prompts)

  # ============================================
  # MLX Profiles (TERTIARY - Apple Silicon Optimization)
  # ============================================
  qwen3_4b_mlx_fp16:
    description: "Qwen3 reranker 4B in MLX FP16 (Apple Silicon) - TERTIARY"
    backend: "mlx"
    model_id: "Lipdog/Qwen3-Reranker-4B-mlx-fp16"
    hf_tokenizer_id: "Qwen/Qwen3-Reranker-4B"
    performance:
      expected_latency_ms: 50-100  # per batch of 8
      memory_gb: 9-10
      throughput_advantage: "2-3x vs PyTorch MPS"
    scoring:
      method: "yes_no_next_token_prob"
      yes_token: "yes"
      no_token: "no"
    prompts:
      prefix: |
        <|im_start|>system
        Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
        <|im_start|>user
      suffix: |
        <|im_end|>
        <|im_start|>assistant
        <think>

        </think>

      query_template: "<Instruct>: {instruction}\n<Query>: {query}\n"
      document_template: "<Document>: {document}"
    limits:
      max_length: 4096
      max_docs_per_request: 200
      max_query_chars: 8000
      max_doc_chars: 20000
    batching:
      batch_size: 8
      max_concurrent_forwards: 1
    mlx_options:
      compile: true  # Use mx.compile() for JIT optimization
    defaults:
      instruction: "Given a web search query, retrieve relevant passages that answer the query"

  qwen3_4b_mlx_8bit:
    description: "Qwen3 reranker 4B in MLX 8-bit (lower memory) - TERTIARY"
    backend: "mlx"
    model_id: "Lipdog/Qwen3-Reranker-4B-mlx-8bit"
    hf_tokenizer_id: "Qwen/Qwen3-Reranker-4B"
    performance:
      expected_latency_ms: 60-120
      memory_gb: 5-6
    # ... (same scoring/prompts as fp16)
```

### 3.9 Environment Variables (Updated)

```bash
# ============================================
# Backend Selection (PyTorch is default - cross-platform)
# ============================================
QWEN_RERANK_BACKEND=auto           # auto | pytorch | vllm | mlx
QWEN_RERANK_PROFILE=qwen3_4b_cuda  # Default profile (PyTorch CUDA)

# ============================================
# Service Configuration
# ============================================
QWEN_RERANK_PORT=9003
QWEN_RERANK_HOST=127.0.0.1        # Use 0.0.0.0 for Docker
QWEN_RERANK_LOG_LEVEL=INFO
QWEN_RERANK_LOG_FORMAT=json        # json | text

# ============================================
# Override Profile Settings
# ============================================
QWEN_RERANK_MAX_LENGTH=4096
QWEN_RERANK_BATCH_SIZE=8
QWEN_RERANK_MAX_CONCURRENT=1

# ============================================
# PyTorch-Specific (Primary)
# ============================================
QWEN_RERANK_DEVICE=auto            # auto | cuda | cuda:0 | mps | cpu
QWEN_RERANK_DTYPE=float16
QWEN_RERANK_FLASH_ATTN=true        # CUDA only
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTORCH_ENABLE_MPS_FALLBACK=1      # Allow CPU fallback for unsupported MPS ops

# ============================================
# vLLM-Specific (Secondary)
# ============================================
QWEN_RERANK_TP_SIZE=1              # Tensor parallel size
QWEN_RERANK_GPU_UTIL=0.8

# ============================================
# MLX-Specific (Tertiary)
# ============================================
QWEN_RERANK_MLX_COMPILE=true       # Enable mx.compile() JIT

# ============================================
# Model Override
# ============================================
QWEN_RERANK_MODEL_ID=              # Override profile model_id
```

---

## Part 4: Implementation Phases (PyTorch-First)

### Phase A: Core + PyTorch Backend (2-3 sessions)

**Goal:** Working PyTorch implementation (CUDA/MPS/CPU cross-platform)

1. Create project scaffold with pyproject.toml
2. Implement config loading (profiles YAML + env overrides)
3. Implement backend abstraction layer (Protocol + registry)
4. **Implement PyTorch backend first** (primary target, cross-platform)
5. Implement prompt builder (templates from config)
6. Implement tokenization with truncation policy
7. Implement yes/no score extraction
8. Add structured JSON logging
9. Implement FastAPI routes
10. Add health check endpoints
11. Add PyTorch warmup routine

**Deliverable:** Service runs on CUDA/MPS/CPU, returns real scores

**Validation:**
```bash
# Test on any platform (CUDA, MPS, or CPU)
curl -X POST http://localhost:9003/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Python?", "documents": ["Python is a programming language.", "Java is also a language."]}'

# Expected: Python doc scores higher
```

### Phase B: vLLM Backend + Parity Testing (1-2 sessions)

**Goal:** High-throughput CUDA option with validated score parity

12. Implement vLLM backend (CUDA high-throughput)
13. Add backend parity tests (PyTorch vs vLLM)
14. Implement automatic backend detection
15. Test vLLM continuous batching performance

**Deliverable:** vLLM backend produces scores within 0.05 of PyTorch

**Key Test:**
```python
# tests/test_backend_parity.py
def test_pytorch_vllm_score_parity():
    """PyTorch and vLLM must produce equivalent scores."""
    query = "What is machine learning?"
    docs = [
        "Machine learning is a subset of AI.",
        "Python is a programming language.",
        "The weather is nice today."
    ]

    pytorch_scores = pytorch_backend.rerank(query, docs)
    vllm_scores = vllm_backend.rerank(query, docs)

    # Scores should be within tolerance
    for i, (pt, vl) in enumerate(zip(pytorch_scores, vllm_scores)):
        assert abs(pt - vl) < 0.05, f"Doc {i}: PyTorch={pt:.4f}, vLLM={vl:.4f}"

    # Ranking order should be identical
    pytorch_order = np.argsort(pytorch_scores)[::-1]
    vllm_order = np.argsort(vllm_scores)[::-1]
    assert list(pytorch_order) == list(vllm_order), "Ranking order mismatch"
```

### Phase C: Production Hardening (1-2 sessions)

**Goal:** Production-ready service

16. Implement batching loop with configurable batch size
17. Add concurrency semaphore (prevent OOM)
18. Implement request limits and validation
19. Add per-request timing logs with correlation IDs
20. Add Docker support (CUDA)
21. Add graceful shutdown handling
22. Implement request queuing (optional)

**Deliverable:** Stable under load, proper error handling

### Phase D: MLX Backend + Evaluation (1 session)

**Goal:** Apple Silicon optimization + quality validation

23. Implement MLX backend (optional, Apple Silicon optimization)
24. Add MLX parity tests (PyTorch vs MLX)
25. Build evaluation harness
26. Run comparative benchmarks on test corpus
27. Document performance characteristics

**Deliverable:** Full backend coverage + quality metrics

---

## Part 5: Testing Strategy (PyTorch as Ground Truth)

### 5.1 Unit Tests

```python
# tests/test_scoring.py
def test_yes_no_extraction_basic():
    """Verify softmax over yes/no produces correct probability."""
    # Logits where yes (idx 1) has higher value than no (idx 0)
    logits = np.array([[0.0, 1.0]])  # yes at idx 1, no at idx 0
    scores = extract_yes_no_scores(logits, yes_token_id=1, no_token_id=0)
    # softmax([0, 1]) ≈ [0.27, 0.73]
    assert 0.7 < scores[0] < 0.8

def test_yes_no_extraction_batch():
    """Verify batch processing works correctly."""
    logits = np.array([
        [2.0, 0.0],  # Strongly no
        [0.0, 2.0],  # Strongly yes
        [0.0, 0.0],  # Uncertain
    ])
    scores = extract_yes_no_scores(logits, yes_token_id=1, no_token_id=0)
    assert scores[0] < 0.2  # No
    assert scores[1] > 0.8  # Yes
    assert 0.4 < scores[2] < 0.6  # Uncertain

def test_yes_no_token_validation():
    """Ensure we fail fast if yes/no are multi-token."""
    class BadTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [1, 2]  # Always returns 2 tokens

    with pytest.raises(ValueError, match="Expected single token"):
        get_yes_no_token_ids(BadTokenizer())
```

### 5.2 PyTorch Backend Tests (Primary)

```python
# tests/test_pytorch_backend.py
import pytest
import torch

def test_pytorch_backend_loads():
    """Verify PyTorch backend loads model successfully."""
    backend = PyTorchBackend()
    backend.load_model("Qwen/Qwen3-Reranker-4B")
    assert backend.is_loaded
    assert backend.backend_name == "pytorch"

def test_pytorch_forward_shape():
    """Verify forward pass returns correct shape."""
    backend = PyTorchBackend()
    backend.load_model("Qwen/Qwen3-Reranker-4B")

    batch_size, seq_len = 4, 128
    input_ids = np.ones((batch_size, seq_len), dtype=np.int64)
    attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)

    logits = backend.forward(input_ids, attention_mask)

    assert logits.shape[0] == batch_size
    assert logits.shape[1] > 100000  # Vocab size

def test_pytorch_warmup():
    """Verify warmup completes."""
    backend = PyTorchBackend()
    backend.load_model("Qwen/Qwen3-Reranker-4B")

    warmup_time = backend.warmup(batch_size=1, seq_len=128)
    assert warmup_time > 0

def test_pytorch_device_selection():
    """Verify device auto-selection works correctly."""
    backend = PyTorchBackend()
    backend.load_model("Qwen/Qwen3-Reranker-4B")

    device_info = backend.device_info()
    assert device_info["backend"] == "pytorch"
    # Should have selected CUDA, MPS, or CPU
    assert device_info["device"] in ["cuda", "mps", "cpu"] or "cuda:" in device_info["device"]
```

### 5.3 Backend Parity Tests (Critical)

```python
# tests/test_backend_parity.py
"""
Backend Parity Tests - Ensure all backends produce equivalent scores.

PyTorch is the reference implementation. vLLM and MLX must match within tolerance.
"""

import pytest
import numpy as np

SCORE_TOLERANCE = 0.05  # Allow 5% difference due to numerical precision


@pytest.fixture
def test_cases():
    """Standard test cases for parity checking."""
    return [
        {
            "query": "What is Python?",
            "docs": [
                "Python is a high-level programming language.",
                "Java is a compiled language.",
                "The sun is a star.",
            ],
            "expected_order": [0, 1, 2],  # Python doc should rank first
        },
        {
            "query": "How to configure RAID?",
            "docs": [
                "RAID configuration involves selecting a RAID level.",
                "Python programming tutorial.",
                "RAID stands for Redundant Array of Independent Disks.",
            ],
            "expected_order": [0, 2, 1],  # RAID docs should rank higher
        },
    ]


class TestBackendParity:
    """Test that all backends produce equivalent scores vs PyTorch reference."""

    def test_vllm_score_parity(self, test_cases):
        """vLLM should produce similar scores to PyTorch (reference)."""
        if not vllm_available():
            pytest.skip("vLLM not available")

        pytorch_backend = get_backend("pytorch")
        vllm_backend = get_backend("vllm")

        pytorch_backend.load_model("Qwen/Qwen3-Reranker-4B")
        vllm_backend.load_model("Qwen/Qwen3-Reranker-4B")

        for case in test_cases:
            pytorch_scores = rerank_with_backend(pytorch_backend, case["query"], case["docs"])
            vllm_scores = rerank_with_backend(vllm_backend, case["query"], case["docs"])

            for i, (pt_s, vl_s) in enumerate(zip(pytorch_scores, vllm_scores)):
                diff = abs(pt_s - vl_s)
                assert diff < SCORE_TOLERANCE, (
                    f"Score mismatch for doc {i}:\n"
                    f"  PyTorch (ref): {pt_s:.4f}\n"
                    f"  vLLM: {vl_s:.4f}\n"
                    f"  Diff: {diff:.4f} (tolerance: {SCORE_TOLERANCE})"
                )

    def test_mlx_score_parity(self, test_cases):
        """MLX should produce similar scores to PyTorch (reference)."""
        if not mlx_available():
            pytest.skip("MLX not available (requires Apple Silicon)")

        pytorch_backend = get_backend("pytorch")
        mlx_backend = get_backend("mlx")

        pytorch_backend.load_model("Qwen/Qwen3-Reranker-4B")
        mlx_backend.load_model("Lipdog/Qwen3-Reranker-4B-mlx-fp16")

        for case in test_cases:
            pytorch_scores = rerank_with_backend(pytorch_backend, case["query"], case["docs"])
            mlx_scores = rerank_with_backend(mlx_backend, case["query"], case["docs"])

            for i, (pt_s, mlx_s) in enumerate(zip(pytorch_scores, mlx_scores)):
                diff = abs(pt_s - mlx_s)
                assert diff < SCORE_TOLERANCE, (
                    f"Score mismatch for doc {i}:\n"
                    f"  PyTorch (ref): {pt_s:.4f}\n"
                    f"  MLX: {mlx_s:.4f}\n"
                    f"  Diff: {diff:.4f} (tolerance: {SCORE_TOLERANCE})"
                )

    def test_ranking_order_matches(self, test_cases):
        """All backends should produce identical ranking order."""
        pytorch_backend = get_backend("pytorch")
        pytorch_backend.load_model("Qwen/Qwen3-Reranker-4B")

        for case in test_cases:
            pytorch_scores = rerank_with_backend(pytorch_backend, case["query"], case["docs"])
            pytorch_order = list(np.argsort(pytorch_scores)[::-1])

            # Test vLLM if available
            if vllm_available():
                vllm_backend = get_backend("vllm")
                vllm_backend.load_model("Qwen/Qwen3-Reranker-4B")
                vllm_scores = rerank_with_backend(vllm_backend, case["query"], case["docs"])
                vllm_order = list(np.argsort(vllm_scores)[::-1])
                assert pytorch_order == vllm_order, f"PyTorch vs vLLM order mismatch"

            # Test MLX if available
            if mlx_available():
                mlx_backend = get_backend("mlx")
                mlx_backend.load_model("Lipdog/Qwen3-Reranker-4B-mlx-fp16")
                mlx_scores = rerank_with_backend(mlx_backend, case["query"], case["docs"])
                mlx_order = list(np.argsort(mlx_scores)[::-1])
                assert pytorch_order == mlx_order, f"PyTorch vs MLX order mismatch"

    def test_expected_ranking(self, test_cases):
        """All available backends should produce expected ranking for known cases."""
        backends_to_test = [("pytorch", "Qwen/Qwen3-Reranker-4B")]
        if vllm_available():
            backends_to_test.append(("vllm", "Qwen/Qwen3-Reranker-4B"))
        if mlx_available():
            backends_to_test.append(("mlx", "Lipdog/Qwen3-Reranker-4B-mlx-fp16"))

        for backend_name, model_id in backends_to_test:
            backend = get_backend(backend_name)
            backend.load_model(model_id)

            for case in test_cases:
                scores = rerank_with_backend(backend, case["query"], case["docs"])
                actual_order = list(np.argsort(scores)[::-1])

                # Top result should match expected
                assert actual_order[0] == case["expected_order"][0], (
                    f"[{backend_name}] Top result mismatch for query: {case['query']}\n"
                    f"  Expected top: {case['expected_order'][0]}\n"
                    f"  Actual top: {actual_order[0]}"
                )
```

### 5.4 API Contract Tests

```python
# tests/test_api_contract.py
def test_rerank_response_shape():
    """Verify response matches wekadocs-matrix expectations."""
    response = client.post("/v1/rerank", json={
        "query": "test query",
        "documents": ["doc1", "doc2", "doc3"],
        "model": "any-string"  # Should be accepted and logged
    })

    assert response.status_code == 200
    data = response.json()

    # Required fields
    assert "results" in data
    assert len(data["results"]) == 3

    # Each result has index and score
    for result in data["results"]:
        assert "index" in result
        assert "score" in result
        assert 0 <= result["index"] <= 2
        assert 0.0 <= result["score"] <= 1.0

    # Results sorted by score descending
    scores = [r["score"] for r in data["results"]]
    assert scores == sorted(scores, reverse=True)

def test_rerank_with_instruction_override():
    """Verify custom instruction is accepted."""
    response = client.post("/v1/rerank", json={
        "query": "test",
        "documents": ["doc1"],
        "instruction": "Custom task instruction"
    })
    assert response.status_code == 200

def test_health_endpoint():
    """Verify health check works."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_ready_endpoint():
    """Verify readiness check returns backend info."""
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert "backend" in data
    assert data["ready"] == True
```

---

## Part 6: Definition of Done (Updated)

### Required
- [ ] PyTorch backend loads Qwen3-Reranker-4B and returns correct scores (PRIMARY)
- [ ] Auto-detection selects CUDA > MPS > CPU for PyTorch
- [ ] Backend parity tests pass (ranking order identical across backends)
- [ ] `POST /v1/rerank` returns sorted results compatible with wekadocs-matrix
- [ ] `GET /health` and `GET /ready` work correctly
- [ ] Logs are JSON with correlation IDs and timing metrics
- [ ] Warmup completes successfully on PyTorch backend

### Secondary
- [ ] vLLM backend for high-throughput CUDA deployment (SECONDARY)
- [ ] vLLM produces scores within 0.05 of PyTorch
- [ ] Docker image builds and runs on CUDA

### Optional (Tertiary)
- [ ] MLX backend for Apple Silicon optimization (TERTIARY)
- [ ] MLX produces scores within 0.05 of PyTorch
- [ ] Quantized MLX models (8-bit, 4-bit) tested
- [ ] Evaluation harness with nDCG/MRR metrics

---

## Appendix A: Backend Comparison Deep Dive

### Why PyTorch is Primary

| Factor | PyTorch | Benefits |
|--------|---------|----------|
| **Cross-platform** | CUDA, MPS, CPU | Deploy anywhere |
| **Ecosystem** | Mature, well-documented | Easy debugging |
| **CUDA Support** | Flash Attention 2 | Fastest on NVIDIA |
| **Familiarity** | Industry standard | Lower learning curve |
| **Tooling** | Best profiling, debugging | Easier development |

### When vLLM is Better (Secondary)

For high-throughput production on CUDA:
1. **Continuous batching** — Better GPU utilization
2. **Tensor parallelism** — Multi-GPU scaling
3. **Optimized kernels** — Lower latency at scale
4. **Production-ready** — Built for serving

### When MLX is Better (Tertiary)

For Apple Silicon local development:
1. **2-3x faster** than PyTorch MPS
2. **Native unified memory** — No transfer overhead
3. **Lower memory footprint** — Lazy evaluation
4. **Metal optimization** — Hardware-specific kernels

### Performance Comparison

| Backend | Platform | Latency/batch | Use Case |
|---------|----------|---------------|----------|
| PyTorch CUDA | Linux/Windows | ~30-50ms | Production (NVIDIA) |
| vLLM CUDA | Linux | ~20-40ms | High-throughput |
| PyTorch MPS | macOS | ~150-300ms | Development |
| MLX | macOS (Apple Silicon) | ~50-100ms | Local optimization |
| PyTorch CPU | Any | ~500ms+ | Fallback only |

### Memory Comparison (Qwen3-4B fp16)

| Backend | Working Set | Peak |
|---------|------------|------|
| PyTorch CUDA | ~9-10GB | ~12GB |
| vLLM CUDA | ~8-9GB | ~10GB |
| MLX | ~9-10GB | ~10GB |
| PyTorch MPS | ~10-12GB | ~14GB |

MLX's lazy evaluation and memory reuse keep footprint lower.

---

## Appendix B: Key References

### Official Documentation
- [Qwen3-Reranker-4B Model Card](https://huggingface.co/Qwen/Qwen3-Reranker-4B)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [vLLM Qwen3 Reranker Example](https://docs.vllm.ai/en/stable/examples/)
- [MLX-LM GitHub](https://github.com/ml-explore/mlx-lm)

### Benchmarks
- [MLX vs MPS vs CUDA Benchmark (TDS)](https://towardsdatascience.com/mlx-vs-mps-vs-cuda-a-benchmark/)
- [Production LLM Inference on Apple Silicon (arxiv 2511.05502)](https://arxiv.org/abs/2511.05502)

### MLX Model Conversions
- [Lipdog/Qwen3-Reranker-4B-mlx-fp16](https://huggingface.co/Lipdog/Qwen3-Reranker-4B-mlx-fp16)

---

*End of Plan v2.0 — PyTorch-First Multi-Backend Architecture*
