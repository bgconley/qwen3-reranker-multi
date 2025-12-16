# Session Context: Qwen3-Reranker Multi-Backend Service Planning

**Session Date:** 2025-12-15
**Working Directory:** `/Users/brennanconley/vibecode/qwen3-reranker-multi`
**Primary Artifact:** `qwen3_reranker_multi_backend_plan_20251215.md`
**Status:** Plan revision complete - Ready for implementation

---

## Executive Summary

This session focused on revising the Qwen3-Reranker Multi-Backend Service implementation plan to change the backend priority order. The original plan used MLX as the primary backend with PyTorch as secondary. Per user request, the plan was modified to use:

| Priority | Backend | Rationale |
|----------|---------|-----------|
| **PRIMARY** | PyTorch | Cross-platform (CUDA/MPS/CPU), mature ecosystem, best debugging tools |
| **SECONDARY** | vLLM | High-throughput CUDA path for production workloads |
| **TERTIARY** | MLX | 2-3x faster than PyTorch MPS on Apple Silicon (optional optimization) |

---

## Project Context: Qwen3-Reranker Service

### Purpose

The Qwen3-Reranker service is a standalone reranking microservice designed to be drop-in compatible with the `wekadocs-matrix` project. It provides neural reranking capabilities using the Qwen3-Reranker-4B model, which significantly outperforms BGE-reranker-v2-m3 especially on:
- Code retrieval (+40 points improvement)
- Instruction-following (+15 points improvement)

### API Contract (wekadocs-matrix Compatible)

The service exposes the following endpoints:

```
POST /v1/rerank     - Main reranking endpoint
GET /health         - Health check
GET /healthz        - Kubernetes-style health probe
GET /ready          - Readiness check with backend info
GET /v1/config      - Configuration information
```

**Request Format:**
```json
{
  "query": "string",
  "documents": ["doc1", "doc2", ...],
  "model": "optional-string",
  "instruction": "optional-custom-instruction"
}
```

**Response Format:**
```json
{
  "results": [
    {"index": 0, "score": 0.95},
    {"index": 2, "score": 0.73},
    {"index": 1, "score": 0.31}
  ]
}
```

Results are sorted by score descending. Scores are in range [0, 1] representing p(yes) - the probability that the document is relevant to the query.

---

## Architecture Overview

### Scoring Mechanism (Verified Correct)

The Qwen3-Reranker uses yes/no token probability scoring:

1. Build prompt with query and document using Qwen3 chat format
2. Run forward pass through causal LM
3. Extract logits at final position (next-token prediction)
4. Get logits for "yes" and "no" tokens only
5. Apply softmax over [no, yes] to get p(yes) ∈ [0, 1]

```python
# Scoring formula
logit_no = logits[:, token_false_id]
logit_yes = logits[:, token_true_id]
p_yes = softmax([logit_no, logit_yes])[1]  # Score in [0, 1]
```

### Prompt Template (Verified Against Official Docs)

```
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {instruction}
<Query>: {query}
<Document>: {document}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

The `<think>` block in the suffix is required for Qwen3's thinking format.

### Backend Abstraction Layer

All backends implement a common `RerankerBackend` Protocol:

```python
class RerankerBackend(Protocol):
    def load_model(self, model_id: str, **kwargs) -> None
    def get_tokenizer(self) -> Any
    def forward(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray
    def device_info(self) -> dict
    def is_loaded(self) -> bool
    def backend_name(self) -> str
```

This abstraction enables:
- Score parity testing between backends
- Graceful degradation when preferred backend unavailable
- Consistent API regardless of underlying implementation

---

## Backend Implementations

### PyTorch Backend (PRIMARY)

**File:** `src/qwen3_reranker/backends/pytorch_backend.py`

**Model:** `Qwen/Qwen3-Reranker-4B` (HuggingFace)

**Device Selection Priority:** CUDA > MPS > CPU

**Features:**
- Cross-platform support (Linux, macOS, Windows)
- Flash Attention 2 support on CUDA (SM 8.0+)
- Automatic device detection
- FP16/BF16/FP32 dtype selection based on device

**Performance:**
- CUDA with Flash Attention: ~30-50ms/batch
- MPS: ~150-300ms/batch
- CPU: ~500ms+/batch (fallback only)

**Memory:** ~9-10GB working set, ~12GB peak (CUDA)

### vLLM Backend (SECONDARY)

**File:** `src/qwen3_reranker/backends/vllm_backend.py`

**Model:** `Qwen/Qwen3-Reranker-4B` (HuggingFace)

**Platform:** CUDA only (Linux)

**Features:**
- Continuous batching for higher throughput
- Tensor parallelism for multi-GPU
- Optimized CUDA kernels
- Prefix caching

**Performance:** ~20-40ms/batch

**Memory:** ~8-9GB working set

### MLX Backend (TERTIARY)

**File:** `src/qwen3_reranker/backends/mlx_backend.py`

**Model:** `Lipdog/Qwen3-Reranker-4B-mlx-fp16` (MLX-converted)

**Platform:** Apple Silicon only (macOS arm64)

**Features:**
- 2-3x faster than PyTorch MPS
- Native unified memory
- Lazy evaluation with JIT compilation (`mx.compile()`)
- Optimized Metal shaders

**Performance:** ~50-100ms/batch

**Memory:** ~9-10GB working set

---

## Backend Registry and Auto-Detection

The registry detects available backends in priority order:

```python
BACKEND_PRIORITY = ["pytorch", "vllm", "mlx"]
```

Detection logic:
1. **PyTorch** - Always checked first, logs CUDA/MPS/CPU availability
2. **vLLM** - Only if CUDA available and vllm package installed
3. **MLX** - Only on Apple Silicon (`platform.machine() == "arm64"`)

Environment variables for override:
```bash
QWEN_RERANK_BACKEND=auto           # auto | pytorch | vllm | mlx
QWEN_RERANK_PROFILE=qwen3_4b_cuda  # Default profile
```

---

## Configuration Profiles

Located in `config/reranker_profiles.yaml`:

### PyTorch Profiles (PRIMARY)

- `qwen3_4b_cuda` - RECOMMENDED for NVIDIA GPUs
- `qwen3_4b_mps` - Apple Silicon via PyTorch

### vLLM Profiles (SECONDARY)

- `qwen3_4b_vllm` - High-throughput CUDA

### MLX Profiles (TERTIARY)

- `qwen3_4b_mlx_fp16` - Apple Silicon optimization
- `qwen3_4b_mlx_8bit` - Lower memory variant

---

## Implementation Phases

### Phase A: Core + PyTorch Backend (2-3 sessions)

**Status:** NOT STARTED

**Goal:** Working PyTorch implementation (CUDA/MPS/CPU cross-platform)

**Tasks:**
1. Create project scaffold with pyproject.toml
2. Implement config loading (profiles YAML + env overrides)
3. Implement backend abstraction layer (Protocol + registry)
4. **Implement PyTorch backend first** (primary target)
5. Implement prompt builder (templates from config)
6. Implement tokenization with truncation policy
7. Implement yes/no score extraction
8. Add structured JSON logging
9. Implement FastAPI routes
10. Add health check endpoints
11. Add PyTorch warmup routine

**Deliverable:** Service runs on CUDA/MPS/CPU, returns real scores

### Phase B: vLLM Backend + Parity Testing (1-2 sessions)

**Status:** NOT STARTED

**Goal:** High-throughput CUDA option with validated score parity

**Tasks:**
12. Implement vLLM backend (CUDA high-throughput)
13. Add backend parity tests (PyTorch vs vLLM)
14. Implement automatic backend detection
15. Test vLLM continuous batching performance

**Deliverable:** vLLM backend produces scores within 0.05 of PyTorch

### Phase C: Production Hardening (1-2 sessions)

**Status:** NOT STARTED

**Goal:** Production-ready service

**Tasks:**
16. Implement batching loop with configurable batch size
17. Add concurrency semaphore (prevent OOM)
18. Implement request limits and validation
19. Add per-request timing logs with correlation IDs
20. Add Docker support (CUDA)
21. Add graceful shutdown handling
22. Implement request queuing (optional)

**Deliverable:** Stable under load, proper error handling

### Phase D: MLX Backend + Evaluation (1 session)

**Status:** NOT STARTED

**Goal:** Apple Silicon optimization + quality validation

**Tasks:**
23. Implement MLX backend (optional, Apple Silicon optimization)
24. Add MLX parity tests (PyTorch vs MLX)
25. Build evaluation harness
26. Run comparative benchmarks on test corpus
27. Document performance characteristics

**Deliverable:** Full backend coverage + quality metrics

---

## Definition of Done

### Required (Must Have)
- [ ] PyTorch backend loads Qwen3-Reranker-4B and returns correct scores
- [ ] Auto-detection selects CUDA > MPS > CPU for PyTorch
- [ ] Backend parity tests pass (ranking order identical across backends)
- [ ] `POST /v1/rerank` returns sorted results compatible with wekadocs-matrix
- [ ] `GET /health` and `GET /ready` work correctly
- [ ] Logs are JSON with correlation IDs and timing metrics
- [ ] Warmup completes successfully on PyTorch backend

### Secondary (Should Have)
- [ ] vLLM backend for high-throughput CUDA deployment
- [ ] vLLM produces scores within 0.05 of PyTorch
- [ ] Docker image builds and runs on CUDA

### Optional/Tertiary (Nice to Have)
- [ ] MLX backend for Apple Silicon optimization
- [ ] MLX produces scores within 0.05 of PyTorch
- [ ] Quantized MLX models (8-bit, 4-bit) tested
- [ ] Evaluation harness with nDCG/MRR metrics

---

## Files Modified This Session

| File | Action | Description |
|------|--------|-------------|
| `qwen3_reranker_multi_backend_plan_20251215.md` | MODIFIED | Changed backend priority from MLX-first to PyTorch-first |

### Specific Changes Made:

1. **Executive Summary** - Updated key design decisions table
2. **Section 3.0** - Design principles changed to PyTorch-First Development
3. **Section 3.1** - Architecture diagram updated backend order
4. **Section 3.3** - Backend Protocol docstring updated (PyTorch as reference)
5. **Section 3.4** - Now contains PyTorch Backend Implementation (PRIMARY)
6. **Section 3.5** - Now contains MLX Backend Implementation (TERTIARY)
7. **Section 3.6** - Backend Registry priority changed to `["pytorch", "vllm", "mlx"]`
8. **Section 3.8** - Configuration profiles reordered (PyTorch RECOMMENDED)
9. **Section 3.9** - Environment variables reordered, default profile changed
10. **Part 4** - Implementation Phases reordered (PyTorch in Phase A)
11. **Part 5** - Testing Strategy updated (PyTorch as ground truth)
12. **Part 6** - Definition of Done restructured (Required/Secondary/Optional)
13. **Appendix A** - Renamed to "Backend Comparison Deep Dive", reframed for PyTorch-first
14. **Appendix B** - Added PyTorch and HuggingFace documentation links

---

## Technical Decisions and Rationale

### Why PyTorch Primary?

1. **Cross-platform compatibility** - Works on Linux (CUDA), macOS (MPS), Windows, any CPU
2. **Mature ecosystem** - Best debugging tools, profiling, documentation
3. **Industry standard** - Lower learning curve for contributors
4. **CUDA performance** - Flash Attention 2 provides excellent performance on NVIDIA
5. **Deployment flexibility** - Same codebase for dev and production

### Why vLLM Secondary?

1. **Production optimization** - Continuous batching maximizes GPU utilization
2. **Throughput** - Better for high-volume production workloads
3. **Scaling** - Tensor parallelism for multi-GPU deployments
4. **Still uses PyTorch** - Consistent scoring with primary backend

### Why MLX Tertiary?

1. **Platform-specific** - Only works on Apple Silicon
2. **Optimization path** - 2-3x faster than PyTorch MPS for local dev
3. **Optional** - Not required for production deployment
4. **Different model format** - Requires MLX-converted weights (Lipdog/)

---

## Score Parity Testing Strategy

PyTorch is the **reference implementation**. All other backends must produce scores within tolerance:

```python
SCORE_TOLERANCE = 0.05  # 5% difference allowed

# Test pattern
pytorch_scores = pytorch_backend.rerank(query, docs)
other_scores = other_backend.rerank(query, docs)

for i, (pt, other) in enumerate(zip(pytorch_scores, other_scores)):
    assert abs(pt - other) < SCORE_TOLERANCE

# Ranking order must be identical
pytorch_order = np.argsort(pytorch_scores)[::-1]
other_order = np.argsort(other_scores)[::-1]
assert list(pytorch_order) == list(other_order)
```

---

## Operational Requirements

### Critical Configuration
- **Single-worker constraint** - Avoid model duplication in memory
- **Left-padding** - Required for causal LM reranking
- **Warmup pass at startup** - Compile kernels, allocate memory
- **Concurrency semaphore** - Essential for stability, prevent OOM

### Resource Requirements (Qwen3-4B fp16)
- **GPU Memory:** ~9-10GB minimum
- **System Memory:** 16GB+ recommended
- **Storage:** ~8GB for model weights

### Batch Configuration
- **Default batch_size:** 8 (PyTorch/MLX), 16 (CUDA), 64 (vLLM)
- **max_length:** 4096 (MLX/MPS), 8192 (CUDA/vLLM)
- **max_docs_per_request:** 200-1000 depending on backend

---

## Next Steps

1. **Begin Phase A Implementation** - Create project scaffold, implement PyTorch backend
2. **Set up test infrastructure** - Unit tests, parity tests, API contract tests
3. **Validate scoring** - Compare against official Qwen examples
4. **Integration with wekadocs-matrix** - Ensure API contract compatibility

---

## Related Projects and Integration Points

### wekadocs-matrix Integration

The Qwen3-Reranker service is designed as a drop-in replacement/enhancement for the existing reranking in wekadocs-matrix. The API contract (`POST /v1/rerank` response format) is specifically designed for compatibility.

### Model Sources

| Backend | Model ID | Source |
|---------|----------|--------|
| PyTorch/vLLM | `Qwen/Qwen3-Reranker-4B` | HuggingFace Hub |
| MLX fp16 | `Lipdog/Qwen3-Reranker-4B-mlx-fp16` | HuggingFace Hub |
| MLX 8bit | `Lipdog/Qwen3-Reranker-4B-mlx-8bit` | HuggingFace Hub |

---

## References

- [Qwen3-Reranker-4B Model Card](https://huggingface.co/Qwen/Qwen3-Reranker-4B)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [vLLM Documentation](https://docs.vllm.ai/en/stable/)
- [MLX-LM GitHub](https://github.com/ml-explore/mlx-lm)
- [MLX vs MPS vs CUDA Benchmark](https://towardsdatascience.com/mlx-vs-mps-vs-cuda-a-benchmark/)

---

*Document generated: 2025-12-15*
*Plan version: 2.0 — PyTorch-First Multi-Backend Architecture*
