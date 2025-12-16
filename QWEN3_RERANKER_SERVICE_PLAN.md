# Qwen3-Reranker (MLX) Standalone Service — Implementation Plan

Goal: implement a **standalone, native macOS (Apple Silicon) HTTP reranker service** running **Qwen/Qwen3-Reranker-4B** in **MLX** format (default: `Lipdog/Qwen3-Reranker-4B-mlx-fp16`) while remaining **drop-in compatible** with the existing `wekadocs-matrix` reranker client/provider (read-only for this effort).

This document is written to be “agentic-coder executable”: an engineer (or coding agent) should be able to implement the service end-to-end from this plan without modifying `wekadocs-matrix`.

---

## 0) Hard constraints (must not break)

### 0.1 Do not modify `wekadocs-matrix`
- Treat `/Users/brennanconley/vibecode/wekadocs-matrix` as **read-only**.
- Compatibility must be achieved by implementing an HTTP API the existing `wekadocs-matrix` provider already speaks.

### 0.2 Required `wekadocs-matrix` compatibility surface (verified)
`wekadocs-matrix/src/providers/rerank/local_bge_service.py` expects:
- `POST /v1/rerank`
  - Request JSON: `{ "query": str, "documents": [str, ...], "model": str }`
  - Response JSON: `{ "results": [ { "index": int, "score": float }, ... ] }`
  - `results` are **sorted by descending score**, with `index` referencing the **input documents list index**.
- `GET /health`
  - Used as a lightweight readiness probe.

Important behavior in the client:
- It may batch many docs per request.
- It treats any status `<500` as “reachable”, but the provider does real work only when `POST /v1/rerank` returns `200`.

Implication for this service:
- Implement `POST /v1/rerank` and `GET /health` exactly.
- Accept extra request fields gracefully (ignore/allow), so future callers don’t get `422`.

---

## 1) Primary sources and the “official” Qwen3 reranker scoring method

### 1.1 Qwen3 reranker scoring is **yes/no next-token probability**
Qwen3 rerankers are **causal LMs used as rerankers** by scoring the probability that the **next token** is `"yes"` rather than `"no"` after a fixed prompt wrapper.

Implementation must follow the official approach:
- Build a prompt with the official **system prefix** and **assistant suffix**.
- Compute logits at the **final position** (next-token distribution).
- Extract `logit_yes` and `logit_no`, compute `softmax([no, yes])`, return `p_yes` as the score.

References:
- Qwen model card for `Qwen/Qwen3-Reranker-4B` (prompt wrapper + scoring example).
  - https://huggingface.co/Qwen/Qwen3-Reranker-4B
- vLLM example clarifies the same prefix/suffix templates.
  - https://docs.vllm.ai/en/stable/examples/offline_inference/qwen3_reranker.html

### 1.2 Canonical templates (keep these as defaults)
The service should embed these templates as defaults (configurable via YAML/env):

**Prefix** (system + user prelude):
```
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
```

**Query template**:
```
{prefix}<Instruct>: {instruction}\n<Query>: {query}\n
```

**Document template + suffix**:
```
<Document>: {doc}{suffix}
```

**Suffix** (assistant “think” block terminator; Qwen expects this):
```
<|im_end|>
<|im_start|>assistant
<think>

</think>

```

Notes:
- Keep `padding_side="left"`.
- Set `pad_token = eos_token` (as in the official example).
- Verify `"yes"` and `"no"` are each **single tokens** for the loaded tokenizer at startup; if not, fail fast with a clear error (don’t silently pick token[0]).

---

## 2) Service requirements (standalone + wekadocs-compatible)

### 2.1 Functional requirements
- Implement a single-process FastAPI service that loads **one** MLX model and exposes:
  - `POST /v1/rerank` (wekadocs-compatible)
  - `GET /health` (wekadocs-compatible)
  - `GET /healthz` (more informative health check; consistent with other local services)
  - `GET /ready` (returns `200` only after model warmup)
  - `GET /v1/config` (introspection of active profile/config; optional but helpful)
- Scores:
  - Output score range: `[0, 1]` (`p_yes`)
  - Higher score ⇒ more relevant.

### 2.2 Non-functional requirements
- No multi-worker deployment (MLX weights are large; multi-worker duplicates memory).
- Predictable latency with guardrails:
  - request size limits (docs, chars, tokens)
  - concurrency limit (semaphore)
  - bounded batching
- Structured logging compatible with `wekadocs-matrix`’s observability style:
  - JSON logs by default
  - include stable fields (`event`, `correlation_id`, timings, counts)
- Runs natively on macOS (Apple Silicon) without Docker.

### 2.3 Compatibility requirements with `wekadocs-matrix`
- Must accept the `model` field in `POST /v1/rerank`.
  - Recommended policy: **accept any string** and treat it as “caller_label” (log it), but always use the single loaded model.
  - Optional stricter mode via env: require `model` to match configured aliases.
- Must return results in the exact shape used by the existing provider client.

---

## 3) Reference implementation layout (repo structure)

Design goal: keep the code small, but modular enough to test prompt/scoring correctness and keep API stable.

### 3.1 Proposed file tree
```
/Users/brennanconley/vibecode/qwen3-reranker/
  README.md
  pyproject.toml                 # or requirements.txt; prefer pyproject for pinning
  config/
    reranker_profiles.yaml        # “profiles” like wekadocs embedding_profiles.yaml
    development.yaml              # optional: service runtime config (port, profile)
  src/
    qwen3_reranker_service/
      __init__.py
      api.py                      # FastAPI app + routes
      api_models.py               # Pydantic request/response models (stable contract)
      config.py                   # env + YAML loading + validation
      logging.py                  # structured logging setup (JSON)
      mlx_backend.py              # model/tokenizer loading via mlx_lm.load()
      prompt.py                   # prompt templates + formatting
      tokenization.py             # truncate/pad/left-pad utilities
      scoring.py                  # yes/no logits -> p_yes
      batching.py                 # in-request batching & concurrency guard
      warmup.py                   # startup warmup routines
      errors.py                   # typed exceptions -> HTTP errors
      version.py                  # version string + build metadata (optional)
  scripts/
    run_dev.sh                    # uvicorn with reload (single worker)
    run_prod.sh                   # uvicorn without reload
    smoke_test.sh                 # curl-based quick check (no eval)
  tests/
    test_prompt_formatting.py
    test_yes_no_token_ids.py
    test_scoring_monotonicity_smoke.py
    test_api_contract.py
  eval/
    README.md
    dataset_schema.md
    build_pool.py                 # builds query + candidate pools from logs/exports
    run_eval.py                   # nDCG/MRR metrics for comparative testing
```

### 3.2 How this mirrors wekadocs conventions (without depending on it)
- **Profiles YAML**: like `wekadocs-matrix/config/embedding_profiles.yaml`, but for rerankers.
- **Health checks**: implement `/health` and `/healthz`, similar to other `wekadocs-matrix/services/*` patterns.
- **Structured logging**: JSON, event-centric, correlation-friendly (similar intent to `wekadocs-matrix/src/shared/observability/logging.py`).

---

## 4) Configuration design (profiles + env + request overrides)

### 4.1 Profiles file: `config/reranker_profiles.yaml`
Format (example):
```yaml
profiles:
  qwen3_4b_mlx_fp16:
    description: "Qwen3 reranker 4B in MLX FP16 (Lipdog conversion)."
    provider: "mlx-lm"
    model_id: "Lipdog/Qwen3-Reranker-4B-mlx-fp16"
    scoring:
      method: "yes_no_next_token_prob"
      yes_token: "yes"
      no_token: "no"
      prefix: "<|im_start|>system\nJudge whether ... \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
      query_template: "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
      document_template: "<Document>: {doc}{suffix}"
      suffix: "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    limits:
      max_length: 4096
      max_docs_per_request: 200
      max_query_chars: 8000
      max_doc_chars: 20000
    batching:
      batch_size: 8
      max_concurrent_forwards: 1
    defaults:
      instruction: "Given a web search query, retrieve relevant passages that answer the query"
```

### 4.2 Runtime env vars (service-side)
Minimum set:
- `QWEN_RERANK_PROFILE` (default: `qwen3_4b_mlx_fp16`)
- `QWEN_RERANK_PORT` (default: `9003` or similar; choose a free port distinct from 9001/9002)
- `QWEN_RERANK_LOG_LEVEL` (default: `INFO`)
- `QWEN_RERANK_MAX_LENGTH` (override profile)
- `QWEN_RERANK_BATCH_SIZE`
- `QWEN_RERANK_MAX_CONCURRENT_FORWARDS`
- `QWEN_RERANK_MODEL_ID` (override profile `model_id`)
- `QWEN_RERANK_MODEL_ALIAS_ALLOWLIST` (optional; comma-separated strings accepted in request `model`)

### 4.3 Request-time overrides (caller-controlled)
Keep `POST /v1/rerank` compatible by default, but optionally accept:
- `instruction` (string)
- `top_n` (int)
- `return_documents` (bool)
- `max_length` (int)

Implementation note:
- Configure Pydantic with `extra="allow"` (or ignore) to accept unknown fields without failing.
- Only apply overrides within safe bounds (e.g., `max_length <= profile.max_length_hard_cap`).

---

## 5) Core MLX implementation details (model loading, tokenization, scoring)

### 5.1 Model loading (MLX-LM)
Use `mlx_lm.load()` to load the MLX model and tokenizer wrapper once at startup.

Upstream signature (reference):
- `mlx_lm.utils.load(path_or_hf_repo, tokenizer_config=None, model_config=None, adapter_path=None, lazy=False, return_config=False, revision=None)`
  - Extracted from `mlx-lm` upstream `mlx_lm/utils.py`.

Reference:
- https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/utils.py

Startup behavior:
1. Load model+tokenizer.
2. Configure tokenizer:
   - left padding
   - set pad token to EOS if needed
3. Precompute:
   - token IDs for `"yes"` and `"no"`
   - tokenized prefix and suffix
4. Run a warmup forward pass (tiny batch) to trigger any first-run compilation/caching.

### 5.2 Prompt building
For each `(query, doc)`:
1. Resolve instruction:
   - if request provides it, use it
   - else use profile default
2. Construct:
   - `query_str = query_template.format(prefix=prefix, instruction=..., query=...)`
   - `doc_str = document_template.format(doc=..., suffix=suffix)`
3. Combine into a **single string** or a **single token list**:
   - Recommended: tokenize `query_str + doc_str` as one sequence to avoid any token-boundary surprises.

### 5.3 Tokenization + truncation policy (must be explicit)
Implement a predictable truncation strategy that matches Qwen’s example intent:
- Hard cap `max_length` (profile default, overrideable within safe bounds).
- Prefer to preserve:
  1) prefix + instruction + query
  2) then as much document as fits
  3) then always include suffix

Recommended truncation algorithm:
1. Tokenize prefix, suffix once.
2. Tokenize the “content” (instruction/query/doc formatted text) without padding.
3. If `len(prefix)+len(content)+len(suffix) > max_length`:
   - truncate **document portion first** until it fits
   - if still too long, truncate query last (but keep at least N tokens)
4. Left-pad batches to the same length.

Implementation detail:
- For correctness, you must know exactly which token position is “final”.
  - With left padding, using `logits[:, -1, :]` is correct if `-1` corresponds to the last token of the (prefix+content+suffix) sequence.

### 5.4 Scoring implementation (yes/no)
Given logits at the final position:
1. `logit_no = logits[:, token_no_id]`
2. `logit_yes = logits[:, token_yes_id]`
3. `p_yes = softmax([logit_no, logit_yes])[yes_index]`
4. Return `p_yes` per document.

Edge cases:
- Ensure token IDs exist and are single-token encodings.
- If MLX returns logits in fp16, cast to fp32 for numeric stability in softmax.

---

## 6) API design (contract + error handling)

### 6.1 `POST /v1/rerank` (wekadocs-compatible)
**Request** (minimum required for wekadocs):
```json
{
  "query": "string",
  "documents": ["doc1", "doc2"],
  "model": "any-string"
}
```

**Response** (required shape):
```json
{
  "results": [
    {"index": 1, "score": 0.9123},
    {"index": 0, "score": 0.1022}
  ]
}
```

Optional response additions (safe if callers ignore unknown keys):
```json
{
  "model": "Lipdog/Qwen3-Reranker-4B-mlx-fp16",
  "results": [...],
  "meta": {
    "max_length": 4096,
    "batch_size": 8,
    "scoring": "p_yes_softmax(no,yes)",
    "truncated_docs": 3,
    "elapsed_ms": 42.1
  }
}
```

Important:
- Always return `results` sorted by `score` desc.
- `index` must refer to the input `documents[]` index.

### 6.2 `GET /health` (wekadocs-compatible)
Return `200` if the server process is up and model is loaded (or if you choose “always up”, at least ensure `POST /v1/rerank` returns `503` when model not ready).

Suggested minimal response:
```json
{ "status": "ok" }
```

### 6.3 `GET /healthz` (diagnostic)
Return details:
- status: `ok|loading|error`
- model id
- profile name
- device info (macOS/Metal info if available)
- uptime seconds

### 6.4 `GET /ready` (strict readiness)
Return `200` only if:
- model loaded
- warmup completed

### 6.5 Error semantics
Use consistent HTTP status codes:
- `400`: invalid request shape, empty docs, oversized inputs (hard cap).
- `413`: payload too large (if body too big).
- `422`: avoid if possible; prefer `400` with clear message.
- `429`: if concurrency guard queue is full (optional).
- `503`: model not loaded or warming up.
- `500`: internal error.

---

## 7) Batching, concurrency, and memory safety

### 7.1 Single-process, single-worker
- Run Uvicorn with `--workers 1`.
- Avoid preloading in multiple processes.

### 7.2 Concurrency guard (must-have)
Implement an `asyncio.Semaphore(max_concurrent_forwards)` around the forward pass to avoid:
- GPU memory spikes
- thrash when multiple large requests arrive concurrently

Default: `max_concurrent_forwards = 1`.

### 7.3 In-request batching
Process `documents` in batches (profile `batch_size`):
- Tokenize and score `batch_size` docs per forward pass.
- Accumulate scores and then sort globally.

### 7.4 Hard limits
Enforce:
- `max_docs_per_request`
- `max_query_chars`
- `max_doc_chars`
- `max_length` hard cap

Expose counts in logs + response meta.

### 7.5 Warmup
On startup:
- run a single forward pass on a trivial `(query, doc)` pair
- mark readiness true

---

## 8) Structured logging (match wekadocs style; keep it lightweight)

### 8.1 Log format
Use JSON logs with an `event` field (so Loki queries can filter by `event="..."`).

Minimum fields per request:
- `event`: e.g. `rerank_request`, `rerank_complete`, `rerank_error`
- `correlation_id`: generate UUID per request if none provided
- `path`, `method`, `status_code`
- `doc_count`, `batch_size`, `max_length`
- `elapsed_ms`
- `truncated_docs`, `truncated_tokens_total` (if tracked)

### 8.2 Correlation ID propagation
Accept optional headers:
- `X-Correlation-Id` (if present, reuse; else generate)
Return it in response headers.

### 8.3 Keep dependencies minimal
Two viable options:
- Use `structlog` (closest to wekadocs approach) with a small setup helper.
- Or stick to stdlib `logging` and manually JSON-encode dicts.

Recommendation:
- Use `structlog` because it is already a proven pattern in wekadocs.
- Do not pull in OpenTelemetry exporters unless explicitly needed; keep hooks optional.

---

## 9) How to run (developer workflow)

### 9.1 Environment
- Python 3.11+ recommended.
- Install dependencies in a local venv (native, not Docker).

Key deps:
- `mlx`, `mlx-lm`
- `fastapi`, `uvicorn`
- `pydantic`
- `pyyaml` (for profiles)
- `structlog` (optional but recommended)

### 9.2 Dev run
- `scripts/run_dev.sh` runs uvicorn with reload, single worker, binds to localhost.

### 9.3 Production-ish run
- `scripts/run_prod.sh` runs uvicorn without reload, single worker.
- Provide example environment variables for profile/port/log level.

---

## 10) Integration with `wekadocs-matrix` (no code changes required)

### 10.1 Configure wekadocs to use the existing “local reranker service” provider
Because `wekadocs-matrix` already supports a local reranker service provider (`bge-reranker-service`), use it as the HTTP client wrapper:
- Set `RERANK_PROVIDER=bge-reranker-service`
- Point `RERANKER_BASE_URL` to this Qwen3 reranker service (e.g. `http://127.0.0.1:9003`)
- Optionally set `RERANK_MODEL` to any string (e.g. `Qwen/Qwen3-Reranker-4B`) for labeling; the service should not require this to match.

### 10.2 Expected behavior
`wekadocs-matrix` will:
- send `POST /v1/rerank` with many candidate docs (possibly batched)
- consume returned `score`s as `rerank_score`
- perform its own logging of `bge_rerank_complete` at the pipeline layer

The service should not attempt to replicate wekadocs pipeline logs; just provide stable API + internal metrics logs.

---

## 11) Testing plan (correctness + contract + smoke)

### 11.1 Unit tests (fast)
- Prompt formatting:
  - ensure prefix/suffix exactly match profile defaults
  - ensure instruction/query/doc are placed correctly
- Token ID sanity:
  - `encode("yes")` and `encode("no")` each produce exactly one token ID
- Deterministic scoring smoke:
  - Use a trivial pair where doc obviously answers query; ensure it scores higher than an unrelated doc.
  - Don’t assert exact numeric values (model updates/MLX changes may shift), just relative ordering.

### 11.2 API contract tests
- `POST /v1/rerank`:
  - returns `200`, valid JSON, `results` present
  - `index` values are within range
  - sorted by score descending
- Health endpoints return expected status.

### 11.3 Manual smoke test script
Add `scripts/smoke_test.sh`:
- curl `/health`
- curl `/ready`
- POST a tiny rerank request and pretty-print results

---

## 12) Evaluation harness (to justify max_length, batching, quantization)

### 12.1 Minimal dataset schema
Define a small JSONL format:
```json
{"query":"...", "positives":["..."], "negatives":["...","..."]}
```

### 12.2 Metrics
- nDCG@10
- MRR@10
- Recall@K (optional)
- Latency p50/p95 at rerank service layer

### 12.3 Comparisons
Compare at least:
- Qwen3 4B MLX fp16 (this service)
- Existing baseline reranker in your stack (e.g. BGE reranker service), if available

Keep constant:
- candidate pools
- instruction string
- max_length
- truncation policy

---

## 13) Performance and operational tuning (macOS / Apple Silicon)

### 13.1 Start conservative
- `max_length=4096`
- `batch_size=8`
- `max_concurrent_forwards=1`

### 13.2 Increase only with measurement
- Try `batch_size=16` if memory headroom remains stable.
- Try `max_length=8192` only if eval shows meaningful gains.

### 13.3 Wired memory (optional, only if you see MLX warnings/perf issues)
Some users tune macOS GPU wired memory limits with:
- `sudo sysctl iogpu.wired_limit_mb=<N>`

This is optional and should be documented as a manual tuning step (include how to revert).

---

## 14) Quantization fallback plan (only if needed)

Preferred: start with fp16 (4B fp16 weights are ~8GB-class).

If memory/latency is problematic:
- Quantize with `mlx_lm.convert -q` (MLX-LM supports quantization workflows).
- Or use a vetted community 4-bit MLX conversion (verify quality with your eval harness).

Implementation consideration:
- Quantized model id should be another entry in `reranker_profiles.yaml`.

---

## 15) Step-by-step implementation checklist (recommended order)

### Phase A — Repository bootstrap (no model yet)
1. Create Python project scaffold (`pyproject.toml`, `src/`, `tests/`, `config/`).
2. Define `reranker_profiles.yaml` and config loader.
3. Implement structured logging.
4. Implement FastAPI app with `/health`, `/healthz`, `/ready`, `/v1/config`.
5. Implement `POST /v1/rerank` with placeholder scoring (return zeros) and contract tests.

### Phase B — MLX model integration
6. Add MLX-LM dependency and implement `mlx_backend.load_model()`.
7. Implement prompt/tokenization pipeline + yes/no token extraction.
8. Implement forward pass to get last-position logits and compute `p_yes`.
9. Add warmup on startup and make `/ready` depend on it.

### Phase C — Production hardening
10. Add request hard limits and clear errors.
11. Add batching loop and concurrency semaphore.
12. Add per-request logging (timings, truncation stats).
13. Add smoke test script and document run instructions.

### Phase D — Eval + tuning
14. Add eval harness and measure quality/latency on your corpus.
15. Tune `max_length`, `batch_size`, truncation policy.
16. Only then consider quantization.

---

## 16) Definition of done (DoD)
- Service starts reliably and loads `Lipdog/Qwen3-Reranker-4B-mlx-fp16`.
- `POST /v1/rerank` matches `wekadocs-matrix` expectations and returns stable, sorted results.
- `GET /health` works for `wekadocs-matrix` provider’s health check.
- `/ready` turns `200` only after warmup.
- Logs are JSON with `event` fields and include request IDs and timings.
- Basic test suite runs locally and includes contract + scoring sanity checks.
- `README.md` documents how to run standalone and how to point `wekadocs-matrix` at it via env vars (without modifying wekadocs code).

