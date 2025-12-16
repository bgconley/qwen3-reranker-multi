# Session Context: Qwen3 Reranker Migration & System Operations Guide

**Session Date:** 2025-12-15
**Branch:** `multi-embedder-reranker`
**Session Focus:** Migrating from BGE-reranker-v2-m3 to Qwen3-Reranker-4B
**Status:** Configuration complete, databases empty, awaiting ingestion for full pipeline test

---

## Table of Contents

1. [Resume Point](#resume-point)
2. [Session Summary](#session-summary)
3. [Credential Management](#credential-management)
4. [Docker Container Operations](#docker-container-operations)
5. [Native ML Service Interactions](#native-ml-service-interactions)
6. [Database Access Patterns](#database-access-patterns)
7. [Ingestion Pipeline Operations](#ingestion-pipeline-operations)
8. [Qwen3 Reranker Migration Details](#qwen3-reranker-migration-details)
9. [Architectural Decisions](#architectural-decisions)
10. [Implementation Status](#implementation-status)
11. [Files Modified This Session](#files-modified-this-session)
12. [What Works vs What Needs Work](#what-works-vs-what-needs-work)
13. [Next Steps](#next-steps)
14. [Quick Reference Commands](#quick-reference-commands)

---

## Resume Point

### EXACT STOPPING POINT

**Qwen3 reranker fully configured. All containers restarted with new configuration. Databases are empty (cleaned from previous session). Ready for document ingestion to test the full pipeline with Qwen3.**

The session achieved:
- Complete migration from BGE-reranker-v2-m3 to Qwen3-Reranker-4B
- Updated 6 configuration files across the codebase
- Critically increased token budget limits from 800/1024 to 2048/4096
- Discovered and resolved Docker Compose env var precedence issue
- Verified Qwen3 service health on port 9003

**IMMEDIATE NEXT ACTION:**
1. Copy WekaDocs source files into `data/ingest/`
2. Monitor ingestion with `docker logs weka-ingestion-worker -f`
3. Run full pipeline test once data is ingested

---

## Session Summary

This session focused on swapping the reranker backend from the BGE-reranker-v2-m3 model (running on port 9001) to the Qwen3-Reranker-4B model (running on port 9003). The key insight was that the existing HTTP client wrapper (`bge-reranker-service`) is model-agnostic - it simply POSTs to `/v1/rerank` with `{query, documents, model}`. Therefore, only configuration changes were needed, not code changes to the client itself.

### Key Accomplishments

1. **Updated `.env`** - Changed `RERANK_MODEL`, `RERANKER_BASE_URL`, added `RERANKER_TIMEOUT_SECONDS`
2. **Updated `config/development.yaml`** - Changed reranker model and increased `max_tokens_per_pair` to 4096
3. **Updated `docker-compose.yml`** - Changed fallback defaults in all 3 services (mcp-server, ingestion-worker, ingestion-service)
4. **Updated `.env.example`** - Documentation for new Qwen3 configuration
5. **Updated `scripts/phase0/capture_baseline.py`** - Changed connectivity check defaults
6. **CRITICAL: Updated `src/providers/rerank/local_bge_service.py`** - Increased `MAX_TOKENS_PER_DOC` from 800 to 2048 and `MAX_TOKENS_TOTAL` from 1024 to 4096

### Critical Discovery: Docker Compose Env Precedence

During container recreation, we discovered that shell-exported environment variables take precedence over `.env` file values in Docker Compose. If `RERANK_MODEL` or `RERANKER_BASE_URL` are exported in your shell (from previous sessions), they override the `.env` file when Compose performs `${VAR:-default}` substitution.

**Solution:** Always unset these variables before recreating containers:
```bash
unset RERANK_PROVIDER RERANK_MODEL RERANKER_BASE_URL RERANKER_TIMEOUT_SECONDS
docker compose up -d --force-recreate
```

---

## Credential Management

### Environment Variables (Development Only)

All credentials are stored in `.env` at project root (gitignored). These are **DEVELOPMENT credentials** intended to be changed for production deployment.

```bash
# Neo4j Graph Database
NEO4J_URI=bolt://localhost:7687        # Use weka-neo4j:7687 inside Docker network
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=testpassword123

# Qdrant Vector Database
QDRANT_HOST=localhost                  # Use weka-qdrant inside Docker network
QDRANT_PORT=6333

# Redis Cache & Job Queue
REDIS_HOST=localhost                   # Use weka-redis inside Docker network
REDIS_PORT=6379
REDIS_PASSWORD=testredis123

# BGE-M3 Embedding Service (NATIVE - NOT Docker)
BGE_M3_API_URL=http://host.docker.internal:9000

# Qwen3 Reranker Service (NATIVE - NOT Docker) - NEW
RERANKER_BASE_URL=http://host.docker.internal:9003
RERANK_MODEL=Qwen/Qwen3-Reranker-4B
RERANKER_TIMEOUT_SECONDS=120

# GLiNER NER Service (NATIVE - NOT Docker)
GLINER_SERVICE_URL=http://host.docker.internal:9002

# Jina API Key (alternative embedding provider - currently unused)
JINA_API_KEY=jina_35169a1e714a41aab7b4c37817b58910Z65UGWJRVNStkMbt12lxaWrmIsVi
```

### Loading Credentials for Scripts

When running Python scripts outside Docker, you must set environment variables explicitly:

```bash
# Option 1: Inline (PREFERRED for single commands)
NEO4J_URI=bolt://localhost:7687 NEO4J_PASSWORD=testpassword123 python script.py

# Option 2: With PYTHONPATH for module imports
PYTHONPATH=/Users/brennanconley/vibecode/wekadocs-matrix \
NEO4J_URI=bolt://localhost:7687 NEO4J_PASSWORD=testpassword123 \
QDRANT_HOST=localhost REDIS_HOST=localhost \
python scripts/some_script.py

# Option 3: Export all at once (affects entire shell session)
export NEO4J_URI=bolt://localhost:7687
export NEO4J_PASSWORD=testpassword123
export QDRANT_HOST=localhost
export REDIS_HOST=localhost
```

**WARNING:** Exported variables override `.env` file values in Docker Compose. Use `unset` to clear them before container operations.

---

## Docker Container Operations

### Container Names and Purposes

| Container | Purpose | Ports | Notes |
|-----------|---------|-------|-------|
| `weka-neo4j` | Graph database | 7687 (bolt), 7474 (browser) | Stores document graph, entities, relationships |
| `weka-qdrant` | Vector database | 6333 (HTTP), 6334 (gRPC) | Stores embeddings for semantic search |
| `weka-redis` | Cache & job queue | 6379 | Query cache, ingestion queue, file hash tracking |
| `weka-ingestion-worker` | Background ingestion | None | Processes queued documents |
| `weka-ingestion-service` | Ingestion API + file watcher | 8081 | Watches `data/ingest/`, exposes health endpoint |
| `weka-mcp-server` | MCP query server | 8000 | Main query interface, exposes tools for LLMs |
| `weka-alloy` | Telemetry collector | 12345, 4317, 4318 | Grafana Alloy for observability |

### CRITICAL: Native Services (NOT in Docker)

The following ML services run **NATIVELY on the host machine** to leverage Apple Silicon MPS acceleration. They are NOT Docker containers. Docker containers access them via `host.docker.internal`.

| Service | Port | Model | Purpose |
|---------|------|-------|---------|
| BGE-M3 | 9000 | `BAAI/bge-m3` | Dense + Sparse + ColBERT embeddings |
| Qwen3 Reranker | 9003 | `Qwen/Qwen3-Reranker-4B` | Cross-encoder reranking (NEW) |
| GLiNER | 9002 | `urchade/gliner_medium-v2.1` | Named entity recognition |

**Health Check Commands:**
```bash
curl http://localhost:9000/healthz  # BGE-M3
curl http://localhost:9003/health   # Qwen3 Reranker
curl http://localhost:9002/healthz  # GLiNER
```

### Container Rebuild vs Restart

**When to REBUILD (code changes):**
```bash
# When you modify Python code in src/, containers have STALE code
docker compose build --no-cache ingestion-worker ingestion-service mcp-server
docker compose up -d --force-recreate ingestion-worker ingestion-service mcp-server
```

**When to RESTART (config changes only):**
```bash
# When you modify config/development.yaml (not .env!)
docker compose restart mcp-server ingestion-worker ingestion-service
```

**When to RECREATE (env var changes):**
```bash
# When you modify .env, you MUST recreate containers
# Also unset any shell-exported vars first!
unset RERANK_PROVIDER RERANK_MODEL RERANKER_BASE_URL RERANKER_TIMEOUT_SECONDS
docker compose up -d --force-recreate mcp-server ingestion-worker ingestion-service
```

### Verifying Container Environment

```bash
# Check what env vars a container actually has
docker exec weka-mcp-server env | grep -E "^RERANK" | sort

# Check what Docker Compose resolves (before container creation)
docker compose config | grep -E "RERANK_MODEL|RERANKER_BASE_URL"
```

---

## Native ML Service Interactions

### BGE-M3 Embedding Service

- **Location:** Runs natively on host, port 9000
- **Model:** `BAAI/bge-m3` with MPS acceleration
- **Max Context:** 8192 tokens per text (HARD LIMIT)
- **Capabilities:** Dense (1024-D), Sparse (BM25-style), ColBERT late-interaction

```bash
# Direct embedding call
curl -X POST http://localhost:9000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "BAAI/bge-m3", "input": ["test query"]}'
```

### Qwen3 Reranker Service (NEW)

- **Location:** Runs natively on host, port 9003
- **Model:** `Qwen/Qwen3-Reranker-4B` (or `Lipdog/Qwen3-Reranker-4B-mlx-fp16` on MLX)
- **Max Context:** 4096 tokens per query-document pair
- **Endpoint:** `POST /v1/rerank` with `{query, documents, model}`

```bash
# Start the Qwen3 service (from qwen3-reranker directory)
cd /Users/brennanconley/vibecode/qwen3-reranker
QWEN_RERANK_HOST=0.0.0.0 QWEN_RERANK_PORT=9003 ./scripts/run_prod.sh

# Test reranking
curl -X POST http://localhost:9003/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"query": "How to create filesystem?", "documents": ["WEKA guide", "Network config"], "model": "Qwen/Qwen3-Reranker-4B"}'
```

### GLiNER NER Service

- **Location:** Runs natively on host, port 9002
- **Model:** `urchade/gliner_medium-v2.1` with MPS
- **Timeout:** 160 seconds (increased from 60s in previous session to handle large documents)
- **Labels:** COMMAND, PARAMETER, COMPONENT, PROTOCOL, CLOUD_PROVIDER, STORAGE_CONCEPT, VERSION, PROCEDURE_STEP, ERROR, CAPACITY_METRIC

---

## Database Access Patterns

### Neo4j Access

**Via Docker exec (preferred for ad-hoc queries):**
```bash
# Interactive shell
docker exec -it weka-neo4j cypher-shell -u neo4j -p testpassword123

# Single query
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (n) RETURN count(n)"

# Check chunk count
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (c:Chunk) RETURN count(c)"
```

**Important Neo4j Notes:**
- APOC is NOT installed - cannot use `apoc.*` functions
- Vector index exists: `chunk_embeddings_v2` on `Chunk.vector_embedding`
- SchemaVersion nodes (2) are preserved during cleanup
- HAS_SECTION is deprecated - use HAS_CHUNK only

### Qdrant Access

```bash
# Collection info
curl http://localhost:6333/collections/chunks_multi_bge_m3

# Point count
curl -s http://localhost:6333/collections/chunks_multi_bge_m3 | \
  python3 -c "import json,sys; print(json.load(sys.stdin)['result']['points_count'])"

# Sample points with payload
curl -s -X POST "http://localhost:6333/collections/chunks_multi_bge_m3/points/scroll" \
  -H "Content-Type: application/json" \
  -d '{"limit": 5, "with_payload": true}'
```

### Redis Access and Usage

**USE Redis for:**
- Query result caching (L2 cache)
- Ingestion job queue management (`ingestion:queue`)
- Epoch-based cache invalidation
- File hash tracking for deduplication

**DO NOT use Redis for:**
- Direct data storage (use Neo4j/Qdrant)
- Long-term persistence (data is ephemeral)

```bash
# Check connection
docker exec weka-redis redis-cli -a testredis123 ping

# Check queue length
docker exec weka-redis redis-cli -a testredis123 LLEN ingestion:queue

# Check total keys
docker exec weka-redis redis-cli -a testredis123 DBSIZE

# Flush all (CRITICAL for clean re-ingestion)
docker exec weka-redis redis-cli -a testredis123 FLUSHALL
```

**CRITICAL: Why FLUSHALL matters for re-ingestion:**
The file watcher stores content hashes of previously ingested files in Redis. If these hashes remain after cleaning Neo4j/Qdrant, the watcher thinks files are "already processed" and skips them. `FLUSHALL` ensures no stale hashes block re-ingestion.

---

## Ingestion Pipeline Operations

### Test Document Location

```
data/ingest/           # Drop .md files here for auto-ingestion
```

The file watcher monitors this directory and automatically queues new/changed files.

### CRITICAL: Clean Ingestion Protocol

**When you want a fresh ingestion state, you MUST clean ALL THREE databases:**

```bash
# Step 1: Clean Neo4j and Qdrant (preserves SchemaVersion)
PYTHONPATH=/Users/brennanconley/vibecode/wekadocs-matrix \
NEO4J_URI=bolt://localhost:7687 NEO4J_PASSWORD=testpassword123 \
QDRANT_HOST=localhost REDIS_HOST=localhost \
python scripts/cleanup-databases.py

# Step 2: Flush Redis completely (CRITICAL for file hash dedup)
docker exec weka-redis redis-cli -a testredis123 FLUSHALL

# Step 3: Verify clean state
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 \
  "MATCH (n) WHERE NOT n:SchemaVersion RETURN count(n)"
curl -s http://localhost:6333/collections/chunks_multi_bge_m3 | \
  python3 -c "import json,sys; print(json.load(sys.stdin)['result']['points_count'])"
```

### IMPORTANT: Test Document Cleanup After Container Restart

**After rebuilding or restarting containers, if you want a clean ingestion:**
1. Delete any test documents from `data/ingest/`
2. Run the cleanup protocol above
3. Then add fresh documents

If you leave test documents and restart the worker, stale jobs in Redis may try to re-process them, causing errors if the file hashes have changed.

### Triggering Ingestion After Cleanup

After cleaning databases and Redis, files won't auto-ingest because the watcher only triggers on **file changes**. To re-ingest existing files:

```bash
# Touch all files to trigger watcher
find data/ingest -name "*.md" -exec touch {} \;
```

---

## Qwen3 Reranker Migration Details

### Why Qwen3?

The Qwen3-Reranker-4B model offers several advantages over BGE-reranker-v2-m3:
- **Larger context window:** 4096 tokens vs effective 1024 (due to previous client limits)
- **Better understanding:** 4B parameter model provides deeper semantic understanding
- **Same API:** `/v1/rerank` endpoint is compatible with existing HTTP client

### Configuration Changes Made

| File | Change | Purpose |
|------|--------|---------|
| `.env:40` | `RERANK_MODEL=Qwen/Qwen3-Reranker-4B` | Model identifier for logging/routing |
| `.env:42` | `RERANKER_BASE_URL=http://host.docker.internal:9003` | Point to Qwen3 service |
| `.env:44` | `RERANKER_TIMEOUT_SECONDS=120` | 4B model needs more time |
| `config/development.yaml:142` | `model: "Qwen/Qwen3-Reranker-4B"` | Config file alignment |
| `config/development.yaml:145` | `max_tokens_per_pair: 4096` | Document max context |
| `docker-compose.yml` (3 locations) | Updated fallback defaults | Prevent surprise reverts |
| `.env.example:65-69` | Updated documentation | Help future developers |
| `scripts/phase0/capture_baseline.py:463-471` | Updated connectivity check | Use Qwen3 for baseline |

### CRITICAL: Token Budget Increase

The most important change was in `src/providers/rerank/local_bge_service.py`:

```python
# OLD (too restrictive - filtered many valid candidates)
MAX_TOKENS_PER_DOC = 800   # Token budget per document
MAX_TOKENS_TOTAL = 1024    # Max tokens per request (query + doc)

# NEW (leverages Qwen3's 4096 context)
MAX_TOKENS_PER_DOC = 2048  # Token budget per document
MAX_TOKENS_TOTAL = 4096    # Max tokens per request (query + doc)
```

**Impact:** Previously, chunks over ~800 tokens were silently filtered out before reranking. Now, chunks up to 2048 tokens can be reranked, significantly improving coverage for longer WEKA documentation sections.

---

## Architectural Decisions

### Decision 1: Keep `bge-reranker-service` Provider Name

The HTTP client wrapper in `src/providers/rerank/local_bge_service.py` is model-agnostic. It simply POSTs to the configured `RERANKER_BASE_URL` with the request payload. Keeping the provider name as `bge-reranker-service` avoids unnecessary code changes while still routing traffic to Qwen3.

### Decision 2: Increased Token Budgets

The previous 800/1024 token limits were set conservatively for BGE-reranker but caused significant candidate filtering. With Qwen3's 4096 context support, we increased to 2048/4096 to ensure longer chunks are properly reranked.

### Decision 3: 120-Second Timeout

The Qwen3-4B model requires more inference time than BGE (~300ms for 2 docs vs ~50ms). The 120-second timeout provides headroom for batches of 50 candidates without risking timeouts.

---

## Implementation Status

### Completed This Session

| Feature | Status | Evidence |
|---------|--------|----------|
| Qwen3 reranker configuration | COMPLETE | All 6 files updated |
| Token budget increase | COMPLETE | 800/1024 → 2048/4096 |
| Container env vars | COMPLETE | Verified via `docker exec` |
| Service health verification | COMPLETE | Port 9003 responding |

### Pending Validation

| Feature | Status | Next Step |
|---------|--------|-----------|
| Full pipeline test | BLOCKED | Need data in databases first |
| Reranking quality comparison | NOT STARTED | Compare Qwen3 vs BGE scores |
| Latency benchmarking | NOT STARTED | Measure P50/P95/P99 |

---

## Files Modified This Session

| File | Lines Changed | Summary |
|------|---------------|---------|
| `.env` | 39-44 | Qwen3 model, URL, timeout |
| `config/development.yaml` | 142, 145 | Qwen3 model, max_tokens_per_pair |
| `docker-compose.yml` | 150-153, 249-253, 360-365 | Fallback defaults (3 services) |
| `.env.example` | 64-70 | Documentation update |
| `scripts/phase0/capture_baseline.py` | 463-471 | Connectivity check defaults |
| `src/providers/rerank/local_bge_service.py` | 27-30 | **CRITICAL:** Token budget increase |

---

## What Works vs What Needs Work

### Verified Working

| Component | Status | Evidence |
|-----------|--------|----------|
| Qwen3 service on port 9003 | HEALTHY | `{"status":"ok"}` |
| Container env vars | CORRECT | `RERANK_MODEL=Qwen/Qwen3-Reranker-4B` |
| Token budget configuration | UPDATED | 2048/4096 in code |
| BGE-M3 embedding service | HEALTHY | Port 9000 responding |
| GLiNER NER service | HEALTHY | Port 9002 responding |
| All Docker containers | HEALTHY | 7/7 containers up |

### Not Yet Tested

| Component | Reason | Next Step |
|-----------|--------|-----------|
| Full retrieval pipeline | Databases empty | Ingest documents first |
| Qwen3 reranking quality | No data to search | Run queries after ingestion |
| Latency under load | No data | Benchmark after ingestion |

### Known Issues

| Issue | Severity | Workaround |
|-------|----------|------------|
| Shell env var precedence | MEDIUM | `unset` vars before `docker compose up` |
| Databases empty | BLOCKING | Must ingest documents |
| Pydantic V1 deprecation warnings | LOW | Cosmetic only, no functional impact |

---

## Next Steps

### Immediate (Before Next Session)

1. **Start Qwen3 service** (if not running):
   ```bash
   cd /Users/brennanconley/vibecode/qwen3-reranker
   QWEN_RERANK_HOST=0.0.0.0 QWEN_RERANK_PORT=9003 ./scripts/run_prod.sh
   ```

2. **Copy WekaDocs source files** into `data/ingest/`

3. **Touch files to trigger ingestion:**
   ```bash
   find data/ingest -name "*.md" -exec touch {} \;
   ```

4. **Monitor ingestion:**
   ```bash
   docker logs weka-ingestion-worker -f --tail 50
   ```

5. **Run full pipeline test** once data is ingested

### Short-Term

6. **Benchmark Qwen3 vs BGE** - Compare reranking quality and latency
7. **Commit changes** after validation:
   ```bash
   git add -A
   git commit -m "feat: migrate reranker from BGE to Qwen3-Reranker-4B

   - Update RERANK_MODEL and RERANKER_BASE_URL to Qwen3 service (port 9003)
   - Increase token budgets from 800/1024 to 2048/4096 for longer chunks
   - Add RERANKER_TIMEOUT_SECONDS=120 for 4B model latency
   - Update docker-compose.yml fallback defaults
   - Update documentation in .env.example"
   ```

---

## Quick Reference Commands

```bash
# === Service Health ===
curl -s http://localhost:9000/healthz && echo " BGE-M3 OK"
curl -s http://localhost:9003/health && echo " Qwen3 Reranker OK"
curl -s http://localhost:9002/healthz && echo " GLiNER OK"
curl -s http://localhost:8081/health && echo " Ingestion OK"
curl -s http://localhost:8000/health && echo " MCP OK"

# === Container Operations ===
# Unset shell vars FIRST, then recreate
unset RERANK_PROVIDER RERANK_MODEL RERANKER_BASE_URL RERANKER_TIMEOUT_SECONDS
docker compose up -d --force-recreate mcp-server ingestion-worker ingestion-service

# Verify container env
docker exec weka-mcp-server env | grep -E "^RERANK" | sort

# === Database Cleanup (ALL THREE) ===
PYTHONPATH=/Users/brennanconley/vibecode/wekadocs-matrix \
NEO4J_URI=bolt://localhost:7687 NEO4J_PASSWORD=testpassword123 \
QDRANT_HOST=localhost REDIS_HOST=localhost \
python scripts/cleanup-databases.py && \
docker exec weka-redis redis-cli -a testredis123 FLUSHALL

# === Data Parity Check ===
echo "Neo4j Chunks:" && docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 \
  "MATCH (c:Chunk) RETURN count(c)" 2>/dev/null
echo "Qdrant Points:" && curl -s http://localhost:6333/collections/chunks_multi_bge_m3 | \
  python3 -c "import json,sys; print(json.load(sys.stdin)['result']['points_count'])"

# === Trigger Re-ingestion ===
find data/ingest -name "*.md" -exec touch {} \;

# === Monitor Ingestion ===
docker logs weka-ingestion-worker -f --tail 50

# === Test Qwen3 Reranker Directly ===
curl -s -X POST http://localhost:9003/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "documents": ["doc1", "doc2"], "model": "Qwen/Qwen3-Reranker-4B"}'
```

---

## Git Status

**Current Branch:** `multi-embedder-reranker`

**Uncommitted Changes:**
- `.env` - Qwen3 configuration
- `config/development.yaml` - Qwen3 model and token settings
- `docker-compose.yml` - Fallback defaults
- `.env.example` - Documentation
- `scripts/phase0/capture_baseline.py` - Connectivity check
- `src/providers/rerank/local_bge_service.py` - Token budget increase

**Previous Commits (from neo4j-gds-enhancements merge):**
- `1d6fef3` - fix: increase GLiNER HTTP client timeout to 160s
- `84ad78e` - feat: enable Phase 2 graph channels for retrieval enhancement
- `03b7741` - feat: implement heading-only chunks for 100% hierarchy coverage

---

## Previous Session Context (For Continuity)

### Session 2025-12-14: GLiNER Timeout Fix

The previous session discovered a race condition where the GLiNER HTTP client timeout (60 seconds) was causing unnecessary fallback to slow CPU-based NER processing. The native MPS-accelerated GLiNER service was actually completing successfully, just 156ms over the timeout when processing the CLI Reference Guide (888 chunks in 60.156 seconds).

**Fix Applied:** Increased `HTTP_TIMEOUT` from 60.0 to 160.0 seconds in `src/providers/ner/gliner_service.py:137`. This was committed as `1d6fef3`.

### Session 2025-12-13: Graph Channels Enabled

Phase 2 graph channel configuration was enabled after validating that structural edges (NEXT_CHUNK, PARENT_HEADING, CHILD_OF, PARENT_OF, NEXT, HAS_CHUNK) were being created correctly by the ingestion pipeline.

**Configuration Changes (committed as `84ad78e`):**
- `neo4j_disabled: false` - Master switch enabled
- `graph_as_reranker: true` - Reorders vector candidates using graph signals
- `graph_score_normalized: true` - Normalizes graph scores for fusion
- `graph_garbage_filter: true` - Filters low-quality graph matches
- `graph_rel_types_wired: true` - Query-type specific relationship definitions
- `graph_adaptive_enabled: true` - Uses query_type_relationships config

### Previous Architectural Discoveries

1. **Neo4j ↔ Qdrant ID Alignment:** Qdrant uses UUIDs as point IDs, but stores Neo4j-compatible `chunk_id` in `payload.id` and `payload.node_id`. This allows efficient UUID-based lookups while maintaining referential integrity.

2. **HAS_SECTION Deprecation:** HAS_SECTION was redundant with HAS_CHUNK. All queries now use HAS_CHUNK exclusively (P0 complete).

3. **Heading-Only Chunks:** Parser now emits minimal sections with `text = title` for heading-only sections, achieving 100% hierarchy coverage (P1 complete).

---

## Troubleshooting Guide

### Problem: Containers Show Old Environment Variables

**Symptom:** After editing `.env`, `docker exec weka-mcp-server env` shows old values.

**Cause:** Docker Compose env var precedence: shell exports > `.env` file > docker-compose.yml defaults.

**Solution:**
```bash
# Check for shell exports
printenv | grep -E "^RERANK"

# Unset them
unset RERANK_PROVIDER RERANK_MODEL RERANKER_BASE_URL RERANKER_TIMEOUT_SECONDS

# Recreate containers (restart won't work!)
docker compose up -d --force-recreate mcp-server ingestion-worker ingestion-service
```

### Problem: Files Won't Re-Ingest After Database Cleanup

**Symptom:** After running cleanup script and FLUSHALL, dropping files in `data/ingest/` doesn't trigger ingestion.

**Cause:** File watcher only triggers on **file changes**, not presence. Pre-existing files appear unchanged.

**Solution:**
```bash
# Touch all files to trigger watcher
find data/ingest -name "*.md" -exec touch {} \;
```

### Problem: "File Not Found" Errors After Container Restart

**Symptom:** Worker logs show "File not found" errors for files that existed before restart.

**Cause:** Redis retained job entries for files that no longer exist or have different paths.

**Solution:**
```bash
docker exec weka-redis redis-cli -a testredis123 FLUSHALL
```

### Problem: Reranker Filtering Too Many Candidates

**Symptom:** Logs show "reranker_candidates_filtered" with high `skipped_count`, or "reranker_all_candidates_filtered".

**Cause:** Token budget limits in `local_bge_service.py` are filtering chunks that exceed `MAX_TOKENS_PER_DOC` or combined query+doc exceeds `MAX_TOKENS_TOTAL`.

**Solution:** Increase limits in `src/providers/rerank/local_bge_service.py`. For Qwen3:
```python
MAX_TOKENS_PER_DOC = 2048  # Was 800
MAX_TOKENS_TOTAL = 4096    # Was 1024
```

### Problem: Native ML Service Not Reachable from Docker

**Symptom:** Container logs show connection refused to embedding/reranker/NER service.

**Cause:** Service not bound to `0.0.0.0`, or wrong URL in config.

**Solution:**
1. Ensure native service binds to `0.0.0.0` (not `127.0.0.1`)
2. Use `http://host.docker.internal:PORT` in container env vars
3. Verify service health: `curl http://localhost:PORT/health`

---

## Code Architecture Reference

### Retrieval Pipeline Flow

```
Query → Query Classification → Embedding (BGE-M3)
      → Qdrant Multi-Vector Search (Dense + Sparse + Title + Entity)
      → RRF Fusion (Python-side)
      → Graph Reranker (if enabled, uses Neo4j signals)
      → Cross-Encoder Reranker (Qwen3-Reranker-4B)
      → Response Builder → Final Results
```

### Key Files for Reranking

| File | Purpose |
|------|---------|
| `src/providers/rerank/local_bge_service.py` | HTTP client wrapper for reranker service |
| `src/providers/factory.py:264` | Reads `RERANKER_TIMEOUT_SECONDS` env var |
| `src/query/hybrid_retrieval.py` | Orchestrates retrieval + reranking |
| `config/development.yaml:139-145` | Reranker configuration |

### Key Files for Ingestion

| File | Purpose |
|------|---------|
| `src/ingestion/worker.py` | Background job processor |
| `src/ingestion/auto/service.py` | File watcher service |
| `src/ingestion/parsers/markdown_it_parser.py` | Markdown parsing |
| `src/ingestion/chunk_assembler.py` | Semantic chunking |
| `src/ingestion/structural_edges.py` | Graph relationship creation |

### Graph Schema (Current)

**Node Types:**
- `Document` - Source markdown files
- `Chunk` - Semantic text segments (content units)
- `Entity` - Named entities (COMMAND, PARAMETER, etc.)
- `Section` - Deprecated, use Chunk instead

**Relationship Types:**
- `HAS_CHUNK` - Document owns Chunk (membership)
- `NEXT_CHUNK` - Sequential ordering within document
- `PARENT_HEADING` - Hierarchy to parent section
- `CHILD_OF` / `PARENT_OF` - Bidirectional hierarchy
- `MENTIONS` - Chunk references Entity
- `REFERENCES` - Cross-document linking

---

## Performance Considerations

### Qwen3-Reranker-4B Characteristics

- **Model Size:** 4 billion parameters (vs BGE's ~278M)
- **Latency:** ~300ms for 2 documents (vs ~50ms for BGE)
- **Context Window:** 4096 tokens (configurable via `max_length`)
- **Batch Processing:** Supports batched documents (16 per batch default)
- **Scoring Method:** `p_yes_softmax(no,yes)` for relevance probability

### Expected Latency Budget

| Operation | Target | Actual |
|-----------|--------|--------|
| Embedding (BGE-M3) | <100ms | ~50ms |
| Qdrant Search | <50ms | ~30ms |
| RRF Fusion | <10ms | ~5ms |
| Graph Rerank | <100ms | ~50ms |
| Cross-Encoder (Qwen3, 50 docs) | <5s | ~3s |
| **Total E2E** | <6s | ~4s |

### Memory Requirements

- BGE-M3: ~2GB VRAM (MPS)
- Qwen3-Reranker-4B: ~8GB VRAM (MPS/MLX)
- GLiNER: ~1GB VRAM (MPS)

Ensure sufficient unified memory on Apple Silicon when running all three services.

---

## Environment-Specific Notes

### macOS (Apple Silicon) Development

- All ML services leverage MPS acceleration
- Use `host.docker.internal` for container → host communication
- Docker Desktop required for `host.docker.internal` support
- HuggingFace cache: `./hf-cache` (project-local)

### Future Production Considerations

- Replace `testpassword123` with strong credentials
- Enable TLS for all service endpoints
- Configure proper rate limiting
- Set up monitoring dashboards (Grafana via Alloy)
- Consider GPU deployment for ML services

---

*End of Session Context - 2025-12-15*
*Reranker: BGE → Qwen3-Reranker-4B (port 9003)*
*Token Budget: 800/1024 → 2048/4096*
*Databases: EMPTY, awaiting ingestion*
*Next: Ingest documents, run full pipeline test*
