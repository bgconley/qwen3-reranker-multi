Runtime switch (the minimum changes that actually route traffic to Qwen3)

  - Start the Qwen3 service on the host in a way Docker containers can reach it:
      - cd /Users/brennanconley/vibecode/qwen3-reranker && QWEN_RERANK_HOST=0.0.0.0 QWEN_RERANK_PORT=9003 ./scripts/run_prod.sh
      - Verify: curl -s http://127.0.0.1:9003/health
  - Update the env vars that wekadocs-matrix uses to construct the HTTP client:
      - wekadocs-matrix/.env:39 keep RERANK_PROVIDER=bge-reranker-service (this is just the client wrapper name)
      - wekadocs-matrix/.env:40 change RERANK_MODEL to a Qwen label (recommended: Qwen/Qwen3-Reranker-4B)
      - wekadocs-matrix/.env:42 change RERANKER_BASE_URL:
          - If running wekadocs-matrix via Docker Compose: http://host.docker.internal:9003
          - If running wekadocs-matrix natively (no Docker): http://127.0.0.1:9003
      - Optional but commonly needed for a 4B reranker: add RERANKER_TIMEOUT_SECONDS=120 (read by wekadocs-matrix/src/providers/factory.py:264)

  That’s enough to make wekadocs-matrix call your Qwen service because the existing provider posts to POST /v1/rerank with {query, documents, model} (which your service supports).

  Config file alignment (if you rely on YAML config instead of env overrides)

  - wekadocs-matrix/config/development.yaml:141 keep provider: "bge-reranker-service"
  - wekadocs-matrix/config/development.yaml:142 change model: "Qwen/Qwen3-Reranker-4B"
  - wekadocs-matrix/config/development.yaml:145 optionally update max_tokens_per_pair to match your Qwen service profile (e.g. 4096), noting: this value currently isn’t referenced anywhere in code (only defined in
    wekadocs-matrix/src/shared/config.py:275).

  Docker Compose defaults (optional, but avoids “surprise reverts” when env vars aren’t set)

  If you want Compose defaults to point to Qwen by default (instead of falling back to BGE on :9001), change these fallback strings:

  - wekadocs-matrix/docker-compose.yml:150 / :151 / :152
  - wekadocs-matrix/docker-compose.yml:249 / :250 / :251
  - wekadocs-matrix/docker-compose.yml:360 / :361 / :362
    from BAAI/bge-reranker-v2-m3 + http://host.docker.internal:9001 to Qwen/Qwen3-Reranker-4B + http://host.docker.internal:9003.

  Docs/examples to update (optional but likely “other spots” you remembered)

  - wekadocs-matrix/.env.example:64 and wekadocs-matrix/.env.example:67 (model + base URL example)
  - wekadocs-matrix/scripts/phase0/capture_baseline.py:463 defaults RERANKER_BASE_URL to :9001 and uses model: "BAAI/bge-reranker-v2-m3" at wekadocs-matrix/scripts/phase0/capture_baseline.py:469 (your service
    currently ignores model, so it will still work, but the script/docs will be misleading).

  One non-config “gotcha” to consider (might affect quality/coverage)

  - The HTTP client used (wekadocs-matrix/src/providers/rerank/local_bge_service.py:27) has hard-coded “token budget” filters (MAX_TOKENS_PER_DOC=800, MAX_TOKENS_TOTAL=1024) based on whitespace splitting, and it
    will skip candidates that exceed them. That can prevent Qwen from seeing longer chunks even if your Qwen service supports max_length=4096+. If you notice lots of candidates being filtered or rerank quality looks
    off, that file is the place that would need adjustment.



      The reranker service won’t print “periodic API results” unless you actually send requests to it. To see request logs, hit it with:

  - ./scripts/smoke_test.sh http://127.0.0.1:9003
  - or curl -s http://127.0.0.1:9003/healthz
  - or a real rerank call: curl -s -X POST http://127.0.0.1:9003/v1/rerank -H 'Content-Type: application/json' -d '{"query":"test","documents":["a","b"],"model":"Qwen/Qwen3-Reranker-4B"}'
    The reranker service won’t print “periodic API results” unless you actually send requests to it. To see request logs, hit it with:

  - ./scripts/smoke_test.sh http://127.0.0.1:9003
  - or curl -s http://127.0.0.1:9003/healthz
  - or a real rerank call: curl -s -X POST http://127.0.0.1:9003/v1/rerank -H 'Content-Type: application/json' -d '{"query":"test","documents":["a","b"],"model":"Qwen/Qwen3-Reranker-4B"}'  The reranker service won’t print “periodic API results” unless you actually send requests to it. To see request logs, hit it with:

  - ./scripts/smoke_test.sh http://127.0.0.1:9003
  - or curl -s http://127.0.0.1:9003/healthz
  - or a real rerank call: curl -s -X POST http://127.0.0.1:9003/v1/rerank -H 'Content-Type: application/json' -d '{"query":"test","documents":["a","b"],"model":"Qwen/Qwen3-Reranker-4B"}'