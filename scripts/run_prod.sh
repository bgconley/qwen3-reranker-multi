#!/usr/bin/env bash
# Production server (no reload, optimized settings)
# Usage: ./scripts/run_prod.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Default environment
export QWEN_RERANK_PROFILE="${QWEN_RERANK_PROFILE:-qwen3_4b_mlx_fp16}"
export QWEN_RERANK_PORT="${QWEN_RERANK_PORT:-9003}"
export QWEN_RERANK_LOG_LEVEL="${QWEN_RERANK_LOG_LEVEL:-INFO}"
export QWEN_RERANK_LOG_FORMAT="${QWEN_RERANK_LOG_FORMAT:-json}"
export QWEN_RERANK_HOST="${QWEN_RERANK_HOST:-127.0.0.1}"

echo "Starting Qwen3 Reranker Service (production mode)"
echo "  Profile: $QWEN_RERANK_PROFILE"
echo "  Port: $QWEN_RERANK_PORT"
echo "  Host: $QWEN_RERANK_HOST"
echo ""

# macOS ships an older bash by default which doesn't support `${var,,}`.
LOG_LEVEL_LOWER="$(printf '%s' "$QWEN_RERANK_LOG_LEVEL" | tr '[:upper:]' '[:lower:]')"

# Run without reload, single worker (MLX constraint)
exec uvicorn qwen3_reranker_service.api:app \
    --host "$QWEN_RERANK_HOST" \
    --port "$QWEN_RERANK_PORT" \
    --workers 1 \
    --log-level "$LOG_LEVEL_LOWER"
