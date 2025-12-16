#!/usr/bin/env bash
# Development server - auto-detects best available backend
# Usage: ./scripts/run_dev.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Auto-detect backend, default profile based on available hardware
export QWEN_RERANK_BACKEND="${QWEN_RERANK_BACKEND:-auto}"
export QWEN_RERANK_PROFILE="${QWEN_RERANK_PROFILE:-qwen3_4b_cuda}"
export QWEN_RERANK_PORT="${QWEN_RERANK_PORT:-9003}"
export QWEN_RERANK_LOG_LEVEL="${QWEN_RERANK_LOG_LEVEL:-DEBUG}"
export QWEN_RERANK_LOG_FORMAT="${QWEN_RERANK_LOG_FORMAT:-console}"
export QWEN_RERANK_HOST="${QWEN_RERANK_HOST:-127.0.0.1}"

echo "Starting Qwen3-Reranker Service (development mode)"
echo "  Backend: $QWEN_RERANK_BACKEND"
echo "  Profile: $QWEN_RERANK_PROFILE"
echo "  Port: $QWEN_RERANK_PORT"
echo "  Log Level: $QWEN_RERANK_LOG_LEVEL"
echo ""

# macOS ships an older bash by default which doesn't support `${var,,}`.
LOG_LEVEL_LOWER="$(printf '%s' "$QWEN_RERANK_LOG_LEVEL" | tr '[:upper:]' '[:lower:]')"

# Run with reload (single worker only)
exec uvicorn qwen3_reranker.api.app:app \
    --host "$QWEN_RERANK_HOST" \
    --port "$QWEN_RERANK_PORT" \
    --reload \
    --reload-dir src \
    --workers 1 \
    --log-level "$LOG_LEVEL_LOWER"
