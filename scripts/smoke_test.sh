#!/usr/bin/env bash
# Smoke test for Qwen3 Reranker Service
# Usage: ./scripts/smoke_test.sh [base_url]

set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:9003}"

echo "=== Qwen3 Reranker Smoke Test ==="
echo "Base URL: $BASE_URL"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
}

fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    exit 1
}

warn() {
    echo -e "${YELLOW}! WARN${NC}: $1"
}

# Test 1: Health endpoint
echo "Test 1: GET /health"
HEALTH_RESPONSE=$(curl -s -w "\n%{http_code}" "$BASE_URL/health")
HEALTH_CODE=$(echo "$HEALTH_RESPONSE" | tail -n1)
HEALTH_BODY=$(echo "$HEALTH_RESPONSE" | sed '$d')

if [ "$HEALTH_CODE" = "200" ]; then
    pass "/health returns 200"
    echo "  Response: $HEALTH_BODY"
else
    fail "/health returned $HEALTH_CODE"
fi
echo ""

# Test 2: Ready endpoint
echo "Test 2: GET /ready"
READY_RESPONSE=$(curl -s -w "\n%{http_code}" "$BASE_URL/ready")
READY_CODE=$(echo "$READY_RESPONSE" | tail -n1)
READY_BODY=$(echo "$READY_RESPONSE" | sed '$d')

if [ "$READY_CODE" = "200" ]; then
    pass "/ready returns 200"
    echo "  Response: $READY_BODY"
else
    warn "/ready returned $READY_CODE (service may still be warming up)"
    echo "  Response: $READY_BODY"
fi
echo ""

# Test 3: Healthz endpoint
echo "Test 3: GET /healthz"
HEALTHZ_RESPONSE=$(curl -s -w "\n%{http_code}" "$BASE_URL/healthz")
HEALTHZ_CODE=$(echo "$HEALTHZ_RESPONSE" | tail -n1)
HEALTHZ_BODY=$(echo "$HEALTHZ_RESPONSE" | sed '$d')

if [ "$HEALTHZ_CODE" = "200" ]; then
    pass "/healthz returns 200"
    echo "  Response: $HEALTHZ_BODY" | python3 -m json.tool 2>/dev/null || echo "  Response: $HEALTHZ_BODY"
else
    fail "/healthz returned $HEALTHZ_CODE"
fi
echo ""

# Test 4: Config endpoint
echo "Test 4: GET /v1/config"
CONFIG_RESPONSE=$(curl -s -w "\n%{http_code}" "$BASE_URL/v1/config")
CONFIG_CODE=$(echo "$CONFIG_RESPONSE" | tail -n1)
CONFIG_BODY=$(echo "$CONFIG_RESPONSE" | sed '$d')

if [ "$CONFIG_CODE" = "200" ]; then
    pass "/v1/config returns 200"
    echo "  Response: $CONFIG_BODY" | python3 -m json.tool 2>/dev/null || echo "  Response: $CONFIG_BODY"
else
    fail "/v1/config returned $CONFIG_CODE"
fi
echo ""

# Test 5: Rerank endpoint
echo "Test 5: POST /v1/rerank"
RERANK_REQUEST='{
    "query": "What is the capital of France?",
    "documents": [
        "Paris is the capital and largest city of France.",
        "Berlin is the capital of Germany.",
        "The Eiffel Tower is located in Paris, France.",
        "London is the capital of the United Kingdom."
    ],
    "model": "qwen3-reranker-4b"
}'

RERANK_RESPONSE=$(curl -s -w "\n%{http_code}" \
    -X POST \
    -H "Content-Type: application/json" \
    -H "X-Correlation-Id: smoke-test-$(date +%s)" \
    -d "$RERANK_REQUEST" \
    "$BASE_URL/v1/rerank")
RERANK_CODE=$(echo "$RERANK_RESPONSE" | tail -n1)
RERANK_BODY=$(echo "$RERANK_RESPONSE" | sed '$d')

if [ "$RERANK_CODE" = "200" ]; then
    pass "/v1/rerank returns 200"
    echo "  Response:"
    echo "$RERANK_BODY" | python3 -m json.tool 2>/dev/null || echo "  $RERANK_BODY"
else
    fail "/v1/rerank returned $RERANK_CODE"
    echo "  Response: $RERANK_BODY"
fi
echo ""

# Test 6: Verify ranking order
echo "Test 6: Verify ranking makes sense"
# Extract first result index
FIRST_INDEX=$(echo "$RERANK_BODY" | python3 -c "import json,sys; r=json.load(sys.stdin); print(r['results'][0]['index'])" 2>/dev/null || echo "")

if [ "$FIRST_INDEX" = "0" ]; then
    pass "Top result is document 0 ('Paris is the capital...') - makes sense!"
elif [ "$FIRST_INDEX" = "2" ]; then
    pass "Top result is document 2 ('Eiffel Tower in Paris...') - acceptable"
else
    warn "Top result is document $FIRST_INDEX - may want to verify ranking quality"
fi
echo ""

echo "=== Smoke Test Complete ==="
echo ""
