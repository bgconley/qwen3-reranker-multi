"""FastAPI application for Qwen3 Reranker Service.

Provides:
- POST /v1/rerank (wekadocs-compatible)
- GET /health (simple health check)
- GET /healthz (detailed health)
- GET /ready (readiness probe)
- GET /v1/config (configuration introspection)
"""

from qwen3_reranker.api.app import app, main

__all__ = ["app", "main"]
