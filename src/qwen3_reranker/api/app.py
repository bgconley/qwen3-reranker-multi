"""FastAPI application for Qwen3 Reranker Service.

Multi-backend support:
- PyTorch (PRIMARY): Cross-platform CUDA/MPS/CPU
- vLLM (SECONDARY): High-throughput CUDA
- MLX (TERTIARY): Apple Silicon optimization
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import structlog
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from qwen3_reranker.api.models import (
    ConfigResponse,
    ErrorResponse,
    HealthResponse,
    HealthzResponse,
    ReadyResponse,
    RerankRequest,
    RerankResponse,
    RerankResponseMeta,
    RerankResult,
)
from qwen3_reranker.backends import get_backend
from qwen3_reranker.backends.base import RerankerBackend
from qwen3_reranker.core.batching import init_concurrency_guard, rerank_documents
from qwen3_reranker.core.config import AppConfig, get_config
from qwen3_reranker.core.errors import (
    DocumentTooLongError,
    EmptyDocumentsError,
    EmptyQueryError,
    ModelAliasNotAllowedError,
    QueryTooLongError,
    RerankerError,
    TooManyDocumentsError,
)
from qwen3_reranker.core.prompt import PromptFormatter
from qwen3_reranker.core.tokenization import RerankerTokenizer, setup_tokenizer
from qwen3_reranker.version import __version__

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Global state
_config: AppConfig | None = None
_backend: RerankerBackend | None = None
_tokenizer: RerankerTokenizer | None = None
_prompt_formatter: PromptFormatter | None = None
_start_time: float = 0.0
_warmup_complete: bool = False


def get_app_config() -> AppConfig:
    """Get the application configuration."""
    global _config
    if _config is None:
        _config = get_config()
    return _config


def get_loaded_backend() -> RerankerBackend:
    """Get the loaded backend."""
    if _backend is None or not _backend.is_loaded:
        raise HTTPException(status_code=503, detail="Backend not loaded")
    return _backend


def get_loaded_tokenizer() -> RerankerTokenizer:
    """Get the loaded tokenizer."""
    if _tokenizer is None:
        raise HTTPException(status_code=503, detail="Tokenizer not loaded")
    return _tokenizer


def get_loaded_formatter() -> PromptFormatter:
    """Get the loaded prompt formatter."""
    if _prompt_formatter is None:
        raise HTTPException(status_code=503, detail="Prompt formatter not loaded")
    return _prompt_formatter


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    global _config, _backend, _tokenizer, _prompt_formatter, _start_time, _warmup_complete

    # Startup
    _start_time = time.time()

    # Load configuration
    config = get_app_config()

    # Configure logging level
    logging.basicConfig(level=getattr(logging, config.settings.log_level))

    logger.info(
        "startup",
        version=__version__,
        profile=config.profile_name,
        backend=config.backend,
        model_id=config.model_id,
    )

    # Initialize concurrency guard
    init_concurrency_guard(config.max_concurrent_forwards)

    # Get backend options from profile
    backend_kwargs: dict[str, Any] = {}
    if config.backend == "pytorch" and config.profile.pytorch_options:
        backend_kwargs["device"] = config.settings.device or config.profile.pytorch_options.device
    elif config.backend == "vllm" and config.profile.vllm_options:
        backend_kwargs["tensor_parallel_size"] = (
            config.settings.tensor_parallel_size
            or config.profile.vllm_options.tensor_parallel_size
        )
        backend_kwargs["gpu_memory_utilization"] = (
            config.settings.gpu_memory_utilization
            or config.profile.vllm_options.gpu_memory_utilization
        )
        backend_kwargs["max_model_len"] = config.profile.vllm_options.max_model_len

    # Get and load backend
    try:
        _backend = get_backend(config.backend, **backend_kwargs)
        logger.info("loading_model", backend=_backend.backend_name, model_id=config.model_id)
        _backend.load_model(config.model_id)
    except Exception as e:
        logger.error("startup_model_load_failed", error=str(e))
        raise

    # Set up tokenizer
    tokenizer = _backend.get_tokenizer()
    scoring_config = config.profile.scoring
    _tokenizer = setup_tokenizer(
        tokenizer=tokenizer,
        prefix=scoring_config.prefix,
        suffix=scoring_config.suffix,
        yes_token=scoring_config.yes_token,
        no_token=scoring_config.no_token,
        max_length=config.max_length,
    )

    # Set up prompt formatter
    _prompt_formatter = PromptFormatter.from_scoring_config(
        prefix=scoring_config.prefix,
        suffix=scoring_config.suffix,
        query_template=scoring_config.query_template,
        default_instruction=config.profile.defaults.instruction,
    )

    # Run warmup
    try:
        warmup_ms = _backend.warmup()
        _warmup_complete = True
        logger.info("startup_warmup_complete", warmup_ms=warmup_ms)
    except Exception as e:
        logger.error("startup_warmup_failed", error=str(e))
        raise

    logger.info("startup_complete", backend=_backend.backend_name)

    yield

    # Shutdown
    logger.info("shutdown")


# Create FastAPI app
app = FastAPI(
    title="Qwen3 Reranker Service",
    description="Multi-backend Qwen3-Reranker-4B HTTP service for reranking documents",
    version=__version__,
    lifespan=lifespan,
)


@app.middleware("http")
async def request_middleware(request: Request, call_next) -> Response:
    """Middleware for request logging and correlation ID handling."""
    config = get_app_config()

    # Get or generate correlation ID
    correlation_id = request.headers.get(config.settings.correlation_header)
    if not correlation_id:
        correlation_id = str(uuid.uuid4())

    # Persist correlation id on request state for downstream handlers
    request.state.correlation_id = correlation_id

    start_time = time.perf_counter()

    try:
        response = await call_next(request)

        # Add correlation ID to response
        response.headers[config.settings.correlation_header] = correlation_id

        # Log request completion
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "request_complete",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            elapsed_ms=round(elapsed_ms, 2),
            correlation_id=correlation_id,
        )

        return response

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "request_error",
            method=request.method,
            path=request.url.path,
            error=str(e),
            elapsed_ms=round(elapsed_ms, 2),
            correlation_id=correlation_id,
        )
        raise


@app.exception_handler(RerankerError)
async def reranker_error_handler(request: Request, exc: RerankerError) -> JSONResponse:
    """Handle RerankerError exceptions."""
    correlation_id = getattr(request.state, "correlation_id", "unknown")
    logger.error(
        "reranker_error",
        error=exc.error_code,
        message=exc.message,
        correlation_id=correlation_id,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTP_ERROR",
            message=str(exc.detail),
            details={},
        ).model_dump(),
    )


def validate_rerank_request(request: RerankRequest, config: AppConfig) -> None:
    """Validate a rerank request against configuration limits."""
    limits = config.profile.limits

    # Check empty query
    if not request.query or not request.query.strip():
        raise EmptyQueryError("Query cannot be empty")

    # Check empty documents
    if not request.documents:
        raise EmptyDocumentsError("Documents list cannot be empty")

    # Check document count
    if len(request.documents) > limits.max_docs_per_request:
        raise TooManyDocumentsError(
            f"Too many documents: {len(request.documents)} > {limits.max_docs_per_request}",
            {"count": len(request.documents), "max": limits.max_docs_per_request},
        )

    # Optional model alias allowlist enforcement
    allowed_aliases = config.settings.get_model_aliases()
    if allowed_aliases is not None and request.model and request.model not in allowed_aliases:
        raise ModelAliasNotAllowedError(
            f"Model alias not allowed: {request.model}",
            {"model": request.model, "allowed": sorted(allowed_aliases)},
        )

    # Check query length
    if len(request.query) > limits.max_query_chars:
        raise QueryTooLongError(
            f"Query too long: {len(request.query)} > {limits.max_query_chars} chars",
            {"length": len(request.query), "max": limits.max_query_chars},
        )

    # Check document lengths
    for i, doc in enumerate(request.documents):
        if len(doc) > limits.max_doc_chars:
            raise DocumentTooLongError(
                f"Document {i} too long: {len(doc)} > {limits.max_doc_chars} chars",
                {"index": i, "length": len(doc), "max": limits.max_doc_chars},
            )


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(
    request: RerankRequest,
    http_request: Request,
    x_correlation_id: str | None = Header(None, alias="X-Correlation-Id"),
) -> RerankResponse:
    """Rerank documents for a query.

    This endpoint is compatible with wekadocs-matrix local_bge_service provider.
    """
    config = get_app_config()
    correlation_id = (
        getattr(http_request.state, "correlation_id", None)
        or x_correlation_id
        or str(uuid.uuid4())
    )

    # Validate request
    validate_rerank_request(request, config)

    # Determine effective max_length
    effective_max_length = config.max_length
    if request.max_length is not None:
        effective_max_length = min(
            request.max_length,
            config.profile.limits.max_length_hard_cap,
        )

    # Log rerank request
    logger.info(
        "rerank_request",
        doc_count=len(request.documents),
        query_chars=len(request.query),
        max_length=effective_max_length,
        batch_size=config.batch_size,
        correlation_id=correlation_id,
    )

    # Get components
    backend = get_loaded_backend()
    tokenizer = get_loaded_tokenizer()
    prompt_formatter = get_loaded_formatter()

    # Apply max_length override without mutating global tokenizer state
    request_tokenizer = tokenizer
    if tokenizer.max_length != effective_max_length:
        request_tokenizer = tokenizer.clone_with_max_length(effective_max_length)

    # Run reranking
    ranked_results, stats = await rerank_documents(
        backend=backend,
        tokenizer=request_tokenizer,
        prompt_formatter=prompt_formatter,
        query=request.query,
        documents=request.documents,
        batch_size=config.batch_size,
        instruction=request.instruction,
        top_n=request.top_n,
        forward_timeout=config.settings.request_timeout,
    )

    # Log completion
    logger.info(
        "rerank_complete",
        doc_count=stats.doc_count,
        batches=stats.batches_processed,
        elapsed_ms=round(stats.elapsed_ms, 2),
        truncated_docs=stats.truncated_docs,
        correlation_id=correlation_id,
    )

    # Build results
    results = []
    for idx, score in ranked_results:
        result = RerankResult(
            index=idx,
            score=score,
            document=request.documents[idx] if request.return_documents else None,
        )
        results.append(result)

    # Build response
    return RerankResponse(
        results=results,
        model=config.model_id,
        meta=RerankResponseMeta(
            max_length=effective_max_length,
            batch_size=config.batch_size,
            scoring="p_yes_softmax(no,yes)",
            truncated_docs=stats.truncated_docs,
            elapsed_ms=round(stats.elapsed_ms, 2),
        ),
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Basic health check (wekadocs-compatible)."""
    if _backend and _backend.is_loaded:
        return HealthResponse(status="ok")
    else:
        return HealthResponse(status="loading")


@app.get("/healthz", response_model=HealthzResponse)
async def healthz() -> HealthzResponse:
    """Detailed health check with diagnostics."""
    config = get_app_config()

    status = "ok"
    if _backend is None or not _backend.is_loaded:
        status = "loading"
    elif not _warmup_complete:
        status = "warming_up"

    uptime = time.time() - _start_time
    backend_name = _backend.backend_name if _backend else "not_initialized"

    return HealthzResponse(
        status=status,
        backend=backend_name,
        model_id=config.model_id,
        profile=config.profile_name,
        model_loaded=_backend.is_loaded if _backend else False,
        warmup_complete=_warmup_complete,
        uptime_seconds=round(uptime, 2),
        version=__version__,
    )


@app.get("/ready", response_model=ReadyResponse)
async def ready() -> ReadyResponse:
    """Readiness probe."""
    if _backend and _backend.is_loaded and _warmup_complete:
        return ReadyResponse(
            ready=True,
            message="Service is ready",
            backend=_backend.backend_name,
        )

    if not _backend or not _backend.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not _warmup_complete:
        raise HTTPException(status_code=503, detail="Warmup not complete")

    raise HTTPException(status_code=503, detail="Service not ready")


@app.get("/v1/config", response_model=ConfigResponse)
async def get_service_config() -> ConfigResponse:
    """Get current service configuration."""
    config = get_app_config()
    backend_name = _backend.backend_name if _backend else config.backend

    return ConfigResponse(
        profile_name=config.profile_name,
        backend=backend_name,
        model_id=config.model_id,
        max_length=config.max_length,
        batch_size=config.batch_size,
        max_concurrent_forwards=config.max_concurrent_forwards,
        max_docs_per_request=config.profile.limits.max_docs_per_request,
        max_query_chars=config.profile.limits.max_query_chars,
        max_doc_chars=config.profile.limits.max_doc_chars,
        default_instruction=config.profile.defaults.instruction,
        scoring_method=config.profile.scoring.method,
    )


def main() -> None:
    """Entry point for the service."""
    config = get_config()

    uvicorn.run(
        "qwen3_reranker.api.app:app",
        host=config.settings.host,
        port=config.settings.port,
        workers=1,  # Always single worker for model inference
        reload=False,
        log_level=config.settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
