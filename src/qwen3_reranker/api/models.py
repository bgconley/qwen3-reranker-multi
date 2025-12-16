"""Pydantic models for API request/response schemas."""

from pydantic import BaseModel, Field


class RerankRequest(BaseModel):
    """Request body for POST /v1/rerank."""

    query: str = Field(..., description="Search query")
    documents: list[str] = Field(..., description="Documents to rerank")
    model: str | None = Field(
        default=None,
        description="Model alias (logged but not used for routing)",
    )
    instruction: str | None = Field(
        default=None,
        description="Custom instruction override",
    )
    top_n: int | None = Field(
        default=None,
        description="Return only top N results",
    )
    max_length: int | None = Field(
        default=None,
        description="Override max sequence length",
    )
    return_documents: bool = Field(
        default=False,
        description="Include document text in response",
    )


class RerankResult(BaseModel):
    """Single result in rerank response."""

    index: int = Field(..., description="Original document index")
    score: float = Field(..., description="Relevance score [0, 1]")
    document: str | None = Field(
        default=None,
        description="Document text (if return_documents=True)",
    )


class RerankResponseMeta(BaseModel):
    """Metadata about the rerank operation."""

    max_length: int
    batch_size: int
    scoring: str = "p_yes_softmax(no,yes)"
    truncated_docs: int
    elapsed_ms: float


class RerankResponse(BaseModel):
    """Response body for POST /v1/rerank."""

    results: list[RerankResult] = Field(..., description="Ranked results")
    model: str = Field(..., description="Model ID used")
    meta: RerankResponseMeta = Field(..., description="Operation metadata")


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = Field(..., description="Health status: ok | loading | error")


class HealthzResponse(BaseModel):
    """Response for GET /healthz."""

    status: str
    backend: str
    model_id: str
    profile: str
    model_loaded: bool
    warmup_complete: bool
    uptime_seconds: float
    version: str


class ReadyResponse(BaseModel):
    """Response for GET /ready."""

    ready: bool
    message: str
    backend: str | None = None


class ConfigResponse(BaseModel):
    """Response for GET /v1/config."""

    profile_name: str
    backend: str
    model_id: str
    max_length: int
    batch_size: int
    max_concurrent_forwards: int
    max_docs_per_request: int
    max_query_chars: int
    max_doc_chars: int
    default_instruction: str
    scoring_method: str


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str
    message: str
    details: dict | None = None
