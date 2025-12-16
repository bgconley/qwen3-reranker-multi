"""Tests for API contract compatibility with wekadocs-matrix."""

import pytest
from pydantic import ValidationError

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


class TestRerankRequest:
    """Tests for RerankRequest model."""

    def test_minimal_request(self) -> None:
        """Test minimal valid request."""
        request = RerankRequest(
            query="test query",
            documents=["doc1", "doc2"],
        )

        assert request.query == "test query"
        assert request.documents == ["doc1", "doc2"]
        assert request.model is None  # Default

    def test_full_request(self) -> None:
        """Test request with all fields."""
        request = RerankRequest(
            query="test query",
            documents=["doc1"],
            model="some-model",
            instruction="custom instruction",
            top_n=10,
            return_documents=True,
            max_length=2048,
        )

        assert request.model == "some-model"
        assert request.instruction == "custom instruction"
        assert request.top_n == 10
        assert request.return_documents is True
        assert request.max_length == 2048

    def test_missing_query_raises(self) -> None:
        """Test that missing query raises validation error."""
        with pytest.raises(ValidationError):
            RerankRequest(documents=["doc1"])

    def test_missing_documents_raises(self) -> None:
        """Test that missing documents raises validation error."""
        with pytest.raises(ValidationError):
            RerankRequest(query="test")


class TestRerankResponse:
    """Tests for RerankResponse model (wekadocs contract)."""

    def test_minimal_response(self) -> None:
        """Test minimal valid response."""
        response = RerankResponse(
            results=[
                RerankResult(index=0, score=0.95),
                RerankResult(index=1, score=0.85),
            ],
            model="test-model",
            meta=RerankResponseMeta(
                max_length=4096,
                batch_size=8,
                scoring="p_yes_softmax(no,yes)",
                truncated_docs=0,
                elapsed_ms=42.5,
            ),
        )

        assert len(response.results) == 2
        assert response.results[0].index == 0
        assert response.results[0].score == 0.95
        assert response.model == "test-model"
        assert response.meta is not None

    def test_full_response(self) -> None:
        """Test response with all fields."""
        response = RerankResponse(
            results=[
                RerankResult(index=1, score=0.99, document="doc text"),
            ],
            model="test-model",
            meta=RerankResponseMeta(
                max_length=4096,
                batch_size=8,
                scoring="p_yes_softmax(no,yes)",
                truncated_docs=2,
                elapsed_ms=42.5,
            ),
        )

        assert response.results[0].document == "doc text"
        assert response.meta is not None
        assert response.meta.truncated_docs == 2

    def test_results_sorted_descending(self) -> None:
        """Verify results can represent sorted order."""
        # The model should return results sorted by score descending
        response = RerankResponse(
            results=[
                RerankResult(index=2, score=0.95),
                RerankResult(index=0, score=0.80),
                RerankResult(index=1, score=0.60),
            ],
            model="test",
            meta=RerankResponseMeta(
                max_length=4096,
                batch_size=8,
                scoring="p_yes_softmax(no,yes)",
                truncated_docs=0,
                elapsed_ms=10.0,
            ),
        )

        scores = [r.score for r in response.results]
        assert scores == sorted(scores, reverse=True)


class TestRerankResult:
    """Tests for individual rerank results."""

    def test_valid_scores(self) -> None:
        """Test that valid scores are accepted."""
        RerankResult(index=0, score=0.0)
        RerankResult(index=0, score=1.0)
        RerankResult(index=0, score=0.5)

    def test_result_with_document(self) -> None:
        """Test result with document text."""
        result = RerankResult(index=0, score=0.9, document="Sample document text")
        assert result.document == "Sample document text"


class TestHealthResponse:
    """Tests for health endpoint responses."""

    def test_health_ok(self) -> None:
        """Test healthy status."""
        response = HealthResponse(status="ok")
        assert response.status == "ok"

    def test_health_loading(self) -> None:
        """Test loading status."""
        response = HealthResponse(status="loading")
        assert response.status == "loading"


class TestHealthzResponse:
    """Tests for detailed health response."""

    def test_healthz_full(self) -> None:
        """Test full healthz response."""
        response = HealthzResponse(
            status="ok",
            backend="pytorch",
            model_id="test/model",
            profile="test_profile",
            model_loaded=True,
            warmup_complete=True,
            uptime_seconds=123.45,
            version="0.1.0",
        )

        assert response.model_loaded is True
        assert response.warmup_complete is True
        assert response.uptime_seconds == 123.45
        assert response.backend == "pytorch"


class TestReadyResponse:
    """Tests for readiness probe response."""

    def test_ready_true(self) -> None:
        """Test ready state."""
        response = ReadyResponse(
            ready=True, message="Service is ready", backend="pytorch"
        )
        assert response.ready is True
        assert response.backend == "pytorch"

    def test_ready_false(self) -> None:
        """Test not ready state."""
        response = ReadyResponse(ready=False, message="Warming up")
        assert response.ready is False


class TestConfigResponse:
    """Tests for config endpoint response."""

    def test_config_response(self) -> None:
        """Test config response structure."""
        response = ConfigResponse(
            profile_name="qwen3_4b_cuda",
            backend="pytorch",
            model_id="Qwen/Qwen3-Reranker-4B",
            max_length=4096,
            batch_size=8,
            max_concurrent_forwards=1,
            max_docs_per_request=200,
            max_query_chars=8000,
            max_doc_chars=20000,
            default_instruction="Given a web search query...",
            scoring_method="yes_no_next_token_prob",
        )

        assert response.profile_name == "qwen3_4b_cuda"
        assert response.max_length == 4096
        assert response.backend == "pytorch"


class TestErrorResponse:
    """Tests for error response format."""

    def test_error_response(self) -> None:
        """Test error response structure."""
        response = ErrorResponse(
            error="VALIDATION_ERROR",
            message="Query cannot be empty",
            details={"field": "query"},
        )

        assert response.error == "VALIDATION_ERROR"
        assert "empty" in response.message.lower()
        assert response.details["field"] == "query"

    def test_error_response_empty_details(self) -> None:
        """Test error response with empty details."""
        response = ErrorResponse(
            error="INTERNAL_ERROR",
            message="Something went wrong",
        )

        assert response.details is None
