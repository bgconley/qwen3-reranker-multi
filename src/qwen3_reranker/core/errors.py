"""Custom exceptions for Qwen3-Reranker service."""

from typing import Any


class RerankerError(Exception):
    """Base exception for reranker errors."""

    error_code: str = "RERANKER_ERROR"
    status_code: int = 500

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON response."""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
        }


# Configuration errors
class ConfigurationError(RerankerError):
    """Configuration error."""

    error_code = "CONFIGURATION_ERROR"
    status_code = 500


# Model errors
class ModelLoadError(RerankerError):
    """Model loading error."""

    error_code = "MODEL_LOAD_ERROR"
    status_code = 500


class ModelNotLoadedError(RerankerError):
    """Model not loaded error."""

    error_code = "MODEL_NOT_LOADED"
    status_code = 503


# Validation errors (4xx)
class ValidationError(RerankerError):
    """Request validation error."""

    error_code = "VALIDATION_ERROR"
    status_code = 400


class EmptyQueryError(ValidationError):
    """Empty query error."""

    error_code = "EMPTY_QUERY"


class EmptyDocumentsError(ValidationError):
    """Empty documents error."""

    error_code = "EMPTY_DOCUMENTS"


class TooManyDocumentsError(ValidationError):
    """Too many documents error."""

    error_code = "TOO_MANY_DOCUMENTS"


class QueryTooLongError(ValidationError):
    """Query too long error."""

    error_code = "QUERY_TOO_LONG"


class DocumentTooLongError(ValidationError):
    """Document too long error."""

    error_code = "DOCUMENT_TOO_LONG"


class ModelAliasNotAllowedError(ValidationError):
    """Model alias not allowed error."""

    error_code = "MODEL_ALIAS_NOT_ALLOWED"


# Processing errors
class ScoringError(RerankerError):
    """Scoring computation error."""

    error_code = "SCORING_ERROR"


class TokenizationError(RerankerError):
    """Tokenization error."""

    error_code = "TOKENIZATION_ERROR"


class InvalidTokenError(RerankerError):
    """Invalid token error."""

    error_code = "INVALID_TOKEN"
    status_code = 500


# Concurrency errors
class ConcurrencyLimitError(RerankerError):
    """Concurrency limit exceeded."""

    error_code = "CONCURRENCY_LIMIT"
    status_code = 503


# Backend errors
class BackendNotAvailableError(RerankerError):
    """Backend not available error."""

    error_code = "BACKEND_NOT_AVAILABLE"
    status_code = 503
