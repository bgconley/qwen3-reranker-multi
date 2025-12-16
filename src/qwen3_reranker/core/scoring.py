"""Scoring Module - Backend-agnostic yes/no probability extraction.

This module implements the Qwen3 reranker scoring method:
1. Extract logits at final position (next-token prediction)
2. Get logits for "yes" and "no" tokens only
3. Apply softmax over [no, yes] to get p(yes) ∈ [0, 1]

The scoring is identical regardless of backend (PyTorch/vLLM/MLX).
All backends return numpy arrays for consistency.
"""

import logging
from typing import Any

import numpy as np

from qwen3_reranker.core.errors import ScoringError

logger = logging.getLogger(__name__)


def extract_yes_no_scores(
    logits: np.ndarray,
    yes_token_id: int,
    no_token_id: int,
) -> np.ndarray:
    """Extract p(yes) scores from logits using softmax over [no, yes].

    Args:
        logits: Shape [batch, vocab_size], logits at final position
        yes_token_id: Token ID for "yes"
        no_token_id: Token ID for "no"

    Returns:
        np.ndarray of shape [batch] with p(yes) scores in [0, 1]

    Note:
        Higher score = more relevant document.
        Score of 0.5 means model is uncertain.
    """
    # Extract relevant logits
    logit_no = logits[:, no_token_id]
    logit_yes = logits[:, yes_token_id]

    # Stack: [batch, 2] where [:, 0] = no, [:, 1] = yes
    stacked = np.stack([logit_no, logit_yes], axis=1)

    # Numerically stable softmax
    max_logits = np.max(stacked, axis=1, keepdims=True)
    exp_logits = np.exp(stacked - max_logits)
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Return p(yes) which is index 1
    scores = softmax_probs[:, 1]

    return scores.astype(np.float32)


def get_yes_no_token_ids(tokenizer: Any) -> tuple[int, int]:
    """Get token IDs for "yes" and "no", with validation.

    Critical: Both "yes" and "no" must be single tokens for the
    Qwen3 reranker scoring method to work correctly.

    Args:
        tokenizer: HuggingFace tokenizer instance

    Returns:
        Tuple of (yes_token_id, no_token_id)

    Raises:
        ValueError: If "yes" or "no" tokenize to multiple tokens
    """
    yes_tokens = tokenizer.encode("yes", add_special_tokens=False)
    no_tokens = tokenizer.encode("no", add_special_tokens=False)

    if len(yes_tokens) != 1:
        raise ValueError(
            f"'yes' tokenizes to {len(yes_tokens)} tokens: {yes_tokens}. "
            "Expected single token. This tokenizer may not be compatible "
            "with Qwen3 reranker scoring."
        )
    if len(no_tokens) != 1:
        raise ValueError(
            f"'no' tokenizes to {len(no_tokens)} tokens: {no_tokens}. "
            "Expected single token. This tokenizer may not be compatible "
            "with Qwen3 reranker scoring."
        )

    logger.debug(f"Token IDs: yes={yes_tokens[0]}, no={no_tokens[0]}")
    return yes_tokens[0], no_tokens[0]


def validate_score_distribution(scores: np.ndarray) -> dict[str, Any]:
    """Analyze score distribution for debugging/monitoring.

    Returns statistics useful for detecting scoring issues.
    """
    return {
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "std": float(np.std(scores)),
        "num_above_0.5": int(np.sum(scores > 0.5)),
        "num_below_0.5": int(np.sum(scores < 0.5)),
    }


def rank_by_scores(
    scores: list[float] | np.ndarray,
    top_n: int | None = None,
) -> list[tuple[int, float]]:
    """Rank documents by scores in descending order.

    Args:
        scores: Scores for each document
        top_n: Return only top N results (None for all)

    Returns:
        List of (original_index, score) tuples sorted by score descending
    """
    # Create (index, score) pairs
    if isinstance(scores, np.ndarray):
        scores_list = scores.tolist()
    else:
        scores_list = list(scores)

    indexed_scores = list(enumerate(scores_list))

    # Sort by score descending
    ranked = sorted(indexed_scores, key=lambda x: x[1], reverse=True)

    # Apply top_n limit if specified
    if top_n is not None and top_n > 0:
        ranked = ranked[:top_n]

    return ranked


class RerankerScorer:
    """Scores query-document pairs using yes/no probability.

    This class encapsulates the scoring logic and provides
    a clean interface for batch scoring.
    """

    def __init__(
        self,
        yes_token_id: int,
        no_token_id: int,
    ) -> None:
        """Initialize the scorer.

        Args:
            yes_token_id: Token ID for "yes"
            no_token_id: Token ID for "no"
        """
        self.yes_token_id = yes_token_id
        self.no_token_id = no_token_id

    def score_logits(self, logits: np.ndarray) -> list[float]:
        """Score a batch of logits.

        Args:
            logits: Logits at last position [batch_size, vocab_size]

        Returns:
            List of scores in [0, 1]

        Raises:
            ScoringError: If scoring computation fails
        """
        try:
            scores = extract_yes_no_scores(
                logits=logits,
                yes_token_id=self.yes_token_id,
                no_token_id=self.no_token_id,
            )
            return scores.tolist()
        except Exception as e:
            raise ScoringError(
                "Failed to compute scores",
                {"error": str(e)},
            ) from e
