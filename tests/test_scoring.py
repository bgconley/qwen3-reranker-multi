"""Tests for scoring computation."""

import numpy as np
import pytest

from qwen3_reranker.core.scoring import (
    RerankerScorer,
    extract_yes_no_scores,
    rank_by_scores,
    validate_score_distribution,
)


class TestExtractYesNoScores:
    """Tests for yes/no score computation using numpy."""

    def test_strong_yes_high_score(self) -> None:
        """Test that high yes logit gives high score."""
        # Logits where yes (token 1) is much higher than no (token 0)
        # Shape: [batch=1, vocab=2]
        logits = np.array([[10.0, -10.0]])
        # With yes_token_id=1, no_token_id=0:
        # logit_no = logits[:, 0] = 10.0
        # logit_yes = logits[:, 1] = -10.0
        # So p(yes) should be low

        # Let me swap to make yes high:
        logits = np.array([[-10.0, 10.0]])  # no=-10, yes=10

        scores = extract_yes_no_scores(logits, yes_token_id=1, no_token_id=0)

        assert len(scores) == 1
        assert scores[0] > 0.99  # Should be very close to 1

    def test_strong_no_low_score(self) -> None:
        """Test that high no logit gives low score."""
        # Logits where no (token 0) is high, yes (token 1) is low
        logits = np.array([[10.0, -10.0]])  # no=10, yes=-10

        scores = extract_yes_no_scores(logits, yes_token_id=1, no_token_id=0)

        assert len(scores) == 1
        assert scores[0] < 0.01  # Should be very close to 0

    def test_equal_logits_half_score(self) -> None:
        """Test that equal logits give ~0.5 score."""
        logits = np.array([[5.0, 5.0]])

        scores = extract_yes_no_scores(logits, yes_token_id=1, no_token_id=0)

        assert len(scores) == 1
        assert 0.49 < scores[0] < 0.51  # Should be very close to 0.5

    def test_batch_processing(self) -> None:
        """Test scoring multiple items in batch."""
        # 3 items with varying yes/no preferences
        logits = np.array(
            [
                [10.0, -10.0],  # Strong no
                [0.0, 0.0],  # Neutral
                [-10.0, 10.0],  # Strong yes
            ]
        )

        scores = extract_yes_no_scores(logits, yes_token_id=1, no_token_id=0)

        assert len(scores) == 3
        assert scores[0] < 0.01  # Strong no -> low score
        assert 0.49 < scores[1] < 0.51  # Neutral -> ~0.5
        assert scores[2] > 0.99  # Strong yes -> high score

    def test_score_range(self) -> None:
        """Test that scores are always in [0, 1]."""
        # Test various extreme values
        test_cases = [
            np.array([[1000.0, -1000.0]]),
            np.array([[-1000.0, 1000.0]]),
            np.array([[0.0, 0.0]]),
            np.array([[1.0, 2.0]]),
        ]

        for logits in test_cases:
            scores = extract_yes_no_scores(logits, yes_token_id=1, no_token_id=0)

            for score in scores:
                assert 0.0 <= score <= 1.0

    def test_larger_vocab(self) -> None:
        """Test with larger vocabulary size (realistic scenario)."""
        vocab_size = 151936  # Typical for Qwen models
        yes_id = 9891  # Example token ID for "yes"
        no_id = 2152  # Example token ID for "no"

        # Create logits with specific values at yes/no positions
        logits = np.zeros((1, vocab_size))
        logits[0, yes_id] = 10.0
        logits[0, no_id] = -10.0

        scores = extract_yes_no_scores(logits, yes_token_id=yes_id, no_token_id=no_id)

        assert scores[0] > 0.99

    def test_numerical_stability(self) -> None:
        """Test numerical stability with very large logits."""
        # This would cause overflow without stable softmax
        logits = np.array([[1e6, 1e6 + 1]])  # Very large but similar values

        scores = extract_yes_no_scores(logits, yes_token_id=1, no_token_id=0)

        # Should not produce nan/inf
        assert np.isfinite(scores[0])
        # The +1 difference on yes should give slight preference
        assert scores[0] > 0.5


class TestRankByScores:
    """Tests for ranking function."""

    def test_basic_ranking(self) -> None:
        """Test basic ranking by score descending."""
        scores = [0.5, 0.9, 0.3, 0.7]
        ranked = rank_by_scores(scores)

        assert len(ranked) == 4
        # Should be sorted by score descending
        assert ranked[0] == (1, 0.9)  # Index 1 had highest score
        assert ranked[1] == (3, 0.7)
        assert ranked[2] == (0, 0.5)
        assert ranked[3] == (2, 0.3)

    def test_ranking_with_numpy_array(self) -> None:
        """Test ranking with numpy array input."""
        scores = np.array([0.5, 0.9, 0.3, 0.7])
        ranked = rank_by_scores(scores)

        assert len(ranked) == 4
        assert ranked[0] == (1, 0.9)

    def test_ranking_with_top_n(self) -> None:
        """Test ranking with top_n limit."""
        scores = [0.5, 0.9, 0.3, 0.7, 0.1]
        ranked = rank_by_scores(scores, top_n=3)

        assert len(ranked) == 3
        assert ranked[0] == (1, 0.9)
        assert ranked[1] == (3, 0.7)
        assert ranked[2] == (0, 0.5)

    def test_ranking_top_n_larger_than_list(self) -> None:
        """Test top_n larger than list returns all."""
        scores = [0.5, 0.9]
        ranked = rank_by_scores(scores, top_n=10)

        assert len(ranked) == 2

    def test_ranking_top_n_none(self) -> None:
        """Test top_n=None returns all."""
        scores = [0.5, 0.9, 0.3]
        ranked = rank_by_scores(scores, top_n=None)

        assert len(ranked) == 3

    def test_ranking_preserves_original_indices(self) -> None:
        """Test that original indices are preserved."""
        scores = [0.1, 0.2, 0.3]
        ranked = rank_by_scores(scores)

        # After sorting by score desc: index 2 (0.3), index 1 (0.2), index 0 (0.1)
        indices = [r[0] for r in ranked]
        assert indices == [2, 1, 0]

    def test_ranking_single_item(self) -> None:
        """Test ranking with single item."""
        scores = [0.75]
        ranked = rank_by_scores(scores)

        assert len(ranked) == 1
        assert ranked[0] == (0, 0.75)

    def test_ranking_empty_list(self) -> None:
        """Test ranking with empty list."""
        scores: list[float] = []
        ranked = rank_by_scores(scores)

        assert len(ranked) == 0

    def test_ranking_ties(self) -> None:
        """Test ranking with tied scores."""
        scores = [0.5, 0.5, 0.5]
        ranked = rank_by_scores(scores)

        assert len(ranked) == 3
        # All should be present, order among ties is implementation-defined
        indices = {r[0] for r in ranked}
        assert indices == {0, 1, 2}


class TestValidateScoreDistribution:
    """Tests for score distribution validation."""

    def test_basic_stats(self) -> None:
        """Test that basic statistics are calculated correctly."""
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        stats = validate_score_distribution(scores)

        assert stats["min"] == pytest.approx(0.1)
        assert stats["max"] == pytest.approx(0.9)
        assert stats["mean"] == pytest.approx(0.5)
        assert stats["median"] == pytest.approx(0.5)
        assert stats["num_above_0.5"] == 2
        assert stats["num_below_0.5"] == 2


class TestRerankerScorer:
    """Tests for the RerankerScorer class."""

    def test_score_logits(self) -> None:
        """Test scoring through the class interface."""
        scorer = RerankerScorer(yes_token_id=1, no_token_id=0)

        logits = np.array(
            [
                [10.0, -10.0],  # Strong no
                [-10.0, 10.0],  # Strong yes
            ]
        )

        scores = scorer.score_logits(logits)

        assert len(scores) == 2
        assert scores[0] < 0.01  # Strong no
        assert scores[1] > 0.99  # Strong yes
