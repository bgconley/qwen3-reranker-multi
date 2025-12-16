"""Tokenization utilities for Qwen3 Reranker - Backend-agnostic.

Implements the official Qwen3 tokenization strategy with:
- Left padding for causal LM reranking
- Truncation that preserves prefix/suffix tokens
- Batch tokenization with padding to uniform length

All outputs are numpy arrays for cross-backend compatibility.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TokenizationResult:
    """Result of tokenizing a batch of prompts."""

    input_ids: np.ndarray  # Shape: [batch_size, seq_len], dtype=int64
    attention_mask: np.ndarray  # Shape: [batch_size, seq_len], dtype=int64
    truncated_count: int  # Number of sequences that were truncated
    original_lengths: list[int]  # Original token counts before truncation


class RerankerTokenizer:
    """Tokenizer wrapper for the Qwen3 Reranker model.

    Handles the specific tokenization requirements:
    - Left padding for efficient batching (critical for causal LM)
    - Truncation that preserves prefix and suffix tokens
    - Batch processing for efficient inference
    """

    def __init__(
        self,
        tokenizer: Any,  # HuggingFace tokenizer
        prefix_tokens: list[int],
        suffix_tokens: list[int],
        yes_token_id: int,
        no_token_id: int,
        max_length: int,
    ) -> None:
        """Initialize the tokenizer wrapper.

        Args:
            tokenizer: HuggingFace tokenizer instance
            prefix_tokens: Pre-tokenized prefix token IDs
            suffix_tokens: Pre-tokenized suffix token IDs
            yes_token_id: Token ID for "yes"
            no_token_id: Token ID for "no"
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.prefix_tokens = prefix_tokens
        self.suffix_tokens = suffix_tokens
        self.yes_token_id = yes_token_id
        self.no_token_id = no_token_id
        self.max_length = max_length

        # Set pad token to EOS if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id

        # Calculate available space for content
        self._prefix_len = len(prefix_tokens)
        self._suffix_len = len(suffix_tokens)
        self._fixed_overhead = self._prefix_len + self._suffix_len

    @property
    def content_max_length(self) -> int:
        """Maximum tokens available for content (excluding prefix/suffix)."""
        return self.max_length - self._fixed_overhead

    def tokenize_content(self, text: str) -> list[int]:
        """Tokenize content text without special tokens.

        Args:
            text: The content text to tokenize

        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(text, add_special_tokens=False)

    def truncate_content(self, content_tokens: list[int]) -> tuple[list[int], bool]:
        """Truncate content tokens to fit within limits.

        Truncation strategy:
        - Remove tokens from the end of the content (document portion)
        - Preserves the beginning which typically contains instruction/query

        Args:
            content_tokens: Token IDs for the content

        Returns:
            Tuple of (truncated tokens, was_truncated)
        """
        max_content = self.content_max_length
        if len(content_tokens) <= max_content:
            return content_tokens, False

        # Truncate from the end (document portion)
        return content_tokens[:max_content], True

    def build_sequence(self, content_tokens: list[int]) -> list[int]:
        """Build complete sequence with prefix and suffix.

        Args:
            content_tokens: Token IDs for the content

        Returns:
            Complete sequence: prefix + content + suffix
        """
        return self.prefix_tokens + content_tokens + self.suffix_tokens

    def tokenize_prompt(self, prompt: str) -> tuple[list[int], bool]:
        """Tokenize a complete prompt with truncation.

        The prompt should already be formatted with instruction/query/doc.
        This function:
        1. Tokenizes the content (without prefix/suffix - those are pre-tokenized)
        2. Truncates if needed
        3. Builds the complete sequence

        Args:
            prompt: Formatted prompt string (content only, without prefix/suffix)

        Returns:
            Tuple of (token IDs, was_truncated)
        """
        # Tokenize content
        content_tokens = self.tokenize_content(prompt)

        # Truncate if needed
        truncated_tokens, was_truncated = self.truncate_content(content_tokens)

        # Build complete sequence
        sequence = self.build_sequence(truncated_tokens)

        return sequence, was_truncated

    def tokenize_batch(
        self,
        prompts: list[str],
    ) -> TokenizationResult:
        """Tokenize a batch of prompts with LEFT padding.

        All prompts are left-padded to the same length.
        For causal LMs, left-padding ensures the actual content ends at position -1,
        which is where we extract next-token predictions.

        Args:
            prompts: List of formatted prompt strings (content only)

        Returns:
            TokenizationResult with padded input_ids and attention_mask
        """
        sequences: list[list[int]] = []
        truncated_count = 0
        original_lengths: list[int] = []

        for prompt in prompts:
            sequence, was_truncated = self.tokenize_prompt(prompt)
            sequences.append(sequence)
            original_lengths.append(len(sequence))
            if was_truncated:
                truncated_count += 1

        # Find max length in this batch
        max_len = max(len(seq) for seq in sequences)

        # LEFT-pad all sequences (critical for causal LM reranking)
        padded_sequences: list[list[int]] = []
        attention_masks: list[list[int]] = []

        for seq in sequences:
            pad_len = max_len - len(seq)
            # LEFT padding: pad tokens first, then real tokens
            padded = [self.pad_token_id] * pad_len + seq
            # Attention mask: 0 for pad tokens, 1 for real tokens
            mask = [0] * pad_len + [1] * len(seq)
            padded_sequences.append(padded)
            attention_masks.append(mask)

        # Convert to numpy arrays
        input_ids = np.array(padded_sequences, dtype=np.int64)
        attention_mask = np.array(attention_masks, dtype=np.int64)

        return TokenizationResult(
            input_ids=input_ids,
            attention_mask=attention_mask,
            truncated_count=truncated_count,
            original_lengths=original_lengths,
        )

    def tokenize_single(self, prompt: str) -> tuple[np.ndarray, np.ndarray, bool]:
        """Tokenize a single prompt.

        Convenience method for single-item processing.

        Args:
            prompt: Formatted prompt string (content only)

        Returns:
            Tuple of (input_ids [1, seq_len], attention_mask [1, seq_len], was_truncated)
        """
        result = self.tokenize_batch([prompt])
        return result.input_ids, result.attention_mask, result.truncated_count > 0

    def clone_with_max_length(self, max_length: int) -> "RerankerTokenizer":
        """Return a new tokenizer wrapper with a different max_length."""
        return RerankerTokenizer(
            tokenizer=self.tokenizer,
            prefix_tokens=self.prefix_tokens,
            suffix_tokens=self.suffix_tokens,
            yes_token_id=self.yes_token_id,
            no_token_id=self.no_token_id,
            max_length=max_length,
        )


def setup_tokenizer(
    tokenizer: Any,
    prefix: str,
    suffix: str,
    yes_token: str,
    no_token: str,
    max_length: int,
) -> RerankerTokenizer:
    """Set up the tokenizer wrapper with validated configuration.

    Args:
        tokenizer: HuggingFace tokenizer instance
        prefix: Prefix template string
        suffix: Suffix template string
        yes_token: String for "yes" token
        no_token: String for "no" token
        max_length: Maximum sequence length

    Returns:
        Configured RerankerTokenizer

    Raises:
        ValueError: If yes/no tokens are not single tokens
    """
    # Configure tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Ensure left padding for causal LM
    tokenizer.padding_side = "left"

    # Pre-tokenize prefix and suffix
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    # Get yes/no token IDs and validate they are single tokens
    yes_tokens = tokenizer.encode(yes_token, add_special_tokens=False)
    no_tokens = tokenizer.encode(no_token, add_special_tokens=False)

    if len(yes_tokens) != 1:
        raise ValueError(
            f"'{yes_token}' is not a single token. Got {len(yes_tokens)} tokens: {yes_tokens}"
        )

    if len(no_tokens) != 1:
        raise ValueError(
            f"'{no_token}' is not a single token. Got {len(no_tokens)} tokens: {no_tokens}"
        )

    yes_token_id = yes_tokens[0]
    no_token_id = no_tokens[0]

    return RerankerTokenizer(
        tokenizer=tokenizer,
        prefix_tokens=prefix_tokens,
        suffix_tokens=suffix_tokens,
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        max_length=max_length,
    )
