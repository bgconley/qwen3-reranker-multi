"""Batching and concurrency management for Qwen3 Reranker.

Provides:
- In-request batching (process docs in batches of batch_size)
- Concurrency guard (limit concurrent forward passes)
- Memory-safe processing for large document sets

Backend-agnostic: works with any RerankerBackend implementation.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from qwen3_reranker.backends.base import RerankerBackend
from qwen3_reranker.core.errors import ConcurrencyLimitError
from qwen3_reranker.core.prompt import PromptFormatter
from qwen3_reranker.core.scoring import RerankerScorer, rank_by_scores
from qwen3_reranker.core.tokenization import RerankerTokenizer

logger = logging.getLogger(__name__)


@dataclass
class RerankStats:
    """Statistics from a rerank operation."""

    doc_count: int
    batches_processed: int
    truncated_docs: int
    elapsed_ms: float
    avg_batch_ms: float


@dataclass
class BatchResult:
    """Result of processing a single batch."""

    scores: list[float]
    truncated_count: int


# Global concurrency semaphore
_forward_semaphore: asyncio.Semaphore | None = None


def init_concurrency_guard(max_concurrent: int) -> None:
    """Initialize the concurrency guard semaphore.

    Args:
        max_concurrent: Maximum number of concurrent forward passes
    """
    global _forward_semaphore
    _forward_semaphore = asyncio.Semaphore(max_concurrent)
    logger.info(f"Concurrency guard initialized: max_concurrent={max_concurrent}")


def get_forward_semaphore() -> asyncio.Semaphore:
    """Get the forward pass semaphore.

    Returns:
        The concurrency limiting semaphore

    Raises:
        RuntimeError: If semaphore not initialized
    """
    if _forward_semaphore is None:
        raise RuntimeError("Concurrency guard not initialized")
    return _forward_semaphore


async def acquire_forward_slot(timeout: float = 60.0) -> bool:
    """Acquire a slot for forward pass.

    Args:
        timeout: Maximum time to wait for a slot

    Returns:
        True if slot acquired

    Raises:
        ConcurrencyLimitError: If timeout exceeded
    """
    semaphore = get_forward_semaphore()
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        raise ConcurrencyLimitError(
            "Timed out waiting for forward pass slot",
            {"timeout_seconds": timeout},
        ) from None


def release_forward_slot() -> None:
    """Release a forward pass slot."""
    semaphore = get_forward_semaphore()
    semaphore.release()


def chunk_list(items: list[Any], chunk_size: int) -> list[list[Any]]:
    """Split a list into chunks of specified size.

    Args:
        items: List to split
        chunk_size: Maximum size of each chunk

    Returns:
        List of chunks
    """
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def process_batch_sync(
    backend: RerankerBackend,
    tokenizer: RerankerTokenizer,
    scorer: RerankerScorer,
    prompts: list[str],
) -> BatchResult:
    """Process a single batch of prompts synchronously.

    This is the core computation that runs under the semaphore.
    Backend-agnostic: works with PyTorch, vLLM, or MLX backends.

    Args:
        backend: Reranker backend instance
        tokenizer: Reranker tokenizer wrapper
        scorer: Reranker scorer
        prompts: Batch of formatted prompts (content only, without prefix/suffix)

    Returns:
        BatchResult with scores and truncation count
    """
    # Tokenize the batch
    tok_result = tokenizer.tokenize_batch(prompts)

    # Forward pass through backend (returns numpy array)
    logits = backend.forward(tok_result.input_ids, tok_result.attention_mask)

    # Compute scores
    scores = scorer.score_logits(logits)

    return BatchResult(
        scores=scores,
        truncated_count=tok_result.truncated_count,
    )


async def rerank_documents(
    backend: RerankerBackend,
    tokenizer: RerankerTokenizer,
    prompt_formatter: PromptFormatter,
    query: str,
    documents: list[str],
    batch_size: int,
    instruction: str | None = None,
    top_n: int | None = None,
    forward_timeout: float = 60.0,
) -> tuple[list[tuple[int, float]], RerankStats]:
    """Rerank documents for a query.

    This is the main entry point for reranking. It:
    1. Formats all query-document pairs
    2. Processes documents in batches
    3. Collects scores and ranks results

    Backend-agnostic: works with any RerankerBackend implementation.

    Args:
        backend: Reranker backend instance
        tokenizer: Reranker tokenizer wrapper
        prompt_formatter: Prompt formatter
        query: Search query
        documents: Documents to rank
        batch_size: Number of docs per batch
        instruction: Optional instruction override
        top_n: Return only top N results
        forward_timeout: Timeout for acquiring forward slot

    Returns:
        Tuple of (ranked results, stats)
    """
    start_time = time.perf_counter()

    # Create scorer
    scorer = RerankerScorer(
        yes_token_id=tokenizer.yes_token_id,
        no_token_id=tokenizer.no_token_id,
    )

    # Format content prompts for all documents
    # (content only, without prefix/suffix - tokenizer handles those)
    content_prompts = [
        prompt_formatter.format_content_only(
            query=query,
            doc=doc,
            instruction=instruction,
        )
        for doc in documents
    ]

    # Split into batches
    batches = chunk_list(content_prompts, batch_size)

    all_scores: list[float] = []
    total_truncated = 0
    batch_times: list[float] = []

    for batch in batches:
        batch_start = time.perf_counter()

        # Acquire concurrency slot
        await acquire_forward_slot(timeout=forward_timeout)

        try:
            # Process batch off the event loop thread to keep health endpoints responsive.
            result = await asyncio.to_thread(
                process_batch_sync, backend, tokenizer, scorer, batch
            )
            all_scores.extend(result.scores)
            total_truncated += result.truncated_count
        finally:
            release_forward_slot()

        batch_times.append((time.perf_counter() - batch_start) * 1000)

    # Rank by scores
    ranked = rank_by_scores(all_scores, top_n=top_n)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    avg_batch_ms = sum(batch_times) / len(batch_times) if batch_times else 0

    stats = RerankStats(
        doc_count=len(documents),
        batches_processed=len(batches),
        truncated_docs=total_truncated,
        elapsed_ms=elapsed_ms,
        avg_batch_ms=avg_batch_ms,
    )

    return ranked, stats
