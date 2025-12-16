# Evaluation Harness

This directory contains tools for evaluating the Qwen3 Reranker service quality.

## Overview

The evaluation harness allows you to:
1. Build evaluation datasets from query logs or manual annotations
2. Measure ranking quality metrics (nDCG@K, MRR@K, Recall@K)
3. Compare different configurations (max_length, batch_size, quantization)

## Dataset Format

Evaluation datasets use JSONL format with the following schema:

```json
{
  "query": "user search query",
  "positives": ["relevant doc 1", "relevant doc 2"],
  "negatives": ["irrelevant doc 1", "irrelevant doc 2"]
}
```

## Scripts

- `build_pool.py`: Build candidate pools from query logs or exports
- `run_eval.py`: Run evaluation and compute metrics

## Usage

```bash
# Build evaluation dataset from query logs
python eval/build_pool.py --input logs.jsonl --output eval_data.jsonl

# Run evaluation
python eval/run_eval.py \
  --dataset eval_data.jsonl \
  --service-url http://127.0.0.1:9003 \
  --output results.json
```

## Metrics

- **nDCG@K**: Normalized Discounted Cumulative Gain at K
- **MRR@K**: Mean Reciprocal Rank at K
- **Recall@K**: Fraction of relevant documents in top K

## Comparing Configurations

To compare different settings:

1. Run evaluation with baseline config
2. Change config (e.g., max_length, batch_size)
3. Run evaluation again
4. Compare metrics

Keep constant across comparisons:
- Same evaluation dataset
- Same candidate pools
- Same instruction
