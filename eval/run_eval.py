#!/usr/bin/env python3
"""Evaluate reranking quality against a JSONL dataset.

Dataset schema:
{"query":"...", "positives":["..."], "negatives":["...","..."]}
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass
class Metrics:
    ndcg_at_k: float
    mrr_at_k: float
    recall_at_k: float


def dcg(rels: list[int]) -> float:
    total = 0.0
    for i, rel in enumerate(rels):
        if rel:
            total += 1.0 / math.log2(i + 2)
    return total


def ndcg_at_k(rels: list[int], k: int) -> float:
    rels_k = rels[:k]
    ideal = sorted(rels_k, reverse=True)
    denom = dcg(ideal)
    return 0.0 if denom == 0 else dcg(rels_k) / denom


def mrr_at_k(rels: list[int], k: int) -> float:
    for i, rel in enumerate(rels[:k]):
        if rel:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(rels: list[int], positives_total: int, k: int) -> float:
    if positives_total <= 0:
        return 0.0
    return sum(rels[:k]) / float(positives_total)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            rows.append(obj)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Dataset JSONL path")
    ap.add_argument(
        "--service-url",
        default="http://127.0.0.1:9003",
        help="Base URL of reranker service",
    )
    ap.add_argument("--k", type=int, default=10, help="K for nDCG/MRR/Recall")
    ap.add_argument("--instruction", default=None, help="Optional instruction override")
    ap.add_argument(
        "--max-length", type=int, default=None, help="Optional max_length override"
    )
    ap.add_argument(
        "--output", default=None, help="Optional output JSON path for summary"
    )
    ap.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout seconds")
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    rows = read_jsonl(dataset_path)

    ndcgs: list[float] = []
    mrrs: list[float] = []
    recalls: list[float] = []

    with httpx.Client(base_url=args.service_url, timeout=args.timeout) as client:
        for i, row in enumerate(rows, start=1):
            query = row.get("query")
            positives = row.get("positives")
            negatives = row.get("negatives")
            if not isinstance(query, str):
                raise SystemExit(f"{dataset_path}:{i}: invalid query")
            if not isinstance(positives, list) or not all(
                isinstance(x, str) for x in positives
            ):
                raise SystemExit(f"{dataset_path}:{i}: invalid positives")
            if not isinstance(negatives, list) or not all(
                isinstance(x, str) for x in negatives
            ):
                raise SystemExit(f"{dataset_path}:{i}: invalid negatives")

            documents = list(positives) + list(negatives)
            positive_set = set(range(len(positives)))

            payload: dict[str, Any] = {
                "query": query,
                "documents": documents,
                "model": "eval",
            }
            if args.instruction is not None:
                payload["instruction"] = args.instruction
            if args.max_length is not None:
                payload["max_length"] = args.max_length

            resp = client.post("/v1/rerank", json=payload)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not isinstance(results, list):
                raise SystemExit("Bad response: missing results list")

            ranked_indices: list[int] = []
            for r in results:
                if not isinstance(r, dict) or "index" not in r:
                    continue
                idx = r["index"]
                if isinstance(idx, int):
                    ranked_indices.append(idx)

            # relevance list aligned to returned order
            rels = [1 if idx in positive_set else 0 for idx in ranked_indices]

            ndcgs.append(ndcg_at_k(rels, args.k))
            mrrs.append(mrr_at_k(rels, args.k))
            recalls.append(recall_at_k(rels, positives_total=len(positives), k=args.k))

    metrics = Metrics(
        ndcg_at_k=sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        mrr_at_k=sum(mrrs) / len(mrrs) if mrrs else 0.0,
        recall_at_k=sum(recalls) / len(recalls) if recalls else 0.0,
    )

    summary = {
        "dataset": str(dataset_path),
        "service_url": args.service_url,
        "k": args.k,
        "count": len(rows),
        "metrics": {
            "ndcg@k": metrics.ndcg_at_k,
            "mrr@k": metrics.mrr_at_k,
            "recall@k": metrics.recall_at_k,
        },
    }

    print(json.dumps(summary, indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
