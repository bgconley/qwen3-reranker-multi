#!/usr/bin/env python3
"""Build an evaluation dataset (JSONL) from an input JSONL export.

This is intentionally lightweight and supports a couple of common shapes:

1) Already-in-schema (passthrough):
   {"query": "...", "positives": ["..."], "negatives": ["..."]}

2) Labeled documents with indices:
   {"query": "...", "documents": ["..."], "relevant_indices": [0,2]}

3) Labeled documents with boolean flags:
   {"query": "...", "documents": [{"text":"...", "relevant": true}, ...]}

If the input format doesn't match one of these, this script exits with a clear error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("expected list")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("expected list of strings")
        out.append(item)
    return out


def _convert_record(obj: dict[str, Any]) -> dict[str, Any]:
    if "query" not in obj or not isinstance(obj["query"], str):
        raise ValueError("missing/invalid 'query'")

    query = obj["query"]

    # 1) passthrough schema
    if "positives" in obj and "negatives" in obj:
        positives = _as_str_list(obj["positives"])
        negatives = _as_str_list(obj["negatives"])
        return {"query": query, "positives": positives, "negatives": negatives}

    # 2) documents + relevant_indices
    if "documents" in obj and "relevant_indices" in obj:
        documents = _as_str_list(obj["documents"])
        rel_idx = obj["relevant_indices"]
        if not isinstance(rel_idx, list) or not all(
            isinstance(i, int) for i in rel_idx
        ):
            raise ValueError("expected 'relevant_indices' as list[int]")
        positives = [documents[i] for i in rel_idx if 0 <= i < len(documents)]
        negatives = [doc for i, doc in enumerate(documents) if i not in set(rel_idx)]
        return {"query": query, "positives": positives, "negatives": negatives}

    # 3) documents as objects with relevance flags
    docs_obj = obj.get("documents")
    if isinstance(docs_obj, list) and docs_obj and isinstance(docs_obj[0], dict):
        positives: list[str] = []
        negatives: list[str] = []
        for d in docs_obj:
            if (
                not isinstance(d, dict)
                or "text" not in d
                or not isinstance(d["text"], str)
            ):
                raise ValueError("documents[] objects must include string 'text'")
            relevant = bool(d.get("relevant", False))
            (positives if relevant else negatives).append(d["text"])
        return {"query": query, "positives": positives, "negatives": negatives}

    raise ValueError("unrecognized input format")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL path")
    ap.add_argument(
        "--output", required=True, help="Output JSONL path (dataset schema)"
    )
    ap.add_argument(
        "--min-negatives",
        type=int,
        default=1,
        help="Drop rows with fewer negatives than this",
    )
    ap.add_argument(
        "--min-positives",
        type=int,
        default=1,
        help="Drop rows with fewer positives than this",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    kept = 0
    dropped = 0

    with (
        in_path.open("r", encoding="utf-8") as fin,
        out_path.open("w", encoding="utf-8") as fout,
    ):
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError("expected JSON object per line")
                rec = _convert_record(obj)
                if len(rec["positives"]) < args.min_positives:
                    dropped += 1
                    continue
                if len(rec["negatives"]) < args.min_negatives:
                    dropped += 1
                    continue
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
            except Exception as e:
                raise SystemExit(f"{in_path}:{line_no}: {e}") from e

    print(f"Wrote {kept} records to {out_path} (dropped {dropped})")


if __name__ == "__main__":
    main()
