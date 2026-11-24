#!/usr/bin/env python3
"""Build a balanced DSpark draft-vocabulary artifact.

The artifact contains only the target ids. The Inkling2 runtime slices the
shared target head on each TP rank and derives the folded Markov projection in
place, so no production checkpoint is modified by this script.

Usage
  python build_draft_vocab.py MODEL_DIR OUT_DIR --size 32768 CORPUS...

Corpus files are plain text, JSONL with common text fields, or JSON arrays of
strings. The output is experimental until held-out coverage and all quality
gates in REPORT_QUANT_OTIMIZADA_INKLING2_SM86.md pass.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer


def documents(path: Path):
    if path.suffix == ".jsonl":
        for line in path.read_text(errors="ignore").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            parts = []
            if isinstance(value, dict):
                for key in ("text", "prompt", "response"):
                    if isinstance(value.get(key), str):
                        parts.append(value[key])
                if isinstance(value.get("messages"), list):
                    parts.extend(
                        item.get("content", "")
                        for item in value["messages"]
                        if isinstance(item, dict) and isinstance(item.get("content"), str)
                    )
            if parts:
                yield "\n".join(parts)
        return
    if path.suffix == ".json":
        value = json.loads(path.read_text())
        if isinstance(value, list):
            yield from (item for item in value if isinstance(item, str))
        elif isinstance(value, dict):
            yield from (item for item in value.values() if isinstance(item, str))
        return
    yield path.read_text(errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--size", type=int, choices=(32768, 40960), required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("corpus", type=Path, nargs="+")
    args = parser.parse_args()
    if args.size % args.tp:
        raise SystemExit("size must be divisible by TP")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    counts = collections.Counter()
    documents_seen = 0
    token_count = 0
    for path in args.corpus:
        for text in documents(path):
            ids = tokenizer(text, add_special_tokens=False).input_ids
            counts.update(ids)
            token_count += len(ids)
            documents_seen += 1

    config = json.loads((args.model_dir / "config.json").read_text())
    vocab_size = int(config.get("text_config", config).get("vocab_size", tokenizer.vocab_size))
    special_ids = set(tokenizer.all_special_ids)
    per_rank = args.size // args.tp
    selected: list[int] = []
    for rank in range(args.tp):
        lo, hi = rank * (vocab_size // args.tp), (rank + 1) * (vocab_size // args.tp)
        mandatory = sorted(i for i in special_ids if lo <= i < hi)
        candidates = sorted(
            (i for i in range(lo, hi) if i not in special_ids),
            key=lambda i: (-counts[i], i),
        )
        rank_ids = mandatory + candidates[: max(0, per_rank - len(mandatory))]
        if len(rank_ids) != per_rank:
            raise SystemExit(f"rank {rank} has only {len(rank_ids)} eligible ids")
        selected.extend(sorted(rank_ids))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(selected, dtype=torch.int32), args.out_dir / "draft_vocab_ids.pt")
    covered = sum(counts[i] for i in selected)
    metadata = {
        "size": len(selected),
        "tp": args.tp,
        "per_rank": per_rank,
        "vocab_size": vocab_size,
        "documents": documents_seen,
        "corpus_tokens": token_count,
        "coverage": covered / token_count if token_count else 0.0,
        "special_ids": sorted(special_ids),
        "source_model": str(args.model_dir),
        "status": "experimental",
    }
    (args.out_dir / "draft_vocab_ids.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
