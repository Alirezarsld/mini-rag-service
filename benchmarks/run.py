"""Compare retrieval modes (bm25 / vector / hybrid) on a labelled question set.

Standalone: needs only numpy, no Django and no database.

    python benchmarks/run.py                              # default hashing embedder
    python benchmarks/run.py --embedder sentence-transformers   # needs that package
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from rag.chunking import chunk_text  # noqa: E402
from rag.embeddings import get_embedder  # noqa: E402
from rag.retrieval import ChunkRecord, HybridRetriever  # noqa: E402


def build(embedder) -> HybridRetriever:
    records, chunk_id = [], 1
    for doc_id, path in enumerate(sorted((ROOT / "corpus").glob("*.md")), 1):
        for index, text in enumerate(chunk_text(path.read_text(encoding="utf-8"), 800, 120)):
            records.append(ChunkRecord(chunk_id, doc_id, path.stem, index, text,
                                       embedder.embed([text])[0]))
            chunk_id += 1
    return HybridRetriever(records, embedder)


def score(retriever, cases, mode, k):
    """Return {kind: (hit_rate@k, mrr)} plus an 'all' row."""
    buckets = defaultdict(list)
    for case in cases:
        hits = retriever.search(case["question"], top_k=k, mode=mode)
        rank = next((r for r, h in enumerate(hits, 1) if h.document_title == case["document"]), 0)
        buckets[case["kind"]].append(rank)
        buckets["all"].append(rank)
    return {
        kind: (sum(1 for r in ranks if r) / len(ranks), sum(1 / r for r in ranks if r) / len(ranks))
        for kind, ranks in buckets.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", default="hashing")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    cases = json.loads((ROOT / "cases.json").read_text(encoding="utf-8"))
    retriever = build(get_embedder(args.embedder))
    kinds = ["lexical", "paraphrase", "typo", "persian", "all"]
    print(f"embedder={args.embedder}  k={args.k}  cases={len(cases)}  chunks={len(retriever.records)}\n")
    print(f"{'mode':8s}" + "".join(f"{kind:>14s}" for kind in kinds))
    print(" " * 8 + "".join(f"{'hit / mrr':>14s}" for _ in kinds))
    for mode in ("bm25", "vector", "hybrid"):
        result = score(retriever, cases, mode, args.k)
        print(f"{mode:8s}" + "".join(f"{result[kind][0]:>8.2f}/{result[kind][1]:.2f}" for kind in kinds))


if __name__ == "__main__":
    main()
