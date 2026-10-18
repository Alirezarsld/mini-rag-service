"""Hybrid retrieval: BM25 + dense vectors, fused with RRF, re-ranked with MMR."""
from dataclasses import dataclass

import numpy as np

from .bm25 import BM25Index
from .embeddings import tokenize


@dataclass
class ChunkRecord:
    id: int
    document_id: int
    document_title: str
    index: int
    text: str
    embedding: np.ndarray


@dataclass
class Hit:
    chunk_id: int
    document_id: int
    document_title: str
    chunk_index: int
    text: str
    score: float          # fused + re-ranked relevance, 0..1
    bm25_score: float
    vector_score: float


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Combine several ranked lists of ids; ids ranked high in many lists win."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank + 1)
    return fused


def mmr_rerank(
    candidates: list[int],
    relevance: dict[int, float],
    vectors: np.ndarray,
    top_k: int,
    lambda_: float = 0.7,
) -> list[int]:
    """Maximal Marginal Relevance: keep relevant results but avoid near-duplicates."""
    selected: list[int] = []
    remaining = list(candidates)
    while remaining and len(selected) < top_k:
        best, best_score = None, float("-inf")
        for c in remaining:
            redundancy = max((float(vectors[c] @ vectors[s]) for s in selected), default=0.0)
            score = lambda_ * relevance[c] - (1 - lambda_) * redundancy
            if score > best_score:
                best, best_score = c, score
        selected.append(best)
        remaining.remove(best)
    return selected


def _trigrams(term: str) -> set[str]:
    padded = f"  {term} "
    return {padded[i:i + 3] for i in range(len(padded) - 2)}


class HybridRetriever:
    MODES = ("hybrid", "bm25", "vector")

    def __init__(self, records: list[ChunkRecord], embedder, min_vector_score: float | None = None):
        self.records = records
        self.embedder = embedder
        self.min_vector_score = (
            embedder.min_score if min_vector_score is None else min_vector_score
        )
        self.bm25 = BM25Index([tokenize(r.text) for r in records])
        self._vocab_trigrams: dict[str, set[str]] | None = None
        self.matrix = (
            np.vstack([r.embedding for r in records]).astype(np.float32)
            if records
            else np.zeros((0, 1), dtype=np.float32)
        )

    def rescue_tokens(self, tokens: list[str], threshold: float = 0.5) -> list[str]:
        """Map query terms that never occur in the corpus to their closest indexed term.

        Similarity is Jaccard over character trigrams, so "recieve" still finds
        "receive". Terms that are shorter than 4 characters or have no close
        match are kept unchanged.
        """
        vocab = self.bm25.doc_freq
        if self._vocab_trigrams is None:
            self._vocab_trigrams = {t: _trigrams(t) for t in vocab}
        fixed = []
        for token in tokens:
            if token in vocab or len(token) < 4:
                fixed.append(token)
                continue
            grams = _trigrams(token)
            best, best_sim = token, threshold
            for candidate, cand_grams in self._vocab_trigrams.items():
                sim = len(grams & cand_grams) / len(grams | cand_grams)
                if sim > best_sim:
                    best, best_sim = candidate, sim
            fixed.append(best)
        return fixed

    def search(self, query: str, top_k: int = 5, candidate_k: int = 20,
               mmr_lambda: float = 0.7, mode: str = "hybrid") -> list[Hit]:
        """``mode`` selects the retrieval signals: ``hybrid`` (default), ``bm25`` or
        ``vector``. The single-signal modes exist so retrieval quality can be compared
        (see ``benchmarks/run.py``)."""
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        if not self.records or not query.strip():
            return []

        bm25_scores = self.bm25.scores(self.rescue_tokens(tokenize(query)))
        query_vec = self.embedder.embed([query])[0]
        vector_scores = (self.matrix @ query_vec).tolist()

        rankings = []
        if mode in ("hybrid", "bm25"):
            rankings.append(sorted((i for i, s in enumerate(bm25_scores) if s > 0),
                                   key=lambda i: bm25_scores[i], reverse=True)[:candidate_k])
        if mode in ("hybrid", "vector"):
            rankings.append(sorted((i for i, s in enumerate(vector_scores)
                                    if s >= self.min_vector_score),
                                   key=lambda i: vector_scores[i], reverse=True)[:candidate_k])

        fused = reciprocal_rank_fusion(rankings)
        if not fused:
            return []
        top = max(fused.values())
        relevance = {i: v / top for i, v in fused.items()}
        candidates = sorted(fused, key=fused.get, reverse=True)[:candidate_k]

        order = mmr_rerank(candidates, relevance, self.matrix, top_k, mmr_lambda)
        hits = []
        for i in order:
            r = self.records[i]
            hits.append(Hit(
                chunk_id=r.id, document_id=r.document_id, document_title=r.document_title,
                chunk_index=r.index, text=r.text, score=round(relevance[i], 4),
                bm25_score=round(bm25_scores[i], 4), vector_score=round(vector_scores[i], 4),
            ))
        return hits
