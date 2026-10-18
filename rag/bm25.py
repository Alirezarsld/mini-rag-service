"""A small Okapi BM25 implementation backed by an inverted index (no external dependency)."""
import math
from collections import Counter, defaultdict


class BM25Index:
    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(docs_tokens)
        self.lengths = [len(d) for d in docs_tokens]
        self.avg_len = (sum(self.lengths) / self.n_docs) if self.n_docs else 0.0
        # term -> [(doc_index, term_frequency)]: a query only touches documents that contain
        # at least one of its terms instead of scanning the whole corpus.
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, tokens in enumerate(docs_tokens):
            for term, freq in Counter(tokens).items():
                self.postings[term].append((i, freq))
        self.doc_freq = {term: len(p) for term, p in self.postings.items()}

    def _idf(self, term: str) -> float:
        n = self.doc_freq.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def scores(self, query_tokens: list[str]) -> list[float]:
        out = [0.0] * self.n_docs
        if not self.n_docs or not self.avg_len:
            return out
        for term in set(query_tokens):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            for i, f in postings:
                norm = f + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_len)
                out[i] += idf * f * (self.k1 + 1) / norm
        return out
