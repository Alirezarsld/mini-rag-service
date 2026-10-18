"""Offline retrieval metrics.

``cases`` is a list of ``(question, expected_document_title)`` pairs. A case is
a hit at rank *r* when the first returned chunk belonging to the expected
document is at position *r* (1-based).
"""
from dataclasses import dataclass


@dataclass
class EvalReport:
    n: int
    hit_rate: float   # share of questions whose expected document is in the top k
    mrr: float        # mean reciprocal rank of the first correct result

    def as_dict(self) -> dict:
        return {"n": self.n, "hit_rate": round(self.hit_rate, 4), "mrr": round(self.mrr, 4)}


def evaluate(search, cases: list[tuple[str, str]], k: int = 5) -> EvalReport:
    """``search(question, top_k)`` must return objects with ``document_title``."""
    if not cases:
        return EvalReport(0, 0.0, 0.0)
    hits = 0
    reciprocal = 0.0
    for question, expected in cases:
        results = search(question, k)
        for rank, hit in enumerate(results, 1):
            if hit.document_title == expected:
                hits += 1
                reciprocal += 1.0 / rank
                break
    return EvalReport(len(cases), hits / len(cases), reciprocal / len(cases))
