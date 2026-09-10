"""BM25 baseline ranker using rank-bm25."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class BM25Ranker:
    def __init__(self, passages: List[Tuple[str, str]]) -> None:
        """
        passages: list of (pid, text) pairs for a single query's candidate set.
        Builds an in-query BM25 index over those candidates.
        """
        self.pids = [p[0] for p in passages]
        tokenized = [tokenize(p[1]) for p in passages]
        self.bm25 = BM25Okapi(tokenized)

    def score(self, query: str) -> Dict[str, float]:
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        return {pid: float(score) for pid, score in zip(self.pids, scores)}

    def rank(self, query: str) -> List[Tuple[str, float]]:
        scored = self.score(query)
        return sorted(scored.items(), key=lambda x: x[1], reverse=True)


def run_bm25_on_candidates(
    queries: Dict[str, str],
    top1000: Dict[str, List[Tuple[str, str]]],
) -> Dict[str, List[Tuple[str, float]]]:
    """Re-rank each query's top-1000 candidates with BM25."""
    results = {}
    for qid, query_text in queries.items():
        if qid not in top1000:
            continue
        ranker = BM25Ranker(top1000[qid])
        results[qid] = ranker.rank(query_text)
    return results
