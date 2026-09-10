"""
Feature extraction for each (query, passage) pair.

Features extracted:
  0  bm25_score          - BM25 relevance score
  1  passage_len         - passage word count
  2  query_len           - query word count
  3  exact_match_count   - exact query terms found in passage
  4  exact_match_ratio   - exact_match_count / query_len
  5  idf_sum             - sum of IDF weights for matched terms
  6  idf_avg             - idf_sum / query_len
  7  title_hit           - 1 if first 10 words of passage contain all query terms
  8  passage_len_log     - log(1 + passage_len)
  9  query_passage_len_ratio - query_len / passage_len
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def compute_idf(corpus_tokens: List[List[str]]) -> Dict[str, float]:
    N = len(corpus_tokens)
    df: Counter = Counter()
    for tokens in corpus_tokens:
        df.update(set(tokens))
    return {term: math.log((N + 1) / (freq + 1)) + 1 for term, freq in df.items()}


class FeatureExtractor:
    def __init__(self, candidate_passages: List[Tuple[str, str]]) -> None:
        self.pids = [p[0] for p in candidate_passages]
        self.texts = [p[1] for p in candidate_passages]
        self._corpus_tokens = [tokenize(t) for t in self.texts]
        self._idf = compute_idf(self._corpus_tokens)

    def extract(
        self,
        query: str,
        bm25_scores: Dict[str, float],
    ) -> Tuple[List[str], np.ndarray]:
        q_tokens = tokenize(query)
        q_len = max(len(q_tokens), 1)
        q_set = set(q_tokens)

        rows = []
        for pid, tokens, text in zip(self.pids, self._corpus_tokens, self.texts):
            p_len = max(len(tokens), 1)
            token_set = set(tokens)

            matched = q_set & token_set
            exact_count = len(matched)
            exact_ratio = exact_count / q_len
            idf_sum = sum(self._idf.get(t, 0.0) for t in matched)
            idf_avg = idf_sum / q_len

            first10 = set(tokens[:10])
            title_hit = float(q_set.issubset(first10))

            row = [
                bm25_scores.get(pid, 0.0),        # 0
                float(p_len),                       # 1
                float(q_len),                       # 2
                float(exact_count),                 # 3
                exact_ratio,                        # 4
                idf_sum,                            # 5
                idf_avg,                            # 6
                title_hit,                          # 7
                math.log1p(p_len),                  # 8
                q_len / p_len,                      # 9
            ]
            rows.append(row)

        return self.pids, np.array(rows, dtype=np.float32)


FEATURE_NAMES = [
    "bm25_score",
    "passage_len",
    "query_len",
    "exact_match_count",
    "exact_match_ratio",
    "idf_sum",
    "idf_avg",
    "title_hit",
    "passage_len_log",
    "query_passage_len_ratio",
]
