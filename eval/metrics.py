"""
Offline evaluation: NDCG@10, MAP@10, MRR@10.
Compares BM25 baseline vs LambdaMART ranker.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.loader import load_qrels, load_queries, load_top1000
from models.bm25_baseline import BM25Ranker
from features.extractor import FeatureExtractor
from models.lambdamart import load_model, predict

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ── pure-Python metric implementations ──────────────────────────────────────

def dcg_at_k(rels: List[int], k: int) -> float:
    return sum(
        (2 ** r - 1) / math.log2(i + 2)
        for i, r in enumerate(rels[:k])
    )


def ndcg_at_k(ranked_rels: List[int], ideal_rels: List[int], k: int) -> float:
    idcg = dcg_at_k(sorted(ideal_rels, reverse=True), k)
    return dcg_at_k(ranked_rels, k) / idcg if idcg > 0 else 0.0


def ap_at_k(ranked_rels: List[int], k: int) -> float:
    hits, precision_sum = 0, 0.0
    for i, r in enumerate(ranked_rels[:k]):
        if r > 0:
            hits += 1
            precision_sum += hits / (i + 1)
    return precision_sum / max(hits, 1)


def rr_at_k(ranked_rels: List[int], k: int) -> float:
    for i, r in enumerate(ranked_rels[:k]):
        if r > 0:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_run(
    run: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 10,
) -> Dict[str, float]:
    ndcg_scores, ap_scores, rr_scores = [], [], []
    for qid, ranked in run.items():
        gold = qrels.get(qid, {})
        if not gold:
            continue
        ranked_rels = [gold.get(pid, 0) for pid, _ in ranked]
        ideal_rels = list(gold.values())
        ndcg_scores.append(ndcg_at_k(ranked_rels, ideal_rels, k))
        ap_scores.append(ap_at_k(ranked_rels, k))
        rr_scores.append(rr_at_k(ranked_rels, k))
    return {
        f"NDCG@{k}": float(np.mean(ndcg_scores)),
        f"MAP@{k}":  float(np.mean(ap_scores)),
        f"MRR@{k}":  float(np.mean(rr_scores)),
        "num_queries": len(ndcg_scores),
    }


# ── main evaluation ──────────────────────────────────────────────────────────

def main() -> None:
    print("Loading dev data ...")
    queries = load_queries(os.path.join(DATA_DIR, "queries.dev.small.tsv"))
    qrels   = load_qrels(os.path.join(DATA_DIR, "qrels.dev.small.tsv"))
    top1000 = load_top1000(os.path.join(DATA_DIR, "top1000.dev.tsv"))

    model = load_model()

    bm25_run: Dict[str, List[Tuple[str, float]]] = {}
    lmart_run: Dict[str, List[Tuple[str, float]]] = {}

    print("Scoring queries ...")
    for qid, query_text in queries.items():
        if qid not in top1000:
            continue
        candidates = top1000[qid]

        bm25 = BM25Ranker(candidates)
        bm25_scores = bm25.score(query_text)
        bm25_run[qid] = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)

        extractor = FeatureExtractor(candidates)
        pids, X = extractor.extract(query_text, bm25_scores)
        lmart_scores = predict(model, X)
        lmart_run[qid] = sorted(
            zip(pids, lmart_scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

    bm25_metrics  = evaluate_run(bm25_run,  qrels)
    lmart_metrics = evaluate_run(lmart_run, qrels)

    print("\n" + "=" * 52)
    print(f"{'Metric':<14} {'BM25':>10} {'LambdaMART':>12} {'Diff%':>8}")
    print("-" * 52)
    for metric in ["NDCG@10", "MAP@10", "MRR@10"]:
        bv = bm25_metrics[metric]
        lv = lmart_metrics[metric]
        delta = (lv - bv) / bv * 100 if bv > 0 else 0.0
        print(f"{metric:<14} {bv:>10.4f} {lv:>12.4f} {delta:>+8.1f}%")
    print("=" * 52)
    print(f"Queries evaluated: {lmart_metrics['num_queries']:,}")


if __name__ == "__main__":
    main()
