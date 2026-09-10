"""
Build train/dev feature matrices from MS MARCO top-1000 candidates.

Outputs (saved to data/):
  X_train.npy, y_train.npy, groups_train.npy
  X_dev.npy,   y_dev.npy,   groups_dev.npy
  qids_dev.npy  (for eval)
"""

from __future__ import annotations

import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.loader import load_collection, load_qrels, load_queries, load_top1000
from features.extractor import FeatureExtractor
from models.bm25_baseline import BM25Ranker

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def build_split(
    queries_path: str,
    qrels_path: str,
    top1000_path: str,
    out_prefix: str,
    max_queries: int = None,
) -> None:
    print(f"\nLoading {os.path.basename(queries_path)} ...")
    queries = load_queries(queries_path)
    qrels = load_qrels(qrels_path)
    print(f"Loading top1000 from {os.path.basename(top1000_path)} ...")
    top1000 = load_top1000(top1000_path)

    qids = [q for q in queries if q in top1000]
    if max_queries:
        qids = qids[:max_queries]

    X_parts, y_parts, groups, qids_out = [], [], [], []

    for qid in tqdm(qids, desc="Extracting features"):
        query_text = queries[qid]
        candidates = top1000[qid]

        extractor = FeatureExtractor(candidates)
        bm25 = BM25Ranker(candidates)
        bm25_scores = bm25.score(query_text)

        pids, X = extractor.extract(query_text, bm25_scores)
        y = np.array(
            [float(qrels.get(qid, {}).get(pid, 0)) for pid in pids],
            dtype=np.float32,
        )

        X_parts.append(X)
        y_parts.append(y)
        groups.append(len(pids))
        qids_out.append(qid)

    X_all = np.vstack(X_parts)
    y_all = np.concatenate(y_parts)
    groups_all = np.array(groups, dtype=np.int32)

    np.save(f"{out_prefix}_X.npy", X_all)
    np.save(f"{out_prefix}_y.npy", y_all)
    np.save(f"{out_prefix}_groups.npy", groups_all)

    if "dev" in out_prefix:
        np.save(f"{out_prefix}_qids.npy", np.array(qids_out))

    print(f"Saved: {X_all.shape[0]} pairs, {X_all.shape[1]} features, {len(qids_out)} queries")


if __name__ == "__main__":
    build_split(
        queries_path=os.path.join(DATA_DIR, "queries.train.tsv"),
        qrels_path=os.path.join(DATA_DIR, "qrels.train.tsv"),
        top1000_path=os.path.join(DATA_DIR, "top1000.train.tsv"),
        out_prefix=os.path.join(DATA_DIR, "train"),
    )
    build_split(
        queries_path=os.path.join(DATA_DIR, "queries.dev.small.tsv"),
        qrels_path=os.path.join(DATA_DIR, "qrels.dev.small.tsv"),
        top1000_path=os.path.join(DATA_DIR, "top1000.dev.tsv"),
        out_prefix=os.path.join(DATA_DIR, "dev"),
    )
