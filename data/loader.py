"""Load MS MARCO files into memory-efficient structures."""

import os
from typing import Dict, List, Tuple

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def load_collection(path: str = None) -> Dict[str, str]:
    path = path or os.path.join(DATA_DIR, "collection.tsv")
    collection = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            pid, text = line.rstrip("\n").split("\t", 1)
            collection[pid] = text
    return collection


def load_queries(path: str) -> Dict[str, str]:
    queries = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            qid, text = line.rstrip("\n").split("\t", 1)
            queries[qid] = text
    return queries


def load_qrels(path: str) -> Dict[str, Dict[str, int]]:
    """Returns {qid: {pid: relevance_score}}."""
    qrels: Dict[str, Dict[str, int]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            qid, _, pid, rel = parts[0], parts[1], parts[2], int(parts[3])
            qrels.setdefault(qid, {})[pid] = rel
    return qrels


def load_top1000(path: str) -> Dict[str, List[Tuple[str, str]]]:
    """Returns {qid: [(pid, passage_text), ...]} — BM25 top-1000 candidates."""
    top: Dict[str, List[Tuple[str, str]]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            qid, pid, _query, passage = parts[0], parts[1], parts[2], parts[3]
            top.setdefault(qid, []).append((pid, passage))
    return top
