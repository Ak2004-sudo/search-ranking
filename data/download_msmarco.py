"""
Download MS MARCO passage ranking data via HuggingFace datasets.
Saves the required TSV files to the data/ folder.
"""

import os
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    queries_path = os.path.join(DATA_DIR, "queries.dev.small.tsv")
    qrels_path   = os.path.join(DATA_DIR, "qrels.dev.small.tsv")
    top1000_path = os.path.join(DATA_DIR, "top1000.dev.tsv")

    if all(os.path.exists(p) for p in [queries_path, qrels_path, top1000_path]):
        print("All files already exist — skipping download.")
        return

    print("Loading MS MARCO from HuggingFace (ms_marco v2.1) ...")
    # passage ranking dataset with queries, passages, and relevance labels
    ds = load_dataset("microsoft/ms_marco", "v2.1", trust_remote_code=True)

    print("Writing queries.dev.small.tsv ...")
    seen_qids = set()
    queries = {}
    qrels   = {}
    top1000 = {}

    dev_data = ds["validation"]
    for row in tqdm(dev_data, desc="Processing dev set"):
        qid  = str(row["query_id"])
        qtext = row["query"]
        if qid in seen_qids:
            continue
        seen_qids.add(qid)
        queries[qid] = qtext

        passages = row["passages"]
        pids  = [str(i) for i in range(len(passages["passage_text"]))]
        texts = passages["passage_text"]
        is_selected = passages["is_selected"]

        top1000[qid] = list(zip(pids, texts))
        for pid, rel in zip(pids, is_selected):
            if rel > 0:
                qrels.setdefault(qid, {})[pid] = int(rel)

    print(f"  {len(queries):,} queries loaded")

    with open(queries_path, "w", encoding="utf-8") as f:
        for qid, text in queries.items():
            f.write(f"{qid}\t{text}\n")

    with open(qrels_path, "w", encoding="utf-8") as f:
        for qid, rels in qrels.items():
            for pid, rel in rels.items():
                f.write(f"{qid} 0 {pid} {rel}\n")

    with open(top1000_path, "w", encoding="utf-8") as f:
        for qid, candidates in top1000.items():
            qtext = queries[qid]
            for pid, ptext in candidates:
                f.write(f"{qid}\t{pid}\t{qtext}\t{ptext}\n")

    print(f"\nAll files saved to: {DATA_DIR}")


if __name__ == "__main__":
    main()
