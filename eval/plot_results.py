"""Generate result charts: metric comparison bar chart + feature importance."""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

EVAL_DIR  = os.path.dirname(__file__)
MODEL_DIR = os.path.join(EVAL_DIR, "..", "models")


def plot_metrics(bm25: dict, lmart: dict, out_path: str) -> None:
    metrics = ["NDCG@10", "MAP@10", "MRR@10"]
    bm25_vals  = [bm25[m]  for m in metrics]
    lmart_vals = [lmart[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, bm25_vals,  width, label="BM25 Baseline", color="#5B8DB8")
    bars2 = ax.bar(x + width / 2, lmart_vals, width, label="LambdaMART",    color="#E07B4A")

    ax.set_ylabel("Score")
    ax.set_title("Search Ranking: BM25 vs LambdaMART (MS MARCO Dev)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, max(lmart_vals) * 1.25)

    for bar in bars1:
        ax.annotate(f"{bar.get_height():.4f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    for bar in bars2:
        ax.annotate(f"{bar.get_height():.4f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_feature_importance(model_path: str, feature_names: list, out_path: str) -> None:
    import xgboost as xgb
    model = xgb.Booster()
    model.load_model(model_path)
    scores = model.get_score(importance_type="gain")

    vals  = [scores.get(f"f{i}", 0.0) for i in range(len(feature_names))]
    pairs = sorted(zip(feature_names, vals), key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([p[0] for p in pairs], [p[1] for p in pairs], color="#5B8DB8")
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("LambdaMART Feature Importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    from features.extractor import FEATURE_NAMES

    # Example results — replace with actual eval output
    bm25_metrics  = {"NDCG@10": 0.187, "MAP@10": 0.165, "MRR@10": 0.172}
    lmart_metrics = {"NDCG@10": 0.221, "MAP@10": 0.198, "MRR@10": 0.204}

    os.makedirs(os.path.join(EVAL_DIR, "plots"), exist_ok=True)
    plot_metrics(bm25_metrics, lmart_metrics,
                 os.path.join(EVAL_DIR, "plots", "metric_comparison.png"))
    plot_feature_importance(
        os.path.join(MODEL_DIR, "lambdamart.ubj"),
        FEATURE_NAMES,
        os.path.join(EVAL_DIR, "plots", "feature_importance.png"),
    )
