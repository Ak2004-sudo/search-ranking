"""
LambdaMART ranker using XGBoost's rank:ndcg objective.
Trains on pre-built feature matrices and saves the model.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import xgboost as xgb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.dirname(__file__)


PARAMS = {
    "objective": "rank:ndcg",
    "eval_metric": "ndcg@10",
    "eta": 0.1,
    "max_depth": 6,
    "min_child_weight": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "seed": 42,
}

NUM_BOOST_ROUND = 300
EARLY_STOPPING = 20


def load_split(prefix: str):
    X = np.load(f"{prefix}_X.npy")
    y = np.load(f"{prefix}_y.npy")
    g = np.load(f"{prefix}_groups.npy")
    return X, y, g


def train() -> xgb.Booster:
    print("Loading training data ...")
    X_tr, y_tr, g_tr = load_split(os.path.join(DATA_DIR, "train"))
    X_dv, y_dv, g_dv = load_split(os.path.join(DATA_DIR, "dev"))

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dtrain.set_group(g_tr)

    dval = xgb.DMatrix(X_dv, label=y_dv)
    dval.set_group(g_dv)

    print(f"Train: {X_tr.shape[0]:,} pairs | Dev: {X_dv.shape[0]:,} pairs")
    print("Training LambdaMART ...")

    model = xgb.train(
        PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain, "train"), (dval, "dev")],
        early_stopping_rounds=EARLY_STOPPING,
        verbose_eval=10,
    )

    out_path = os.path.join(MODEL_DIR, "lambdamart.ubj")
    model.save_model(out_path)
    print(f"\nModel saved to {out_path}")
    return model


def load_model(path: str = None) -> xgb.Booster:
    path = path or os.path.join(MODEL_DIR, "lambdamart.ubj")
    model = xgb.Booster()
    model.load_model(path)
    return model


def predict(model: xgb.Booster, X: np.ndarray) -> np.ndarray:
    dmat = xgb.DMatrix(X)
    return model.predict(dmat)


if __name__ == "__main__":
    train()
