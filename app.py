import os
import sys
import subprocess

import streamlit as st
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from data.demo_data import write_demo_files
from data.loader import load_queries, load_qrels, load_top1000
from models.bm25_baseline import BM25Ranker
from features.extractor import FeatureExtractor, FEATURE_NAMES
from models.lambdamart import load_model, predict, train
from eval.metrics import evaluate_run

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

st.set_page_config(page_title="Search Ranking — LambdaMART", page_icon="🔍", layout="wide")

st.markdown("""
<style>
.main-title { font-size:2.2rem; font-weight:800; color:#00D4FF; text-align:center; }
.sub-title  { text-align:center; color:#888; margin-bottom:1rem; }
.rank-card  { background:#1E2A3A; border-left:4px solid #00D4FF;
              padding:10px 14px; border-radius:6px; margin:6px 0; }
.rank-card-bm25 { background:#1E2A3A; border-left:4px solid #E07B4A;
                  padding:10px 14px; border-radius:6px; margin:6px 0; }
.rel-badge  { background:#1a4a1a; color:#4caf50; padding:2px 8px;
              border-radius:10px; font-size:0.75rem; }
.irr-badge  { background:#3a1a1a; color:#f44336; padding:2px 8px;
              border-radius:10px; font-size:0.75rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔍 Search Ranking: BM25 vs LambdaMART</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enter a query — see how LambdaMART re-ranks results vs keyword-based BM25</div>', unsafe_allow_html=True)
st.divider()


# ── helpers ───────────────────────────────────────────────────────────────────

DATA_READY  = all(os.path.exists(os.path.join(DATA_DIR, f))
                  for f in ["queries.dev.small.tsv", "qrels.dev.small.tsv", "top1000.dev.tsv"])
MODEL_READY = os.path.exists(os.path.join(MODEL_DIR, "lambdamart.ubj"))


@st.cache_resource(show_spinner="Loading data ...")
def get_data():
    queries = load_queries(os.path.join(DATA_DIR, "queries.dev.small.tsv"))
    qrels   = load_qrels(os.path.join(DATA_DIR, "qrels.dev.small.tsv"))
    top1000 = load_top1000(os.path.join(DATA_DIR, "top1000.dev.tsv"))
    return queries, qrels, top1000


@st.cache_resource(show_spinner="Loading model ...")
def get_model():
    return load_model()


def rank_query(query_text, candidates, model):
    bm25 = BM25Ranker(candidates)
    bm25_scores = bm25.score(query_text)
    bm25_ranked = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)

    extractor = FeatureExtractor(candidates)
    pids, X = extractor.extract(query_text, bm25_scores)
    lm_scores = predict(model, X)
    lm_ranked = sorted(zip(pids, lm_scores.tolist()), key=lambda x: x[1], reverse=True)

    return bm25_ranked, lm_ranked, bm25_scores, dict(zip(pids, lm_scores.tolist()))


# ── sidebar: setup ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Setup")

    if not DATA_READY or not MODEL_READY:
        st.warning("Data or model not ready. Click below to generate.")
        if st.button("Generate Data + Train Model", use_container_width=True, type="primary"):
            with st.spinner("Generating synthetic data (800 train / 200 dev queries)..."):
                write_demo_files()
            with st.spinner("Building features..."):
                subprocess.run([sys.executable,
                    os.path.join(os.path.dirname(__file__), "features", "build_dataset.py")],
                    check=True)
            with st.spinner("Training LambdaMART..."):
                subprocess.run([sys.executable,
                    os.path.join(os.path.dirname(__file__), "models", "lambdamart.py")],
                    check=True)
            st.cache_resource.clear()
            st.success("Ready!")
            st.rerun()
    else:
        st.success("Data + Model ready")
        if st.button("Retrain from scratch", use_container_width=True):
            for f in ["queries.train.tsv","queries.dev.small.tsv","qrels.train.tsv",
                      "qrels.dev.small.tsv","top1000.train.tsv","top1000.dev.tsv",
                      "train_X.npy","train_y.npy","train_groups.npy",
                      "dev_X.npy","dev_y.npy","dev_groups.npy","dev_qids.npy"]:
                path = os.path.join(DATA_DIR, f)
                if os.path.exists(path): os.remove(path)
            mp = os.path.join(MODEL_DIR, "lambdamart.ubj")
            if os.path.exists(mp): os.remove(mp)
            st.cache_resource.clear()
            st.rerun()

    st.divider()
    st.markdown("## 📊 Model Info")
    st.markdown("""
| | |
|---|---|
| **Algorithm** | LambdaMART |
| **Library** | XGBoost |
| **Objective** | rank:ndcg |
| **Features** | 10 per pair |
| **Train queries** | 800 |
| **Dev queries** | 200 |
    """)

    st.divider()
    top_k = st.slider("Results to show (top-K)", 3, 10, 5)


# ── main area ─────────────────────────────────────────────────────────────────
if not DATA_READY or not MODEL_READY:
    st.info("Click **Generate Data + Train Model** in the sidebar to get started.")
    st.stop()

queries, qrels, top1000 = get_data()
model = get_model()

# ── tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Live Search", "📊 Metrics", "📈 Feature Importance"])

# ── TAB 1: Live Search ────────────────────────────────────────────────────────
with tab1:
    query_list  = [(qid, qt) for qid, qt in queries.items()]
    query_labels = [qt for _, qt in query_list]

    st.markdown("#### Type your own query or pick one from the dataset")
    col_txt, col_or, col_drop = st.columns([2, 0.15, 2])

    with col_txt:
        typed = st.text_input("Type a query:", placeholder="e.g. what is machine learning")
    with col_or:
        st.markdown("<br><center>or</center>", unsafe_allow_html=True)
    with col_drop:
        selected = st.selectbox("Pick from dataset:", options=["— select —"] + query_labels)

    # typed query takes priority; else use dropdown
    if typed.strip():
        query_input = typed.strip()
        matched_qid = next((qid for qid, qt in query_list if qt.lower() == query_input.lower()), None)
    elif selected != "— select —":
        query_input = selected
        matched_qid = next((qid for qid, qt in query_list if qt == selected), None)
    else:
        query_input = None
        matched_qid = None

    if query_input:
        if matched_qid:
            candidates = top1000[matched_qid]
            gold = qrels.get(matched_qid, {})
        else:
            # custom typed query — use first available candidate pool
            import random as _rnd
            sample_qid = _rnd.choice(list(top1000.keys()))
            candidates = top1000[sample_qid]
            gold = {}

        bm25_ranked, lm_ranked, bm25_scores, lm_scores = rank_query(query_input, candidates, model)
        pid_to_text = {pid: text for pid, text in candidates}
        if not gold:
            st.info("Custom query — no gold relevance labels available. Showing ranked results only.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🟠 BM25 Baseline")
            st.caption("Keyword matching — ranks by word overlap")
            for rank, (pid, score) in enumerate(bm25_ranked[:top_k], 1):
                text = pid_to_text.get(pid, "")
                rel  = gold.get(pid, 0)
                badge = '<span class="rel-badge">Relevant</span>' if rel > 0 else '<span class="irr-badge">Not relevant</span>'
                st.markdown(f"""<div class="rank-card-bm25">
                    <b>#{rank}</b> &nbsp; Score: {score:.3f} &nbsp; {badge}<br>
                    <small>{text[:180]}{'...' if len(text)>180 else ''}</small>
                </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown("### 🔵 LambdaMART")
            st.caption("Learned ranking — uses 10 features")
            for rank, (pid, score) in enumerate(lm_ranked[:top_k], 1):
                text = pid_to_text.get(pid, "")
                rel  = gold.get(pid, 0)
                badge = '<span class="rel-badge">Relevant</span>' if rel > 0 else '<span class="irr-badge">Not relevant</span>'
                st.markdown(f"""<div class="rank-card">
                    <b>#{rank}</b> &nbsp; Score: {score:.3f} &nbsp; {badge}<br>
                    <small>{text[:180]}{'...' if len(text)>180 else ''}</small>
                </div>""", unsafe_allow_html=True)


# ── TAB 2: Metrics ───────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Offline Evaluation — 200 Dev Queries")
    if st.button("Run Evaluation", type="primary"):
        with st.spinner("Evaluating BM25 vs LambdaMART on 200 dev queries..."):
            bm25_run, lm_run = {}, {}
            for qid, query_text in queries.items():
                if qid not in top1000:
                    continue
                candidates = top1000[qid]
                bm25_r, lm_r, _, _ = rank_query(query_text, candidates, model)
                bm25_run[qid] = bm25_r
                lm_run[qid]   = lm_r

            bm25_m = evaluate_run(bm25_run, qrels)
            lm_m   = evaluate_run(lm_run,   qrels)

        col1, col2, col3 = st.columns(3)
        for col, metric in zip([col1, col2, col3], ["NDCG@10", "MAP@10", "MRR@10"]):
            bv = bm25_m[metric]
            lv = lm_m[metric]
            delta = (lv - bv) / bv * 100
            with col:
                st.metric(label=f"BM25 {metric}",      value=f"{bv:.4f}")
                st.metric(label=f"LambdaMART {metric}", value=f"{lv:.4f}", delta=f"{delta:+.1f}%")

        import matplotlib.pyplot as plt
        metrics = ["NDCG@10", "MAP@10", "MRR@10"]
        x = np.arange(len(metrics))
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("#0E1117")
        ax.set_facecolor("#0E1117")
        ax.bar(x - 0.18, [bm25_m[m] for m in metrics], 0.35, label="BM25",       color="#E07B4A")
        ax.bar(x + 0.18, [lm_m[m]   for m in metrics], 0.35, label="LambdaMART", color="#00D4FF")
        ax.set_xticks(x); ax.set_xticklabels(metrics, color="white")
        ax.tick_params(colors="white"); ax.spines[["top","right"]].set_visible(False)
        for spine in ax.spines.values(): spine.set_color("#444")
        ax.set_ylabel("Score", color="white"); ax.legend(facecolor="#1E2A3A", labelcolor="white")
        ax.set_title("BM25 vs LambdaMART", color="white")
        st.pyplot(fig)
    else:
        st.info("Click **Run Evaluation** to compute NDCG@10, MAP@10, MRR@10 across all 200 dev queries.")


# ── TAB 3: Feature Importance ────────────────────────────────────────────────
with tab3:
    st.markdown("### What features does LambdaMART rely on most?")
    import xgboost as xgb
    import matplotlib.pyplot as plt

    booster = xgb.Booster()
    booster.load_model(os.path.join(MODEL_DIR, "lambdamart.ubj"))
    scores = booster.get_score(importance_type="gain")
    vals = [scores.get(f"f{i}", 0.0) for i in range(len(FEATURE_NAMES))]
    pairs = sorted(zip(FEATURE_NAMES, vals), key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")
    bars = ax.barh([p[0] for p in pairs], [p[1] for p in pairs], color="#00D4FF")
    ax.tick_params(colors="white"); ax.spines[["top","right"]].set_visible(False)
    for spine in ax.spines.values(): spine.set_color("#444")
    ax.set_xlabel("Importance (Gain)", color="white")
    ax.set_title("Feature Importance", color="white")
    st.pyplot(fig)

    st.markdown("**Feature descriptions:**")
    descs = {
        "bm25_score":              "BM25 relevance score for the passage",
        "passage_len":             "Number of words in the passage",
        "query_len":               "Number of words in the query",
        "exact_match_count":       "Number of query words found in passage",
        "exact_match_ratio":       "Fraction of query words matched",
        "idf_sum":                 "Sum of IDF weights of matched terms",
        "idf_avg":                 "Average IDF weight per query term",
        "title_hit":               "All query terms appear in first 10 words",
        "passage_len_log":         "Log of passage length (smoothed)",
        "query_passage_len_ratio": "Query length divided by passage length",
    }
    for name, desc in descs.items():
        st.markdown(f"- **{name}**: {desc}")
