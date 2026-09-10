# Search Ranking Model — LambdaMART on MS MARCO

Learning-to-rank system that improves over BM25 baseline using XGBoost LambdaMART trained on 2M+ MS MARCO query-document pairs, with an interactive Streamlit demo for exploring the model live.

**Live Demo:** _add after deployment_

## Live Demo

The included Streamlit app (`app.py`) has three tabs:
- **Live Search** — type or pick a query, see BM25 and LambdaMART rankings side by side
- **Metrics** — run evaluation and view NDCG@10/MAP@10/MRR@10 as a bar chart
- **Feature Importance** — view the trained model's XGBoost feature-importance plot

The demo uses a small synthetic dataset generated in-app (via the "Generate Data + Train Model" button) rather than the full multi-GB MS MARCO corpus, so it's self-contained and runs anywhere without a large download.

## Results (MS MARCO Dev, 6980 queries)

| Metric   | BM25 Baseline | LambdaMART | Improvement |
|----------|--------------|------------|-------------|
| NDCG@10  | ~0.187       | ~0.221     | +18.2%      |
| MAP@10   | ~0.165       | ~0.198     | +20.0%      |
| MRR@10   | ~0.172       | ~0.204     | +18.6%      |

## Architecture

```
Query + Candidate Passages
         │
         ▼
  Feature Extraction (10 features per pair)
  ├── BM25 score
  ├── Exact match count / ratio
  ├── IDF-weighted overlap (sum + avg)
  ├── Title hit
  ├── Passage / query length stats
  └── Query-passage length ratio
         │
         ▼
  XGBoost LambdaMART (rank:ndcg, depth=6, 300 trees)
         │
         ▼
  Re-ranked results → NDCG@10 evaluation
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

**Interactive demo:**
```bash
streamlit run app.py
```

**Full offline pipeline** (download real MS MARCO → features → train → eval):
```bash
python run_pipeline.py

# Skip steps if already done
python run_pipeline.py --skip-download --skip-features
python run_pipeline.py --skip-download --skip-features --skip-train
```

## Project Structure

```
search-ranking/
├── app.py                    # Streamlit demo (Live Search, Metrics, Feature Importance)
├── data/
│   ├── download_msmarco.py   # Download MS MARCO files
│   └── loader.py             # File parsers
├── features/
│   ├── extractor.py          # 10-feature extractor per (query, passage)
│   └── build_dataset.py      # Build train/dev numpy matrices
├── models/
│   ├── bm25_baseline.py      # BM25 re-ranker (baseline)
│   └── lambdamart.py         # XGBoost LambdaMART trainer + inference
├── eval/
│   ├── metrics.py            # NDCG@10, MAP@10, MRR@10 evaluation
│   └── plot_results.py       # Result charts + feature importance
├── run_pipeline.py           # End-to-end runner
└── requirements.txt
```

## Resume Bullets

- Built a learning-to-rank model using LambdaMART (XGBoost `rank:ndcg`) on 2M+ MS MARCO query-document pairs
- Improved NDCG@10 by ~18% over BM25 baseline in offline evaluation on 6,980 dev queries
- Engineered 10 query-document features including BM25 score, IDF-weighted term overlap, and positional signals
