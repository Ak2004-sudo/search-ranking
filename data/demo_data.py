"""
Generate large realistic synthetic MS MARCO-style data for demo/offline use.
Produces 800 training queries + 200 dev queries, each with 20 candidate passages.
Relevant passages share vocabulary with the query; distractors do not.
"""

import os
import random

random.seed(42)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ── topic bank ────────────────────────────────────────────────────────────────
TOPICS = [
    ("machine learning", [
        "Machine learning enables systems to learn patterns from data without explicit programming.",
        "Supervised learning trains models on labeled examples to predict outputs for new inputs.",
        "Unsupervised learning discovers hidden structure in data without predefined labels.",
        "Reinforcement learning trains agents to maximize cumulative reward through trial and error.",
        "Feature engineering transforms raw data into informative representations for ML models.",
    ]),
    ("neural networks", [
        "Neural networks consist of layers of interconnected nodes that process information.",
        "Deep learning uses multiple hidden layers to learn hierarchical feature representations.",
        "Backpropagation computes gradients by applying the chain rule through network layers.",
        "Convolutional neural networks use filters to detect local patterns in images and text.",
        "Recurrent neural networks process sequential data by maintaining hidden state across steps.",
    ]),
    ("natural language processing", [
        "NLP enables computers to understand analyze and generate human language automatically.",
        "Tokenization splits text into words or subword units for downstream processing.",
        "Named entity recognition identifies people places and organizations in text.",
        "Sentiment analysis classifies text as positive negative or neutral in opinion mining.",
        "Text summarization produces concise descriptions of longer documents automatically.",
    ]),
    ("information retrieval", [
        "Information retrieval finds relevant documents from large collections given a user query.",
        "Inverted indexes map terms to documents for fast query processing at scale.",
        "BM25 ranks documents by term frequency and inverse document frequency weighting.",
        "Vector space models represent queries and documents as vectors for similarity ranking.",
        "Query expansion adds related terms to improve recall in document retrieval systems.",
    ]),
    ("transformer models", [
        "Transformers use self-attention to process all tokens in parallel without recurrence.",
        "BERT is pre-trained on masked language modeling and next sentence prediction tasks.",
        "GPT models generate text autoregressively by predicting the next token in sequence.",
        "Attention mechanisms allow models to focus on relevant parts of the input sequence.",
        "Positional encodings inject sequence order information into transformer representations.",
    ]),
    ("gradient descent", [
        "Gradient descent minimizes loss by iteratively updating parameters in the negative gradient direction.",
        "Stochastic gradient descent updates parameters using a single random sample per step.",
        "Mini-batch gradient descent balances computational efficiency and gradient noise.",
        "Adam optimizer adapts learning rates for each parameter using first and second moments.",
        "Learning rate schedules decay the step size over training for better convergence.",
    ]),
    ("search ranking", [
        "Learning to rank trains models to order documents by relevance to a given query.",
        "LambdaMART optimizes ranking metrics like NDCG using gradient boosted trees.",
        "Pointwise ranking models score each document independently of other candidates.",
        "Pairwise ranking learns from preference pairs of more and less relevant documents.",
        "Listwise ranking directly optimizes the ordering of an entire candidate document list.",
    ]),
    ("recommendation systems", [
        "Collaborative filtering recommends items based on preferences of similar users.",
        "Content-based filtering matches item features to user preference profiles.",
        "Matrix factorization decomposes user-item interaction matrices into latent factors.",
        "Hybrid recommenders combine collaborative and content-based approaches for better accuracy.",
        "Click-through rate prediction estimates the probability a user clicks a recommended item.",
    ]),
    ("computer vision", [
        "Object detection locates and classifies multiple objects within a single image.",
        "Image segmentation assigns a class label to every pixel in an image.",
        "Transfer learning reuses features from pretrained vision models for new tasks.",
        "Data augmentation artificially expands training sets with rotations crops and flips.",
        "ResNet introduced skip connections to train very deep networks without degradation.",
    ]),
    ("data preprocessing", [
        "Normalization scales features to a common range to stabilize model training.",
        "Missing value imputation fills gaps in data using statistical or model-based methods.",
        "One-hot encoding converts categorical variables into binary indicator vectors.",
        "Principal component analysis reduces dimensionality by projecting onto principal axes.",
        "Train test split partitions data into training validation and held-out test sets.",
    ]),
    ("evaluation metrics", [
        "NDCG measures ranking quality by rewarding relevant documents placed higher in the list.",
        "Precision at k measures the fraction of relevant items in the top k results.",
        "Mean average precision averages precision at each relevant document position.",
        "F1 score is the harmonic mean of precision and recall for classification tasks.",
        "ROC AUC measures classifier discrimination across all classification thresholds.",
    ]),
    ("overfitting regularization", [
        "Overfitting occurs when a model memorizes training data and fails to generalize.",
        "Dropout randomly deactivates neurons during training to prevent co-adaptation.",
        "L2 regularization penalizes large weights by adding squared norms to the loss.",
        "Early stopping halts training when validation loss stops improving to avoid overfitting.",
        "Cross validation estimates generalization performance by rotating the held-out fold.",
    ]),
    ("knowledge graphs", [
        "Knowledge graphs store entities and relationships as structured triples.",
        "Entity linking maps text mentions to nodes in a knowledge graph.",
        "Graph neural networks aggregate neighborhood information for node representation learning.",
        "Relation extraction identifies semantic relationships between entities in text.",
        "Ontologies define hierarchical concept schemas for knowledge representation.",
    ]),
    ("federated learning", [
        "Federated learning trains models across distributed devices without sharing raw data.",
        "Differential privacy adds calibrated noise to model updates to protect individual records.",
        "Model aggregation combines local gradients from edge devices into a global model.",
        "Communication efficiency reduces bandwidth by compressing or sparsifying gradient updates.",
        "Non-IID data distribution across clients poses challenges for federated optimization.",
    ]),
    ("time series forecasting", [
        "ARIMA models capture autoregressive integrated moving average patterns in time series.",
        "LSTM networks model long-range temporal dependencies in sequential data.",
        "Seasonal decomposition separates trend seasonal and residual components of a series.",
        "Anomaly detection identifies unusual patterns that deviate from expected behavior.",
        "Fourier transforms convert time domain signals into frequency domain representations.",
    ]),
]

DISTRACTORS = [
    "The recipe calls for two cups of flour and one teaspoon of baking powder.",
    "Football teams compete in a league system with promotion and relegation.",
    "The stock market closed higher following positive earnings announcements.",
    "Tourists flock to the coastal city during the summer holiday season.",
    "The chef prepared a three-course meal for the anniversary celebration.",
    "Rainfall this month exceeded the historical average by thirty percent.",
    "The new legislation requires companies to disclose environmental impact reports.",
    "Astronomers observed a rare conjunction of planets visible to the naked eye.",
    "The orchestra performed a symphony composed in the eighteenth century.",
    "Scientists discovered a new species of amphibian in the tropical rainforest.",
    "The architect designed a sustainable building with solar panels and green roofs.",
    "Marathon runners trained for months to prepare for the city race event.",
    "The documentary explored ancient civilizations of the Mediterranean region.",
    "Farmers adopted precision agriculture techniques to improve crop yields.",
    "The museum acquired a collection of impressionist paintings from a private donor.",
]


def generate_data(n_queries_train=800, n_queries_dev=200, n_candidates=20):
    all_queries, all_qrels, all_top1000 = {}, {}, {}
    qid_counter = 0

    for _ in range(n_queries_train + n_queries_dev):
        topic_name, relevant_passages = random.choice(TOPICS)
        # slight query variation
        words = topic_name.split()
        if random.random() < 0.5:
            words = ["what", "is"] + words
        elif random.random() < 0.5:
            words = ["explain"] + words
        else:
            words = ["how", "does"] + words + ["work"]
        query_text = " ".join(words)

        qid = f"q{qid_counter}"
        qid_counter += 1

        # pick 2 relevant passages + fill rest with distractors
        rel_passages = random.sample(relevant_passages, min(2, len(relevant_passages)))
        distractor_pool = DISTRACTORS + [p for t, ps in TOPICS if t != topic_name for p in ps]
        distractors = random.sample(distractor_pool, n_candidates - len(rel_passages))

        candidates = []
        pid_counter = 0
        for p in rel_passages:
            candidates.append((f"{qid}_p{pid_counter}", p))
            pid_counter += 1
        for p in distractors:
            candidates.append((f"{qid}_p{pid_counter}", p))
            pid_counter += 1
        random.shuffle(candidates)

        all_queries[qid] = query_text
        all_qrels[qid] = {
            pid: 1 for pid, text in candidates if text in rel_passages
        }
        all_top1000[qid] = candidates

    train_qids = list(all_queries.keys())[:n_queries_train]
    dev_qids   = list(all_queries.keys())[n_queries_train:]

    return all_queries, all_qrels, all_top1000, train_qids, dev_qids


def write_demo_files() -> None:
    queries, qrels, top1000, train_qids, dev_qids = generate_data()

    def write_queries(qids, path):
        with open(path, "w", encoding="utf-8") as f:
            for qid in qids:
                f.write(f"{qid}\t{queries[qid]}\n")

    def write_qrels(qids, path):
        with open(path, "w", encoding="utf-8") as f:
            for qid in qids:
                for pid, rel in qrels.get(qid, {}).items():
                    f.write(f"{qid} 0 {pid} {rel}\n")

    def write_top1000(qids, path):
        with open(path, "w", encoding="utf-8") as f:
            for qid in qids:
                for pid, ptext in top1000[qid]:
                    f.write(f"{qid}\t{pid}\t{queries[qid]}\t{ptext}\n")

    write_queries(train_qids, os.path.join(DATA_DIR, "queries.train.tsv"))
    write_queries(dev_qids,   os.path.join(DATA_DIR, "queries.dev.small.tsv"))
    write_qrels(train_qids,   os.path.join(DATA_DIR, "qrels.train.tsv"))
    write_qrels(dev_qids,     os.path.join(DATA_DIR, "qrels.dev.small.tsv"))
    write_top1000(train_qids, os.path.join(DATA_DIR, "top1000.train.tsv"))
    write_top1000(dev_qids,   os.path.join(DATA_DIR, "top1000.dev.tsv"))

    print(f"Demo data written: {len(train_qids)} train / {len(dev_qids)} dev queries")
    print(f"Location: {DATA_DIR}")


if __name__ == "__main__":
    write_demo_files()
