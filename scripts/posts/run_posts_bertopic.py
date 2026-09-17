#!/usr/bin/env python3
"""
Single-model BERTopic runner for posts corpus (merged, 70k+).
Identical parameters to comments pipeline.
Usage: python run_posts_bertopic.py --model-name <short_name> --embedding-path <path>
"""
import argparse, json, os, time, warnings, re, sys
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize as sk_normalize
import umap, hdbscan
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
import scipy.sparse as sp

warnings.filterwarnings('ignore')

UMAP_CONFIG = {"n_neighbors": 15, "n_components": 5, "min_dist": 0.0, "metric": "cosine", "random_state": 42, "verbose": False}
HDBSCAN_CONFIG = {"min_cluster_size": 100, "min_samples": 10, "metric": "euclidean", "cluster_selection_method": "leaf", "prediction_data": True}
CV_CONFIG = {"min_df": 1, "max_df": 1.0, "ngram_range": (1, 1), "max_features": 10000}
REDUCE_OUTLIERS = True; RO_STRATEGY = "c-tf-idf"; RO_THRESHOLD = 0.05

CULTURAL_PHRASES = [
    "chud_gaye", "padhle_bsdk", "tier_3", "chud_gaye_guru", "nta_ki_mkc",
    "iit_madras", "iit_kanpur", "double_dropper", "double_drop", "partial_dropper", "partial_drop",
    "mar_jana_hai", "jeene_ka_mann_nahi", "sab_khatam", "zindagi_barbaad", "chud_gaye",
    "kuch_nahi_ho_sakta", "padhle_bhai", "padhle_bsdc", "bhosdike", "madarchod", "bhenchod",
    "chutiya", "bkl", "mc", "bc", "iit_bombay", "iit_delhi", "jee_mains", "jee_advanced",
    "dropper_life", "dropper", "coaching_adda", "allen_kota", "physics_wallah", "pw_vip",
    "arre_bhai", "bhai_yaar", "mains_ki_tayari", "board_exam", "cbse_board", "10th_result",
    "general_category", "obc_category", "meme_distress", "cooked", "canon_event", "joever",
]

ARTIFACT_PATTERNS = [r"format=png", r"format=pjpg", r"format=jpg", r"auto=webp", r"x200b",
    r"highlightedupdate", r"urn%3a", r"s=[a-f0-9]{32,}", r"[a-f0-9]{40,}", r"preview\.redd", r"redd\.it"]

def evaluate_model(topics, topic_model, texts):
    topic_counts = Counter(topics)
    total = len(topics)
    n_topics = len([t for t in topic_counts if t != -1])
    outlier_rate = topic_counts.get(-1, 0) / total
    non_outlier = [c for t, c in topic_counts.items() if t != -1]
    if not non_outlier: return None
    counts = np.array(non_outlier)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    norm_entropy = entropy / np.log2(len(probs)) if len(probs) > 1 else 0
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    cumsum = np.cumsum(sorted_counts)
    gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
    top3 = np.sort(counts)[-3:].sum() / counts.sum() if len(counts) >= 3 else 1.0
    c_tf_idf = topic_model.c_tf_idf_
    coherence_proxy = float(np.mean(np.array(c_tf_idf.max(axis=1).todense()).flatten())) if c_tf_idf is not None and sp.issparse(c_tf_idf) else 0.0
    top_words = []
    for topic_id in range(n_topics):
        words = topic_model.get_topic(topic_id)
        if words: top_words.extend([w for w, _ in words[:10]])
    diversity = len(set(top_words)) / len(top_words) if top_words else 0
    mid = n_topics // 2
    first, second = set(), set()
    for tid in range(n_topics):
        words = topic_model.get_topic(tid)
        if words:
            wset = set([w for w, _ in words[:10]])
            (first if tid < mid else second).update(wset)
    union = len(first | second)
    stability = len(first & second) / union if union > 0 else 0
    phrases_found = 0
    for tid in range(n_topics):
        words = topic_model.get_topic(tid)
        if words:
            for w, _ in words:
                if any(p in w for p in CULTURAL_PHRASES): phrases_found += 1; break
    artifact_topics = 0
    for tid in range(n_topics):
        words = topic_model.get_topic(tid)
        if words:
            for w, _ in words[:5]:
                if any(re.search(p, w, re.I) for p in ARTIFACT_PATTERNS): artifact_topics += 1; break
    return {
        "total_docs": total, "n_topics": n_topics, "outlier_rate": round(float(outlier_rate), 4),
        "norm_entropy": round(float(norm_entropy), 4), "gini_coeff": round(float(gini), 4),
        "top3_concentration": round(float(top3), 4), "coherence_proxy": round(float(coherence_proxy), 4),
        "topic_diversity": round(float(diversity), 4), "keyword_stability": round(float(stability), 4),
        "phrases_found": phrases_found, "artifact_topics": artifact_topics,
    }

def run_bertopic(model_name, embedding_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    result_json = os.path.join(output_dir, f"{model_name.lower().replace('/', '_').replace('-', '_')}_results.json")
    if os.path.exists(result_json):
        print(f"Result already exists: {result_json}. Skipping.")
        return True

    print(f"\n{'='*60}")
    print(f"Running BERTopic for: {model_name}")
    print(f"Embedding: {embedding_path}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    df_text = pd.read_csv("data/processed/posts_merged_final_preprocessed.csv", usecols=["cleaned_text"])
    texts = df_text["cleaned_text"].fillna("").tolist()
    print(f"Loaded {len(texts):,} post texts")

    embeddings = np.load(embedding_path).astype(np.float32)
    print(f"Embeddings shape: {embeddings.shape}")
    embeddings = sk_normalize(embeddings, norm='l2', axis=1)
    print("L2-normalized")

    t0 = time.time()
    umap_model = umap.UMAP(**UMAP_CONFIG)
    umap_embeddings = umap_model.fit_transform(embeddings)
    print(f"UMAP: {time.time()-t0:.1f}s | Shape: {umap_embeddings.shape}")

    t0 = time.time()
    hdbscan_model = hdbscan.HDBSCAN(**HDBSCAN_CONFIG)
    topics = hdbscan_model.fit_predict(umap_embeddings)
    n_clusters = len(set(topics)) - (1 if -1 in topics else 0)
    print(f"HDBSCAN: {time.time()-t0:.1f}s | Clusters: {n_clusters}")

    t0 = time.time()
    vectorizer = CountVectorizer(**CV_CONFIG)
    keybert = KeyBERTInspired(top_n_words=15)
    mmr = MaximalMarginalRelevance(diversity=0.3)
    topic_model = BERTopic(
        umap_model=umap_model, hdbscan_model=hdbscan_model, vectorizer_model=vectorizer,
        representation_model=[keybert, mmr], embedding_model="all-MiniLM-L6-v2",
        calculate_probabilities=False, verbose=False,
    )
    topics, _ = topic_model.fit_transform(texts, embeddings)
    print(f"BERTopic fit: {time.time()-t0:.1f}s | Topics: {len(set(topics)) - (1 if -1 in topics else 0)}")

    if REDUCE_OUTLIERS and -1 in topics:
        n_out_before = sum(1 for t in topics if t == -1)
        print(f"Outliers before: {n_out_before:,} ({n_out_before/len(topics)*100:.1f}%)")
        topics = topic_model.reduce_outliers(texts, topics, strategy=RO_STRATEGY, threshold=RO_THRESHOLD)
        topic_model.update_topics(texts, topics=topics)
        n_out_after = sum(1 for t in topics if t == -1)
        print(f"Outliers after: {n_out_after:,} ({n_out_after/len(topics)*100:.1f}%)")

    print("Evaluating...")
    metrics = evaluate_model(topics, topic_model, texts)
    print(f"Metrics: {json.dumps(metrics, indent=2)}")

    model_dir = os.path.join(output_dir, model_name.lower().replace('/', '_').replace('-', '_'))
    os.makedirs(model_dir, exist_ok=True)
    pd.DataFrame({"topic": topics}).to_csv(os.path.join(model_dir, "topics.csv"), index=False)
    topic_model.get_topic_info().to_csv(os.path.join(model_dir, "topic_info.csv"), index=False)
    keywords = []
    for tid in range(metrics["n_topics"]):
        words = topic_model.get_topic(tid)
        if words:
            keywords.append({"topic_id": tid, "keywords": ", ".join([w for w, _ in words[:15]]),
                             "scores": ", ".join([str(round(s, 4)) for _, s in words[:15]])})
    pd.DataFrame(keywords).to_csv(os.path.join(model_dir, "topic_keywords.csv"), index=False)

    with open(result_json, 'w') as f:
        json.dump({"model_name": model_name, "metrics": metrics}, f, indent=2)
    print(f"Saved results to {result_json}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--embedding-path", required=True)
    parser.add_argument("--output-dir", default="final_outputs/posts_benchmark")
    args = parser.parse_args()
    success = run_bertopic(args.model_name, args.embedding_path, args.output_dir)
    sys.exit(0 if success else 1)
