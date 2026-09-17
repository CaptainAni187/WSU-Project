#!/usr/bin/env python3
"""
Regenerate the DEFINITIVE BERTopic outputs for the single best configuration
selected by the hyperparameter search (src/pipeline/bertopic_search.py).

Winning configuration (rank 1 by the count-independent quality composite):

    UMAP     : n_neighbors=50, n_components=10, min_dist=0.0, metric=cosine
    HDBSCAN  : min_cluster_size=300, min_samples=10, cluster_selection_method=leaf
    Represent: improved vectorizer (corpus-level stopwords + bigrams + df filter)
               + KeyBERTInspired + MaximalMarginalRelevance
    Outliers : reduced with the c-TF-IDF strategy (as in the frozen pipeline)

This regenerates ONLY BERTopic outputs, figures and reports.  It does not touch
embeddings, preprocessing, retrieval or the corpus.

Run:  python src/pipeline/bertopic_finalize.py
"""
import json
import logging
import os
import re
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bertopic_search as bs  # reuse metric + vocabulary helpers

LOGGER = logging.getLogger("bertopic_finalize")
FINAL_DIR = Path("final_outputs/topic_modeling/bertopic_final")
FIG_DIR = FINAL_DIR / "figures"

BEST = {
    "config_id": "u_nn50_c10__h_mcs300_ms10_leaf",
    "umap": dict(n_neighbors=50, n_components=10, min_dist=0.0, metric="cosine"),
    "hdbscan": dict(min_cluster_size=300, min_samples=10, cluster_selection_method="leaf"),
}
# Previous production 98-topic BGE-m3 model labels (for a live, apples-to-apples
# re-measurement through THIS metric pipeline -- see measure_baseline()).
BASELINE_LABELS_CSV = "final_outputs/posts_benchmark/bge_m3/topics.csv"


def anonymize(text):
    text = re.sub(r"u/[A-Za-z0-9_-]+", "[user]", str(text))
    text = re.sub(r"https?://\S+", "[link]", text)
    return re.sub(r"\s+", " ", text).strip()[:600]


def build_final_model(emb, texts):
    import umap
    import hdbscan
    from sklearn.feature_extraction.text import CountVectorizer
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance

    # Fixed corpus-level vocabulary so BERTopic's class-based c-TF-IDF inherits
    # proper document-frequency filtering (see bertopic_search for rationale).
    base_cv = CountVectorizer(**bs.VECTORIZER_CONFIG)
    base_cv.fit(texts)
    fixed_vocab = base_cv.vocabulary_
    LOGGER.info("Fixed corpus vocabulary: %d terms", len(fixed_vocab))
    vectorizer_model = CountVectorizer(
        vocabulary=fixed_vocab, ngram_range=bs.VECTORIZER_CONFIG["ngram_range"],
        token_pattern=bs.VECTORIZER_CONFIG["token_pattern"], lowercase=True,
    )

    umap_model = umap.UMAP(random_state=42, low_memory=True, verbose=False, **BEST["umap"])
    hdbscan_model = hdbscan.HDBSCAN(
        metric="euclidean", prediction_data=True, core_dist_n_jobs=-1, **BEST["hdbscan"]
    )
    representation_model = {
        "KeyBERT": KeyBERTInspired(top_n_words=15),
        "MMR": MaximalMarginalRelevance(diversity=0.3),
    }
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        embedding_model="all-MiniLM-L6-v2",   # used only by KeyBERTInspired (cached)
        calculate_probabilities=False,
        verbose=True,
    )
    t0 = time.time()
    topics, _ = topic_model.fit_transform(texts, emb)
    LOGGER.info("BERTopic fit: %.0fs | natural topics=%d", time.time() - t0,
                len(set(topics)) - (1 if -1 in topics else 0))

    n_out = int(np.sum(np.array(topics) == -1))
    LOGGER.info("Outliers (natural): %d (%.1f%%)", n_out, 100 * n_out / len(topics))
    topics = topic_model.reduce_outliers(texts, topics, strategy="c-tf-idf", threshold=0.05)
    topic_model.update_topics(texts, topics=topics, vectorizer_model=vectorizer_model,
                              representation_model=representation_model)
    n_out2 = int(np.sum(np.array(topics) == -1))
    LOGGER.info("Outliers (after reduction): %d (%.1f%%)", n_out2, 100 * n_out2 / len(topics))
    return topic_model, np.array(topics)


def compute_metrics(emb, texts, labels, tok, dictionary, cv):
    tw, ctfidf, _ = bs.topic_keywords(texts, labels, cv, top_n=10)
    c_v, c_npmi = bs.coherence(tw, tok, dictionary)
    div = bs.diversity(tw)
    sep = bs.separation_metrics(emb, labels, ctfidf)
    sizes = bs.size_stats(labels)
    return {
        **sizes,
        "c_v": round(c_v, 4), "c_npmi": round(c_npmi, 4), "topic_diversity": round(div, 4),
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in sep.items()},
    }


def save_outputs(topic_model, labels, texts, ids):
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    info = topic_model.get_topic_info()
    info.to_csv(FINAL_DIR / "topic_info.csv", index=False)
    pd.DataFrame({"id": ids, "topic_id": labels}).to_csv(FINAL_DIR / "topics.csv", index=False)

    rows, rep_rows = [], []
    for tid in sorted(t for t in set(labels.tolist()) if t != -1):
        words = topic_model.get_topic(tid) or []
        rows.append({
            "topic_id": tid,
            "size": int(np.sum(labels == tid)),
            "keywords": ", ".join(w for w, _ in words[:15]),
            "scores": ", ".join(str(round(float(s), 4)) for _, s in words[:15]),
        })
        try:
            reps = topic_model.get_representative_docs(tid) or []
        except Exception:
            reps = []
        for doc in reps[:4]:
            rep_rows.append({"topic_id": tid, "representative_doc": anonymize(doc)})
    pd.DataFrame(rows).to_csv(FINAL_DIR / "topic_keywords.csv", index=False)
    pd.DataFrame(rep_rows).to_csv(FINAL_DIR / "representative_docs.csv", index=False)

    # topic hierarchy (addresses weak-hierarchy P7)
    try:
        hier = topic_model.hierarchical_topics(texts)
        hier.to_csv(FINAL_DIR / "topic_hierarchy.csv", index=False)
        LOGGER.info("Saved topic hierarchy (%d merges)", len(hier))
    except Exception as e:
        LOGGER.warning("Hierarchy skipped: %s", str(e)[:80])
    return rows


def make_figures(labels, ctfidf_norm_sim, keyword_rows):
    import matplotlib.pyplot as plt

    # topic size distribution
    sizes = sorted((int(np.sum(labels == t)) for t in set(labels.tolist()) if t != -1), reverse=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(sizes)), sizes, color="#2c6fbb")
    ax.set_xlabel("Topic (sorted by size)"); ax.set_ylabel("Documents")
    ax.set_title(f"Final topic size distribution — {len(sizes)} topics")
    fig.tight_layout(); fig.savefig(FIG_DIR / "topic_sizes.png", dpi=170); plt.close(fig)

    # inter-topic keyword-similarity heatmap (distinctiveness)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(ctfidf_norm_sim, cmap="magma", vmin=0, vmax=1)
    ax.set_title("Inter-topic keyword (c-TF-IDF) cosine similarity")
    ax.set_xlabel("Topic"); ax.set_ylabel("Topic")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(FIG_DIR / "topic_similarity_heatmap.png", dpi=170); plt.close(fig)
    LOGGER.info("Saved figures to %s", FIG_DIR)


def write_reports(metrics, baseline, keyword_rows):
    with open(FINAL_DIR / "final_metrics.json", "w") as f:
        json.dump({"final_41": {"config_id": BEST["config_id"], **metrics},
                   "baseline_98_same_pipeline": baseline,
                   "note": "Both measured via identical metric pipeline on final "
                           "all-assigned labels. Embedding-centroid separation/inter_sim_emb "
                           "excluded from selection (confounded with topic count)."}, f, indent=2)
    b = baseline
    lines = [
        "# Final BERTopic Model — Quality Report", "",
        f"Selected configuration: **`{BEST['config_id']}`**", "",
        "## Before / after vs the previous 98-topic model", "",
        "Both models measured through the IDENTICAL metric pipeline on their final",
        "(all-documents-assigned) labels, so the comparison is apples-to-apples.", "",
        "| Metric | 98-topic baseline | Final | Verdict |",
        "|---|---:|---:|---|",
        f"| Topics | {b['n_topics']} | {metrics['n_topics']} | fewer — fragmentation resolved (P1) |",
        f"| c_v (coherence) | {b['c_v']} | {metrics['c_v']} | higher is better |",
        f"| c_npmi | {b['c_npmi']} | {metrics['c_npmi']} | higher is better |",
        f"| Topic diversity | {b['topic_diversity']} | {metrics['topic_diversity']} | higher is better |",
        f"| Tiny topics (<200) | {b['topics_lt_200']} | {metrics['topics_lt_200']} | fewer is better (P5) |",
        f"| Smallest / largest topic | {b['smallest_topic']} / {b['largest_topic']} | {metrics['smallest_topic']} / {metrics['largest_topic']} | — |",
        f"| Redundant pairs (centroid >0.90) | {b['redundant_pairs']} | {metrics['redundant_pairs']} | fewer is better |",
        f"| Inter-topic keyword sim (c-TF-IDF) | {b['inter_sim_ctfidf']} | {metrics['inter_sim_ctfidf']} | trade-off (broader topics) |",
        f"| Intra-topic sim (compactness) | {b['intra_sim']} | {metrics['intra_sim']} | trade-off (broader topics) |",
        f"| Outlier rate (after reduction) | {b['outlier_rate']} | {metrics['outlier_rate']} | both near-zero |",
        "",
        "> Embedding-centroid `separation` / `inter_sim_emb` are confounded with topic",
        "> count and full-assignment (corr(n_topics, inter_sim_emb) = -0.88), so they were",
        "> excluded from selection and are omitted here. Selection was anchored on",
        "> count-independent quality (coherence, stability) and semantic interpretability.",
        "",
        "## Configuration", "",
        f"- UMAP: {BEST['umap']}",
        f"- HDBSCAN: {BEST['hdbscan']}",
        "- Representation: corpus-level stopword+bigram vectorizer + KeyBERTInspired + MMR",
        "- Outlier reduction: c-TF-IDF strategy, threshold 0.05",
        "",
        "## Final topics", "",
        "| Topic | Size | Keywords |", "|---:|---:|---|",
    ]
    for r in sorted(keyword_rows, key=lambda x: x["size"], reverse=True):
        lines.append(f"| {r['topic_id']} | {r['size']} | {r['keywords']} |")
    (FINAL_DIR / "topic_quality_report.md").write_text("\n".join(lines) + "\n")
    LOGGER.info("Wrote %s", FINAL_DIR / "topic_quality_report.md")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    for noisy in ("gensim", "umap", "numba", "sentence_transformers", "BERTopic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    from sklearn.preprocessing import normalize as sk_normalize
    emb = sk_normalize(np.load(bs.EMBEDDING_PATH).astype(np.float32), axis=1)
    df = pd.read_csv(bs.TEXT_PATH, usecols=["id", "cleaned_text"])
    texts = df["cleaned_text"].fillna("").astype(str).tolist()
    ids = df["id"].tolist()

    topic_model, labels = build_final_model(emb, texts)
    keyword_rows = save_outputs(topic_model, labels, texts, ids)

    # shared coherence reference + vectorizer for identical measurement of both models
    tok, dictionary = bs.build_coherence_reference(texts)
    cv = bs.build_vectorizer(texts)

    metrics = compute_metrics(emb, texts, labels, tok, dictionary, cv)
    LOGGER.info("FINAL metrics: %s", json.dumps(metrics))

    base_labels = pd.read_csv(BASELINE_LABELS_CSV)["topic"].values
    baseline = compute_metrics(emb, texts, base_labels, tok, dictionary, cv)
    LOGGER.info("BASELINE (98-topic, same pipeline): %s", json.dumps(baseline))

    # inter-topic keyword similarity for the heatmap
    _, ctfidf, _ = bs.topic_keywords(texts, labels, cv, top_n=10)
    ct = ctfidf / (np.linalg.norm(ctfidf, axis=1, keepdims=True) + 1e-12)
    make_figures(labels, ct @ ct.T, keyword_rows)
    write_reports(metrics, baseline, keyword_rows)
    LOGGER.info("Finalization complete -> %s", FINAL_DIR)


if __name__ == "__main__":
    main()
