#!/usr/bin/env python3
"""
Systematic BERTopic hyperparameter search on the FROZEN BGE-m3 post embeddings.

Nothing here regenerates embeddings, preprocessing, retrieval or the corpus.
It only explores the BERTopic search space (UMAP + HDBSCAN + c-TF-IDF
representation) to maximise semantic topic quality: fewer redundant topics,
stronger coherence, better separation, more compact clusters.

Design
------
* UMAP reductions are expensive, so each UMAP config is fit ONCE per seed and
  cached to disk. HDBSCAN configs are then swept cheaply on the cached reduction.
* Genuine topic stability is measured as adjusted_rand_score between two UMAP
  seeds clustered with identical HDBSCAN params (agreement under UMAP stochasticity).
* Topic keywords use BERTopic's own ClassTfidfTransformer so the search-time
  representation matches what the final BERTopic model produces (plus a stronger
  stopword / n-gram vectorizer that is itself part of the representation search).
* All metrics required by the protocol are recorded per config, results are
  written incrementally (resumable), then ranked jointly by a z-score composite.

Run:  python src/pipeline/bertopic_search.py
"""
import json
import logging
import os
import re
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

LOGGER = logging.getLogger("bertopic_search")

# ----------------------------------------------------------------------------
# Paths (all frozen inputs; only SEARCH_DIR is written to)
# ----------------------------------------------------------------------------
EMBEDDING_PATH = "data/embeddings/posts/bge_m3_embeddings.npy"
TEXT_PATH = "data/processed/posts_merged_final_preprocessed.csv"
SEARCH_DIR = Path("final_outputs/topic_modeling/bertopic_search")
UMAP_CACHE = SEARCH_DIR / "umap_cache"
RESULTS_JSONL = SEARCH_DIR / "search_results.jsonl"

# ----------------------------------------------------------------------------
# Vocabulary control (part of the representation search - drives keyword quality)
# ----------------------------------------------------------------------------
ENGLISH_EXTRA_STOPWORDS = {
    "im", "ive", "dont", "didnt", "cant", "wont", "isnt", "thats", "youre",
    "theyre", "weve", "would", "could", "should", "also", "just", "really",
    "like", "get", "got", "getting", "know", "think", "feel", "want", "need",
    "people", "person", "student", "students", "year", "years", "day", "days",
    "month", "months", "time", "thing", "things", "good", "bad", "nice", "help",
    "pls", "please", "thanks", "much", "many", "even", "still", "way", "going",
    "guys", "guy", "someone", "anyone", "everyone", "something", "anything",
    "lot", "actually", "basically", "literally", "gonna", "wanna", "yeah", "okay",
    "ok", "well", "make", "made", "take", "use", "using", "one", "two", "back",
    "give", "sure", "tell", "ask", "asked", "come", "went", "put", "let", "said",
}
HINGLISH_STOPWORDS = {
    "hai", "ki", "ke", "ka", "se", "aur", "bhi", "mai", "main", "nahi", "nhi",
    "tha", "thi", "the", "ho", "hua", "hui", "kar", "kya", "ye", "woh", "wo",
    "is", "us", "apna", "apni", "apne", "mera", "meri", "mere", "tera", "teri",
    "tere", "ham", "hum", "tum", "tumhara", "hu", "haan", "ya", "koi", "bhai",
    "bro", "toh", "kuch", "hi", "mein", "ko", "mujhe", "ek", "ab", "h", "par",
    "to", "do", "ni", "na", "hoga", "hota", "hoti", "raha", "rha", "rahi", "gaya",
    "gya", "gaye", "gye", "liye", "jo", "jab", "tab", "yaar", "abhi", "sab",
    "bahut", "bohot", "acha", "achha", "kaise", "kaisa", "kyun", "kyu", "matlab",
    # grammatical connectives / light verbs that otherwise dominate the largest
    # Hinglish-narrative cluster without describing its content
    "pe", "maine", "ne", "kiya", "diya", "fir", "aa", "bas", "aaj", "kal", "pata",
    "karu", "karun", "aayega", "aya", "ayi", "lo", "le", "de", "dena", "kare",
    "hun", "hoon", "kr", "krna", "krne", "krta", "krti", "hoga", "raha", "rahe",
    "aur", "wala", "wali", "wale", "kyunki", "lekin", "phir", "bhai", "yrr",
}
ARTIFACT_STOPWORDS = {
    "deleted", "removed", "amp", "title", "post", "comment", "reddit",
    "subreddit", "24tard", "25tard", "26tard", "educational", "info", "edit",
    "https", "http", "www", "com", "org", "www", "nbsp", "gt", "lt", "x200b",
}
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOPWORDS = set(ENGLISH_STOP_WORDS) | ENGLISH_EXTRA_STOPWORDS | HINGLISH_STOPWORDS | ARTIFACT_STOPWORDS

VECTORIZER_CONFIG = dict(
    stop_words=list(STOPWORDS),
    ngram_range=(1, 2),
    min_df=15,
    max_df=0.45,
    max_features=20000,
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z_0-9]+\b",
)

# ----------------------------------------------------------------------------
# Search space  (carefully selected, not a blind full grid)
# ----------------------------------------------------------------------------
# UMAP: vary how much global structure is preserved.  min_dist=0.0 is best for
# clustering; larger n_neighbors -> broader, more semantically merged topics.
UMAP_CONFIGS = [
    {"name": "u_nn15_c5",  "n_neighbors": 15, "n_components": 5,  "min_dist": 0.0, "metric": "cosine"},
    {"name": "u_nn30_c5",  "n_neighbors": 30, "n_components": 5,  "min_dist": 0.0, "metric": "cosine"},
    {"name": "u_nn50_c5",  "n_neighbors": 50, "n_components": 5,  "min_dist": 0.0, "metric": "cosine"},
    {"name": "u_nn30_c10", "n_neighbors": 30, "n_components": 10, "min_dist": 0.0, "metric": "cosine"},
    {"name": "u_nn50_c10", "n_neighbors": 50, "n_components": 10, "min_dist": 0.0, "metric": "cosine"},
]
# HDBSCAN: 'eom' (excess of mass) yields fewer, broader clusters than 'leaf'.
# Larger min_cluster_size reduces fragmentation; SMALLER min_samples keeps the
# natural outlier rate manageable (large min_samples over-flags noise under eom).
HDBSCAN_CONFIGS = [
    {"name": "h_mcs100_ms5_eom",  "min_cluster_size": 100, "min_samples": 5,  "cluster_selection_method": "eom"},
    {"name": "h_mcs150_ms5_eom",  "min_cluster_size": 150, "min_samples": 5,  "cluster_selection_method": "eom"},
    {"name": "h_mcs250_ms10_eom", "min_cluster_size": 250, "min_samples": 10, "cluster_selection_method": "eom"},
    {"name": "h_mcs400_ms10_eom", "min_cluster_size": 400, "min_samples": 10, "cluster_selection_method": "eom"},
    {"name": "h_mcs150_ms10_leaf", "min_cluster_size": 150, "min_samples": 10, "cluster_selection_method": "leaf"},
    {"name": "h_mcs300_ms10_leaf", "min_cluster_size": 300, "min_samples": 10, "cluster_selection_method": "leaf"},
]

SEEDS = [42, 7]                       # two seeds -> genuine stability via ARI
COHERENCE_SAMPLE = 20000              # reference corpus size for c_v / c_npmi
# The confident-core outlier rate is intrinsically ~50% for these embeddings
# (HDBSCAN flags low-density points as noise); the pipeline resolves this later
# with outlier reduction.  So we DO NOT reject on outlier rate here beyond a
# loose sanity bound -- instead quality is scored on the confident core and the
# outlier rate enters the joint ranking as a penalty.
MAX_OUTLIER_RATE = 0.70               # loose sanity bound only
MIN_TOPICS = 6                        # reject collapsed runs


# ----------------------------------------------------------------------------
def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    for noisy in ("gensim", "gensim.corpora.dictionary", "gensim.topic_coherence",
                  "gensim.models", "umap", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def load_inputs():
    from sklearn.preprocessing import normalize as sk_normalize

    LOGGER.info("Loading frozen BGE-m3 embeddings: %s", EMBEDDING_PATH)
    emb = np.load(EMBEDDING_PATH).astype(np.float32)
    emb = sk_normalize(emb, norm="l2", axis=1)
    LOGGER.info("Embeddings: %s", emb.shape)

    LOGGER.info("Loading texts (cleaned_text): %s", TEXT_PATH)
    texts = pd.read_csv(TEXT_PATH, usecols=["cleaned_text"])["cleaned_text"].fillna("").astype(str).tolist()
    assert len(texts) == emb.shape[0], f"text/embedding mismatch {len(texts)} vs {emb.shape[0]}"
    return emb, texts


def build_coherence_reference(texts):
    """One reference corpus + dictionary, reused for every config (big speedup)."""
    from gensim.corpora import Dictionary

    rng = np.random.RandomState(42)
    idx = rng.choice(len(texts), min(COHERENCE_SAMPLE, len(texts)), replace=False)
    tok = [re.findall(r"\b[a-z][a-z_0-9]+\b", texts[i].lower()) for i in idx]
    tok = [[w for w in doc if w not in STOPWORDS] for doc in tok]
    dictionary = Dictionary(tok)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    LOGGER.info("Coherence reference: %d docs, dictionary=%d", len(tok), len(dictionary))
    return tok, dictionary


# ----------------------------------------------------------------------------
# UMAP with on-disk caching
# ----------------------------------------------------------------------------
def get_umap(emb, ucfg, seed):
    import umap

    UMAP_CACHE.mkdir(parents=True, exist_ok=True)
    cache = UMAP_CACHE / f"{ucfg['name']}_seed{seed}.npy"
    if cache.exists():
        return np.load(cache)
    t0 = time.time()
    model = umap.UMAP(
        n_neighbors=ucfg["n_neighbors"],
        n_components=ucfg["n_components"],
        min_dist=ucfg["min_dist"],
        metric=ucfg["metric"],
        random_state=seed,
        low_memory=True,
        verbose=False,
    )
    reduced = model.fit_transform(emb).astype(np.float32)
    np.save(cache, reduced)
    LOGGER.info("UMAP %s seed=%d -> %s (%.0fs)", ucfg["name"], seed, reduced.shape, time.time() - t0)
    return reduced


def cluster(reduced, hcfg):
    import hdbscan

    model = hdbscan.HDBSCAN(
        min_cluster_size=hcfg["min_cluster_size"],
        min_samples=hcfg["min_samples"],
        cluster_selection_method=hcfg["cluster_selection_method"],
        metric="euclidean",
        core_dist_n_jobs=-1,
    )
    return model.fit_predict(reduced)


# ----------------------------------------------------------------------------
# Representation + metrics
# ----------------------------------------------------------------------------
def build_vectorizer(texts):
    """Fit the CountVectorizer ONCE on the full post corpus so min_df / max_df
    are proper corpus-level document frequencies (not per-topic counts), then
    reuse the fixed vocabulary for every config's class-based c-TF-IDF."""
    from sklearn.feature_extraction.text import CountVectorizer

    cv = CountVectorizer(**VECTORIZER_CONFIG)
    cv.fit(texts)
    LOGGER.info("Global vectorizer fitted: vocab=%d", len(cv.get_feature_names_out()))
    return cv


def topic_keywords(texts, labels, cv, top_n=10):
    """BERTopic-faithful c-TF-IDF top words per (non-outlier) topic, using the
    pre-fit corpus-level vocabulary."""
    from bertopic.vectorizers import ClassTfidfTransformer

    labels = np.asarray(labels)
    topic_ids = sorted(t for t in set(labels.tolist()) if t != -1)
    joined = []
    for t in topic_ids:
        docs = [texts[i] for i in np.where(labels == t)[0]]
        joined.append(" ".join(docs))
    X = cv.transform(joined)                            # n_topics x V (fixed vocab)
    ctfidf = ClassTfidfTransformer().fit_transform(X)  # n_topics x V
    vocab = np.array(cv.get_feature_names_out())
    ctfidf = ctfidf.toarray()
    out = {}
    for row, t in enumerate(topic_ids):
        order = np.argsort(ctfidf[row])[::-1][:top_n]
        out[t] = [vocab[j] for j in order if ctfidf[row][j] > 0]
    return out, ctfidf, vocab


def coherence(topic_words, tok, dictionary):
    from gensim.models import CoherenceModel

    topics = []
    for words in topic_words.values():
        valid = [w for w in words if w in dictionary.token2id]
        if len(valid) >= 3:
            topics.append(valid)
    if len(topics) < 2:
        return 0.0, 0.0
    cv = CoherenceModel(topics=topics, texts=tok, dictionary=dictionary, coherence="c_v", topn=10, processes=1).get_coherence()
    npmi = CoherenceModel(topics=topics, texts=tok, dictionary=dictionary, coherence="c_npmi", topn=10, processes=1).get_coherence()
    return float(cv), float(npmi)


def diversity(topic_words):
    words = [w for ws in topic_words.values() for w in ws[:10]]
    return len(set(words)) / len(words) if words else 0.0


def separation_metrics(emb, labels, ctfidf):
    """Compactness (intra) and distinctiveness (inter) in embedding + keyword space."""
    labels = np.asarray(labels)
    topic_ids = sorted(t for t in set(labels.tolist()) if t != -1)
    centroids, intra = [], []
    for t in topic_ids:
        members = emb[labels == t]
        c = members.mean(axis=0)
        cn = c / (np.linalg.norm(c) + 1e-12)
        centroids.append(cn)
        intra.append(float(np.mean(members @ cn)))          # members already L2-normed
    centroids = np.vstack(centroids)
    sim = centroids @ centroids.T
    n = len(topic_ids)
    iu = np.triu_indices(n, k=1)
    inter_emb = float(sim[iu].mean()) if n > 1 else 0.0
    redundant_pairs = int((sim[iu] > 0.90).sum()) if n > 1 else 0

    ct = ctfidf / (np.linalg.norm(ctfidf, axis=1, keepdims=True) + 1e-12)
    ksim = ct @ ct.T
    inter_kw = float(ksim[iu].mean()) if n > 1 else 0.0

    return {
        "intra_sim": float(np.mean(intra)),
        "inter_sim_emb": inter_emb,
        "inter_sim_ctfidf": inter_kw,
        "separation": float(np.mean(intra)) - inter_emb,
        "redundant_pairs": redundant_pairs,
    }


def size_stats(labels):
    labels = np.asarray(labels)
    total = len(labels)
    n_out = int((labels == -1).sum())
    sizes = np.array([int((labels == t).sum()) for t in set(labels.tolist()) if t != -1])
    if sizes.size == 0:
        return None
    return {
        "n_topics": int(sizes.size),
        "outlier_rate": round(n_out / total, 4),
        "avg_topic_size": round(float(sizes.mean()), 1),
        "median_topic_size": int(np.median(sizes)),
        "smallest_topic": int(sizes.min()),
        "largest_topic": int(sizes.max()),
        "size_std": round(float(sizes.std()), 1),
        "topics_lt_100": int((sizes < 100).sum()),
        "topics_lt_200": int((sizes < 200).sum()),
    }


def stability_ari(labels_a, labels_b):
    from sklearn.metrics import adjusted_rand_score

    a, b = np.asarray(labels_a), np.asarray(labels_b)
    mask = (a != -1) & (b != -1)
    if mask.sum() < 100:
        return 0.0
    return round(float(adjusted_rand_score(a[mask], b[mask])), 4)


# ----------------------------------------------------------------------------
# Main search
# ----------------------------------------------------------------------------
def already_done():
    done = set()
    if RESULTS_JSONL.exists():
        for line in RESULTS_JSONL.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["config_id"])
    return done


def run_search():
    setup_logging()
    SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    emb, texts = load_inputs()
    tok, dictionary = build_coherence_reference(texts)
    cv = build_vectorizer(texts)
    done = already_done()
    LOGGER.info("Already completed configs: %d", len(done))

    total = len(UMAP_CONFIGS) * len(HDBSCAN_CONFIGS)
    n = 0
    for ucfg in UMAP_CONFIGS:
        reduced_a = get_umap(emb, ucfg, SEEDS[0])
        reduced_b = None  # lazily computed only when a config needs stability
        for hcfg in HDBSCAN_CONFIGS:
            n += 1
            config_id = f"{ucfg['name']}__{hcfg['name']}"
            if config_id in done:
                LOGGER.info("[%d/%d] skip (done): %s", n, total, config_id)
                continue
            t0 = time.time()
            labels_a = cluster(reduced_a, hcfg)
            sizes = size_stats(labels_a)
            if sizes is None or sizes["n_topics"] < MIN_TOPICS or sizes["outlier_rate"] > MAX_OUTLIER_RATE:
                rec = {"config_id": config_id, "valid": False,
                       "reason": "no clusters" if sizes is None else
                                 f"n_topics={sizes['n_topics'] if sizes else 0} outlier={sizes['outlier_rate'] if sizes else 1}",
                       **_params(ucfg, hcfg), **(sizes or {})}
                _append(rec)
                LOGGER.info("[%d/%d] %s INVALID (%s)", n, total, config_id, rec["reason"])
                continue

            tw, ctfidf, _ = topic_keywords(texts, labels_a, cv, top_n=10)
            cv_score, npmi = coherence(tw, tok, dictionary)
            div = diversity(tw)
            sep = separation_metrics(emb, labels_a, ctfidf)

            if reduced_b is None:
                reduced_b = get_umap(emb, ucfg, SEEDS[1])
            labels_b = cluster(reduced_b, hcfg)
            stab = stability_ari(labels_a, labels_b)

            rec = {
                "config_id": config_id, "valid": True, "reason": "",
                **_params(ucfg, hcfg), **sizes,
                "c_v": round(cv_score, 4), "c_npmi": round(npmi, 4),
                "topic_diversity": round(div, 4), "topic_stability": stab,
                **{k: round(v, 4) if isinstance(v, float) else v for k, v in sep.items()},
                "elapsed_s": round(time.time() - t0, 1),
                "top_topics_preview": " | ".join(", ".join(ws[:6]) for ws in list(tw.values())[:6]),
            }
            _append(rec)
            LOGGER.info("[%d/%d] %s | topics=%d out=%.3f c_v=%.3f npmi=%.3f div=%.3f stab=%.3f sep=%.3f (%.0fs)",
                        n, total, config_id, sizes["n_topics"], sizes["outlier_rate"],
                        cv_score, npmi, div, stab, sep["separation"], rec["elapsed_s"])

    LOGGER.info("Search complete. Ranking...")
    rank_and_report()


def _params(ucfg, hcfg):
    return {
        "umap_n_neighbors": ucfg["n_neighbors"], "umap_n_components": ucfg["n_components"],
        "umap_min_dist": ucfg["min_dist"], "umap_metric": ucfg["metric"],
        "hdbscan_min_cluster_size": hcfg["min_cluster_size"],
        "hdbscan_min_samples": hcfg["min_samples"],
        "hdbscan_cluster_selection_method": hcfg["cluster_selection_method"],
    }


def _append(rec):
    with open(RESULTS_JSONL, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------
# Joint ranking + reports
# ----------------------------------------------------------------------------
# metric -> +1 higher is better, -1 lower is better
#
# NOTE ON THE RANKING METRICS.  Embedding-centroid `separation` and `inter_sim_emb`
# are deliberately EXCLUDED from the composite (they are still recorded/reported).
# Empirically corr(n_topics, inter_sim_emb) = -0.88 and separation likewise rises
# with topic count: with fewer topics each centroid averages a broader region and
# drifts toward the global mean, inflating mutual similarity.  Ranking on them would
# reward fragmentation -- the opposite of the stated objective (P1: reduce the ~98
# over-split topics).  We instead rank on count-independent quality signals plus a
# direct penalty on topic count and near-duplicate topics.
#   Anchor: c_v, c_npmi and topic_stability are count-independent quality signals
#   that PEAK in the middle -- collapse (few mega-blobs) tanks coherence + stability,
#   while over-fragmentation raises redundant_pairs and tiny-topic counts.  We
#   therefore rank on this trio plus direct penalties on keyword overlap, near-
#   duplicate topics and tiny topics.  Metrics that are artifacts of collapse
#   (diversity, outlier_rate) or of fragmentation (separation, inter_sim_emb,
#   intra_sim) are recorded and reported but kept OUT of the ranking.
RANK_METRICS = {
    "c_v": 1, "c_npmi": 1, "topic_stability": 1,   # count-independent quality anchor
    "inter_sim_ctfidf": -1,            # keyword-space distinctiveness
    "redundant_pairs": -1,             # near-duplicate topics (centroid cos > 0.90)
    "topics_lt_200": -1,               # small / unstable topics
}


def rank_and_report():
    recs = [json.loads(l) for l in RESULTS_JSONL.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(recs)
    df.to_csv(SEARCH_DIR / "all_experiments.csv", index=False)
    valid = df[df["valid"]].copy()
    if valid.empty:
        LOGGER.warning("No valid configs to rank.")
        return

    # z-score composite across valid configs
    comp = np.zeros(len(valid))
    for m, sign in RANK_METRICS.items():
        if m not in valid:
            continue
        x = valid[m].astype(float).values
        s = x.std()
        z = (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)
        comp += sign * z
    valid["composite_score"] = np.round(comp / len(RANK_METRICS), 4)
    valid = valid.sort_values("composite_score", ascending=False).reset_index(drop=True)
    valid["rank"] = valid.index + 1
    valid.to_csv(SEARCH_DIR / "ranking.csv", index=False)

    best = valid.iloc[0]
    with open(SEARCH_DIR / "best_config.json", "w") as f:
        json.dump(best.to_dict(), f, indent=2, default=str)

    _write_report(df, valid, best)
    LOGGER.info("BEST: %s | topics=%s c_v=%s npmi=%s stab=%s sep=%s score=%s",
                best["config_id"], best["n_topics"], best["c_v"], best["c_npmi"],
                best["topic_stability"], best["separation"], best["composite_score"])


def _write_report(df, valid, best):
    lines = ["# BERTopic Hyperparameter Search — Comparison Report", ""]
    lines += [
        f"- Embeddings (frozen): `{EMBEDDING_PATH}` — 70,425 × 1024, L2-normalized",
        f"- Configurations evaluated: {len(df)} ({len(valid)} valid, {len(df) - len(valid)} rejected)",
        f"- Stability = adjusted Rand index between UMAP seeds {SEEDS[0]} and {SEEDS[1]}",
        "- Objective: maximise semantic quality (coherence, separation, compactness,",
        "  interpretability) while reducing redundant fragmentation. Joint z-score ranking.",
        "",
        "## Best Configuration",
        "",
        f"**`{best['config_id']}`** (rank 1, composite {best['composite_score']})",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| UMAP n_neighbors | {best['umap_n_neighbors']} |",
        f"| UMAP n_components | {best['umap_n_components']} |",
        f"| UMAP min_dist | {best['umap_min_dist']} |",
        f"| UMAP metric | {best['umap_metric']} |",
        f"| HDBSCAN min_cluster_size | {best['hdbscan_min_cluster_size']} |",
        f"| HDBSCAN min_samples | {best['hdbscan_min_samples']} |",
        f"| HDBSCAN cluster_selection_method | {best['hdbscan_cluster_selection_method']} |",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Topics | {best['n_topics']} |",
        f"| Outlier rate | {best['outlier_rate']} |",
        f"| c_v | {best['c_v']} |",
        f"| c_npmi | {best['c_npmi']} |",
        f"| Topic diversity | {best['topic_diversity']} |",
        f"| Topic stability (ARI) | {best['topic_stability']} |",
        f"| Intra-topic sim (compactness) | {best['intra_sim']} |",
        f"| Inter-topic sim (embedding) | {best['inter_sim_emb']} |",
        f"| Inter-topic sim (c-TF-IDF) | {best['inter_sim_ctfidf']} |",
        f"| Separation (intra − inter) | {best['separation']} |",
        f"| Redundant pairs (>0.90) | {best['redundant_pairs']} |",
        f"| Avg / median / min / max size | {best['avg_topic_size']} / {best['median_topic_size']} / {best['smallest_topic']} / {best['largest_topic']} |",
        "",
        "## Ranking (top 15 by joint composite)",
        "",
        "| Rank | Config | Topics | Out% | c_v | c_npmi | Div | Stab | Intra | InterEmb | Sep | Redund | Score |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in valid.head(15).iterrows():
        lines.append(
            f"| {r['rank']} | {r['config_id']} | {r['n_topics']} | {r['outlier_rate']} | "
            f"{r['c_v']} | {r['c_npmi']} | {r['topic_diversity']} | {r['topic_stability']} | "
            f"{r['intra_sim']} | {r['inter_sim_emb']} | {r['separation']} | {r['redundant_pairs']} | "
            f"{r['composite_score']} |"
        )
    lines += ["", "## Full parameter × metric table", "",
              "See `all_experiments.csv` and `ranking.csv` for every configuration.", ""]
    (SEARCH_DIR / "search_report.md").write_text("\n".join(lines) + "\n")
    LOGGER.info("Wrote %s", SEARCH_DIR / "search_report.md")


if __name__ == "__main__":
    run_search()
