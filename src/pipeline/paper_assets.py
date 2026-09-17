#!/usr/bin/env python3
"""
Regenerate the COMPLETE publication asset set from the finalized 41-topic BERTopic
model, into final_outputs/paper_assets/.  Everything here is derived exclusively from the
finalized model (final_outputs/topic_modeling/bertopic_final) plus read-only frozen corpus metadata.

Nothing upstream is touched: embeddings, preprocessing, phrases, corpus, retrieval,
annotations and embedding benchmarks are all left as-is.  Historical experiment
folders are preserved; this writes only under final_outputs/paper_assets/.

Run:  python src/pipeline/paper_assets.py
"""
import json
import logging
import os
import re
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bertopic_search as bs

LOGGER = logging.getLogger("paper_assets")

SRC = Path("final_outputs/topic_modeling/bertopic_final")
OUT = Path("final_outputs/paper_assets")
DIRS = {k: OUT / k for k in
        ["figures", "tables", "reports", "taxonomy", "representative_documents", "metadata"]}

EMB_PATH = "data/embeddings/posts/bge_m3_embeddings.npy"
META_PATH = "data/processed/posts_merged_final_preprocessed.csv"

# ---------------------------------------------------------------------------
# Higher-level themes — reconstructed from the 41-topic model.
# Groupings were seeded by agglomerative clustering of topic centroids in the
# frozen BGE-m3 space (which surfaced the tight coping-meme, study-resource and
# coaching sub-clusters) and finalized by reading each topic's keywords and
# representative documents.  This is NOT the old hand mapping.
# ---------------------------------------------------------------------------
THEME_MAP = {
    "Entrance Exam Preparation": [3, 9, 12, 16, 18, 27, 31, 32, 35, 38, 39],
    "Coaching & Study Resources": [8, 15, 17, 25, 33, 40],
    "College & Academic Decisions": [5, 21, 28, 29, 36, 37],
    "Careers, Placements & Jobs": [0, 4, 7, 19, 20, 23],
    "Higher Studies & Global Mobility": [1, 10, 13],
    "Mental Health & Wellbeing": [6, 26, 34],
    "Coping & Meme Culture": [11, 14, 22, 24, 30],
    "General Venting": [2],
}
THEME_DESC = {
    "Entrance Exam Preparation":
        "Preparation, attempts, marks and dropping decisions across India's high-stakes "
        "entrance exams (JEE Main/Advanced, NEET, CAT, UPSC, boards, improvement exams) "
        "and subject-level study for them.",
    "Coaching & Study Resources":
        "Coaching institutes (Allen, PW, FIITJEE, Aakash), lectures and faculty, study "
        "material (modules, HCV, Cengage), exam-day logistics and result-day discussion.",
    "College & Academic Decisions":
        "Branch and stream selection, engineering disciplines, CGPA/backlogs and everyday "
        "college academic life — the decision and performance side of undergraduate study.",
    "Careers, Placements & Jobs":
        "Placements and recruitment drives, salary/CTC, interviews and hiring processes, "
        "internships, resumes and coding/DSA skill-building for jobs.",
    "Higher Studies & Global Mobility":
        "Post-graduate pathways: GATE/M.Tech, MBA/IIM admissions and profiles, and studying "
        "abroad (master's, visas, universities).",
    "Mental Health & Wellbeing":
        "Explicit emotional distress (depression, suicidal ideation), burnout, sleep and "
        "anxiety, and social isolation / relationships.",
    "Coping & Meme Culture":
        "Community coping through humour and in-group memes — hopium/copium, 'cooked', "
        "'canon event', fatalistic 'barbaad' venting and the ALECC coaching-meme subculture.",
    "General Venting":
        "Free-form Hinglish personal narrative and venting that is not tied to a single "
        "concrete sub-topic (the corpus's natural catch-all).",
}

# ---------------------------------------------------------------------------
# Distress taxonomy — per-topic distress type (controlled vocabulary).
# ---------------------------------------------------------------------------
DISTRESS_MAP = {
    0: "Career Distress", 1: "Career Distress", 2: "Coping Expression",
    3: "Academic Distress", 4: "Career Distress", 5: "Informational / Low-Distress",
    6: "Emotional Distress", 7: "Career Distress", 8: "Academic Distress",
    9: "Academic Distress", 10: "Career Distress", 11: "Coping Expression",
    12: "Academic Distress", 13: "Career Distress", 14: "Coping Expression",
    15: "Informational / Low-Distress", 16: "Academic Distress", 17: "Academic Distress",
    18: "Academic Distress", 19: "Career Distress", 20: "Career Distress",
    21: "Informational / Low-Distress", 22: "Coping Expression", 23: "Career Distress",
    24: "Coping Expression", 25: "Institutional Distress", 26: "Emotional Distress",
    27: "Academic Distress", 28: "Informational / Low-Distress",
    29: "Informational / Low-Distress", 30: "Coping Expression", 31: "Academic Distress",
    32: "Academic Distress", 33: "Institutional Distress", 34: "Emotional Distress",
    35: "Academic Distress", 36: "Academic Distress", 37: "Informational / Low-Distress",
    38: "Academic Distress", 39: "Academic Distress", 40: "Institutional Distress",
}
THEME_DISTRESS_CATEGORY = {
    "Entrance Exam Preparation": "Academic Distress",
    "Coaching & Study Resources": "Institutional / Academic Support",
    "College & Academic Decisions": "Academic Planning (Low-Distress)",
    "Careers, Placements & Jobs": "Career Distress",
    "Higher Studies & Global Mobility": "Career Planning",
    "Mental Health & Wellbeing": "Emotional Distress",
    "Coping & Meme Culture": "Coping / Cultural Expression",
    "General Venting": "Coping / Cultural Expression",
}
DISTRESS_DEFN = {
    "Academic Distress": "Pressure, failure and anxiety directly tied to exams, marks and studying.",
    "Career Distress": "Uncertainty and stress about placements, jobs, salary and future pathways.",
    "Emotional Distress": "Explicit mental-health signals: depression, suicidal ideation, burnout, loneliness.",
    "Institutional Distress": "Frustration with coaching institutes, exam administration and logistics.",
    "Coping Expression": "In-group memes, humour and venting used to process distress collectively.",
    "Informational / Low-Distress": "Guidance, decisions and resource-sharing with limited affective distress.",
}

TOPIC_TO_THEME = {t: theme for theme, ts in THEME_MAP.items() for t in ts}
FINAL_CONFIG = {
    "config_id": "u_nn50_c10__h_mcs300_ms10_leaf",
    "umap": {"n_neighbors": 50, "n_components": 10, "min_dist": 0.0, "metric": "cosine", "random_state": 42},
    "hdbscan": {"min_cluster_size": 300, "min_samples": 10, "cluster_selection_method": "leaf", "metric": "euclidean"},
    "representation": "corpus-level stopword+bigram CountVectorizer + KeyBERTInspired + MaximalMarginalRelevance(0.3)",
    "outlier_reduction": "reduce_outliers(strategy='c-tf-idf', threshold=0.05)",
    "embeddings": "BAAI/bge-m3 (FROZEN), 70425 x 1024, L2-normalized",
}

PALETTE = ["#2c6fbb", "#e07b39", "#3a9679", "#b5446e", "#8a6bbf", "#c9a227",
           "#4f9d9d", "#d1495b", "#6a8caf", "#9c6644"]


def setup():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    for n in ("gensim", "matplotlib", "PIL"):
        logging.getLogger(n).setLevel(logging.WARNING)
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)


def anonymize(text, n=320):
    text = re.sub(r"\[(deleted|removed)\]", " ", str(text), flags=re.I)
    text = re.sub(r"u/[A-Za-z0-9_-]+", "[user]", text)
    text = re.sub(r"r/[A-Za-z0-9_-]+", "[sub]", text)
    text = re.sub(r"https?://\S+", "[link]", text)
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[email]", text)
    text = re.sub(r"\+?\d[\d\s-]{8,}\d", "[phone]", text)
    return re.sub(r"\s+", " ", text).strip()[:n]


def save_fig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(DIRS["figures"] / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    LOGGER.info("figure: %s.{png,pdf}", name)


# ===========================================================================
def load_all():
    from sklearn.preprocessing import normalize
    topics = pd.read_csv(SRC / "topics.csv")
    kw = pd.read_csv(SRC / "topic_keywords.csv")
    info = pd.read_csv(SRC / "topic_info.csv")
    meta = pd.read_csv(META_PATH, usecols=["id", "subreddit", "lifecycle_stage", "full_text", "cleaned_text"])
    df = topics.merge(meta, on="id", how="left")
    emb = normalize(np.load(EMB_PATH).astype(np.float32), axis=1)
    return df, kw, info, emb


def topic_table(df, kw):
    total = len(df)
    kw_map = kw.set_index("topic_id")["keywords"].to_dict()
    rows = []
    for tid in sorted(t for t in df["topic_id"].unique() if t != -1):
        sub = df[df["topic_id"] == tid]
        keywords = kw_map.get(tid, "")
        name = ", ".join(str(keywords).split(", ")[:4])
        rows.append({
            "topic_id": tid,
            "topic_name": name,
            "documents": len(sub),
            "percentage": round(100 * len(sub) / total, 2),
            "keywords": keywords,
            "theme": TOPIC_TO_THEME.get(tid, "Unassigned"),
            "distress_type": DISTRESS_MAP.get(tid, "Unclassified"),
            "top_subreddit": sub["subreddit"].mode().iat[0] if not sub["subreddit"].mode().empty else "",
            "top_lifecycle": sub["lifecycle_stage"].mode().iat[0] if not sub["lifecycle_stage"].mode().empty else "",
        })
    return pd.DataFrame(rows).sort_values("documents", ascending=False)


def theme_table(topic_tbl, total):
    rows = []
    for theme in THEME_MAP:
        sub = topic_tbl[topic_tbl["theme"] == theme]
        rows.append({
            "theme": theme,
            "n_topics": len(sub),
            "documents": int(sub["documents"].sum()),
            "percentage": round(100 * sub["documents"].sum() / total, 2),
            "distress_category": THEME_DISTRESS_CATEGORY[theme],
            "topic_ids": ", ".join(map(str, sorted(sub["topic_id"]))),
            "description": THEME_DESC[theme],
        })
    return pd.DataFrame(rows).sort_values("documents", ascending=False)


def distress_table(topic_tbl, total):
    rows = []
    for dtype in ["Academic Distress", "Career Distress", "Emotional Distress",
                  "Institutional Distress", "Coping Expression", "Informational / Low-Distress"]:
        sub = topic_tbl[topic_tbl["distress_type"] == dtype]
        rows.append({
            "distress_type": dtype,
            "definition": DISTRESS_DEFN[dtype],
            "n_topics": len(sub),
            "documents": int(sub["documents"].sum()),
            "percentage": round(100 * sub["documents"].sum() / total, 2),
            "topic_ids": ", ".join(map(str, sorted(sub["topic_id"]))),
        })
    return pd.DataFrame(rows).sort_values("documents", ascending=False)


# ---------------------------------------------------------------------------
def _content_bearing(s, min_words=12):
    """Reject [deleted]/[removed] stubs and very short posts for publication quotes."""
    s2 = re.sub(r"\[(deleted|removed)\]", " ", str(s), flags=re.I)
    s2 = re.sub(r"[^A-Za-zऀ-ॿ ]", " ", s2)
    return len(s2.split()) >= min_words


def representative_docs(df, emb, topic_tbl, per_topic=5):
    """Highest-confidence (closest to centroid) + diverse representative posts,
    restricted to content-bearing posts (no deleted/removed stubs)."""
    rows = []
    txt = df["full_text"].fillna(df["cleaned_text"]).fillna("").astype(str).values
    for tid in sorted(t for t in df["topic_id"].unique() if t != -1):
        idx = np.where(df["topic_id"].values == tid)[0]
        c = emb[idx].mean(0)
        c /= (np.linalg.norm(c) + 1e-12)
        sim = emb[idx] @ c
        order = idx[np.argsort(sim)[::-1]]           # global indices, high->low confidence
        good = [gi for gi in order if _content_bearing(txt[gi])]
        pool = good if len(good) >= per_topic else list(order)  # fallback if too few
        picks = list(pool[:3])                        # top-3 highest confidence
        for q in (0.55, 0.35):                         # 2 diverse, still on-topic
            j = int((1 - q) * (len(pool) - 1))
            if pool[j] not in picks:
                picks.append(pool[j])
        for rank, gi in enumerate(picks[:per_topic], 1):
            local = np.where(idx == gi)[0][0]
            rows.append({
                "topic_id": tid,
                "theme": TOPIC_TO_THEME.get(tid, ""),
                "distress_type": DISTRESS_MAP.get(tid, ""),
                "rank": rank,
                "confidence": round(float(sim[local]), 4),
                "subreddit": df["subreddit"].values[gi],
                "lifecycle_stage": df["lifecycle_stage"].values[gi],
                "representative_document": anonymize(txt[gi]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def per_topic_coherence(df, emb):
    from gensim.models import CoherenceModel
    tok, dictionary = bs.build_coherence_reference(df["cleaned_text"].fillna("").astype(str).tolist())
    cv = bs.build_vectorizer(df["cleaned_text"].fillna("").astype(str).tolist())
    labels = df["topic_id"].values
    tw, _, _ = bs.topic_keywords(df["cleaned_text"].fillna("").astype(str).tolist(), labels, cv, top_n=10)
    tids = sorted(tw)
    topics = [[w for w in tw[t] if w in dictionary.token2id] for t in tids]
    keep = [i for i, t in enumerate(topics) if len(t) >= 3]
    cm = CoherenceModel(topics=[topics[i] for i in keep], texts=tok, dictionary=dictionary,
                        coherence="c_v", topn=10, processes=1)
    per = cm.get_coherence_per_topic()
    out = {tids[keep[i]]: round(float(per[i]), 4) for i in range(len(keep))}
    return out


# ===========================================================================
# Figures
# ===========================================================================
def fig_topic_sizes(topic_tbl):
    import matplotlib.pyplot as plt
    t = topic_tbl.sort_values("documents", ascending=False)
    colors = [PALETTE[list(THEME_MAP).index(th) % len(PALETTE)] for th in t["theme"]]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(range(len(t)), t["documents"], color=colors)
    ax.set_xticks(range(len(t)))
    ax.set_xticklabels(t["topic_id"], fontsize=7)
    ax.set_xlabel("Topic ID (colored by theme)"); ax.set_ylabel("Documents")
    ax.set_title(f"Topic size distribution — {len(t)} topics ({int(t['documents'].sum()):,} posts)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[i % len(PALETTE)]) for i in range(len(THEME_MAP))]
    ax.legend(handles, list(THEME_MAP), fontsize=7, ncol=2, loc="upper right")
    save_fig(fig, "fig01_topic_size_distribution")


def fig_similarity_heatmap(df, emb, topic_tbl):
    import matplotlib.pyplot as plt
    tids = sorted(t for t in df["topic_id"].unique() if t != -1)
    order_theme = sorted(tids, key=lambda t: (list(THEME_MAP).index(TOPIC_TO_THEME[t]), t))
    cent = []
    for t in order_theme:
        c = emb[df["topic_id"].values == t].mean(0)
        cent.append(c / (np.linalg.norm(c) + 1e-12))
    cent = np.vstack(cent)
    sim = cent @ cent.T
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(sim, cmap="magma", vmin=sim.min(), vmax=1)
    ax.set_xticks(range(len(order_theme))); ax.set_xticklabels(order_theme, fontsize=6, rotation=90)
    ax.set_yticks(range(len(order_theme))); ax.set_yticklabels(order_theme, fontsize=6)
    ax.set_title("Inter-topic cosine similarity (centroids, grouped by theme)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_fig(fig, "fig02_topic_similarity_heatmap")


def fig_hierarchy(df, emb):
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    tids = sorted(t for t in df["topic_id"].unique() if t != -1)
    cent = []
    for t in tids:
        c = emb[df["topic_id"].values == t].mean(0)
        cent.append(c / (np.linalg.norm(c) + 1e-12))
    cent = np.vstack(cent)
    D = 1 - cent @ cent.T
    np.fill_diagonal(D, 0); D = (D + D.T) / 2
    Z = linkage(squareform(D, checks=False), method="average")
    labels = [f"{t}: {TOPIC_TO_THEME[t][:18]}" for t in tids]
    fig, ax = plt.subplots(figsize=(10, 12))
    dendrogram(Z, labels=labels, orientation="right", ax=ax, color_threshold=0.5, leaf_font_size=7)
    ax.set_title("Hierarchical topic tree (avg-linkage, cosine on centroids)")
    ax.set_xlabel("Cosine distance")
    save_fig(fig, "fig03_hierarchical_topic_tree")


def fig_theme_distribution(theme_tbl):
    import matplotlib.pyplot as plt
    t = theme_tbl.sort_values("documents", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = [PALETTE[list(THEME_MAP).index(th) % len(PALETTE)] for th in t["theme"]]
    ax.barh(t["theme"], t["documents"], color=colors)
    for i, (d, p) in enumerate(zip(t["documents"], t["percentage"])):
        ax.text(d + max(t["documents"]) * 0.01, i, f"{d:,} ({p}%)", va="center", fontsize=8)
    ax.set_xlabel("Documents"); ax.set_title("Theme distribution (8 higher-level themes)")
    ax.set_xlim(0, max(t["documents"]) * 1.18)
    save_fig(fig, "fig04_theme_distribution")


def fig_theme_by_lifecycle(topic_tbl, df):
    import matplotlib.pyplot as plt
    df = df.copy()
    df["theme"] = df["topic_id"].map(TOPIC_TO_THEME)
    ct = pd.crosstab(df["theme"], df["lifecycle_stage"])
    ct = ct.reindex([t for t in THEME_MAP if t in ct.index])
    ctp = ct.div(ct.sum(1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(ctp))
    for i, col in enumerate(ctp.columns):
        ax.barh(ctp.index, ctp[col], left=bottom, label=col, color=PALETTE[i % len(PALETTE)])
        bottom += ctp[col].values
    ax.set_xlabel("Share of theme's posts (%)")
    ax.set_title("Theme composition by lifecycle stage")
    ax.legend(fontsize=8, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.18))
    save_fig(fig, "fig05_theme_by_lifecycle_stage")


def fig_distress_categories(distress_tbl):
    import matplotlib.pyplot as plt
    t = distress_tbl.sort_values("documents", ascending=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.1, 1]})
    ax1.pie(t["documents"], labels=t["distress_type"], autopct="%1.1f%%",
            colors=PALETTE[:len(t)], textprops={"fontsize": 8}, pctdistance=0.8)
    ax1.set_title("Distribution of distress types (by posts)")
    ax2.barh(t["distress_type"][::-1], t["n_topics"][::-1], color=PALETTE[:len(t)][::-1])
    ax2.set_xlabel("Number of topics"); ax2.set_title("Topics per distress type")
    save_fig(fig, "fig06_distress_taxonomy")


def fig_coherence_ranking(coh, topic_tbl):
    import matplotlib.pyplot as plt
    items = sorted(coh.items(), key=lambda x: x[1], reverse=True)
    tids = [t for t, _ in items]; vals = [v for _, v in items]
    colors = [PALETTE[list(THEME_MAP).index(TOPIC_TO_THEME[t]) % len(PALETTE)] for t in tids]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(range(len(tids)), vals, color=colors)
    ax.axhline(np.mean(vals), ls="--", c="gray", lw=1, label=f"mean c_v={np.mean(vals):.3f}")
    ax.set_xticks(range(len(tids))); ax.set_xticklabels(tids, fontsize=7)
    ax.set_xlabel("Topic ID"); ax.set_ylabel("Per-topic c_v coherence")
    ax.set_title("Per-topic coherence ranking"); ax.legend()
    save_fig(fig, "fig07_topic_coherence_ranking")


def _rgba(hex_color, a=0.5):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


def fig_sankey(topic_tbl):
    import plotly.graph_objects as go
    themes = list(THEME_MAP)
    cats = list(dict.fromkeys(THEME_DISTRESS_CATEGORY.values()))
    t_nodes = [f"T{tid}" for tid in topic_tbl.sort_values("topic_id")["topic_id"]]
    nodes = t_nodes + themes + cats
    idx = {n: i for i, n in enumerate(nodes)}
    src, tgt, val, lc = [], [], [], []
    for _, r in topic_tbl.iterrows():
        ci = list(THEME_MAP).index(r["theme"]) % len(PALETTE)
        src.append(idx[f"T{r['topic_id']}"]); tgt.append(idx[r["theme"]]); val.append(int(r["documents"]))
        lc.append(PALETTE[ci])
    theme_docs = topic_tbl.groupby("theme")["documents"].sum()
    for theme in themes:
        cat = THEME_DISTRESS_CATEGORY[theme]
        src.append(idx[theme]); tgt.append(idx[cat]); val.append(int(theme_docs.get(theme, 0)))
        lc.append(PALETTE[themes.index(theme) % len(PALETTE)])
    node_colors = ["#bbbbbb"] * len(t_nodes) + \
                  [PALETTE[i % len(PALETTE)] for i in range(len(themes))] + \
                  ["#444444"] * len(cats)
    fig = go.Figure(go.Sankey(
        node=dict(label=nodes, color=node_colors, pad=6, thickness=12,
                  line=dict(color="white", width=0.5)),
        link=dict(source=src, target=tgt, value=val, color=[_rgba(c, 0.5) for c in lc])))
    fig.update_layout(title="Topic → Theme → Distress category", font_size=9,
                      width=1200, height=1100)
    fig.write_image(str(DIRS["figures"] / "fig08_topic_theme_sankey.png"), scale=2)
    fig.write_image(str(DIRS["figures"] / "fig08_topic_theme_sankey.pdf"))
    LOGGER.info("figure: fig08_topic_theme_sankey.{png,pdf}")


def fig_before_after():
    import matplotlib.pyplot as plt
    fm = json.load(open(SRC / "final_metrics.json"))
    b, f = fm["baseline_98_same_pipeline"], fm["final_41"]
    metrics = [("Topics", b["n_topics"], f["n_topics"], False),
               ("c_v", b["c_v"], f["c_v"], True),
               ("c_npmi", b["c_npmi"], f["c_npmi"], True),
               ("Diversity", b["topic_diversity"], f["topic_diversity"], True),
               ("Redundant\npairs", b["redundant_pairs"], f["redundant_pairs"], False),
               ("Tiny topics\n(<200)", b["topics_lt_200"], f["topics_lt_200"], False)]
    fig, axes = plt.subplots(1, 6, figsize=(15, 4.4))
    for ax, (name, bv, fv, up) in zip(axes, metrics):
        ax.bar(["98", "41"], [bv, fv], color=["#b0b0b0", "#2c6fbb"], width=0.6)
        ax.set_title(name, fontsize=10, pad=8)
        for i, v in enumerate([bv, fv]):
            ax.text(i, v, f"{v}", ha="center", va="bottom", fontsize=8.5)
        ax.margins(y=0.28)
        ax.tick_params(labelsize=9)
    fig.suptitle("Before (98 topics) vs After (41 topics) — identical metric pipeline",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig09_before_after_refinement")


# ===========================================================================
def write_tables_md(topic_tbl, theme_tbl, distress_tbl, coh):
    total = int(topic_tbl["documents"].sum())
    fm = json.load(open(SRC / "final_metrics.json"))
    b, f = fm["baseline_98_same_pipeline"], fm["final_41"]
    L = ["# Publication Tables — Final 41-Topic Model", ""]

    L += ["## Table 1 — Topic Summary", "",
          "| Topic | Name | Docs | % | Theme | Distress type | Keywords |",
          "|---:|---|---:|---:|---|---|---|"]
    for _, r in topic_tbl.iterrows():
        L.append(f"| {r['topic_id']} | {r['topic_name']} | {r['documents']:,} | {r['percentage']} | "
                 f"{r['theme']} | {r['distress_type']} | {', '.join(str(r['keywords']).split(', ')[:8])} |")

    L += ["", "## Table 2 — Theme Summary", "",
          "| Theme | Topics | Docs | % | Distress category |", "|---|---:|---:|---:|---|"]
    for _, r in theme_tbl.iterrows():
        L.append(f"| {r['theme']} | {r['n_topics']} | {r['documents']:,} | {r['percentage']} | {r['distress_category']} |")

    L += ["", "## Table 3 — Distress Taxonomy", "",
          "| Distress type | Topics | Docs | % | Definition |", "|---|---:|---:|---:|---|"]
    for _, r in distress_tbl.iterrows():
        L.append(f"| {r['distress_type']} | {r['n_topics']} | {r['documents']:,} | {r['percentage']} | {r['definition']} |")

    L += ["", "## Table 5 — Topic Quality Summary", "",
          f"- Topics: {f['n_topics']} | c_v: {f['c_v']} | c_npmi: {f['c_npmi']} | "
          f"diversity: {f['topic_diversity']} | outliers: {f['outlier_rate']} | redundant pairs: {f['redundant_pairs']}",
          f"- Topic sizes — smallest {f['smallest_topic']:,}, largest {f['largest_topic']:,}, "
          f"median {f['median_topic_size']:,}, mean {f['avg_topic_size']:,.0f}; tiny (<200): {f['topics_lt_200']}",
          f"- Most / least coherent topic: {max(coh, key=coh.get)} (c_v={max(coh.values()):.3f}) / "
          f"{min(coh, key=coh.get)} (c_v={min(coh.values()):.3f})", ""]

    L += ["## Table 6 — Before vs After BERTopic Refinement", "",
          "| Metric | Before (98) | After (41) |", "|---|---:|---:|",
          f"| Topics | {b['n_topics']} | {f['n_topics']} |",
          f"| c_v | {b['c_v']} | {f['c_v']} |",
          f"| c_npmi | {b['c_npmi']} | {f['c_npmi']} |",
          f"| Topic diversity | {b['topic_diversity']} | {f['topic_diversity']} |",
          f"| Outlier rate | {b['outlier_rate']} | {f['outlier_rate']} |",
          f"| Redundant pairs (>0.90) | {b['redundant_pairs']} | {f['redundant_pairs']} |",
          f"| Tiny topics (<200) | {b['topics_lt_200']} | {f['topics_lt_200']} |", ""]

    L += ["## Table 7 — Final BERTopic Configuration", ""]
    for k, v in FINAL_CONFIG.items():
        L.append(f"- **{k}**: {v}")
    (DIRS["tables"] / "all_tables.md").write_text("\n".join(L) + "\n")
    LOGGER.info("wrote all_tables.md")


def write_reports(topic_tbl, theme_tbl, distress_tbl, coh):
    total = int(topic_tbl["documents"].sum())
    # theme descriptions report
    L = ["# Higher-Level Themes — Reconstructed from the 41-Topic Model", "",
         "Themes were seeded by agglomerative clustering of topic centroids in the frozen",
         "BGE-m3 embedding space and finalized by reading each topic's keywords and",
         "representative documents. Percentages are share of all assigned posts.", ""]
    for _, r in theme_tbl.iterrows():
        L += [f"## {r['theme']}  —  {r['documents']:,} posts ({r['percentage']}%)",
              f"*Distress category: {r['distress_category']}. Topics: {r['topic_ids']}.*", "",
              THEME_DESC[r["theme"]], ""]
    (DIRS["reports"] / "theme_descriptions.md").write_text("\n".join(L) + "\n")

    # taxonomy report + csvs
    L2 = ["# Distress Taxonomy — Final 41-Topic Model", "",
          "Two-level taxonomy: each topic is assigned a distress *type*; each higher-level",
          "theme rolls up to a distress *category*.", "",
          "## Topic → Distress Type", "", "| Topic | Keywords | Theme | Distress type |",
          "|---:|---|---|---|"]
    kwmap = topic_tbl.set_index("topic_id")
    for tid in sorted(kwmap.index):
        r = kwmap.loc[tid]
        L2.append(f"| {tid} | {', '.join(str(r['keywords']).split(', ')[:6])} | {r['theme']} | {r['distress_type']} |")
    L2 += ["", "## Theme → Distress Category", "", "| Theme | Distress category |", "|---|---|"]
    for theme, cat in THEME_DISTRESS_CATEGORY.items():
        L2.append(f"| {theme} | {cat} |")
    (DIRS["taxonomy"] / "distress_taxonomy.md").write_text("\n".join(L2) + "\n")

    # per-topic coherence csv
    pd.DataFrame([{"topic_id": t, "c_v": c, "theme": TOPIC_TO_THEME.get(t, "")}
                  for t, c in sorted(coh.items(), key=lambda x: -x[1])]).to_csv(
        DIRS["tables"] / "topic_coherence_ranking.csv", index=False)


def write_metadata(topic_tbl, theme_tbl, distress_tbl, coh):
    fm = json.load(open(SRC / "final_metrics.json"))
    total = int(topic_tbl["documents"].sum())
    meta = {
        "model": "BERTopic on frozen BAAI/bge-m3 post embeddings",
        "config": FINAL_CONFIG,
        "n_topics": int(len(topic_tbl)),
        "n_documents_corpus": 70425,
        "n_documents_assigned": total,
        "n_outliers": 70425 - total,
        "n_documents": total,  # backward-compat alias = assigned
        "n_themes": len(THEME_MAP),
        "quality": fm["final_41"],
        "baseline_98_same_pipeline": fm["baseline_98_same_pipeline"],
        "theme_frequencies": {r["theme"]: {"documents": int(r["documents"]), "percentage": r["percentage"]}
                               for _, r in theme_tbl.iterrows()},
        "distress_frequencies": {r["distress_type"]: {"documents": int(r["documents"]), "percentage": r["percentage"]}
                                  for _, r in distress_tbl.iterrows()},
        "mean_per_topic_c_v": round(float(np.mean(list(coh.values()))), 4),
    }
    json.dump(meta, open(DIRS["metadata"] / "bertopic_metadata.json", "w"), indent=2)
    json.dump(FINAL_CONFIG, open(DIRS["metadata"] / "final_config.json", "w"), indent=2)
    # canonical topic->theme->distress map
    topic_tbl[["topic_id", "topic_name", "documents", "percentage", "theme", "distress_type", "keywords"]].to_csv(
        DIRS["metadata"] / "topic_theme_distress_map.csv", index=False)


def copy_canonical_exports():
    """Copy the canonical BERTopic exports into the paper metadata folder."""
    for f in ["topics.csv", "topic_info.csv", "topic_keywords.csv",
              "topic_hierarchy.csv", "representative_docs.csv", "final_metrics.json"]:
        if (SRC / f).exists():
            shutil.copy(SRC / f, DIRS["metadata"] / f)


# ===========================================================================
def main():
    setup()
    LOGGER.info("Loading finalized 41-topic model + frozen metadata (read-only)...")
    df, kw, info, emb = load_all()
    total = int((df["topic_id"] != -1).sum())

    topic_tbl = topic_table(df, kw)
    theme_tbl = theme_table(topic_tbl, total)
    distress_tbl = distress_table(topic_tbl, total)
    LOGGER.info("Themes: %d | topics: %d | assigned posts: %d", len(theme_tbl), len(topic_tbl), total)

    # tables (csv)
    topic_tbl.to_csv(DIRS["tables"] / "table1_topic_summary.csv", index=False)
    theme_tbl.to_csv(DIRS["tables"] / "table2_theme_summary.csv", index=False)
    distress_tbl.to_csv(DIRS["taxonomy"] / "table3_distress_taxonomy.csv", index=False)

    reps = representative_docs(df, emb, topic_tbl)
    reps.to_csv(DIRS["representative_documents"] / "representative_documents.csv", index=False)
    _write_reps_md(reps, topic_tbl)

    LOGGER.info("Computing per-topic coherence...")
    coh = per_topic_coherence(df, emb)

    # figures
    LOGGER.info("Rendering figures...")
    fig_topic_sizes(topic_tbl)
    fig_similarity_heatmap(df, emb, topic_tbl)
    fig_hierarchy(df, emb)
    fig_theme_distribution(theme_tbl)
    fig_theme_by_lifecycle(topic_tbl, df)
    fig_distress_categories(distress_tbl)
    fig_coherence_ranking(coh, topic_tbl)
    fig_sankey(topic_tbl)
    fig_before_after()

    # tables md, reports, metadata
    write_tables_md(topic_tbl, theme_tbl, distress_tbl, coh)
    write_reports(topic_tbl, theme_tbl, distress_tbl, coh)
    write_metadata(topic_tbl, theme_tbl, distress_tbl, coh)
    copy_canonical_exports()
    LOGGER.info("Paper assets complete -> %s", OUT)


def _write_reps_md(reps, topic_tbl):
    kwmap = topic_tbl.set_index("topic_id")
    L = ["# Representative Documents — Final 41-Topic Model", "",
         "Usernames, links, emails and phone numbers removed. `confidence` = cosine",
         "similarity to the topic centroid in the frozen BGE-m3 space.", ""]
    for tid in sorted(reps["topic_id"].unique()):
        r = kwmap.loc[tid]
        L += [f"## Topic {tid} — {r['theme']} / {r['distress_type']}",
              f"*Keywords: {', '.join(str(r['keywords']).split(', ')[:8])}*", ""]
        for _, x in reps[reps["topic_id"] == tid].iterrows():
            L.append(f"- ({x['confidence']:.2f}, {x['subreddit']}/{x['lifecycle_stage']}) “{x['representative_document']}”")
        L.append("")
    (DIRS["representative_documents"] / "representative_documents.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
