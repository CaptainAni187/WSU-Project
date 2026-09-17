#!/usr/bin/env python3
"""
Additional publication figures + tables + a computed-numbers file, all derived
from the finalized 41-topic model.  Nothing upstream is rerun; this only reads the
canonical model outputs and frozen metadata (read-only) and writes under
final_outputs/paper_assets/.

Run:  python src/pipeline/paper_extras.py
"""
import json
import logging
import os
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
import paper_assets as pa

LOGGER = logging.getLogger("paper_extras")
DIRS = pa.DIRS
PALETTE = pa.PALETTE
THEME_MAP = pa.THEME_MAP
TOPIC_TO_THEME = pa.TOPIC_TO_THEME
DISTRESS_MAP = pa.DISTRESS_MAP


def save_fig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(DIRS["figures"] / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    LOGGER.info("figure: %s", name)


def theme_color(theme):
    return PALETTE[list(THEME_MAP).index(theme) % len(PALETTE)]


# ---------------------------------------------------------------------------
def load():
    from sklearn.preprocessing import normalize
    topics = pd.read_csv(pa.SRC / "topics.csv")
    meta = pd.read_csv(pa.META_PATH, usecols=["id", "subreddit", "lifecycle_stage", "cleaned_text"])
    df = topics.merge(meta, on="id", how="left")
    df["theme"] = df["topic_id"].map(TOPIC_TO_THEME)
    df["distress_type"] = df["topic_id"].map(DISTRESS_MAP)
    emb = normalize(np.load(pa.EMB_PATH).astype(np.float32), axis=1)
    tbl = pd.read_csv(DIRS["tables"] / "table1_topic_summary.csv")
    return df, emb, tbl


def compute(df, emb, tbl):
    texts = df["cleaned_text"].fillna("").astype(str).tolist()
    labels = df["topic_id"].values
    tids = sorted(t for t in set(labels) if t != -1)

    # per-topic coherence
    from gensim.models import CoherenceModel
    tok, dictionary = bs.build_coherence_reference(texts)
    cv = bs.build_vectorizer(texts)
    tw, ctfidf, vocab = bs.topic_keywords(texts, labels, cv, top_n=10)
    ct_rows = sorted(tw)
    topics_words = [[w for w in tw[t] if w in dictionary.token2id] for t in ct_rows]
    keep = [i for i, t in enumerate(topics_words) if len(t) >= 3]
    cm = CoherenceModel(topics=[topics_words[i] for i in keep], texts=tok, dictionary=dictionary,
                        coherence="c_v", topn=10, processes=1)
    per = cm.get_coherence_per_topic()
    coh = {ct_rows[keep[i]]: round(float(per[i]), 4) for i in range(len(keep))}

    # per-topic centroid, per-doc confidence, redundancy
    cent, conf = {}, {}
    for t in tids:
        idx = np.where(labels == t)[0]
        c = emb[idx].mean(0); c /= (np.linalg.norm(c) + 1e-12)
        cent[t] = c
        conf[t] = emb[idx] @ c
    C = np.vstack([cent[t] for t in tids])
    S = C @ C.T
    iu = np.triu_indices(len(tids), 1)
    pair_sims = [(tids[i], tids[j], float(S[i, j])) for i, j in zip(*iu)]

    # per-topic diversity (unique top-10 fraction across its own words is 1; use theme-level)
    df["confidence"] = np.nan
    for t in tids:
        idx = np.where(labels == t)[0]
        df.loc[df.index[idx], "confidence"] = conf[t]

    return dict(coh=coh, cent=cent, conf=conf, tids=tids, C=C, S=S,
                pair_sims=pair_sims, tw=tw, tok=tok, dictionary=dictionary, cv=cv)


# ===========================================================================
# FIGURES (Part 4)
# ===========================================================================
def fig_cumulative(tbl):
    import matplotlib.pyplot as plt
    t = tbl.sort_values("documents", ascending=False).reset_index(drop=True)
    total = t["documents"].sum()
    cum = np.cumsum(t["documents"]) / total * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, len(t) + 1), cum, marker="o", ms=4, color="#2c6fbb")
    for frac, lbl in [(50, "50%"), (80, "80%")]:
        k = int((cum >= frac).argmax()) + 1
        ax.axhline(frac, ls=":", c="gray", lw=1)
        ax.annotate(f"top {k} topics → {frac}%", (k, frac), fontsize=9,
                    xytext=(k + 2, frac - 8), arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_xlabel("Topic rank (largest → smallest)"); ax.set_ylabel("Cumulative % of assigned posts")
    ax.set_title("Cumulative topic coverage (41 topics)")
    ax.grid(alpha=0.25)
    save_fig(fig, "fig10_cumulative_topic_coverage")


def fig_size_vs_coherence(tbl, coh):
    import matplotlib.pyplot as plt
    t = tbl[tbl["topic_id"].isin(coh)].copy()
    t["cv"] = t["topic_id"].map(coh)
    colors = [theme_color(th) for th in t["theme"]]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(t["documents"], t["cv"], c=colors, s=60, edgecolor="white", linewidth=0.5)
    for _, r in t.iterrows():
        ax.annotate(str(r["topic_id"]), (r["documents"], r["cv"]), fontsize=6.5,
                    xytext=(3, 3), textcoords="offset points")
    r = np.corrcoef(t["documents"], t["cv"])[0, 1]
    ax.set_xlabel("Topic size (documents)"); ax.set_ylabel("Per-topic c_v coherence")
    ax.set_title(f"Topic size vs coherence (Pearson r = {r:.2f})")
    ax.grid(alpha=0.25)
    handles = [plt.Line2D([], [], marker="o", ls="", color=theme_color(th), label=th) for th in THEME_MAP]
    ax.legend(handles=handles, fontsize=7, loc="lower right")
    save_fig(fig, "fig11_topic_size_vs_coherence")
    return round(float(r), 3)


def fig_theme_quality(df, coh, tbl):
    import matplotlib.pyplot as plt
    rows = []
    for theme in THEME_MAP:
        ts = [t for t in THEME_MAP[theme] if t in coh]
        mean_cv = np.mean([coh[t] for t in ts]) if ts else np.nan
        conf = df[df["theme"] == theme]["confidence"].mean()
        rows.append((theme, mean_cv, conf))
    rows.sort(key=lambda x: x[1], reverse=True)
    names = [r[0] for r in rows]; cvs = [r[1] for r in rows]; confs = [r[2] for r in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - 0.2, cvs, 0.4, label="Mean per-topic c_v", color="#2c6fbb")
    ax.bar(x + 0.2, confs, 0.4, label="Mean centroid confidence", color="#e07b39")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Score"); ax.set_title("Theme-level coherence and cluster compactness")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25)
    save_fig(fig, "fig12_theme_coherence_compactness")
    return {n: {"mean_cv": round(float(c), 4), "mean_confidence": round(float(cf), 4)}
            for n, c, cf in rows}


def fig_distress_by_lifecycle(df):
    import matplotlib.pyplot as plt
    order = ["school", "competitive_exam", "undergraduate", "career", "higher_education"]
    dtypes = ["Academic Distress", "Career Distress", "Emotional Distress",
              "Institutional Distress", "Coping Expression", "Informational / Low-Distress"]
    ct = pd.crosstab(df["lifecycle_stage"], df["distress_type"])
    ct = ct.reindex(index=[o for o in order if o in ct.index], columns=[d for d in dtypes if d in ct.columns])
    ctp = ct.div(ct.sum(1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(ctp))
    for i, col in enumerate(ctp.columns):
        ax.bar(ctp.index, ctp[col], bottom=bottom, label=col, color=PALETTE[i % len(PALETTE)])
        bottom += ctp[col].values
    ax.set_ylabel("Share of stage's posts (%)")
    ax.set_title("Distress-type composition across lifecycle stages")
    ax.legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    ax.tick_params(axis="x", rotation=15)
    save_fig(fig, "fig13_distress_by_lifecycle")
    return {stage: {c: round(float(ctp.loc[stage, c]), 2) for c in ctp.columns} for stage in ctp.index}


def fig_subreddit_theme(df):
    import matplotlib.pyplot as plt
    ct = pd.crosstab(df["theme"], df["subreddit"])
    ct = ct.reindex([t for t in THEME_MAP if t in ct.index])
    ctp = ct.div(ct.sum(0), axis=1) * 100  # column-normalized: each subreddit's theme mix
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(ctp.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(ctp.columns))); ax.set_xticklabels(ctp.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(ctp.index))); ax.set_yticklabels(ctp.index, fontsize=8)
    for i in range(ctp.shape[0]):
        for j in range(ctp.shape[1]):
            v = ctp.values[i, j]
            if v >= 10:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 40 else "black")
    ax.set_title("Theme composition by subreddit (% of each subreddit's posts)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="%")
    save_fig(fig, "fig14_subreddit_theme_heatmap")


def fig_confidence(df):
    import matplotlib.pyplot as plt
    themes = [t for t in THEME_MAP if (df["theme"] == t).any()]
    data = [df[df["theme"] == t]["confidence"].dropna().values for t in themes]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bp = ax.boxplot(data, vert=True, patch_artist=True, showfliers=False, widths=0.6)
    for patch, th in zip(bp["boxes"], themes):
        patch.set_facecolor(theme_color(th)); patch.set_alpha(0.75)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_xticklabels(themes, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Per-post centroid confidence (cosine)")
    ax.set_title("Within-topic confidence distribution by theme (cluster tightness)")
    ax.grid(axis="y", alpha=0.25)
    save_fig(fig, "fig15_topic_confidence_distribution")


def fig_redundancy_graph(tids, S, tbl):
    import matplotlib.pyplot as plt
    thr = 0.92   # baseline centroid similarity is ~0.7-0.8 in this single domain;
                 # show only genuine near-duplicates to keep the graph informative
    themes = list(THEME_MAP)
    # circular layout grouped by theme
    ordered = sorted(tids, key=lambda t: (themes.index(TOPIC_TO_THEME[t]), t))
    ang = {t: 2 * np.pi * i / len(ordered) for i, t in enumerate(ordered)}
    pos = {t: (np.cos(ang[t]), np.sin(ang[t])) for t in ordered}
    size_map = tbl.set_index("topic_id")["documents"].to_dict()
    idx = {t: i for i, t in enumerate(tids)}
    fig, ax = plt.subplots(figsize=(9, 9))
    n_edges = 0
    for a in range(len(tids)):
        for b in range(a + 1, len(tids)):
            s = S[a, b]
            if s > thr:
                ta, tb = tids[a], tids[b]
                x = [pos[ta][0], pos[tb][0]]; y = [pos[ta][1], pos[tb][1]]
                ax.plot(x, y, color="#c0392b", alpha=min(1, (s - thr) / (1 - thr) * 0.7 + 0.3), lw=1.4)
                n_edges += 1
    for t in ordered:
        c = theme_color(TOPIC_TO_THEME[t])
        ax.scatter(*pos[t], s=40 + size_map.get(t, 0) / 30, color=c, edgecolor="white", zorder=3)
        ax.annotate(str(t), pos[t], fontsize=7, ha="center", va="center", zorder=4)
    ax.set_title(f"Topic redundancy graph (edges: centroid cos > {thr}; {n_edges} edges)")
    ax.axis("off")
    handles = [plt.Line2D([], [], marker="o", ls="", color=theme_color(th), label=th) for th in THEME_MAP]
    ax.legend(handles=handles, fontsize=7, loc="lower left", bbox_to_anchor=(-0.05, -0.05))
    save_fig(fig, "fig16_topic_redundancy_graph")
    return n_edges


# ===========================================================================
# TABLES (Part 5)
# ===========================================================================
def tables(tbl, coh, df, S, tids, pair_sims, theme_q):
    T = DIRS["tables"]
    t = tbl.copy()
    t["c_v"] = t["topic_id"].map(coh)
    t["mean_confidence"] = t["topic_id"].map(
        lambda x: round(float(df[df["topic_id"] == x]["confidence"].mean()), 4))
    t["most_distinctive_keywords"] = t["keywords"].map(lambda k: ", ".join(str(k).split(", ")[:3]))

    t.sort_values("documents", ascending=False).head(10).to_csv(T / "table08_top10_largest_topics.csv", index=False)
    tc = t.dropna(subset=["c_v"])
    tc.sort_values("c_v", ascending=False).head(10).to_csv(T / "table09_top10_highest_coherence.csv", index=False)
    tc.sort_values("c_v").head(10).to_csv(T / "table10_lowest_coherence_topics.csv", index=False)

    # dataset statistics
    ds = pd.DataFrame([
        ["Posts (corpus)", 70425], ["Posts assigned to a topic", int((df["topic_id"] != -1).sum())],
        ["Outliers (topic -1)", int((df["topic_id"] == -1).sum())],
        ["Topics", len(tbl)], ["Themes", len(THEME_MAP)], ["Distress types", 6],
        ["Subreddits", df["subreddit"].nunique()], ["Lifecycle stages", df["lifecycle_stage"].nunique()],
        ["Median topic size", int(tbl["documents"].median())],
        ["Largest topic", int(tbl["documents"].max())], ["Smallest topic", int(tbl["documents"].min())],
    ], columns=["statistic", "value"])
    ds.to_csv(T / "table11_dataset_statistics.csv", index=False)

    life = df[df["topic_id"] != -1].groupby("lifecycle_stage").agg(
        posts=("id", "count")).reset_index()
    life["percentage"] = round(life["posts"] / life["posts"].sum() * 100, 2)
    life.sort_values("posts", ascending=False).to_csv(T / "table12_lifecycle_statistics.csv", index=False)

    sub = df[df["topic_id"] != -1].groupby("subreddit").agg(posts=("id", "count")).reset_index()
    sub["percentage"] = round(sub["posts"] / sub["posts"].sum() * 100, 2)
    sub.sort_values("posts", ascending=False).to_csv(T / "table13_subreddit_statistics.csv", index=False)

    # theme quality
    tq = []
    for theme in THEME_MAP:
        sub_t = t[t["theme"] == theme]
        tq.append({
            "theme": theme, "n_topics": len(sub_t), "documents": int(sub_t["documents"].sum()),
            "percentage": round(sub_t["documents"].sum() / (df["topic_id"] != -1).sum() * 100, 2),
            "mean_c_v": theme_q[theme]["mean_cv"], "mean_confidence": theme_q[theme]["mean_confidence"],
            "distress_category": pa.THEME_DISTRESS_CATEGORY[theme],
        })
    pd.DataFrame(tq).sort_values("documents", ascending=False).to_csv(T / "table14_theme_quality.csv", index=False)

    # most distinctive keywords (top-1/2 c-TF-IDF per topic)
    t[["topic_id", "theme", "distress_type", "most_distinctive_keywords"]].to_csv(
        T / "table15_most_distinctive_keywords.csv", index=False)

    # topic quality ranking
    t.sort_values("c_v", ascending=False)[
        ["topic_id", "topic_name", "documents", "percentage", "c_v", "mean_confidence", "theme", "distress_type"]
    ].to_csv(T / "table16_topic_quality_ranking.csv", index=False)

    # topic redundancy ranking (highest centroid similarity pairs)
    pr = pd.DataFrame(pair_sims, columns=["topic_a", "topic_b", "centroid_cosine"])
    pr["theme_a"] = pr["topic_a"].map(TOPIC_TO_THEME); pr["theme_b"] = pr["topic_b"].map(TOPIC_TO_THEME)
    pr.sort_values("centroid_cosine", ascending=False).head(20).round(4).to_csv(
        T / "table17_topic_redundancy_top20.csv", index=False)

    _write_extra_md(t, tc, ds, life, sub, tq, pr)
    return dict(life=life, sub=sub, tq=tq, pr=pr)


def _write_extra_md(t, tc, ds, life, sub, tq, pr):
    L = ["# Additional Publication Tables (Extras)", ""]
    L += ["## Table 8 — Top 10 Largest Topics", "", "| Topic | Name | Docs | % | Theme |", "|---:|---|---:|---:|---|"]
    for _, r in t.sort_values("documents", ascending=False).head(10).iterrows():
        L.append(f"| {r['topic_id']} | {r['topic_name']} | {r['documents']:,} | {r['percentage']} | {r['theme']} |")
    L += ["", "## Table 9 — Top 10 Highest-Coherence Topics", "", "| Topic | Name | c_v | Docs | Theme |", "|---:|---|---:|---:|---|"]
    for _, r in tc.sort_values("c_v", ascending=False).head(10).iterrows():
        L.append(f"| {r['topic_id']} | {r['topic_name']} | {r['c_v']} | {r['documents']:,} | {r['theme']} |")
    L += ["", "## Table 10 — Lowest-Coherence Topics", "", "| Topic | Name | c_v | Docs | Theme |", "|---:|---|---:|---:|---|"]
    for _, r in tc.sort_values("c_v").head(10).iterrows():
        L.append(f"| {r['topic_id']} | {r['topic_name']} | {r['c_v']} | {r['documents']:,} | {r['theme']} |")
    L += ["", "## Table 11 — Dataset Statistics", "", "| Statistic | Value |", "|---|---:|"]
    for _, r in ds.iterrows():
        L.append(f"| {r['statistic']} | {r['value']:,} |" if isinstance(r['value'], (int, np.integer)) else f"| {r['statistic']} | {r['value']} |")
    L += ["", "## Table 12 — Lifecycle Statistics", "", "| Stage | Posts | % |", "|---|---:|---:|"]
    for _, r in life.sort_values("posts", ascending=False).iterrows():
        L.append(f"| {r['lifecycle_stage']} | {r['posts']:,} | {r['percentage']} |")
    L += ["", "## Table 13 — Subreddit Statistics", "", "| Subreddit | Posts | % |", "|---|---:|---:|"]
    for _, r in sub.sort_values("posts", ascending=False).iterrows():
        L.append(f"| {r['subreddit']} | {r['posts']:,} | {r['percentage']} |")
    L += ["", "## Table 14 — Theme Quality", "", "| Theme | Topics | Docs | % | Mean c_v | Mean confidence |", "|---|---:|---:|---:|---:|---:|"]
    for r in sorted(tq, key=lambda x: -x["documents"]):
        L.append(f"| {r['theme']} | {r['n_topics']} | {r['documents']:,} | {r['percentage']} | {r['mean_c_v']} | {r['mean_confidence']} |")
    L += ["", "## Table 17 — Most Similar Topic Pairs (residual redundancy)", "", "| Topic A | Topic B | Centroid cosine |", "|---:|---:|---:|"]
    for _, r in pr.sort_values("centroid_cosine", ascending=False).head(12).iterrows():
        L.append(f"| {int(r['topic_a'])} | {int(r['topic_b'])} | {r['centroid_cosine']:.3f} |")
    (DIRS["tables"] / "all_extra_tables.md").write_text("\n".join(L) + "\n")


# ===========================================================================
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    for n in ("gensim", "matplotlib", "PIL", "numba"):
        logging.getLogger(n).setLevel(logging.WARNING)
    LOGGER.info("Loading canonical model + frozen metadata (read-only)...")
    df, emb, tbl = load()
    C = compute(df, emb, tbl)
    coh = C["coh"]; tids = C["tids"]; S = C["S"]

    LOGGER.info("Rendering additional figures...")
    fig_cumulative(tbl)
    r_size_coh = fig_size_vs_coherence(tbl, coh)
    theme_q = fig_theme_quality(df, coh, tbl)
    dist_life = fig_distress_by_lifecycle(df)
    fig_subreddit_theme(df)
    fig_confidence(df)
    n_red_edges = fig_redundancy_graph(tids, S, tbl)

    LOGGER.info("Writing additional tables...")
    aux = tables(tbl, coh, df, S, tids, C["pair_sims"], theme_q)

    # computed numbers for the paper text
    t = tbl.copy(); t["c_v"] = t["topic_id"].map(coh)
    cum = np.cumsum(t.sort_values("documents", ascending=False)["documents"].values) / t["documents"].sum()
    sizes = t["documents"].values
    size_p = sizes / sizes.sum()
    theme_sizes = t.groupby("theme")["documents"].sum().values
    theme_p = theme_sizes / theme_sizes.sum()
    pair = C["pair_sims"]
    nums = {
        "n_corpus": 70425, "n_assigned": int((df["topic_id"] != -1).sum()),
        "n_outliers": int((df["topic_id"] == -1).sum()),
        "top5_coverage_pct": round(float(cum[4] * 100), 1),
        "top10_coverage_pct": round(float(cum[9] * 100), 1),
        "size_min": int(sizes.min()), "size_max": int(sizes.max()),
        "size_median": int(np.median(sizes)), "size_mean": round(float(sizes.mean()), 1),
        "size_cv_pct": round(float(sizes.std() / sizes.mean() * 100), 1),
        "topic_size_entropy_norm": round(float(-(size_p * np.log(size_p)).sum() / np.log(len(size_p))), 4),
        "theme_size_entropy_norm": round(float(-(theme_p * np.log(theme_p)).sum() / np.log(len(theme_p))), 4),
        "most_coherent_topic": int(max(coh, key=coh.get)), "most_coherent_cv": max(coh.values()),
        "least_coherent_topic": int(min(coh, key=coh.get)), "least_coherent_cv": min(coh.values()),
        "mean_per_topic_cv": round(float(np.mean(list(coh.values()))), 4),
        "mean_confidence": round(float(df["confidence"].mean()), 4),
        "size_coherence_pearson_r": r_size_coh,
        "redundancy_graph_threshold": 0.92,
        "redundancy_edges_at_threshold": n_red_edges,
        "pairs_gt_0_90": int(sum(1 for _, _, s in pair if s > 0.90)),
        "pairs_gt_0_80": int(sum(1 for _, _, s in pair if s > 0.80)),
        "max_pair_cosine": round(max(s for _, _, s in pair), 4),
        "theme_quality": theme_q,
        "distress_by_lifecycle": dist_life,
        "lifecycle_counts": {r["lifecycle_stage"]: int(r["posts"]) for _, r in aux["life"].iterrows()},
        "subreddit_counts": {r["subreddit"]: int(r["posts"]) for _, r in aux["sub"].iterrows()},
    }
    json.dump(nums, open(DIRS["metadata"] / "computed_numbers.json", "w"), indent=2)
    LOGGER.info("Extras complete. computed_numbers.json written.")


if __name__ == "__main__":
    main()
