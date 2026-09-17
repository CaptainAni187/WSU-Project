import argparse
import itertools
import json
import logging
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
FINAL_DIR = Path("final_outputs/topic_modeling")
REFINED_DIR = FINAL_DIR / "bertopic_refinement"

ENGLISH_EXTRA_STOPWORDS = {
    "im", "i'm", "ive", "i've", "dont", "don't", "didnt", "didn't", "cant", "can't",
    "wont", "won't", "isnt", "isn't", "thats", "that's", "youre", "you're", "theyre",
    "they're", "weve", "we've", "would", "could", "should", "also", "just", "really",
    "like", "get", "got", "getting", "know", "think", "feel", "want", "need", "people",
    "person", "student", "students", "year", "years", "day", "days", "month", "months",
    "time", "thing", "things", "good", "bad", "nice", "help", "pls", "please", "thanks",
}

HINGLISH_STOPWORDS = {
    "hai", "ki", "ke", "ka", "se", "aur", "bhi", "mai", "main", "nahi", "nhi", "tha",
    "thi", "the", "ho", "hua", "hui", "kar", "kya", "ye", "woh", "wo", "is", "us",
    "apna", "apni", "apne", "mera", "meri", "mere", "tera", "teri", "tere", "ham",
    "hum", "tum", "tumhara", "hu", "haan", "ya", "koi", "bhai", "bro",
    "toh", "kuch", "hi", "mein", "ko", "mujhe", "ek", "ab", "h", "par", "to", "do",
}

ARTIFACT_STOPWORDS = {
    "deleted", "removed", "amp", "title", "post", "comment", "reddit", "subreddit",
    "removed need", "deleted title", "removed help", "removed post", "deleted post",
    "24tard", "25tard", "26tard", "educational_info", "educational", "info",
}

PHRASE_MAP = {
    r"\bkya\s+karu\b": "kya_karu",
    r"\bnahi\s+ho\s+raha\b": "nahi_ho_raha",
    r"\bnhi\s+ho\s+rha\b": "nhi_ho_rha",
    r"\bchud\s+gaye\s+guru\b": "chud_gaye_guru",
    r"\bchud\s+gye\s+guru\b": "chud_gaye_guru",
    r"\bcanon\s+event\b": "canon_event",
    r"\blife\s+barbaad\b": "life_barbaad",
    r"\bdrop\s+year\s+barbaad\b": "drop_year_barbaad",
    r"\bdrop\s+year\b": "drop_year",
    r"\bplacement\s+anxiety\b": "placement_anxiety",
    r"\bcareer\s+uncertainty\b": "career_uncertainty",
    r"\bfamily\s+pressure\b": "family_pressure",
    r"\bmental\s+breakdown\b": "mental_breakdown",
    r"\bdouble\s+dropper\b": "double_dropper",
    r"\bthak\s+gaya\b": "thak_gaya",
    r"\blag\s+gaye\b": "lag_gaye",
    r"\blag\s+gye\b": "lag_gaye",
}

IMPLICIT_MARKERS = [
    "cooked",
    "canon_event",
    "kya_karu",
    "nahi_ho_raha",
    "nhi_ho_rha",
    "thak_gaya",
    "lag_gaye",
    "chud_gaye_guru",
    "life_barbaad",
    "drop_year_barbaad",
]

THEME_HIERARCHY = {
    "Academic Distress": ["Competitive Exam Pressure", "Academic Failure", "Burnout"],
    "Career Distress": ["Placement Anxiety", "Career Uncertainty"],
    "Social Distress": ["Family Expectations", "Social Isolation", "Identity Crisis"],
    "Institutional Distress": ["Coaching Frustration", "NTA Frustration", "College Administration Issues"],
    "Cultural Expression": ["Meme Distress", "Coping Humor", "Community Coping Rituals"],
}

THEME_PATTERNS = {
    "Competitive Exam Pressure": r"\b(jee|neet|mains|advanced|rank|percentile|mock|test_series|exam)\b",
    "Academic Failure": r"\b(fail|failed|failure|backlog|cgpa|marks|paper|semester|drop_year)\b",
    "Burnout": r"\b(burnout|thak_gaya|tired|exhausted|nahi_ho_raha|nhi_ho_rha|not_able)\b",
    "Placement Anxiety": r"\b(placement|placements|unplaced|off_campus|package|internship|interview)\b",
    "Career Uncertainty": r"\b(career|future|job|jobs|gate|mba|mtech|career_uncertainty)\b",
    "Family Expectations": r"\b(parent|parents|family|ghar|papa|mummy|family_pressure|forced)\b",
    "Social Isolation": r"\b(lonely|alone|no_friends|isolated|hostel|social)\b",
    "Identity Crisis": r"\b(identity|worth|purpose|life_barbaad|career_khatam|kya_karu)\b",
    "Coaching Frustration": r"\b(coaching|allen|aakash|pw|fiitjee|teacher|faculty)\b",
    "NTA Frustration": r"\b(nta|normalization|rank_inflation|paper_leak|scam)\b",
    "College Administration Issues": r"\b(attendance|admin|fees|assignment|lab|faculty|college)\b",
    "Meme Distress": r"\b(cooked|canon_event|joever|chud_gaye_guru|lag_gaye|barbaad|khatam)\b|[💀😭🤡☠🫠]",
    "Coping Humor": r"\b(hopium|copium|meme|joke|funny|lol|lmao)\b",
    "Community Coping Rituals": r"\b(alecc|daddy|ritual|daily|challenge|thread)\b",
}


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Controlled BERTopic refinement protocol.")
    parser.add_argument("--phase", choices=["audit", "prepare", "grid", "quality", "quotes", "markers", "all-light"], default="audit")
    parser.add_argument("--text", default="data/processed/posts_merged_final_preprocessed.csv")
    parser.add_argument("--embeddings", default="data/processed/muril_embeddings.npy")
    parser.add_argument("--topic-info", default="final_outputs/topic_modeling/bertopic_topic_info.csv")
    parser.add_argument("--topics", default="final_outputs/topic_modeling/bertopic_topics.csv")
    parser.add_argument("--representatives", default="final_outputs/topic_modeling/bertopic_representatives.csv")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_text(path):
    df = pd.read_csv(path, low_memory=False)
    if "combined_text" not in df.columns:
        if "phrase_text" in df.columns:
            df["combined_text"] = df["phrase_text"].fillna("").astype(str)
        elif "cleaned_text" in df.columns:
            df["combined_text"] = df["cleaned_text"].fillna("").astype(str)
        elif "full_text" in df.columns:
            df["combined_text"] = df["full_text"].fillna("").astype(str)
        if {"title", "selftext"}.issubset(df.columns):
            empty = df["combined_text"].fillna("").astype(str).str.strip().eq("")
            df.loc[empty, "combined_text"] = (
                df.loc[empty, "title"].fillna("").astype(str) + " " + df.loc[empty, "selftext"].fillna("").astype(str)
            ).str.strip()
        if "combined_text" not in df.columns:
            raise ValueError(f"{path} must contain combined_text or title/selftext columns.")
    for col in ["id", "subreddit", "lifecycle_stage", "timestamp"]:
        if col not in df.columns:
            df[col] = ""
    return df


def normalize_for_topics(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\[.*?\]\(.*?\)", " ", text)
    text = re.sub(r"\b(removed need|deleted title|removed help|removed post|deleted post)\b", " ", text)
    for pattern, replacement in PHRASE_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.I)
    text = re.sub(r"[^a-zA-Z0-9_\s\u0900-\u097F💀😭🤡☠🫠]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = []
    stopwords = ENGLISH_EXTRA_STOPWORDS | HINGLISH_STOPWORDS | ARTIFACT_STOPWORDS
    for token in text.split():
        if token in stopwords:
            continue
        if len(token) <= 1:
            continue
        tokens.append(token)
    return " ".join(tokens)


def vocabulary_audit(text_path, limit=None):
    REFINED_DIR.mkdir(parents=True, exist_ok=True)
    df = load_text(text_path)
    if limit:
        df = df.head(limit).copy()
    df["topic_text"] = df["combined_text"].fillna("").map(normalize_for_topics)
    tokens = Counter()
    raw_tokens = Counter()
    for raw, cleaned in zip(df["combined_text"], df["topic_text"]):
        raw_tokens.update(str(raw).lower().split())
        tokens.update(cleaned.split())

    stopword_hits = {word: raw_tokens[word] for word in sorted(HINGLISH_STOPWORDS | ARTIFACT_STOPWORDS) if raw_tokens[word]}
    phrase_hits = {replacement: int(df["topic_text"].str.contains(re.escape(replacement), regex=True).sum()) for replacement in PHRASE_MAP.values()}

    pd.DataFrame(tokens.most_common(500), columns=["token", "frequency"]).to_csv(REFINED_DIR / "vocabulary_top_tokens.csv", index=False)
    pd.DataFrame(stopword_hits.items(), columns=["stopword_or_artifact", "raw_frequency"]).to_csv(
        REFINED_DIR / "stopword_artifact_audit.csv", index=False
    )
    pd.DataFrame(phrase_hits.items(), columns=["preserved_phrase", "post_frequency"]).to_csv(
        REFINED_DIR / "preserved_phrase_audit.csv", index=False
    )
    df[["id", "subreddit", "lifecycle_stage", "timestamp", "combined_text", "topic_text"]].to_csv(
        REFINED_DIR / "topic_model_input.csv", index=False
    )

    lines = [
        "# BERTopic Preprocessing Audit",
        "",
        f"- Rows audited: {len(df):,}",
        f"- Unique cleaned tokens: {len(tokens):,}",
        f"- Output text column for BERTopic: `topic_text`",
        f"- Empty topic_text rows: {int(df['topic_text'].str.strip().eq('').sum()):,}",
        "",
        "## Top Preserved Phrases",
        "",
    ]
    for phrase, count in sorted(phrase_hits.items(), key=lambda item: item[1], reverse=True)[:20]:
        lines.append(f"- {phrase}: {count:,}")
    lines.extend(["", "## Most Frequent Raw Stopword/Artifact Hits", ""])
    for word, count in sorted(stopword_hits.items(), key=lambda item: item[1], reverse=True)[:30]:
        lines.append(f"- {word}: {count:,}")
    (REFINED_DIR / "preprocessing_audit.md").write_text("\n".join(lines) + "\n")
    return df


def grid_search_config():
    umap_neighbors = [10, 15, 25, 50]
    min_dist = [0.0, 0.1, 0.3]
    n_components = [5, 10]
    min_cluster_size = [30, 50, 100, 150]
    min_samples = [5, 10, 15]
    rows = []
    for idx, params in enumerate(itertools.product(umap_neighbors, min_dist, n_components, min_cluster_size, min_samples), start=1):
        rows.append(
            {
                "run_id": idx,
                "umap_n_neighbors": params[0],
                "umap_min_dist": params[1],
                "umap_n_components": params[2],
                "hdbscan_min_cluster_size": params[3],
                "hdbscan_min_samples": params[4],
                "status": "PENDING",
            }
        )
    REFINED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(REFINED_DIR / "bertopic_grid_config.csv", index=False)
    (REFINED_DIR / "grid_search_protocol.md").write_text(
        "# BERTopic Grid Search Protocol\n\n"
        "- Reject runs with outlier rate > 40%.\n"
        "- Reject stopword-dominated or artifact-dominated major topics.\n"
        "- Select for interpretability, Hinglish phrase preservation, and cultural relevance before coherence.\n"
        "- Do not merge Career Uncertainty and Placement Anxiety.\n"
    )


def topic_quality_review(topic_info_path, topics_path):
    topic_info = pd.read_csv(topic_info_path)
    topics = pd.read_csv(topics_path)
    topic_col = "Topic" if "Topic" in topic_info.columns else "topic_id"
    count_col = "Count" if "Count" in topic_info.columns else "topic_size"
    outlier_rate = float((topics["topic_id"] == -1).mean()) if "topic_id" in topics.columns else 0.0
    rows = []
    for _, row in topic_info.iterrows():
        topic_id = int(row[topic_col])
        if topic_id == -1:
            continue
        top_terms = str(row.get("top_words", row.get("Name", "")))
        terms = re.split(r"[;_]", top_terms)
        stopword_count = sum(t.strip().lower() in (HINGLISH_STOPWORDS | ARTIFACT_STOPWORDS) for t in terms)
        recommendation = "KEEP"
        reason = "passes automatic checks"
        if stopword_count >= 3:
            recommendation = "REMOVE"
            reason = "stopword/artifact dominated"
        elif int(row[count_col]) < 30:
            recommendation = "MERGE"
            reason = "small topic; inspect for merge"
        rows.append(
            {
                "topic_id": topic_id,
                "topic_size": int(row[count_col]),
                "top_terms": top_terms,
                "assigned_theme": infer_theme(top_terms),
                "confidence_score": confidence_score(top_terms),
                "recommendation": recommendation,
                "reason": reason,
            }
        )
    quality = pd.DataFrame(rows)
    quality.to_csv(REFINED_DIR / "topic_label_validation.csv", index=False)
    lines = [
        "# Topic Quality Review",
        "",
        f"- Topic count: {len(quality):,}",
        f"- Outlier percentage: {outlier_rate:.2%}",
        f"- Accepted by automatic checks: {int((quality['recommendation'] == 'KEEP').sum()):,}",
        f"- Merge candidates: {int((quality['recommendation'] == 'MERGE').sum()):,}",
        f"- Remove candidates: {int((quality['recommendation'] == 'REMOVE').sum()):,}",
        "",
        "| Topic | Size | Theme | Confidence | Recommendation | Reason |",
        "|---|---:|---|---:|---|---|",
    ]
    for _, row in quality.iterrows():
        lines.append(
            f"| {row['topic_id']} | {row['topic_size']} | {row['assigned_theme']} | {row['confidence_score']} | {row['recommendation']} | {row['reason']} |"
        )
    (REFINED_DIR / "topic_quality_review.md").write_text("\n".join(lines) + "\n")


def infer_theme(text):
    scores = {theme: len(re.findall(pattern, str(text), flags=re.I)) for theme, pattern in THEME_PATTERNS.items()}
    theme, score = max(scores.items(), key=lambda item: item[1])
    return theme if score else "Needs Manual Review"


def confidence_score(text):
    scores = [len(re.findall(pattern, str(text), flags=re.I)) for pattern in THEME_PATTERNS.values()]
    best = max(scores) if scores else 0
    return round(min(1.0, best / 4), 2)


def implicit_marker_analysis(text_path, topics_path=None):
    df = load_text(text_path)
    df["topic_text"] = df["combined_text"].fillna("").map(normalize_for_topics)
    if topics_path and Path(topics_path).exists():
        topics = pd.read_csv(topics_path)
        if "id" in topics.columns and "topic_id" in topics.columns:
            df = df.merge(topics[["id", "topic_id"]], on="id", how="left")
    if "topic_id" not in df.columns:
        df["topic_id"] = ""
    rows = []
    for marker in IMPLICIT_MARKERS:
        mask = df["topic_text"].str.contains(re.escape(marker), regex=True, na=False)
        subset = df[mask]
        rows.append(
            {
                "marker": marker,
                "frequency": int(mask.sum()),
                "subreddit_distribution": json.dumps(subset["subreddit"].value_counts().to_dict(), ensure_ascii=False),
                "lifecycle_distribution": json.dumps(subset["lifecycle_stage"].value_counts().to_dict(), ensure_ascii=False),
                "associated_topics": json.dumps(subset["topic_id"].value_counts().head(10).to_dict(), ensure_ascii=False),
                "representative_quotes": " || ".join(anonymize_quote(q) for q in subset["combined_text"].head(5)),
            }
        )
    marker_df = pd.DataFrame(rows).sort_values("frequency", ascending=False)
    marker_df.to_csv(REFINED_DIR / "implicit_distress_marker_analysis.csv", index=False)
    plot_marker_frequency(marker_df)
    write_marker_report(marker_df)


def anonymize_quote(text):
    text = re.sub(r"u/[A-Za-z0-9_-]+", "[user]", str(text))
    text = re.sub(r"https?://\S+", "[link]", text)
    return re.sub(r"\s+", " ", text).strip()[:900]


def plot_marker_frequency(marker_df):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(marker_df["marker"], marker_df["frequency"])
    ax.set_title("Implicit Distress Marker Frequency")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(REFINED_DIR / "implicit_marker_frequency.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_marker_report(marker_df):
    lines = ["# Implicit Distress Marker Analysis", ""]
    for _, row in marker_df.iterrows():
        lines.append(f"- {row['marker']}: {row['frequency']} posts")
    (REFINED_DIR / "implicit_marker_analysis.md").write_text("\n".join(lines) + "\n")


def quote_repository(representatives_path, topics_path=None):
    reps = pd.read_csv(representatives_path)
    text_col = "quote" if "quote" in reps.columns else "text"
    rows = []
    for topic_id, group in reps.groupby("topic_id"):
        for _, row in group.head(20).iterrows():
            quote = anonymize_quote(row[text_col])
            rows.append(
                {
                    "topic_id": topic_id,
                    "theme": infer_theme(quote),
                    "subreddit": row.get("subreddit", ""),
                    "lifecycle_stage": row.get("lifecycle_stage", ""),
                    "quote": quote,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(REFINED_DIR / "representative_quote_repository.csv", index=False)
    lines = ["# Representative Quote Repository", ""]
    for theme, group in out.groupby("theme"):
        lines.append(f"## {theme}")
        for quote in group["quote"].head(10):
            lines.append(f"- “{quote}”")
        lines.append("")
    (REFINED_DIR / "representative_quote_repository.md").write_text("\n".join(lines) + "\n")


def hierarchy_template():
    rows = []
    for level_1, children in THEME_HIERARCHY.items():
        for level_2 in children:
            rows.append({"level_1": level_1, "level_2": level_2, "requires_supporting_posts": True})
    pd.DataFrame(rows).to_csv(REFINED_DIR / "theme_hierarchy_template.csv", index=False)


def main():
    setup_logging()
    args = parse_args()
    REFINED_DIR.mkdir(parents=True, exist_ok=True)
    if args.phase in {"audit", "all-light"}:
        vocabulary_audit(args.text, args.limit)
        grid_search_config()
        hierarchy_template()
    if args.phase in {"prepare", "all-light"}:
        vocabulary_audit(args.text, args.limit)
    if args.phase == "grid":
        grid_search_config()
    if args.phase in {"quality", "all-light"} and Path(args.topic_info).exists() and Path(args.topics).exists():
        topic_quality_review(args.topic_info, args.topics)
    if args.phase in {"markers", "all-light"}:
        implicit_marker_analysis(args.text, args.topics)
    if args.phase in {"quotes", "all-light"} and Path(args.representatives).exists():
        quote_repository(args.representatives, args.topics)
    LOGGER.info("BERTopic refinement phase `%s` complete", args.phase)


if __name__ == "__main__":
    main()
