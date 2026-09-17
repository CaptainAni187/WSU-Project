#!/usr/bin/env python3
"""
Materialize the 7 CORE publication tables as standalone, consistently named CSVs in
final_outputs/paper_assets/tables/ (table01..table07), so every table referenced in the paper
exists as its own CSV alongside the extras (table08..table17).

Reads only canonical finalized-model outputs; reruns nothing. Idempotent.

Run:  python src/pipeline/paper_tables.py
"""
import json
from pathlib import Path
import pandas as pd

OUT = Path("final_outputs/paper_assets")
T = OUT / "tables"


def main():
    T.mkdir(parents=True, exist_ok=True)

    # Table 1 — Topic Summary (already exists as table1_topic_summary.csv -> pad name)
    topic = pd.read_csv(T / "table1_topic_summary.csv")
    topic.to_csv(T / "table01_topic_summary.csv", index=False)

    # Table 2 — Theme Summary
    theme = pd.read_csv(T / "table2_theme_summary.csv")
    theme.to_csv(T / "table02_theme_summary.csv", index=False)

    # Table 3 — Distress Taxonomy (lives in taxonomy/)
    dist = pd.read_csv(OUT / "taxonomy" / "table3_distress_taxonomy.csv")
    dist.to_csv(T / "table03_distress_taxonomy.csv", index=False)

    # Table 4 — Representative Documents (lives in representative_documents/)
    reps = pd.read_csv(OUT / "representative_documents" / "representative_documents.csv")
    reps.to_csv(T / "table04_representative_documents.csv", index=False)

    # Table 5 — Topic Quality Summary (aggregate + per-topic coherence extremes)
    fm = json.load(open(OUT / "metadata" / "final_metrics.json"))["final_41"]
    cn = json.load(open(OUT / "metadata" / "computed_numbers.json"))
    q = [
        ("Topics", fm["n_topics"]),
        ("Assigned posts", cn["n_assigned"]),
        ("Outliers", cn["n_outliers"]),
        ("Outlier rate", fm["outlier_rate"]),
        ("c_v (aggregate)", fm["c_v"]),
        ("c_npmi (aggregate)", fm["c_npmi"]),
        ("Mean per-topic c_v", cn["mean_per_topic_cv"]),
        ("Topic diversity", fm["topic_diversity"]),
        ("Mean centroid confidence", cn["mean_confidence"]),
        ("Redundant pairs (>0.90)", fm["redundant_pairs"]),
        ("Smallest topic", fm["smallest_topic"]),
        ("Largest topic", fm["largest_topic"]),
        ("Median topic size", fm["median_topic_size"]),
        ("Mean topic size", fm["avg_topic_size"]),
        ("Tiny topics (<200)", fm["topics_lt_200"]),
        ("Most coherent topic (c_v)", f"T{cn['most_coherent_topic']} ({cn['most_coherent_cv']})"),
        ("Least coherent topic (c_v)", f"T{cn['least_coherent_topic']} ({cn['least_coherent_cv']})"),
        ("Top-10 topic coverage", f"{cn['top10_coverage_pct']}%"),
    ]
    pd.DataFrame(q, columns=["metric", "value"]).to_csv(T / "table05_topic_quality_summary.csv", index=False)

    # Table 6 — Before vs After BERTopic Refinement (identical pipeline)
    fmj = json.load(open(OUT / "metadata" / "final_metrics.json"))
    b, f = fmj["baseline_98_same_pipeline"], fmj["final_41"]
    rows = [
        ("Topics", b["n_topics"], f["n_topics"]),
        ("c_v", b["c_v"], f["c_v"]),
        ("c_npmi", b["c_npmi"], f["c_npmi"]),
        ("Topic diversity", b["topic_diversity"], f["topic_diversity"]),
        ("Outlier rate", b["outlier_rate"], f["outlier_rate"]),
        ("Redundant pairs (>0.90)", b["redundant_pairs"], f["redundant_pairs"]),
        ("Tiny topics (<200)", b["topics_lt_200"], f["topics_lt_200"]),
        ("Smallest topic", b["smallest_topic"], f["smallest_topic"]),
        ("Largest topic", b["largest_topic"], f["largest_topic"]),
    ]
    pd.DataFrame(rows, columns=["metric", "before_98", "after_41"]).to_csv(
        T / "table06_before_vs_after.csv", index=False)

    # Table 7 — Final BERTopic Configuration
    cfg = json.load(open(OUT / "metadata" / "final_config.json"))
    flat = []
    for k, v in cfg.items():
        flat.append((k, json.dumps(v) if isinstance(v, dict) else str(v)))
    pd.DataFrame(flat, columns=["parameter", "value"]).to_csv(
        T / "table07_final_configuration.csv", index=False)

    # remove the old unpadded duplicates now superseded by table01/02
    for old in ["table1_topic_summary.csv", "table2_theme_summary.csv"]:
        p = T / old
        if p.exists():
            p.unlink()

    print("Core tables written: table01..table07 (padded). Full set:")
    for f in sorted(T.glob("table*.csv")):
        print("  ", f.name)


if __name__ == "__main__":
    main()
