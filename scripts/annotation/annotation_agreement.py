#!/usr/bin/env python3
"""
Agreement between automated labels and human annotation.

Inputs : data/annotations/posts_distress_annotation.csv       (300 posts)
         data/annotations/comments_interaction_annotation.csv (300 comments)
         Each file has `assistant_annotation` (automated) and `human_annotation` columns.
Output : final_outputs/annotation_validation/annotation_summary.csv

Metrics: raw agreement, Cohen's kappa, macro precision / recall / F1
         (human label treated as the reference), plus a seeded bootstrap CI for agreement.

Usage (from repo root): python scripts/annotation/annotation_agreement.py
"""
import argparse, os
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANNOT = os.path.join(REPO, "data", "annotations")
TASKS = [("Posts Distress", "posts_distress_annotation.csv"),
         ("Comments Interaction", "comments_interaction_annotation.csv")]


def bootstrap_ci(agree, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    means = [rng.choice(agree, size=len(agree), replace=True).mean() for _ in range(n_boot)]
    return np.percentile(means, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "final_outputs", "annotation_validation",
                                                  "annotation_summary.csv"))
    args = ap.parse_args()

    rows = []
    for name, fname in TASKS:
        d = pd.read_csv(os.path.join(ANNOT, fname)).dropna(subset=["assistant_annotation", "human_annotation"])
        y_true, y_pred = d["human_annotation"], d["assistant_annotation"]
        agree = (y_true == y_pred).to_numpy(dtype=float)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        lo, hi = bootstrap_ci(agree)
        rows.append({"Dataset": name, "Samples": len(d), "Agreement": agree.mean(),
                     "Cohen_Kappa": cohen_kappa_score(y_true, y_pred),
                     "Macro_Precision": p, "Macro_Recall": r, "Macro_F1": f1})
        print(f"{name:22s} n={len(d)}  agreement={agree.mean():.4f} "
              f"(95% bootstrap CI {lo:.3f}-{hi:.3f})  kappa={rows[-1]['Cohen_Kappa']:.4f}  macroF1={f1:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
