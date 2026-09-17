#!/usr/bin/env python3
"""
Inferential statistics for community response to suicidal-ideation posts.

Input : final_outputs/comments_analysis/comments_taxonomy.csv
        (comment-level primary_category, parent distress_type, post_id, up-vote score)
Output: final_outputs/comments_analysis/suicidal_interaction_stats.json

Reports
  - chi-square test of independence (parent distress x response type) and Cramer's V
  - relative rate of crisis-escalation / support replies on suicidal posts vs. the corpus
  - post-level coverage on suicidal posts (no supportive reply / any harmful / any crisis reply)
  - up-vote endorsement of crisis-escalating vs. supportive replies (one-sided Mann-Whitney U)

Usage (from repo root): python scripts/comments/suicidal_interaction_stats.py
"""
import argparse, json, os
import numpy as np, pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CDIR = os.path.join(REPO, "final_outputs", "comments_analysis")

SUPPORT = ["support_empathy", "encouragement_motivation"]
HARM = ["toxic_abusive", "dismissive_minimizing", "blame_criticism", "crisis_escalation"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(CDIR, "suicidal_interaction_stats.json"))
    args = ap.parse_args()

    d = pd.read_csv(os.path.join(CDIR, "comments_taxonomy.csv"),
                    usecols=["post_id", "distress_type", "score", "primary_category"])

    # association between parent distress and response type
    ct = pd.crosstab(d["distress_type"], d["primary_category"])
    chi2, p, dof, _ = chi2_contingency(ct)
    cramers_v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))

    # relative rates on suicidal posts vs. the whole comment corpus
    base = d["primary_category"].value_counts(normalize=True)
    sui = d[d["distress_type"] == "suicidal"]
    sui_rate = sui["primary_category"].value_counts(normalize=True)

    # post-level coverage on suicidal posts
    per_post = sui.groupby("post_id")["primary_category"].agg(set)
    zero_support = 100 * per_post.apply(lambda s: not (s & set(SUPPORT))).mean()
    any_harm = 100 * per_post.apply(lambda s: bool(s & set(HARM))).mean()
    any_crisis = 100 * per_post.apply(lambda s: "crisis_escalation" in s).mean()

    # community endorsement: crisis-escalating vs supportive replies
    crisis_up = sui.loc[sui["primary_category"] == "crisis_escalation", "score"]
    support_up = sui.loc[sui["primary_category"].isin(SUPPORT), "score"]
    _, mwu_p = mannwhitneyu(crisis_up, support_up, alternative="greater")

    stats = {
        "chi2": int(round(chi2)),
        "dof": int(dof),
        "p": round(float(p), 4),
        "cramers_v": round(float(cramers_v), 3),
        "RR_crisis": round(float(sui_rate["crisis_escalation"] / base["crisis_escalation"]), 2),
        "RR_support": round(float(sui_rate["support_empathy"] / base["support_empathy"]), 2),
        "sui_zero_support_pct": round(float(zero_support), 1),
        "sui_any_harm_pct": round(float(any_harm), 1),
        "sui_any_crisis_pct": round(float(any_crisis), 1),
        "crisis_upvote": round(float(crisis_up.mean()), 2),
        "support_upvote": round(float(support_up.mean()), 2),
        "upvote_mwu_p": round(float(mwu_p), 4),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(stats, f, indent=2)
    for k, v in stats.items():
        print(f"  {k:22s} {v}")
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
