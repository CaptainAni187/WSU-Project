#!/usr/bin/env python3
"""
Cross-corpus analysis: link community RESPONSES (comments) to the DISTRESS (posts)
they answer. Produces the dual-corpus findings for the paper.

Inputs : final_outputs/comments_analysis/comments_taxonomy.csv  (comment-level primary_category,
         parent distress_type, reddit upvote `score`)
         data/processed/posts_merged_final.csv (post count, for engagement asymmetry — optional)
Outputs: final_outputs/comments_analysis/cross_corpus_{engagement,response_mix,endorsement}.csv
         final_outputs/comments_analysis/cross_corpus_summary.md
"""
import json, os
import numpy as np, pandas as pd

def to_md(df):
    df = df.reset_index()
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(out)

CDIR = "final_outputs/comments_analysis"
df = pd.read_csv(f"{CDIR}/comments_taxonomy.csv",
                 usecols=["post_id", "distress_type", "score", "primary_category"])

SUPPORTIVE = ["support_empathy", "encouragement_motivation", "personal_experience",
              "advice_guidance", "information_resources"]
HARMFUL = ["toxic_abusive", "dismissive_minimizing", "blame_criticism", "crisis_escalation"]
# narrow "emotional support" (excludes instrumental advice/info) for a conservative ratio
EMO_SUPPORT = ["support_empathy", "encouragement_motivation"]

order = ["suicidal", "depression_anxiety", "academic_burnout", "exam_stress",
         "career_anxiety", "institutional_anger", "meme_distress"]

# ── 1. Engagement asymmetry: how many comments (and per post) each distress attracts ──
eng = (df.groupby("distress_type")
         .agg(comments=("post_id", "size"), posts=("post_id", "nunique"))
         .reindex(order))
eng["comments_per_post"] = (eng["comments"] / eng["posts"]).round(1)
eng["pct_of_comments"] = (100 * eng["comments"] / eng["comments"].sum()).round(1)
eng.to_csv(f"{CDIR}/cross_corpus_engagement.csv")

# ── 2. Response-mix per distress (row-normalised %) ──
mix = pd.crosstab(df["distress_type"], df["primary_category"], normalize="index").mul(100).round(1)
mix = mix.reindex(order)
mix.to_csv(f"{CDIR}/cross_corpus_response_mix.csv")

# ── 3. Support / harm ratios per distress ──
ratio = pd.DataFrame(index=order)
cnt = pd.crosstab(df["distress_type"], df["primary_category"]).reindex(order).fillna(0)
ratio["emo_support_%"] = (100 * cnt[EMO_SUPPORT].sum(1) / cnt.sum(1)).round(1)
ratio["harm_%"] = (100 * cnt[HARMFUL].sum(1) / cnt.sum(1)).round(1)
ratio["humor_%"] = (100 * cnt["humor_meme_coping"] / cnt.sum(1)).round(1)
ratio["emo_support_to_harm"] = (cnt[EMO_SUPPORT].sum(1) / cnt[HARMFUL].sum(1)).round(2)
overall_ratio = cnt[EMO_SUPPORT].sum().sum() / cnt[HARMFUL].sum().sum()
ratio.to_csv(f"{CDIR}/cross_corpus_support_harm.csv")

# ── 4. Community ENDORSEMENT: mean upvotes per response type (overall + on suicidal posts) ──
end_all = df.groupby("primary_category")["score"].mean().round(2).sort_values(ascending=False)
sui = df[df["distress_type"] == "suicidal"]
end_sui = sui.groupby("primary_category")["score"].mean().round(2).sort_values(ascending=False)
endorse = pd.DataFrame({"mean_upvotes_all": end_all, "mean_upvotes_on_suicidal": end_sui})
endorse.to_csv(f"{CDIR}/cross_corpus_endorsement.csv")

# ── Markdown summary ──
L = []
L.append("# Cross-Corpus Analysis: Community Response to Distress\n")
L.append(f"- Comments analysed: **{len(df):,}** over **{df['post_id'].nunique():,}** parent posts\n")
L.append(f"- Overall emotional-support-to-harm ratio: **{overall_ratio:.2f} : 1** "
         "(support_empathy+encouragement vs toxic+dismissive+blame+crisis)\n")
L.append("\n## 1. Engagement asymmetry (which distress attracts replies)\n")
L.append(to_md(eng))
L.append("\n\n## 2. Emotional support vs harm, by parent distress\n")
L.append(to_md(ratio))
L.append("\n\n## 3. Community endorsement (mean reddit upvotes by response type)\n")
L.append(to_md(endorse.fillna(0)))
L.append("\n\n## Notes\n")
L.append("- Response labels come from a **keyword/heuristic classifier** (12 categories); "
         "human-validation agreement is modest, so *emotional support is likely under-counted* "
         "(natural empathy phrasing is missed) and *neutral over-counted*. Treat absolute rates "
         "as exploratory; the **relative** patterns across distress types are the robust signal.\n")
with open(f"{CDIR}/cross_corpus_summary.md", "w") as f:
    f.write("".join(L))

print("=== ENGAGEMENT ==="); print(eng)
print("\n=== SUPPORT/HARM ==="); print(ratio)
print(f"\nOverall emo-support:harm = {overall_ratio:.2f}:1")
print("\n=== ENDORSEMENT (mean upvotes) ==="); print(endorse.fillna(0))
print("\nWrote cross_corpus_* to", CDIR)
