#!/usr/bin/env python3
"""Publication figure for the cross-corpus community-response finding."""
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

CDIR = "final_outputs/comments_analysis"
mix = pd.read_csv(f"{CDIR}/cross_corpus_response_mix.csv", index_col=0)
end = pd.read_csv(f"{CDIR}/cross_corpus_endorsement.csv", index_col=0)

order = ["suicidal","depression_anxiety","academic_burnout","exam_stress",
         "career_anxiety","institutional_anger","meme_distress"]
labels = ["Suicidal","Depression/\nAnxiety","Academic\nBurnout","Exam\nStress",
          "Career\nAnxiety","Institutional\nAnger","Meme\nDistress"]
mix = mix.reindex(order)

# group the 12 response types into 4 readable bands
groups = {
    "Emotional support": ["support_empathy","encouragement_motivation"],
    "Instrumental (advice/info/exp.)": ["advice_guidance","information_resources","personal_experience","self_disclosure"],
    "Neutral": ["neutral_discussion"],
    "Humor / meme coping": ["humor_meme_coping"],
    "Harmful (toxic/dismissive/blame/crisis)": ["toxic_abusive","dismissive_minimizing","blame_criticism","crisis_escalation"],
}
band = pd.DataFrame({g: mix[cols].sum(axis=1) for g, cols in groups.items()}, index=mix.index)
colors = {"Emotional support":"#2E7D32","Instrumental (advice/info/exp.)":"#1565C0",
          "Neutral":"#9E9E9E","Humor / meme coping":"#F9A825",
          "Harmful (toxic/dismissive/blame/crisis)":"#C62828"}

plt.rcParams.update({"font.family":"sans-serif","font.size":10,"savefig.dpi":300,"savefig.bbox":"tight"})
fig = plt.figure(figsize=(12,4.6))
gs = GridSpec(1,2, width_ratios=[1.55,1], wspace=0.28)

# Panel A: stacked response mix by parent distress
axA = fig.add_subplot(gs[0])
bottom = np.zeros(len(order))
for g in groups:
    axA.bar(range(len(order)), band[g].values, bottom=bottom, label=g, color=colors[g], width=0.72)
    bottom += band[g].values
axA.set_xticks(range(len(order))); axA.set_xticklabels(labels, fontsize=8.5)
axA.set_ylabel("Share of comments (%)"); axA.set_ylim(0,100)
axA.set_title("(a) Community response mix by parent-post distress", fontsize=10, loc="left", pad=10)
axA.legend(fontsize=7.4, loc="lower center", bbox_to_anchor=(0.5,-0.32), ncol=2, frameon=False)
for sp in ["top","right"]: axA.spines[sp].set_visible(False)

# Panel B: endorsement on suicidal posts (mean upvotes by response type)
# colour each response by the SAME 5-group scheme as panel (a) for consistency
grp_color = {
    "support_empathy":"#2E7D32","encouragement_motivation":"#2E7D32",           # Emotional support
    "advice_guidance":"#1565C0","information_resources":"#1565C0",
    "personal_experience":"#1565C0","self_disclosure":"#1565C0",                # Instrumental
    "neutral_discussion":"#9E9E9E",                                             # Neutral
    "humor_meme_coping":"#F9A825",                                              # Humor / meme coping
    "toxic_abusive":"#C62828","dismissive_minimizing":"#C62828",
    "blame_criticism":"#C62828","crisis_escalation":"#C62828",                  # Harmful
}
axB = fig.add_subplot(gs[1])
es = end["mean_upvotes_on_suicidal"].dropna().sort_values()
bar_colors = [grp_color.get(k, "#9E9E9E") for k in es.index]
axB.barh(range(len(es)), es.values, color=bar_colors)
axB.set_yticks(range(len(es))); axB.set_yticklabels([k.replace("_"," ") for k in es.index], fontsize=8)
axB.set_xlabel("Mean upvotes on suicidal posts")
axB.set_title("(b) Community endorsement of replies to suicidal posts", fontsize=10, loc="left", pad=10)
axB.axvline(es.median(), ls="--", lw=0.8, color="#888")
for i,v in enumerate(es.values): axB.text(v+0.03, i, f"{v:.1f}", va="center", fontsize=7.5)
for sp in ["top","right"]: axB.spines[sp].set_visible(False)

fig.savefig(f"{CDIR}/fig_cross_corpus_response.png", dpi=300)
fig.savefig(f"{CDIR}/fig_cross_corpus_response.pdf")
print("wrote", f"{CDIR}/fig_cross_corpus_response.png")
print("\ncrisis_escalation upvotes on suicidal:", round(end.loc['crisis_escalation','mean_upvotes_on_suicidal'],2))
print("encouragement upvotes on suicidal:", round(end.loc['encouragement_motivation','mean_upvotes_on_suicidal'],2))
