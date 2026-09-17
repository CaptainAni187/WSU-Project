#!/usr/bin/env python3
"""Regenerate Fig 3 (theme distribution) with larger, bold text."""
import matplotlib.pyplot as plt
import numpy as np

# from bertopic_metadata.json theme_frequencies (documents desc)
themes = ["Entrance Exam Preparation","Careers, Placements & Jobs","College & Academic Decisions",
          "Coaching & Study Resources","Higher Studies & Global Mobility","Mental Health & Wellbeing",
          "General Venting","Coping & Meme Culture"]
docs = [19136,11277,10376,7937,7445,4824,4770,4080]
# exact labels as in the original figure (metadata percentages)
plabel = ["27.4%","16.15%","14.86%","11.36%","10.66%","6.91%","6.83%","5.84%"]
colors = ["#4C72B0","#B5405A","#55A868","#DD8452","#8172B3","#CCB974","#C44E52","#5AA6A6"]

plt.rcParams.update({"font.family":"DejaVu Sans","savefig.dpi":300,"savefig.bbox":"tight"})
fig, ax = plt.subplots(figsize=(10.5, 5.6))
y = np.arange(len(themes))[::-1]
ax.barh(y, docs, color=colors, height=0.70, edgecolor="white", linewidth=0.7)

for yi, d, p in zip(y, docs, plabel):
    ax.text(d + 220, yi, f"{d:,} ({p})", va="center", ha="left",
            fontsize=20, fontweight="bold", color="#111")

ax.set_yticks(y)
ax.set_yticklabels(themes, fontsize=20.5, fontweight="bold")
ax.set_xlabel("Number of posts", fontsize=21, fontweight="bold")
ax.set_title("Theme distribution (8 higher-level themes)", fontsize=22, fontweight="bold", pad=12)
ax.set_xlim(0, 26500)
ax.tick_params(axis="x", labelsize=17)
for lbl in ax.get_xticklabels():
    lbl.set_fontweight("bold")
for sp in ["top","right"]:
    ax.spines[sp].set_visible(False)
ax.grid(axis="x", alpha=0.25)

fig.savefig("final_outputs/paper_assets/figures/fig04_theme_distribution.png", dpi=300)
fig.savefig("final_outputs/paper_assets/figures/fig04_theme_distribution.pdf")
print("saved final_outputs/paper_assets/figures/fig04_theme_distribution.png / .pdf")
