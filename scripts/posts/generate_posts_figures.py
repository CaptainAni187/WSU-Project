#!/usr/bin/env python3
"""Generate all publication figures for posts pipeline."""
import json, os, sys, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

warnings.filterwarnings('ignore')

OKABE_ITO = ['#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC', '#56B4E9']
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['figure.constrained_layout.use'] = True
sns.set_style("whitegrid")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTDIR = os.path.join(REPO, "final_outputs", "benchmark_figures")
os.makedirs(OUTDIR, exist_ok=True)

BASE = os.path.join(REPO, "final_outputs")
MODELS = ["muril", "mentalbert", "hingbert_mixed_v2", "hingroberta_mixed", "bge_m3"]
LABELS = ["MuRIL", "MentalBERT", "HingBERT-Mixed-v2", "HingRoBERTa-Mixed", "BGE-m3"]

# Load all metrics
metrics = {}
for m in MODELS:
    with open(f"{BASE}/posts_benchmark/{m}_results.json") as f:
        metrics[m] = json.load(f)["metrics"]
    with open(f"{BASE}/posts_benchmark/{m}_coherence.json") as f:
        metrics[m].update(json.load(f))
    with open(f"{BASE}/posts_retrieval/{m}_retrieval.json") as f:
        metrics[m]["retrieval"] = json.load(f)["aggregated"]

# Figure 1: Benchmark comparison (bar chart)
def fig_benchmark():
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    
    plot_data = {
        "c_v": [metrics[m]["c_v"] for m in MODELS],
        "c_npmi": [metrics[m]["c_npmi"] for m in MODELS],
        "topic_diversity": [metrics[m]["topic_diversity"] for m in MODELS],
        "keyword_stability": [metrics[m]["keyword_stability"] for m in MODELS],
        "retrieval_p@10": [metrics[m]["retrieval"]["p@10"]["mean"] for m in MODELS],
        "retrieval_ndcg@10": [metrics[m]["retrieval"]["ndcg@10"]["mean"] for m in MODELS],
    }
    titles = ["Coherence (c_v)", "Coherence (c_npmi)", "Topic Diversity", "Keyword Stability", "Retrieval P@10", "Retrieval nDCG@10"]
    
    for ax, (key, title) in zip(axes, zip(plot_data.keys(), titles)):
        bars = ax.bar(range(len(LABELS)), plot_data[key], color=OKABE_ITO[:len(LABELS)], edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(LABELS)))
        ax.set_xticklabels(LABELS, rotation=30, ha='right', fontsize=8)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel('Score', fontsize=9)
        for bar, val in zip(bars, plot_data[key]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', 
                   ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    plt.suptitle('Posts Corpus Benchmark (70,425 posts, 256 tokens)', fontsize=14, fontweight='bold', y=1.02)
    fig.savefig(f"{OUTDIR}/fig_posts_benchmark.png", format='png')
    fig.savefig(f"{OUTDIR}/fig_posts_benchmark.pdf", format='pdf')
    plt.close()
    print(f"Saved: fig_posts_benchmark")

# Figure 2: Topic distribution comparison
fig_benchmark()

# Figure 3: Radar chart comparison
from math import pi

def fig_radar():
    categories = ["c_v", "c_npmi", "topic_diversity", "keyword_stability", "retrieval_p@10", "retrieval_ndcg@10"]
    cat_labels = ["c_v", "c_npmi", "Diversity", "Stability", "P@10", "nDCG@10"]
    
    # Normalize each metric to 0-1 for radar
    normalized = {}
    for cat in categories:
        if cat in ["retrieval_p@10", "retrieval_ndcg@10"]:
            key = cat.replace("retrieval_", "")
            vals = [metrics[m]["retrieval"][key]["mean"] for m in MODELS]
        else:
            vals = [metrics[m][cat] for m in MODELS]
        minv, maxv = min(vals), max(vals)
        normalized[cat] = [(v - minv) / (maxv - minv) if maxv > minv else 0.5 for v in vals]
    
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, (label, color) in enumerate(zip(LABELS, OKABE_ITO)):
        values = [normalized[cat][i] for cat in categories]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cat_labels, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.title('Posts Corpus: Normalized Metric Comparison', fontsize=12, fontweight='bold', pad=20)
    fig.savefig(f"{OUTDIR}/fig_posts_radar.png", format='png')
    fig.savefig(f"{OUTDIR}/fig_posts_radar.pdf", format='pdf')
    plt.close()
    print(f"Saved: fig_posts_radar")

fig_radar()

# Figure 4: Topic counts and outlier rates
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
n_topics = [metrics[m]["n_topics"] for m in MODELS]
outliers = [metrics[m]["outlier_rate"] * 100 for m in MODELS]

axes[0].bar(range(len(LABELS)), n_topics, color=OKABE_ITO[:len(LABELS)], edgecolor='black', linewidth=0.5)
axes[0].set_xticks(range(len(LABELS)))
axes[0].set_xticklabels(LABELS, rotation=30, ha='right', fontsize=8)
axes[0].set_title('Number of Topics', fontsize=10, fontweight='bold')
axes[0].set_ylabel('Topics', fontsize=9)
for i, v in enumerate(n_topics):
    axes[0].text(i, v + 1, str(v), ha='center', va='bottom', fontsize=8, fontweight='bold')

axes[1].bar(range(len(LABELS)), outliers, color=OKABE_ITO[:len(LABELS)], edgecolor='black', linewidth=0.5)
axes[1].set_xticks(range(len(LABELS)))
axes[1].set_xticklabels(LABELS, rotation=30, ha='right', fontsize=8)
axes[1].set_title('Outlier Rate (%)', fontsize=10, fontweight='bold')
axes[1].set_ylabel('%', fontsize=9)
for i, v in enumerate(outliers):
    axes[1].text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.suptitle('Posts Corpus: Topic Structure', fontsize=12, fontweight='bold', y=1.05)
fig.savefig(f"{OUTDIR}/fig_posts_topic_structure.png", format='png')
fig.savefig(f"{OUTDIR}/fig_posts_topic_structure.pdf", format='pdf')
plt.close()
print(f"Saved: fig_posts_topic_structure")

# Figure 5: Retrieval by category (BGE-m3 best model)
def fig_retrieval_by_category():
    with open(f"{BASE}/posts_retrieval/bge_m3_retrieval.json") as f:
        data = json.load(f)
    
    cat_scores = defaultdict(list)
    for q in data["queries"]:
        cat_scores[q["category"]].append(q["metrics"]["ndcg@10"])
    
    cats = sorted(cat_scores.keys())
    means = [np.mean(cat_scores[c]) for c in cats]
    stds = [np.std(cat_scores[c]) for c in cats]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(cats)), means, xerr=stds, color=OKABE_ITO[0], edgecolor='black', linewidth=0.5, capsize=3)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats, fontsize=9)
    ax.set_xlabel('nDCG@10', fontsize=10)
    ax.set_title('BGE-m3: Retrieval Performance by Category (Posts)', fontsize=11, fontweight='bold')
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + 0.01, i, f'{m:.3f}', va='center', fontsize=8, fontweight='bold')
    fig.savefig(f"{OUTDIR}/fig_posts_retrieval_by_category.png", format='png')
    fig.savefig(f"{OUTDIR}/fig_posts_retrieval_by_category.pdf", format='pdf')
    plt.close()
    print(f"Saved: fig_posts_retrieval_by_category")

fig_retrieval_by_category()

# Figure 6: Gini vs Topic Count
fig, ax = plt.subplots(figsize=(7, 5))
ginis = [metrics[m]["gini_coeff"] for m in MODELS]
ntopics = [metrics[m]["n_topics"] for m in MODELS]
ax.scatter(ntopics, ginis, s=200, c=OKABE_ITO[:len(LABELS)], edgecolors='black', linewidth=1.5, zorder=3)
for i, label in enumerate(LABELS):
    ax.annotate(label, (ntopics[i], ginis[i]), textcoords="offset points", xytext=(5, 5), fontsize=9, fontweight='bold')
ax.set_xlabel('Number of Topics', fontsize=10)
ax.set_ylabel('Gini Coefficient (inequality)', fontsize=10)
ax.set_title('Topic Count vs Distribution Inequality (Posts)', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)
fig.savefig(f"{OUTDIR}/fig_posts_gini_vs_topics.png", format='png')
fig.savefig(f"{OUTDIR}/fig_posts_gini_vs_topics.pdf", format='pdf')
plt.close()
print(f"Saved: fig_posts_gini_vs_topics")

print(f"\nAll 6 figures saved to {OUTDIR}")
