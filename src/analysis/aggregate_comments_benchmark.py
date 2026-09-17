import json, os, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import warnings; warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────────────────────
BENCH_DIR = "final_outputs/comments_benchmark"
FIG_DIR = "final_outputs/comments_analysis/figures_benchmark"
os.makedirs(FIG_DIR, exist_ok=True)

MODELS = ["MuRIL", "MentalBERT", "HingBERT-Mixed-v2", "HingRoBERTa-Mixed", "BGE-m3"]
MODEL_DISPLAY = {m: m for m in MODELS}

# Okabe-Ito colorblind palette
CB_COLORS = {
    "MuRIL": "#0173B2", "MentalBERT": "#CC78BC",
    "HingBERT-Mixed-v2": "#029E73", "HingRoBERTa-Mixed": "#D55E00",
    "BGE-m3": "#56B4E9",
}

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11, 'axes.titlesize': 14,
    'axes.labelsize': 12, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
})

def load_json(path):
    with open(path) as f: return json.load(f)

# ─── Load all evidence ────────────────────────────────────────────────────────
print("[1] Loading all evidence...")

metrics = {}
for m in MODELS:
    metrics[m] = load_json(f"{BENCH_DIR}/{m.lower().replace('-','_').replace('/','_')}_results.json")['metrics']

coherence = {}
for m in MODELS:
    coherence[m] = load_json(f"{BENCH_DIR}/{m.lower().replace('-','_').replace('/','_')}_coherence.json")

retrieval = {}
for m in MODELS:
    data = load_json(f"{BENCH_DIR}/{m.lower().replace('-','_').replace('/','_')}_retrieval.json")
    retrieval[m] = sum(r['avg_relevance'] for r in data['results']) / len(data['results'])

stability = {}
for m in MODELS:
    stability[m] = load_json(f"{BENCH_DIR}/{m.lower().replace('-','_').replace('/','_')}_stability.json")['mean_ari']

# Build evidence table
evidence = {}
for m in MODELS:
    evidence[m] = {
        'c_v': coherence[m]['c_v'],
        'c_npmi': coherence[m]['c_npmi'],
        'retrieval': retrieval[m],
        'stability_ari': stability[m],
        'topic_diversity': metrics[m]['topic_diversity'],
        'outlier_rate': metrics[m]['outlier_rate'],
        'n_topics': metrics[m]['n_topics'],
        'phrases_found': metrics[m]['phrases_found'],
        'gini': metrics[m]['gini_coeff'],
        'entropy': metrics[m]['norm_entropy'],
        'top3_conc': metrics[m]['top3_concentration'],
    }

print(f"    Evidence loaded for {len(MODELS)} models")
for m in MODELS:
    print(f"    {m}: c_v={evidence[m]['c_v']:.4f}, retrieval={evidence[m]['retrieval']:.3f}, ARI={evidence[m]['stability_ari']:.4f}")

# ─── Weighted Scoring ─────────────────────────────────────────────────────────
print("\n[2] Computing weighted rankings...")

def normalize_metric(values_dict):
    vals = np.array(list(values_dict.values()))
    mn, mx = vals.min(), vals.max()
    if mx == mn: return {k: 0.5 for k in values_dict}
    return {k: (v - mn) / (mx - mn) for k, v in values_dict.items()}

score_components = {
    'c_v': normalize_metric({m: evidence[m]['c_v'] for m in MODELS}),
    'c_npmi': normalize_metric({m: evidence[m]['c_npmi'] for m in MODELS if not np.isnan(evidence[m]['c_npmi'])}),
    'retrieval': normalize_metric({m: evidence[m]['retrieval'] for m in MODELS}),
    'stability': normalize_metric({m: evidence[m]['stability_ari'] for m in MODELS}),
    'diversity': normalize_metric({m: evidence[m]['topic_diversity'] for m in MODELS}),
    'phrases': normalize_metric({m: evidence[m]['phrases_found'] for m in MODELS}),
}
# Handle NaN c_npmi for MentalBERT
for m in MODELS:
    if m not in score_components['c_npmi']:
        score_components['c_npmi'][m] = 0.0

outlier_min = min(evidence[m]['outlier_rate'] for m in MODELS)
outlier_max = max(evidence[m]['outlier_rate'] for m in MODELS)
score_components['outlier'] = {m: 1 - (evidence[m]['outlier_rate'] - outlier_min) / (outlier_max - outlier_min + 1e-9) for m in MODELS}

WEIGHTS = {'c_v': 0.25, 'c_npmi': 0.10, 'retrieval': 0.20, 'stability': 0.15, 'diversity': 0.10, 'outlier': 0.05, 'phrases': 0.15}

weighted_scores = {}
for m in MODELS:
    weighted_scores[m] = sum(score_components[metric][m] * weight for metric, weight in WEIGHTS.items())

ranked = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)

print(f"\n{'='*60}")
print("WEIGHTED RANKINGS (Comments Corpus)")
print(f"{'='*60}")
for rank, (m, score) in enumerate(ranked, 1):
    print(f"  {rank}. {m:<22} Score: {score:.4f}")

# Save metrics CSV
scores_df = pd.DataFrame({
    'model': MODELS, 'c_v': [evidence[m]['c_v'] for m in MODELS],
    'c_npmi': [evidence[m]['c_npmi'] for m in MODELS],
    'retrieval': [evidence[m]['retrieval'] for m in MODELS],
    'stability_ari': [evidence[m]['stability_ari'] for m in MODELS],
    'topic_diversity': [evidence[m]['topic_diversity'] for m in MODELS],
    'outlier_rate': [evidence[m]['outlier_rate'] for m in MODELS],
    'phrases_found': [evidence[m]['phrases_found'] for m in MODELS],
    'weighted_score': [weighted_scores[m] for m in MODELS],
})
scores_df = scores_df.sort_values('weighted_score', ascending=False)
scores_df.to_csv(f"{BENCH_DIR}/comments_benchmark_metrics.csv", index=False)
print(f"\nSaved to {BENCH_DIR}/comments_benchmark_metrics.csv")

# ─── Visualizations ───────────────────────────────────────────────────────────
print("\n[3] Generating visualizations...")

# Fig 1: Weighted Score Ranking
fig, ax = plt.subplots(figsize=(10, 7))
ranked_models = [m for m, _ in ranked]
colors = [CB_COLORS[m] for m in ranked_models]
ax.barh(range(len(ranked_models)), [weighted_scores[m] for m in ranked_models], color=colors, alpha=0.9, edgecolor='white', linewidth=1.5, height=0.6)
ax.set_yticks(range(len(ranked_models)))
ax.set_yticklabels([f"{i+1}. {MODEL_DISPLAY[m]}" for i, m in enumerate(ranked_models)], fontsize=12)
ax.invert_yaxis()
ax.set_xlabel('Weighted Evidence Score (0–1)', fontsize=12)
ax.set_title('Embedding Model Ranking: Comments Corpus\n(c_v=0.25, retrieval=0.20, phrases=0.15, stability=0.15, diversity=0.10, c_npmi=0.10, outlier=0.05)', fontsize=13, fontweight='bold', pad=15)
for i, m in enumerate(ranked_models): ax.text(weighted_scores[m] + 0.01, i, f'{weighted_scores[m]:.3f}', va='center', fontsize=11, fontweight='bold')
ax.set_xlim(0, 1.05)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig1_weighted_ranking.png", dpi=300); plt.savefig(f"{FIG_DIR}/fig1_weighted_ranking.pdf", dpi=300); plt.close()
print(f"  Saved: fig1_weighted_ranking")

# Fig 2: Normalized Metric Heatmap
fig, ax = plt.subplots(figsize=(12, 8))
heatmap_data = []
for metric in ['c_v', 'c_npmi', 'retrieval', 'stability', 'diversity', 'outlier', 'phrases']:
    heatmap_data.append([score_components[metric][m] for m in MODELS])
heatmap_df = pd.DataFrame(heatmap_data, columns=[MODEL_DISPLAY[m] for m in MODELS], index=['c_v Coherence', 'c_npmi Coherence', 'Retrieval Relevance', 'Stability (ARI)', 'Topic Diversity', 'Coverage (1−outlier)', 'Cultural Phrases'])
heatmap_df = heatmap_df[[MODEL_DISPLAY[m] for m in ranked_models]]
sns.heatmap(heatmap_df, annot=True, fmt='.2f', cmap='YlGnBu', linewidths=0.5, cbar_kws={'label': 'Normalized Score (0–1)'}, ax=ax, vmin=0, vmax=1)
ax.set_title('Normalized Metric Comparison: Comments Corpus\n(Higher = Better for All Metrics)', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig2_metric_heatmap.png", dpi=300); plt.savefig(f"{FIG_DIR}/fig2_metric_heatmap.pdf", dpi=300); plt.close()
print(f"  Saved: fig2_metric_heatmap")

# Fig 3: Raw Metric Grouped Bars
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
metrics_to_plot = [
    ('c_v', 'c_v Coherence', 'higher'), ('c_npmi', 'c_npmi Coherence', 'higher'),
    ('retrieval', 'Retrieval Relevance', 'higher'), ('stability_ari', 'Stability (ARI)', 'higher'),
    ('topic_diversity', 'Topic Diversity', 'higher'), ('outlier_rate', 'Outlier Rate', 'lower'),
    ('phrases_found', 'Cultural Phrases Found', 'higher'),
]
for idx, (metric, label, direction) in enumerate(metrics_to_plot):
    ax = axes[idx // 4, idx % 4]
    sorted_models = sorted(MODELS, key=lambda m: evidence[m][metric], reverse=(direction=='higher'))
    sorted_vals = [evidence[m][metric] for m in sorted_models]
    ax.bar(range(len(sorted_models)), sorted_vals, color=[CB_COLORS[m] for m in sorted_models], alpha=0.9, edgecolor='white', linewidth=1)
    ax.set_xticks(range(len(sorted_models))); ax.set_xticklabels([MODEL_DISPLAY[m] for m in sorted_models], rotation=45, ha='right', fontsize=9)
    ax.set_title(label, fontsize=11, fontweight='bold'); ax.set_ylabel('Value', fontsize=10)
    for i, v in enumerate(sorted_vals): ax.text(i, v + (max(sorted_vals)-min(sorted_vals))*0.02, f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')
axes[1, 3].axis('off')
plt.suptitle('Raw Metric Comparison: Comments Corpus', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig3_raw_metrics_grouped.png", dpi=300); plt.savefig(f"{FIG_DIR}/fig3_raw_metrics_grouped.pdf", dpi=300); plt.close()
print(f"  Saved: fig3_raw_metrics_grouped")

# Fig 4: Retrieval per Query
fig, ax = plt.subplots(figsize=(14, 8))
query_names = ['jee burnout', 'nta mkc', 'drop year', 'placement anxiety', 'suicidal jee', 'life barbaad']
x = np.arange(len(query_names)); width = 0.15
for i, m in enumerate(MODELS):
    data = load_json(f"{BENCH_DIR}/{m.lower().replace('-','_').replace('/','_')}_retrieval.json")
    vals = [r['avg_relevance'] for r in data['results']]
    ax.bar(x + i*width - width*2, vals, width, label=MODEL_DISPLAY[m], color=CB_COLORS[m], alpha=0.9, edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(query_names, fontsize=11)
ax.set_ylabel('Average Relevance Score (0–1)', fontsize=12)
ax.set_title('Retrieval Performance: Comments Corpus\nAverage Relevance per Query', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=10, framealpha=0.9, loc='upper right'); ax.set_ylim(0, 0.35)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig4_retrieval_per_query.png", dpi=300); plt.savefig(f"{FIG_DIR}/fig4_retrieval_per_query.pdf", dpi=300); plt.close()
print(f"  Saved: fig4_retrieval_per_query")

# Fig 5: Stability with Error Bars
fig, ax = plt.subplots(figsize=(10, 7))
stability_data = []
for m in MODELS:
    data = load_json(f"{BENCH_DIR}/{m.lower().replace('-','_').replace('/','_')}_stability.json")
    aris = list(data['ari_pairs'].values())
    stability_data.append({'model': MODEL_DISPLAY[m], 'mean': data['mean_ari'], 'std': data['std_ari'], 'min': min(aris), 'max': max(aris)})
stdf = pd.DataFrame(stability_data).sort_values('mean', ascending=True)
y_pos = np.arange(len(stdf))
ax.barh(y_pos, stdf['mean'], color=[CB_COLORS[MODELS[i]] for i in stdf.index], alpha=0.9, edgecolor='white', linewidth=1.5, height=0.5)
ax.errorbar(stdf['mean'], y_pos, xerr=stdf['std'], fmt='none', color='black', capsize=4, capthick=1.5, elinewidth=1.5)
ax.set_yticks(y_pos); ax.set_yticklabels(stdf['model'], fontsize=11)
ax.set_xlabel('Adjusted Rand Index (ARI)', fontsize=12)
ax.set_title('Topic Stability: Comments Corpus\n(3 Random Seeds, Mean ± Std)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlim(0.55, 0.72)
for i, row in stdf.iterrows(): ax.text(row['mean'] + row['std'] + 0.005, y_pos[i], f"{row['mean']:.3f} ± {row['std']:.3f}", va='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig5_stability_variance.png", dpi=300); plt.savefig(f"{FIG_DIR}/fig5_stability_variance.pdf", dpi=300); plt.close()
print(f"  Saved: fig5_stability_variance")

# Fig 6: Topics vs Coherence Scatter (Comments vs Posts comparison)
# Load post data for comparison
post_metrics = {}
for m in MODELS:
    try:
        post_metrics[m] = load_json(f"final_outputs/posts_benchmark/{m.lower().replace('-','_').replace('/','_')}_results.json")['metrics']
    except:
        pass

if post_metrics:
    fig, ax = plt.subplots(figsize=(10, 8))
    for m in MODELS:
        if m in post_metrics:
            ax.scatter(post_metrics[m]['n_topics'], post_metrics[m]['coherence_proxy'], s=200, color=CB_COLORS[m], alpha=0.6, edgecolors='white', linewidth=2, marker='o', label=f'{MODEL_DISPLAY[m]} (Posts)')
        ax.scatter(metrics[m]['n_topics'], metrics[m]['coherence_proxy'], s=200, color=CB_COLORS[m], alpha=0.9, edgecolors='white', linewidth=2, marker='^', label=f'{MODEL_DISPLAY[m]} (Comments)')
    
    # Add labels
    for m in MODELS:
        if m in post_metrics:
            ax.annotate(f'{MODEL_DISPLAY[m]}', (post_metrics[m]['n_topics'], post_metrics[m]['coherence_proxy']), xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.7)
        ax.annotate(f'{MODEL_DISPLAY[m]}', (metrics[m]['n_topics'], metrics[m]['coherence_proxy']), xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Number of Topics Discovered', fontsize=12)
    ax.set_ylabel('Coherence Proxy (c-TF-IDF)', fontsize=12)
    ax.set_title('Comments vs Posts: Topic Count vs Coherence\n(Circles = Posts, Triangles = Comments)', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=9, framealpha=0.9, loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig6_comments_vs_posts.png", dpi=300); plt.savefig(f"{FIG_DIR}/fig6_comments_vs_posts.pdf", dpi=300); plt.close()
    print(f"  Saved: fig6_comments_vs_posts")

print(f"\n{'='*60}")
print("ALL BENCHMARK VISUALIZATIONS GENERATED")
print(f"{'='*60}")
for f in sorted(os.listdir(FIG_DIR)):
    if f.endswith('.png'): print(f"  {f} ({os.path.getsize(f'{FIG_DIR}/{f}')/1024:.1f} KB)")
