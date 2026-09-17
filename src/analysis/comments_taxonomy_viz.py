import os, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import warnings; warnings.filterwarnings('ignore')

OUTPUT_DIR = "final_outputs/comments_analysis/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11, 'axes.titlesize': 14,
    'axes.labelsize': 12, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# Okabe-Ito colorblind palette
CB_COLORS = {
    "support_empathy": "#0173B2", "advice_guidance": "#DE8F05",
    "personal_experience": "#029E73", "encouragement_motivation": "#D55E00",
    "information_resources": "#CC78BC", "neutral_discussion": "#56B4E9",
    "humor_meme_coping": "#E69F00", "dismissive_minimizing": "#F0E442",
    "toxic_abusive": "#E63946", "blame_criticism": "#6D597A",
    "self_disclosure": "#2A9D8F", "crisis_escalation": "#9B2226",
}

# Load data
df = pd.read_csv("final_outputs/comments_analysis/comments_taxonomy.csv")
distress_dist = pd.read_csv("final_outputs/comments_analysis/taxonomy_per_distress.csv", index_col=0)

print(f"[1] Loaded {len(df):,} classified comments")

# Fig 1: Overall Distribution - Horizontal Bar
dist = df['primary_category'].value_counts()
total = len(df)

fig, ax = plt.subplots(figsize=(12, 8))
y_pos = np.arange(len(dist))
colors = [CB_COLORS.get(cat, '#888888') for cat in dist.index]
bars = ax.barh(y_pos, dist.values, color=colors, alpha=0.9, edgecolor='white', linewidth=1.5, height=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels([c.replace('_', ' ').title() for c in dist.index], fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('Number of Comments', fontsize=12)
ax.set_title('Comment Interaction Taxonomy Distribution (N=146,715)\n12-Category Heuristic Classification', fontsize=14, fontweight='bold', pad=15)

for i, (bar, val) in enumerate(zip(bars, dist.values)):
    pct = val / total * 100
    ax.text(val + 800, i, f'{val:,} ({pct:.1f}%)', va='center', fontsize=10, fontweight='bold')

ax.set_xlim(0, max(dist.values) * 1.25)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig1_taxonomy_distribution.png", dpi=300)
plt.savefig(f"{OUTPUT_DIR}/fig1_taxonomy_distribution.pdf", dpi=300)
plt.close()
print(f"  Saved: fig1_taxonomy_distribution")

# Fig 2: Stacked Bar - Per Distress Type
fig, ax = plt.subplots(figsize=(14, 9))

distress_types = ['suicidal', 'depression_anxiety', 'academic_burnout', 'exam_stress', 'institutional_anger', 'meme_distress', 'career_anxiety']
distress_labels = ['Suicidal', 'Depression/Anxiety', 'Academic Burnout', 'Exam Stress', 'Institutional Anger', 'Meme Distress', 'Career Anxiety']

# Reorder categories for better visual
ordered_cats = ['neutral_discussion', 'advice_guidance', 'humor_meme_coping', 'information_resources', 'toxic_abusive', 'encouragement_motivation', 'support_empathy', 'personal_experience', 'self_disclosure', 'dismissive_minimizing', 'crisis_escalation', 'blame_criticism']

x = np.arange(len(distress_types))
width = 0.7
bottom = np.zeros(len(distress_types))

for cat in ordered_cats:
    vals = [distress_dist.loc[dt, cat] if dt in distress_dist.index and cat in distress_dist.columns else 0 for dt in distress_types]
    vals = np.array(vals)
    pct = vals / vals.sum() * 100 if vals.sum() > 0 else vals
    ax.bar(x, vals, width, bottom=bottom, label=cat.replace('_', ' ').title(), color=CB_COLORS.get(cat, '#888888'), alpha=0.9, edgecolor='white', linewidth=0.5)
    bottom += vals

ax.set_xticks(x)
ax.set_xticklabels(distress_labels, fontsize=11, rotation=30, ha='right')
ax.set_ylabel('Number of Comments', fontsize=12)
ax.set_title('Community Response Patterns by Parent Distress Type\n(Stacked 12-Category Taxonomy)', fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='upper right', fontsize=8, framealpha=0.9, ncol=2)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig2_per_distress_stacked.png", dpi=300)
plt.savefig(f"{OUTPUT_DIR}/fig2_per_distress_stacked.pdf", dpi=300)
plt.close()
print(f"  Saved: fig2_per_distress_stacked")

# Fig 3: Normalized Percentage Heatmap
distress_pct = distress_dist.div(distress_dist.sum(axis=1), axis=0) * 100
# Reorder rows
distress_pct = distress_pct.reindex(distress_types)
# Reorder columns
ordered_cols = [c for c in ordered_cats if c in distress_pct.columns]
distress_pct = distress_pct[ordered_cols]

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(distress_pct, annot=True, fmt='.1f', cmap='YlOrRd', linewidths=0.5,
            cbar_kws={'label': 'Percentage of Comments (%)'}, ax=ax, vmin=0, vmax=60)
ax.set_xticklabels([c.replace('_', ' ').title() for c in ordered_cols], rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(distress_labels, rotation=0, fontsize=11)
ax.set_title('Community Response Percentage by Distress Type\n(Row-Normalized)', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig3_distress_heatmap.png", dpi=300)
plt.savefig(f"{OUTPUT_DIR}/fig3_distress_heatmap.pdf", dpi=300)
plt.close()
print(f"  Saved: fig3_distress_heatmap")

# Fig 4: Sensitive vs Non-Sensitive Distress Comparison
# Group distress types into sensitive vs non-sensitive
sensitive = ['suicidal', 'depression_anxiety', 'academic_burnout']
non_sensitive = ['exam_stress', 'institutional_anger', 'meme_distress', 'career_anxiety']

sens_df = df[df['distress_type'].isin(sensitive)]
non_sens_df = df[df['distress_type'].isin(non_sensitive)]

sens_dist = sens_df['primary_category'].value_counts(normalize=True) * 100
non_sens_dist = non_sens_df['primary_category'].value_counts(normalize=True) * 100

# Align indices
all_cats = ordered_cats
sens_vals = [sens_dist.get(c, 0) for c in all_cats]
non_sens_vals = [non_sens_dist.get(c, 0) for c in all_cats]

fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(all_cats))
width = 0.35
ax.bar(x - width/2, sens_vals, width, label=f'Sensitive (n={len(sens_df):,})', color='#E63946', alpha=0.85, edgecolor='white')
ax.bar(x + width/2, non_sens_vals, width, label=f'Non-Sensitive (n={len(non_sens_df):,})', color='#457B9D', alpha=0.85, edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels([c.replace('_', ' ').title() for c in all_cats], rotation=45, ha='right', fontsize=10)
ax.set_ylabel('Percentage of Comments (%)', fontsize=12)
ax.set_title('Community Response: Sensitive vs Non-Sensitive Distress Posts', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11, framealpha=0.9)

# Add value labels
for i, (sv, nsv) in enumerate(zip(sens_vals, non_sens_vals)):
    ax.text(i - width/2, sv + 0.5, f'{sv:.1f}%', ha='center', fontsize=8, fontweight='bold')
    ax.text(i + width/2, nsv + 0.5, f'{nsv:.1f}%', ha='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig4_sensitive_vs_nonsensitive.png", dpi=300)
plt.savefig(f"{OUTPUT_DIR}/fig4_sensitive_vs_nonsensitive.pdf", dpi=300)
plt.close()
print(f"  Saved: fig4_sensitive_vs_nonsensitive")

# Fig 5: Crisis Escalation Rate by Distress Type
crisis_by_distress = {}
for dt in distress_types:
    subset = df[df['distress_type'] == dt]
    if len(subset) > 0:
        crisis_rate = (subset['primary_category'] == 'crisis_escalation').sum() / len(subset) * 100
        crisis_by_distress[dt] = crisis_rate

# Also compute support rate and toxic rate
support_by_distress = {}
toxic_by_distress = {}
for dt in distress_types:
    subset = df[df['distress_type'] == dt]
    if len(subset) > 0:
        support_rate = (subset['primary_category'] == 'support_empathy').sum() / len(subset) * 100
        toxic_rate = (subset['primary_category'] == 'toxic_abusive').sum() / len(subset) * 100
        support_by_distress[dt] = support_rate
        toxic_by_distress[dt] = toxic_rate

fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(distress_types))
width = 0.25

ax.bar(x - width, [support_by_distress.get(dt, 0) for dt in distress_types], width, label='Support/Empathy', color='#0173B2', alpha=0.85, edgecolor='white')
ax.bar(x, [crisis_by_distress.get(dt, 0) for dt in distress_types], width, label='Crisis Escalation', color='#9B2226', alpha=0.85, edgecolor='white')
ax.bar(x + width, [toxic_by_distress.get(dt, 0) for dt in distress_types], width, label='Toxic/Abusive', color='#E63946', alpha=0.85, edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels(distress_labels, fontsize=11, rotation=30, ha='right')
ax.set_ylabel('Percentage of Comments (%)', fontsize=12)
ax.set_title('Critical Response Rates by Distress Type\n(Support, Crisis Escalation, Toxicity)', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11, framealpha=0.9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig5_critical_rates.png", dpi=300)
plt.savefig(f"{OUTPUT_DIR}/fig5_critical_rates.pdf", dpi=300)
plt.close()
print(f"  Saved: fig5_critical_rates")

print(f"\n{'='*60}")
print("ALL TAXONOMY VISUALIZATIONS GENERATED")
print(f"{'='*60}")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith('.png'):
        size = os.path.getsize(f"{OUTPUT_DIR}/{f}") / 1024
        print(f"  {f} ({size:.1f} KB)")
