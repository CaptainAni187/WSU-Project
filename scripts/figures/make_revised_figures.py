#!/usr/bin/env python3
"""Revised publication figures:
  - fig_posts_benchmark_compact    compact 5-model embedding benchmark
  - fig04_distress_taxonomy_v2     distress-type composition (shares, post counts, topic counts)
  - fig_annotation_pipeline        human-annotation / validation framework
Saved alongside the originals. Run from the repo root."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np, json, os

OUT_FIG = "final_outputs/paper_assets/figures"
OUT_PUB = "final_outputs/benchmark_figures"
plt.rcParams.update({"font.family":"sans-serif","savefig.dpi":300,"savefig.bbox":"tight"})

# ---------------------------------------------------------------- FIG 4 (redesigned)
# distress type : (posts, pct, n_topics)
dt = [("Academic Distress",24045,34.4,14),
      ("Career Distress",18722,26.8,9),
      ("Informational /\nLow-Distress",10689,15.3,6),
      ("Coping Expression",8850,12.7,6),
      ("Emotional Distress",4824,6.9,3),
      ("Institutional Distress",2715,3.9,3)]
colors=["#1f77b4","#ff7f0e","#2ca02c","#9467bd","#8c564b","#d62728"]
names=[x[0] for x in dt]; pcts=[x[2] for x in dt]; posts=[x[1] for x in dt]; tops=[x[3] for x in dt]
fig,ax=plt.subplots(figsize=(6.6,2.9))
y=np.arange(len(dt))[::-1]
ax.barh(y,pcts,color=colors,height=0.62)
for yi,p,n,t in zip(y,pcts,posts,tops):
    ax.text(p+0.5,yi,f"{p:.1f}%   ({n:,} posts · {t} topics)",va="center",fontsize=8.2)
ax.set_yticks(y); ax.set_yticklabels(names,fontsize=8.5)
ax.set_xlabel("Share of posts (%)",fontsize=9); ax.set_xlim(0,46)
ax.set_title("Distress-type composition of the corpus (69,845 posts, 41 topics)",fontsize=9.5)
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.tick_params(axis="x",labelsize=8)
fig.savefig(f"{OUT_FIG}/fig04_distress_taxonomy_v2.png"); fig.savefig(f"{OUT_FIG}/fig04_distress_taxonomy_v2.pdf")
plt.close(fig); print("wrote fig04_distress_taxonomy_v2")

# ---------------------------------------------------------------- FIG 1 (compact benchmark)
MODELS=["MuRIL","MentalBERT","HingBERT-Mx-v2","HingRoBERTa-Mx","BGE-m3"]
C=["#0173B2","#CC78BC","#029E73","#D55E00","#56B4E9"]
data={
 "Coherence $c_v$":[0.524,0.498,0.487,0.535,0.566],
 "Coherence $c_{npmi}$":[0.080,0.050,0.044,0.079,0.099],
 "Topic diversity":[0.413,0.364,0.334,0.444,0.487],
 "Retrieval P@10":[0.612,0.444,0.544,0.844,0.844],
 "Retrieval MRR":[0.791,0.601,0.816,0.970,1.000],
 "Retrieval nDCG@10":[0.218,0.149,0.189,0.342,0.437],
}
fig,axs=plt.subplots(2,3,figsize=(7.4,3.6))
for ax,(title,vals) in zip(axs.flat,data.items()):
    ax.bar(range(5),vals,color=C,width=0.72)
    for i,v in enumerate(vals): ax.text(i,v,f"{v:.3f}",ha="center",va="bottom",fontsize=6)
    ax.set_title(title,fontsize=8.5); ax.set_xticks([]); ax.tick_params(axis="y",labelsize=6.5)
    ax.set_ylim(0,max(vals)*1.22)
    for s in ["top","right"]: ax.spines[s].set_visible(False)
handles=[plt.Rectangle((0,0),1,1,color=c) for c in C]
fig.legend(handles,MODELS,loc="lower center",ncol=5,fontsize=7.5,frameon=False,bbox_to_anchor=(0.5,-0.04))
fig.suptitle("Five-model embedding benchmark (70,425 posts, 256 tokens)",fontsize=10,y=1.0)
fig.tight_layout(rect=[0,0.03,1,0.98])
fig.savefig(f"{OUT_PUB}/fig_posts_benchmark_compact.png"); fig.savefig(f"{OUT_PUB}/fig_posts_benchmark_compact.pdf")
plt.close(fig); print("wrote fig_posts_benchmark_compact")

# ---------------------------------------------------------------- NEW: annotation pipeline figure
fig,ax=plt.subplots(figsize=(7.6,3.3)); ax.set_xlim(0,102); ax.set_ylim(0,60); ax.axis("off")
def box(x,y,w,h,text,fc,fs=8.2,ec="#333"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.6,rounding_size=2",
                 fc=fc,ec=ec,lw=1.1))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs)
def arrow(x0,y0,x1,y1):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",mutation_scale=12,lw=1.1,color="#555"))
# left: automated outputs
box(0.5,19,20,16,"Automated\npipeline outputs\n(topics, themes,\ndistress labels,\nretrieval,\ncomments)","#E8EEF7",7.6)
arrow(20.5,27,25,27)
# middle: 5 annotation tasks (stacked)
tasks=["Distress identification","Topic validation","Theme validation",
       "Retrieval-relevance assessment","Comment-interaction analysis"]
tc=["#1f77b4","#2ca02c","#9467bd","#ff7f0e","#d62728"]
ty=np.linspace(45,1,5)
for t,c,yy in zip(tasks,tc,ty):
    box(25,yy,31,7.2,t,"#FFFFFF",7.6,ec=c)
    arrow(56,yy+3.6,64.5,27)
ax.text(40.5,54.5,"Human annotation  (independent, predefined criteria)",ha="center",fontsize=8.4,style="italic")
# right: consensus + kappa
box(64.5,19,17,16,"Disagreements\nresolved by\nconsensus","#FFF3E0",7.8)
arrow(81.5,27,86,27)
box(86,19,15.5,16,"Cohen's κ\nagreement\n→ Results\n(§IV)","#E8F5E9",7.8)
fig.savefig(f"{OUT_FIG}/fig_annotation_pipeline.png"); fig.savefig(f"{OUT_FIG}/fig_annotation_pipeline.pdf")
plt.close(fig); print("wrote fig_annotation_pipeline")
print("\nPATHS:")
for p in [f"{OUT_FIG}/fig04_distress_taxonomy_v2.png",f"{OUT_PUB}/fig_posts_benchmark_compact.png",f"{OUT_FIG}/fig_annotation_pipeline.png"]:
    print("  ", os.path.abspath(p))
