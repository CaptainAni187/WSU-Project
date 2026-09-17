# BERTopic Refinement — Executive Summary

**Goal.** Reduce the over-fragmented ~98-topic BERTopic model on the frozen BGE-m3
post embeddings into fewer, broader, semantically coherent, human-interpretable
topics — with minimal manual merging/renaming, and without touching any frozen
artifact (embeddings, preprocessing, phrases, retrieval, corpus, annotations).

**Result.** A **41-topic** model that improves every count-independent quality
signal simultaneously and eliminates near-duplicate and tiny topics.

| Metric (identical pipeline, final labels) | Baseline (98) | Final (41) |
|---|---:|---:|
| Topics | 98 | **41**  (−58%) |
| c_v coherence | 0.626 | **0.652** |
| c_npmi | 0.136 | **0.142** |
| Topic diversity | 0.641 | **0.720** |
| Tiny topics (<200 docs) | 4 | **0** |
| Smallest topic | 149 | **498** |
| Redundant pairs (centroid cos >0.90) | 280 | **128** |
| Outlier rate (post-reduction) | 0.03% | 0.82% |

Diversity rising while topic count falls is the clearest evidence the 98-topic
model was redundantly fragmented (many clusters repeating the same keywords);
consolidation removed that redundancy without hurting coherence.

## Selected configuration

```
UMAP      : n_neighbors=50, n_components=10, min_dist=0.0, metric=cosine, seed=42
HDBSCAN   : min_cluster_size=300, min_samples=10, cluster_selection_method=leaf
Vectorizer: EN+Hinglish+artifact stopwords, ngram (1,2), corpus-level min_df=15 / max_df=0.45
Represent : KeyBERTInspired + MaximalMarginalRelevance (diversity=0.3)
Outliers  : reduce_outliers(strategy="c-tf-idf", threshold=0.05)
```

Config id `u_nn50_c10__h_mcs300_ms10_leaf` — rank 1 of 30 by the joint composite,
**and** confirmed best by direct semantic inspection of the topics.

## How it was chosen

30 configs were searched (5 UMAP × 6 HDBSCAN). UMAP reductions were cached and two
seeds (42, 7) were used so **topic stability = adjusted Rand index between seeds**
is a genuine robustness measure. Every config recorded: topic count, outlier rate,
c_v, c_npmi, diversity, stability, size distribution, intra/inter-topic similarity
(embedding + keyword space), and near-duplicate pairs.

**Key methodological decision.** Embedding-centroid `separation` and `inter_sim_emb`
are *confounded with topic count* — with fewer topics each centroid averages a
broader region and drifts toward the global mean, mechanically inflating mutual
similarity (empirically corr(n_topics, inter_sim_emb) = −0.88 across the 30 runs).
Ranking on them would reward fragmentation, the opposite of the objective. Selection
was therefore anchored on **count-independent quality** — coherence (c_v, c_npmi),
run-to-run stability, keyword-space distinctiveness, near-duplicate pairs and tiny-
topic count — which penalise *both* collapse (few incoherent mega-blobs) and
over-fragmentation. These confounded metrics are still recorded for transparency.

## What the 41 topics look like

Clear, single-concept topics with the P2 overlaps consolidated: JEE/NEET/boards,
CAT prep vs MBA-profile, placements/CGPA, TCS drives, salary/LPA, interview process,
internships, coding/DSA, coaching, droppers, UPSC, GATE, study-abroad, plus distress-
specific clusters (explicit suicidal/depression, exam anxiety & sleep, social
isolation, and Hinglish coping-meme culture — cooked/hopium/chud/ALECC). One large
Hinglish personal-venting cluster (topic 2) remains the natural catch-all; splitting
it further would require manual intervention, which the objective asks to minimise.

## Deliverables

| Path | Contents |
|---|---|
| `outputs/final/bertopic_search/search_report.md` | parameter, metric & ranking tables |
| `outputs/final/bertopic_search/all_experiments.csv` / `ranking.csv` | every config, every metric |
| `outputs/final/bertopic_search/best_config.json` | winning parameters |
| `outputs/final/bertopic_final/topic_quality_report.md` | before/after + per-topic keyword table |
| `outputs/final/bertopic_final/topic_info.csv` / `topics.csv` | BERTopic info + per-doc assignment |
| `outputs/final/bertopic_final/topic_keywords.csv` | KeyBERT+MMR keywords per topic |
| `outputs/final/bertopic_final/representative_docs.csv` | representative posts per topic |
| `outputs/final/bertopic_final/topic_hierarchy.csv` | hierarchical topic tree (P7) |
| `outputs/final/bertopic_final/final_metrics.json` | final vs baseline, same pipeline |
| `outputs/final/bertopic_final/figures/` | topic sizes + inter-topic similarity heatmap |

Reproduce: `python src/pipeline/bertopic_search.py` then
`python src/pipeline/bertopic_finalize.py`. Neither script touches any frozen input.
