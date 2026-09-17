# Posts Corpus Benchmark Report

## Executive Summary

This report presents the complete benchmark evaluation of five embedding models on the merged posts corpus (70,425 posts, 256 tokens, 11 subreddits, 5 lifecycle stages). The benchmark covers topic modeling quality, coherence, retrieval performance, and structural metrics.

**Key Finding:** BGE-m3 is the clear winner across the majority of metrics, with HingRoBERTa-Mixed as a strong second choice for topic interpretation tasks.

---

## Dataset Description

| Property | Value |
|----------|-------|
| Total posts | 70,425 |
| Token length | 256 (max) |
| Subreddits | 11 |
| Lifecycle stages | 5 (competitive_exam, undergraduate, career, school, higher_education) |
| Distress posts | ~16,000 (22.7%) |
| Source files | phase1.csv (48,213) + extra.csv (23,660) |
| Deduplication | 1,448 removed (209 by ID, 1,239 by exact text) |

---

## Model Results Summary

| Model | Topics | Outlier% | c_v | c_npmi | Diversity | Stability | P@10 | MAP | MRR | nDCG@10 | Gini | Top3% | Phrases | Artifacts |
|-------|--------|----------|-----|--------|-----------|-----------|------|-----|-----|---------|------|-------|---------|-----------|
| **BGE-m3** | 98 | 0.03% | **0.566** | **0.099** | **0.487** | 0.256 | **0.844** | **0.933** | **1.000** | **0.437** | 0.372 | 9.94% | 5 | 0 |
| HingRoBERTa-Mixed | 63 | 0.10% | 0.535 | 0.079 | 0.444 | 0.218 | 0.844 | 0.928 | 0.970 | 0.342 | 0.447 | 16.08% | 6 | 0 |
| MuRIL | 79 | 0.08% | 0.524 | 0.080 | 0.413 | 0.261 | 0.612 | 0.732 | 0.791 | 0.218 | 0.375 | 12.69% | 6 | 1 |
| MentalBERT | 62 | 0.06% | 0.498 | 0.050 | 0.365 | **0.314** | 0.444 | 0.543 | 0.601 | 0.149 | **0.300** | 11.35% | 4 | 0 |
| HingBERT-Mixed-v2 | 92 | 0.10% | 0.487 | 0.044 | 0.334 | 0.300 | 0.544 | 0.721 | 0.816 | 0.189 | 0.339 | 10.47% | 6 | 0 |

---

## Per-Metric Ranking (1=best, 5=worst)

| Model | c_v | c_npmi | Diversity | Stability | P@10 | MAP | MRR | nDCG@10 | Gini | Top3% | Artifacts | **Sum** |
|-------|-----|--------|-----------|-----------|------|-----|-----|---------|------|-------|-----------|---------|
| **BGE-m3** | **1** | **1** | **1** | 4 | **1** | **1** | **1** | **1** | 3 | **1** | **1** | **16** |
| HingRoBERTa-Mixed | 2 | 3 | 2 | 5 | **1** | 2 | 2 | 2 | 5 | 5 | **1** | 30 |
| MuRIL | 3 | 2 | 3 | 3 | 3 | 3 | 4 | 3 | 4 | 4 | 5 | 37 |
| MentalBERT | 4 | 4 | 4 | **1** | 5 | 5 | 5 | 5 | **1** | 3 | **1** | 38 |
| HingBERT-Mixed-v2 | 5 | 5 | 5 | 2 | 4 | 4 | 3 | 4 | 2 | 2 | **1** | 39 |

---

## Key Findings

1. **BGE-m3 dominates retrieval**: Perfect MRR (1.0), highest nDCG@10 (0.437), tied highest P@10 (0.844). This is the embedding model of choice for semantic search on academic distress posts.

2. **BGE-m3 also leads coherence**: Highest c_v (0.566) and c_npmi (0.099), indicating the most semantically coherent topic clusters. This validates that the dense retrieval performance correlates with topic quality.

3. **HingRoBERTa-Mixed excels at topic interpretation**: Highest coherence_proxy (0.136), tied for best retrieval P@10, but lower keyword stability (0.218) and higher Gini (0.447) indicating more concentrated topic sizes. This model produces interpretable, focused topics that may be more domain-specific.

4. **MentalBERT has the most uniform topic distribution**: Lowest Gini (0.300), highest keyword stability (0.314), but weakest retrieval performance. Good for balanced topic exploration, poor for targeted search.

5. **HingBERT-Mixed-v2 produces the most topics** (92) but lowest coherence scores (c_v=0.487, c_npmi=0.044). May be over-segmenting the corpus.

6. **MuRIL has 1 artifact topic**: The only model with a contaminated topic (likely containing Reddit markup/formatting artifacts). This is a quality concern despite otherwise solid middle-of-pack performance.

7. **All models achieve <0.1% outlier rate after reduction**: Excellent outlier handling across the board, suggesting the HDBSCAN + c-TF-IDF reduction strategy is effective.

---

## Retrieval Category Analysis (BGE-m3)

BGE-m3 performs best on placement-related queries (nDCG@10=0.52) and worst on implicit_distress (nDCG@10=0.18), confirming that explicit domain-specific language retrieves better than implicit emotional expressions.

---

## Recommendations

- **For production semantic search**: Use **BGE-m3** embeddings
- **For topic exploration and dashboard**: Use **HingRoBERTa-Mixed** for focused, interpretable topics
- **For balanced overview**: Use **MentalBERT** for most evenly distributed topic model
- **For Hinglish-heavy content**: HingRoBERTa-Mixed and BGE-m3 both handle Hinglish well

---

## Files Generated

| File | Description |
|------|-------------|
| `data/embeddings/posts/*_embeddings.npy` | 5 model embeddings (70,425 x dim) |
| `outputs/posts_benchmark/*_results.json` | BERTopic metrics per model |
| `outputs/posts_benchmark/*_coherence.json` | c_v and c_npmi per model |
| `outputs/posts_retrieval/*_retrieval.json` | 25-query retrieval evaluation per model |
| `outputs/posts_benchmark/posts_benchmark_metrics.csv` | Aggregated benchmark table |
| `outputs/figures_publication/fig_posts_*.png/pdf` | 6 publication figures |

---

*Report generated: 2026-07-03*
