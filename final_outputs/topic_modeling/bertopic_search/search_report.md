# BERTopic Hyperparameter Search — Comparison Report

- Embeddings (frozen): `data/embeddings/posts/bge_m3_embeddings.npy` — 70,425 × 1024, L2-normalized
- Configurations evaluated: 30 (30 valid, 0 rejected)
- Stability = adjusted Rand index between UMAP seeds 42 and 7
- Objective: maximise semantic quality (coherence, separation, compactness,
  interpretability) while reducing redundant fragmentation. Joint z-score ranking.

## Best Configuration

**`u_nn50_c10__h_mcs300_ms10_leaf`** (rank 1, composite 0.4182)

| Parameter | Value |
|---|---|
| UMAP n_neighbors | 50 |
| UMAP n_components | 10 |
| UMAP min_dist | 0.0 |
| UMAP metric | cosine |
| HDBSCAN min_cluster_size | 300 |
| HDBSCAN min_samples | 10 |
| HDBSCAN cluster_selection_method | leaf |

| Metric | Value |
|---|---|
| Topics | 41 |
| Outlier rate | 0.5927 |
| c_v | 0.6307 |
| c_npmi | 0.1404 |
| Topic diversity | 0.7878 |
| Topic stability (ARI) | 0.9731 |
| Intra-topic sim (compactness) | 0.7513 |
| Inter-topic sim (embedding) | 0.7566 |
| Inter-topic sim (c-TF-IDF) | 0.154 |
| Separation (intra − inter) | -0.0053 |
| Redundant pairs (>0.90) | 9 |
| Avg / median / min / max size | 699.6 / 513 / 309 / 2155 |

## Ranking (top 15 by joint composite)

| Rank | Config | Topics | Out% | c_v | c_npmi | Div | Stab | Intra | InterEmb | Sep | Redund | Score |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | u_nn50_c10__h_mcs300_ms10_leaf | 41 | 0.5927 | 0.6307 | 0.1404 | 0.7878 | 0.9731 | 0.7513 | 0.7566 | -0.0053 | 9 | 0.4182 |
| 2 | u_nn30_c5__h_mcs300_ms10_leaf | 38 | 0.5836 | 0.6311 | 0.1359 | 0.8026 | 0.9795 | 0.7483 | 0.758 | -0.0098 | 11 | 0.4141 |
| 3 | u_nn50_c10__h_mcs150_ms5_eom | 59 | 0.5631 | 0.6231 | 0.1319 | 0.7542 | 0.9723 | 0.7591 | 0.7265 | 0.0326 | 14 | 0.3701 |
| 4 | u_nn30_c10__h_mcs300_ms10_leaf | 38 | 0.6333 | 0.6269 | 0.1355 | 0.7789 | 0.9411 | 0.7525 | 0.7509 | 0.0016 | 11 | 0.3609 |
| 5 | u_nn50_c5__h_mcs300_ms10_leaf | 42 | 0.5858 | 0.6241 | 0.1316 | 0.75 | 0.9226 | 0.7521 | 0.7521 | 0.0 | 14 | 0.2797 |
| 6 | u_nn50_c5__h_mcs150_ms10_leaf | 70 | 0.5955 | 0.6146 | 0.1325 | 0.7414 | 0.9537 | 0.7645 | 0.7136 | 0.051 | 18 | 0.2776 |
| 7 | u_nn50_c10__h_mcs150_ms10_leaf | 66 | 0.6232 | 0.6182 | 0.1307 | 0.7682 | 0.9522 | 0.7642 | 0.7172 | 0.0469 | 16 | 0.2775 |
| 8 | u_nn30_c5__h_mcs150_ms10_leaf | 69 | 0.6219 | 0.616 | 0.1323 | 0.7522 | 0.9728 | 0.7645 | 0.7238 | 0.0407 | 16 | 0.2624 |
| 9 | u_nn50_c10__h_mcs250_ms10_eom | 42 | 0.5413 | 0.6255 | 0.1379 | 0.7643 | 0.7197 | 0.7497 | 0.7585 | -0.0089 | 11 | 0.2366 |
| 10 | u_nn30_c5__h_mcs250_ms10_eom | 38 | 0.4912 | 0.6112 | 0.1305 | 0.7842 | 0.7799 | 0.7477 | 0.7514 | -0.0037 | 10 | 0.2276 |
| 11 | u_nn50_c5__h_mcs150_ms5_eom | 63 | 0.5363 | 0.6096 | 0.1285 | 0.7524 | 0.8775 | 0.7599 | 0.725 | 0.0348 | 16 | 0.2013 |
| 12 | u_nn50_c5__h_mcs250_ms10_eom | 39 | 0.4982 | 0.5943 | 0.1196 | 0.7513 | 0.7991 | 0.7492 | 0.7469 | 0.0024 | 6 | 0.1814 |
| 13 | u_nn30_c10__h_mcs150_ms10_leaf | 72 | 0.6311 | 0.6127 | 0.1323 | 0.7278 | 0.9421 | 0.7653 | 0.7213 | 0.044 | 18 | 0.1323 |
| 14 | u_nn15_c5__h_mcs250_ms10_eom | 47 | 0.4967 | 0.6207 | 0.1318 | 0.734 | 0.9631 | 0.7512 | 0.7632 | -0.0119 | 20 | 0.1276 |
| 15 | u_nn50_c10__h_mcs100_ms5_eom | 79 | 0.5784 | 0.6192 | 0.1342 | 0.738 | 0.9697 | 0.7685 | 0.7079 | 0.0606 | 18 | 0.1257 |

## Full parameter × metric table

See `all_experiments.csv` and `ranking.csv` for every configuration.

