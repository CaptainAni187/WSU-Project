#!/usr/bin/env python3
"""Topic stability analysis for comments (3 random seeds, ARI)."""
import argparse, os, json, time, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.metrics import adjusted_rand_score
import umap, hdbscan
warnings.filterwarnings('ignore')

UMAP_CONFIG_BASE = {"n_neighbors": 15, "n_components": 5, "min_dist": 0.0, "metric": "cosine", "verbose": False}
HDBSCAN_CONFIG = {"min_cluster_size": 100, "min_samples": 10, "metric": "euclidean", "cluster_selection_method": "leaf", "prediction_data": True}
SEEDS = [42, 123, 999]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--embedding-path', required=True)
    parser.add_argument('--output-dir', default='final_outputs/comments_benchmark')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    model_slug = args.model.lower().replace('-', '_').replace('/', '_')
    out_path = f"{args.output_dir}/{model_slug}_stability.json"
    if os.path.exists(out_path):
        print(f"Already exists: {out_path}")
        return
    
    print(f"[1] Loading embeddings for {args.model}...")
    embeddings = np.load(args.embedding_path).astype(np.float32)
    embeddings = sk_normalize(embeddings, norm='l2', axis=1)
    print(f"    Shape: {embeddings.shape}")
    
    assignments = {}
    n_topics = {}
    
    for seed in SEEDS:
        t0 = time.time()
        config = UMAP_CONFIG_BASE.copy()
        config['random_state'] = seed
        
        umap_model = umap.UMAP(**config)
        umap_emb = umap_model.fit_transform(embeddings)
        
        hdbscan_model = hdbscan.HDBSCAN(**HDBSCAN_CONFIG)
        topics = hdbscan_model.fit_predict(umap_emb)
        
        n = len([t for t in set(topics) if t != -1])
        assignments[seed] = topics
        n_topics[seed] = n
        print(f"    Seed {seed}: {n} topics, {sum(t == -1 for t in topics)} outliers, {time.time()-t0:.1f}s")
    
    aris = {}
    pairs = [(42, 123), (42, 999), (123, 999)]
    for s1, s2 in pairs:
        ari = adjusted_rand_score(assignments[s1], assignments[s2])
        aris[f"{s1}_vs_{s2}"] = round(ari, 4)
        print(f"    ARI({s1} vs {s2}): {ari:.4f}")
    
    mean_ari = sum(aris.values()) / len(aris)
    std_ari = np.std(list(aris.values()))
    
    result = {
        'model': args.model,
        'seeds': SEEDS,
        'n_topics': {str(k): v for k, v in n_topics.items()},
        'ari_pairs': aris,
        'mean_ari': round(mean_ari, 4),
        'std_ari': round(float(std_ari), 4),
    }
    
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"[2] Saved to {out_path}")

if __name__ == '__main__':
    main()
