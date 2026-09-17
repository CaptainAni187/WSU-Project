#!/usr/bin/env python3
"""Retrieval evaluation for posts corpus (5 models, 25 queries, 17 domains)."""
import sys, os, json, re, time, warnings
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
warnings.filterwarnings("ignore")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 32
MAX_LENGTH = 256
CACHE_DIR = "model_cache"
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
OUT_DIR = os.path.join(BASE, "final_outputs", "posts_retrieval")
os.makedirs(OUT_DIR, exist_ok=True)

QUERIES = [
    ("jee mains advanced preparation stress", "exams"),
    ("neet biology physics chemistry preparation", "exams"),
    ("academic burnout can not study anymore", "burnout"),
    ("padhai nahi ho rahi hai exhausted", "burnout"),
    ("drop year waste of life regret", "drop_year"),
    ("double dropper partial dropper failure", "drop_year"),
    ("low cgpa backlogs academic probation", "cgpa"),
    ("semester failure cgpa khatam", "cgpa"),
    ("no job offers placement season failure", "placements"),
    ("tier 3 college no placement package", "placements"),
    ("no internship experience coding skills", "internships"),
    ("tech layoffs recession job loss", "layoffs"),
    ("fresher layoff company shut down", "layoffs"),
    ("leetcode dsa unable to solve easy", "coding"),
    ("competitive programming beginner stuck", "coding"),
    ("branch regret cse mech civil", "branch_regret"),
    ("hostel life mess food terrible", "hostel"),
    ("lonely no friends college isolated", "loneliness"),
    ("depression therapy medication need help", "depression"),
    ("anxiety panic attacks before exam", "anxiety"),
    ("want to end my life suicide", "suicidal_ideation"),
    ("mar jana hai sab khatam hopeless", "suicidal_ideation"),
    ("sab barbaad ho gaya kuch nahi bacha", "implicit_distress"),
    ("nta unfair result delay cheating", "nta"),
    ("coaching institute wasted money time", "coaching"),
]

def load_texts():
    df = pd.read_csv(os.path.join(BASE, "data", "processed", "posts_merged_final_preprocessed.csv"), usecols=["cleaned_text"])
    return df["cleaned_text"].fillna("").astype(str).tolist()

def tokenize_keywords(text):
    text = text.lower()
    words = re.findall(r'[a-z_0-9]+', text)
    return [w for w in words if len(w) > 1]

def compute_keyword_relevance(query, doc_text):
    q_words = set(tokenize_keywords(query))
    if not q_words: return 0.0
    d_words = set(tokenize_keywords(doc_text))
    return len(q_words & d_words) / len(q_words)

def compute_metrics(relevances, top_indices):
    top_rel = relevances[top_indices]
    binary_rel = (relevances > 0).astype(int)
    top_binary = binary_rel[top_indices]
    total_relevant = binary_rel.sum()
    p10 = top_binary.sum() / 10.0
    r10 = top_binary.sum() / total_relevant if total_relevant > 0 else 0.0
    first_rel = next((i+1 for i, r in enumerate(top_binary) if r == 1), None)
    mrr = 1.0 / first_rel if first_rel else 0.0
    if top_binary.sum() == 0:
        ap = 0.0
    else:
        precisions = []
        rel_count = 0
        for rank, rel in enumerate(top_binary, start=1):
            if rel == 1:
                rel_count += 1
                precisions.append(rel_count / rank)
        ap = np.mean(precisions)
    dcg = sum(rel / np.log2(i+2) for i, rel in enumerate(top_rel))
    ideal = np.sort(relevances)[::-1][:10]
    idcg = sum(rel / np.log2(i+2) for i, rel in enumerate(ideal))
    ndcg10 = dcg / idcg if idcg > 0 else 0.0
    return {"p@10": float(p10), "r@10": float(r10), "map": float(ap), "mrr": float(mrr), "ndcg@10": float(ndcg10), "avg_relevance": float(top_rel.mean())}

def evaluate_model(model_label, hf_name, embed_dim, embed_path, texts):
    out_file = os.path.join(OUT_DIR, f"{model_label.lower().replace('/', '_').replace('-', '_')}_retrieval.json")
    if os.path.exists(out_file):
        print(f"  {model_label}: already done")
        return
    
    print(f"\n{'='*50}\n{model_label}\n{'='*50}")
    doc_embeddings = np.load(os.path.join(BASE, embed_path)).astype(np.float32)
    print(f"  Embeddings: {doc_embeddings.shape}")
    
    tokenizer = AutoTokenizer.from_pretrained(hf_name, cache_dir=CACHE_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(hf_name, cache_dir=CACHE_DIR, trust_remote_code=True)
    model.eval().to(DEVICE)
    
    query_texts = [q[0] for q in QUERIES]
    n = len(query_texts)
    query_embeddings = np.zeros((n, embed_dim), dtype=np.float32)
    for i in range(0, n, BATCH_SIZE):
        batch = query_texts[i:i+BATCH_SIZE]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model(**inputs)
            query_embeddings[i:i+len(batch)] = out.last_hidden_state[:, 0, :].cpu().numpy()
    print(f"  Query embeddings done")
    
    similarities = cosine_similarity(query_embeddings, doc_embeddings)
    print(f"  Similarities: {similarities.shape}")
    
    all_results = []
    query_metrics = []
    for q_idx, (query, cat) in enumerate(zip(query_texts, [q[1] for q in QUERIES])):
        relevances = np.array([compute_keyword_relevance(query, t) for t in texts], dtype=np.float32)
        top10 = np.argsort(similarities[q_idx])[::-1][:10]
        metrics = compute_metrics(relevances, top10)
        query_metrics.append(metrics)
        all_results.append({"query": query, "category": cat, "metrics": metrics})
        if (q_idx + 1) % 5 == 0:
            print(f"    {q_idx+1}/{n} queries done")
    
    agg = {k: {"mean": float(np.mean([m[k] for m in query_metrics])), "std": float(np.std([m[k] for m in query_metrics]))} for k in ["p@10", "r@10", "map", "mrr", "ndcg@10", "avg_relevance"]}
    
    with open(out_file, "w") as f:
        json.dump({"model": model_label, "aggregated": agg, "queries": all_results}, f, indent=2)
    print(f"  Saved: {out_file}")
    
    del model, tokenizer
    if DEVICE.type == "mps": torch.mps.empty_cache()
    elif DEVICE.type == "cuda": torch.cuda.empty_cache()

if __name__ == "__main__":
    texts = load_texts()
    print(f"Loaded {len(texts):,} post texts")
    
    MODELS = [
        ("MuRIL", "google/muril-base-cased", 768, "data/embeddings/posts/muril_embeddings.npy"),
        ("MentalBERT", "mental/mental-bert-base-uncased", 768, "data/embeddings/posts/mentalbert_embeddings.npy"),
        ("HingBERT-Mixed-v2", "l3cube-pune/hingbert-mixed-v2", 768, "data/embeddings/posts/hingbert_mixed_v2_embeddings.npy"),
        ("HingRoBERTa-Mixed", "l3cube-pune/hing-roberta-mixed", 768, "data/embeddings/posts/hingroberta_mixed_embeddings.npy"),
        ("BGE-m3", "BAAI/bge-m3", 1024, "data/embeddings/posts/bge_m3_embeddings.npy"),
    ]
    
    for label, hf, dim, path in MODELS:
        evaluate_model(label, hf, dim, path, texts)
    
    # Aggregate
    rows = []
    for label, _, _, _ in MODELS:
        f = os.path.join(OUT_DIR, f"{label.lower().replace('/', '_').replace('-', '_')}_retrieval.json")
        with open(f) as fh:
            data = json.load(fh)
        rows.append({"model": label, **{k: v["mean"] for k, v in data["aggregated"].items()}})
    
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "posts_retrieval_summary.csv"), index=False)
    print(f"\nSummary saved to {OUT_DIR}/posts_retrieval_summary.csv")
