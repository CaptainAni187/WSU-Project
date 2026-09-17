#!/usr/bin/env python3
"""Retrieval evaluation for comments corpus."""
import argparse, json, os, re, warnings, time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize
warnings.filterwarnings('ignore')

MODEL_IDS = {
    "MuRIL": "google/muril-base-cased",
    "HingBERT": "l3cube-pune/hing-bert",
    "HingBERT-Mixed-v2": "l3cube-pune/hingbert-mixed-v2",
    "HingRoBERTa-Mixed": "l3cube-pune/hing-roberta-mixed",
    "MentalBERT": "mental/mental-bert-base-uncased",
    "BGE-m3": "BAAI/bge-m3",
}
QUERIES = [
    "jee burnout exam stress unable to study",
    "nta ki mkc unfair exam result frustration",
    "drop year regret feeling wasted my life",
    "placement anxiety no job offers tier 3 college",
    "suicidal after jee feel like dying no hope",
    "life barbaad sab khatam failed everything",
]
RELEVANCE_KEYWORDS = {
    "jee burnout exam stress unable to study": ["jee", "burnout", "stress", "study", "padh", "exam", "tired", "exhausted", "thak"],
    "nta ki mkc unfair exam result frustration": ["nta", "mkc", "unfair", "result", "frustration", "answer", "key"],
    "drop year regret feeling wasted my life": ["drop", "year", "regret", "waste", "life", "barbaad", "khatam"],
    "placement anxiety no job offers tier 3 college": ["placement", "job", "offer", "tier", "anxiety", "package", "lpa", "unplaced"],
    "suicidal after jee feel like dying no hope": ["suicide", "die", "kill", "hope", "mar", "jeene", "zindagi"],
    "life barbaad sab khatam failed everything": ["barbaad", "khatam", "fail", "sab", "life", "ruined", "destroyed"],
}

def encode_query(tokenizer, model, query):
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--embedding-path', required=True)
    parser.add_argument('--output-dir', default='final_outputs/comments_benchmark')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    model_slug = args.model.lower().replace('-', '_').replace('/', '_')
    out_path = f"{args.output_dir}/{model_slug}_retrieval.json"
    if os.path.exists(out_path):
        print(f"Already done: {out_path}")
        return
    
    model_id = MODEL_IDS[args.model]
    
    print(f"[1] Loading {args.model} ({model_id})...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()
    print(f"    Loaded in {time.time()-t0:.1f}s")
    
    print(f"[2] Loading corpus and embeddings...")
    df_text = pd.read_csv("final_outputs/comments_analysis/comments_preprocessed.csv")
    texts = df_text["clean_body"].fillna("").tolist()
    embeddings = np.load(args.embedding_path).astype(np.float32)
    embeddings_norm = normalize(embeddings, norm='l2', axis=1)
    print(f"    Corpus: {len(texts):,} docs, embeddings: {embeddings.shape}")
    
    results = []
    print(f"[3] Processing {len(QUERIES)} queries...")
    for query in QUERIES:
        t0 = time.time()
        q_emb = encode_query(tokenizer, model, query)
        q_emb = q_emb.reshape(1, -1)
        q_emb = normalize(q_emb, norm='l2')[0]
        
        similarities = embeddings_norm @ q_emb
        top10_idx = np.argsort(similarities)[::-1][:10]
        top10_docs = [str(texts[i])[:150] for i in top10_idx]
        top10_scores = [float(similarities[i]) for i in top10_idx]
        
        keywords = RELEVANCE_KEYWORDS[query]
        relevance_scores = []
        for doc in top10_docs:
            doc_lower = doc.lower()
            matches = sum(1 for kw in keywords if kw in doc_lower)
            relevance_scores.append(matches / len(keywords))
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        
        results.append({
            'query': query,
            'top1_doc': top10_docs[0],
            'top1_score': round(top10_scores[0], 4),
            'top1_relevance': round(relevance_scores[0], 3),
            'avg_relevance': round(avg_relevance, 3),
            'top10_scores': [round(s, 4) for s in top10_scores],
        })
        print(f"    '{query[:40]}...': {time.time()-t0:.1f}s, avg_relevance={avg_relevance:.3f}")
    
    with open(out_path, 'w') as f:
        json.dump({'model': args.model, 'results': results}, f, indent=2)
    print(f"[4] Saved to {out_path}")

if __name__ == '__main__':
    main()
