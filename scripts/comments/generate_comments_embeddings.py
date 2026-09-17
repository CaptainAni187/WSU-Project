#!/usr/bin/env python3
"""
Embedding generator for comments corpus (146,715 comments).
Supports configurable max_length, batch_size, chunk_size.
Usage: python generate_comments_embeddings.py --model <name> --output <path> --embed-dim <N> --max-length 512 --batch-size 32
"""
import argparse, os, time, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModel

def generate_embeddings(model_name, output_path, embed_dim, max_length, batch_size, chunk_size):
    print(f"\n{'='*60}")
    print(f"EMBEDDING GENERATION: {model_name}")
    print(f"Dimension: {embed_dim} | Max length: {max_length} | Batch: {batch_size} | Chunk: {chunk_size}")
    print(f"{'='*60}")
    
    DATA_PATH = "data/processed/comments_preprocessed.csv"
    TEXT_COL = "clean_body"
    CACHE_DIR = "model_cache"
    
    df = pd.read_csv(DATA_PATH, usecols=[TEXT_COL])
    texts = df[TEXT_COL].fillna("").astype(str).tolist()
    n = len(texts)
    print(f"Loaded {n:,} comment texts from {DATA_PATH}")
    
    print(f"Loading tokenizer and model...")
    t0_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, cache_dir=CACHE_DIR, trust_remote_code=True)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()
    print(f"Model loaded in {time.time()-t0_load:.1f}s → device: {device}")
    
    # Verify actual dimension
    test = tokenizer("test", return_tensors="pt", max_length=max_length, truncation=True)
    test = {k: v.to(device) for k, v in test.items()}
    with torch.inference_mode():
        actual_dim = model(**test).last_hidden_state.shape[-1]
    if actual_dim != embed_dim:
        print(f"WARNING: Expected dim {embed_dim}, actual {actual_dim}. Using actual.")
        embed_dim = actual_dim
    
    total_start = time.time()
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_path = f"{output_path}.chunk_{start}_{end}.npy"
        
        if os.path.exists(chunk_path):
            print(f"  Chunk {start}-{end} already exists, skipping")
            continue
        
        sub_texts = texts[start:end]
        sub_n = len(sub_texts)
        embeddings = np.zeros((sub_n, embed_dim), dtype=np.float32)
        
        lengths = [
            len(tokenizer.encode(t, add_special_tokens=True, truncation=True, max_length=max_length))
            if isinstance(t, str) and t.strip() else 0
            for t in sub_texts
        ]
        sorted_indices = sorted(range(sub_n), key=lambda idx: lengths[idx])
        sorted_texts = [sub_texts[idx] if isinstance(sub_texts[idx], str) else "" for idx in sorted_indices]
        
        print(f"  Processing chunk {start}-{end} ({sub_n} texts, lengths: {min(lengths)}-{max(lengths)} tokens)...")
        t0 = time.time()
        batch_count = 0
        
        for i in range(0, sub_n, batch_size):
            batch_texts = sorted_texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.inference_mode():
                outputs = model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            for j, emb in enumerate(cls_embeddings):
                orig_idx = sorted_indices[i + j]
                embeddings[orig_idx] = emb
            
            batch_count += 1
            if batch_count % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + len(batch_texts)) / elapsed
                print(f"    Batch {batch_count} | {i+len(batch_texts)}/{sub_n} | {elapsed:.1f}s | {rate:.1f} texts/s")
        
        total_time = time.time() - t0
        print(f"  Chunk done in {total_time:.1f}s | {sub_n/total_time:.1f} texts/s")
        np.save(chunk_path, embeddings)
        print(f"  Saved: {chunk_path} | Shape: {embeddings.shape}")
    
    print(f"\n  Assembling full embeddings...")
    full = np.zeros((n, embed_dim), dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_path = f"{output_path}.chunk_{start}_{end}.npy"
        if not os.path.exists(chunk_path):
            print(f"  ERROR: Missing chunk {chunk_path}")
            return False
        chunk = np.load(chunk_path)
        full[start:end] = chunk
    
    np.save(output_path, full)
    file_size = os.path.getsize(output_path) / (1024**2)
    print(f"  Saved: {output_path} | Shape: {full.shape} | {file_size:.1f} MB")
    
    del model, tokenizer
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    
    total_time = time.time() - total_start
    print(f"  TOTAL: {total_time:.1f}s | {n/total_time:.1f} texts/s")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--embed-dim", type=int, default=768)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=10000)
    args = parser.parse_args()
    
    success = generate_embeddings(args.model, args.output, args.embed_dim, args.max_length, args.batch_size, args.chunk_size)
    import sys
    sys.exit(0 if success else 1)
