#!/usr/bin/env python3
"""
Embedding generator for posts corpus (merged, 70k+ posts).
Adapted from generate_comments_embeddings.py for posts-specific paths.
Usage: python generate_posts_embeddings.py --model <name> --short-name <name> --dim <N>
"""
import argparse, os, time, json, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModel

BATCH_SIZE = 64
MAX_LENGTH = 256
CHUNK_SIZE = 10000
DATA_PATH = "data/processed/posts_merged_final_preprocessed.csv"
TEXT_COL = "cleaned_text"
OUTPUT_DIR = "data/embeddings/posts"
CACHE_DIR = "model_cache"

def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def generate_embeddings(model_name, short_name, expected_dim, chunk_size=CHUNK_SIZE, batch_size=BATCH_SIZE, max_length=MAX_LENGTH):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    output_path = os.path.join(OUTPUT_DIR, f"{short_name}_embeddings.npy")
    
    print(f"\n{'='*60}")
    print(f"EMBEDDING GENERATION: {short_name}")
    print(f"Model: {model_name}")
    print(f"Expected dim: {expected_dim}")
    print(f"Max length: {max_length}")
    print(f"Batch size: {batch_size}")
    print(f"{'='*60}")
    
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, usecols=[TEXT_COL])
    texts = df[TEXT_COL].fillna("").tolist()
    n = len(texts)
    print(f"Loaded {n:,} post texts")
    
    print(f"Loading tokenizer and model...")
    t0_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, cache_dir=CACHE_DIR, trust_remote_code=True)
    device = get_device()
    model.to(device)
    model.eval()
    print(f"Model loaded in {time.time()-t0_load:.1f}s → device: {device}")
    
    # Verify dimension
    test_inputs = tokenizer("test", return_tensors="pt", max_length=max_length, truncation=True)
    test_inputs = {k: v.to(device) for k, v in test_inputs.items()}
    with torch.inference_mode():
        test_out = model(**test_inputs)
    actual_dim = test_out.last_hidden_state.shape[-1]
    print(f"Actual embedding dimension: {actual_dim}")
    if actual_dim != expected_dim:
        print(f"WARNING: Dimension mismatch! Expected {expected_dim}, got {actual_dim}. Using actual.")
    
    total_start = time.time()
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_path = os.path.join(OUTPUT_DIR, f"{short_name}_chunk_{start}_{end}.npy")
        
        if os.path.exists(chunk_path):
            print(f"  Chunk {start}-{end} already exists, skipping")
            continue
        
        sub_texts = texts[start:end]
        sub_n = len(sub_texts)
        embeddings = np.zeros((sub_n, actual_dim), dtype=np.float32)
        
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
            if batch_count % 50 == 0:
                elapsed = time.time() - t0
                rate = (i + len(batch_texts)) / elapsed
                print(f"    Batch {batch_count} | {i+len(batch_texts)}/{sub_n} | {elapsed:.1f}s | {rate:.1f} texts/s")
        
        total_time = time.time() - t0
        print(f"  Chunk done in {total_time:.1f}s | {sub_n/total_time:.1f} texts/s")
        np.save(chunk_path, embeddings)
        print(f"  Saved: {chunk_path} | Shape: {embeddings.shape}")
    
    # Assemble
    print(f"\n  Assembling full embeddings...")
    full = np.zeros((n, actual_dim), dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_path = os.path.join(OUTPUT_DIR, f"{short_name}_chunk_{start}_{end}.npy")
        if not os.path.exists(chunk_path):
            print(f"  ERROR: Missing chunk {chunk_path}")
            return False
        chunk = np.load(chunk_path)
        full[start:end] = chunk
    
    np.save(output_path, full)
    file_size = os.path.getsize(output_path) / (1024**2)
    print(f"  Saved full: {output_path} | Shape: {full.shape} | {file_size:.2f} MB")
    
    # Cleanup
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
    parser.add_argument("--short-name", required=True)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    args = parser.parse_args()
    
    success = generate_embeddings(args.model, args.short_name, args.dim, chunk_size=args.chunk_size, batch_size=args.batch_size, max_length=args.max_length)
    import sys
    sys.exit(0 if success else 1)
