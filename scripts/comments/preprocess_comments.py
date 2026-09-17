#!/usr/bin/env python3
"""
Comments preprocessing — mirrors the posts pipeline (src/pipeline/run_preprocessing.py)
so comment text is cleaned identically to posts, enabling a fair cross-corpus comparison.

Input : data/raw/reddit_comments.csv   (columns include: comment_id, post_id, topic_id,
        topic_label, distress_type, parent_id, author, score, created_utc, body)
Output: comments_preprocessed.csv with a `clean_body` column, written to BOTH locations
        the downstream scripts expect:
          - data/processed/comments_preprocessed.csv              (embedding generator)
          - final_outputs/comments_analysis/comments_preprocessed.csv (bertopic/coherence/
                                                                     retrieval/taxonomy)

Cleaning chain (identical to posts):
  ArtifactRemover(aggressive_mode=True).clean  ->  normalize_text  ->  preserve_phrases

Filtering:
  - drop [deleted] / [removed] / automod bodies
  - drop rows whose clean_body is empty or shorter than --min-chars (default 10)

Usage: python preprocess_comments.py            # full run
       python preprocess_comments.py --sample 2000   # quick smoke test
"""
import argparse, os, sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]  # scripts/comments/ -> repo root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.chdir(project_root)  # relative data/ and final_outputs/ paths resolve from the repo root

import pandas as pd

from src.preprocessing.artifact_remover import ArtifactRemover
from src.preprocessing.expanded_hinglish_dict import normalize_text
from src.preprocessing.phrase_tokenizer import preserve_phrases

RAW = "data/raw/reddit_comments.csv"
OUT_PATHS = [
    "data/processed/comments_preprocessed.csv",
    "final_outputs/comments_analysis/comments_preprocessed.csv",
]
KEEP_META = ["comment_id", "post_id", "topic_id", "topic_label",
             "distress_type", "parent_id", "author", "score", "created_utc"]
DROP_BODIES = {"[deleted]", "[removed]", "[ Removed by Reddit ]"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-chars", type=int, default=10,
                    help="Drop comments whose clean_body is shorter than this (default 10).")
    ap.add_argument("--sample", type=int, default=0,
                    help="Only process the first N rows (smoke test). 0 = full corpus.")
    args = ap.parse_args()

    print(f"Loading {RAW} ...")
    df = pd.read_csv(RAW, low_memory=False)
    if args.sample:
        df = df.head(args.sample).copy()
    n0 = len(df)
    print(f"  {n0:,} raw comments")

    # 1. Drop deleted/removed/automod
    body = df["body"].fillna("").astype(str)
    mask_del = body.str.strip().isin(DROP_BODIES)
    mask_auto = body.str.lower().str.contains("i am a bot|automoderator", regex=True)
    df = df[~(mask_del | mask_auto)].copy()
    print(f"  removed {int((mask_del | mask_auto).sum()):,} deleted/removed/automod comments")

    # 2. Clean -> normalize -> preserve phrases  (same chain as posts)
    import time
    remover = ArtifactRemover(verbose=False, aggressive_mode=True)

    def apply_with_progress(series, fn, label):
        """Single-threaded apply with a heartbeat so it's clearly not frozen."""
        n = len(series)
        vals = series.tolist()
        out = [None] * n
        t0 = time.time()
        step = max(5000, n // 20)
        print(f"  {label} ... ({n:,} rows)", flush=True)
        for i, v in enumerate(vals):
            out[i] = fn(v)
            if (i + 1) % step == 0 or i + 1 == n:
                el = time.time() - t0
                rate = (i + 1) / el
                eta = (n - i - 1) / rate if rate else 0
                print(f"    {i+1:,}/{n:,}  ({rate:,.0f}/s, ETA {eta:4.0f}s)", flush=True)
        return out

    body = df["body"].fillna("").astype(str)
    cleaned = apply_with_progress(body, remover.clean, "cleaning (ArtifactRemover)")
    normalized = apply_with_progress(pd.Series(cleaned), normalize_text, "normalizing (Hinglish)")
    df["clean_body"] = apply_with_progress(pd.Series(normalized), preserve_phrases, "preserving community phrases")

    # 3. Length filter
    L = df["clean_body"].str.len()
    df = df[L >= args.min_chars].copy()
    print(f"  kept {len(df):,} comments after >= {args.min_chars}-char filter "
          f"({n0 - len(df):,} dropped total)")

    # 4. Output columns: metadata that exists + clean_body
    cols = [c for c in KEEP_META if c in df.columns] + ["clean_body"]
    out = df[cols]

    for p in OUT_PATHS:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        out.to_csv(p, index=False)
        print(f"  wrote {p}  ({len(out):,} rows, {os.path.getsize(p)/1e6:.1f} MB)")

    print("DONE.")


if __name__ == "__main__":
    main()
