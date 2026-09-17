#!/bin/bash
# Generate all 5 comment embedding models at 256 tokens (same rate as posts).
# Sequential (not parallel) to avoid MPS out-of-memory. Each model checkpoints per
# 10k-comment chunk, so if the machine sleeps or you Ctrl+C, just re-run this script.
#
# Usage:  bash scripts/comments/run_comments_embeddings_256.sh   (from repo root)
# Monitor: tail -f logs/comments_*_256.log   (in another terminal)

set -e
cd "$(dirname "$0")/../.."   # always run from the repo root
mkdir -p logs

PYTHON="${PYTHON:-.venv/bin/python}"   # override with: PYTHON=python bash ...
SCRIPT="scripts/comments/generate_comments_embeddings.py"
OUTDIR="data/embeddings/comments"
mkdir -p "$OUTDIR"

echo "========================================================================"
echo "COMMENTS EMBEDDING GENERATION - 256 tokens"
echo "5 models, sequential, checkpointed. Est. ~4-6 h total on Apple M-series."
echo "========================================================================"

echo ""
echo "[1/5] MuRIL (google/muril-base-cased, 768d, batch=64, max_length=256)"
$PYTHON $SCRIPT --model google/muril-base-cased \
  --output $OUTDIR/muril_embeddings.npy \
  --embed-dim 768 --max-length 256 --batch-size 64 --chunk-size 10000 \
  2>&1 | tee logs/comments_muril_256.log

echo ""
echo "[2/5] MentalBERT (mental/mental-bert-base-uncased, 768d, batch=64, max_length=256)"
$PYTHON $SCRIPT --model mental/mental-bert-base-uncased \
  --output $OUTDIR/mentalbert_embeddings.npy \
  --embed-dim 768 --max-length 256 --batch-size 64 --chunk-size 10000 \
  2>&1 | tee logs/comments_mentalbert_256.log

echo ""
echo "[3/5] HingBERT-Mixed-v2 (l3cube-pune/hingbert-mixed-v2, 768d, batch=64, max_length=256)"
$PYTHON $SCRIPT --model l3cube-pune/hingbert-mixed-v2 \
  --output $OUTDIR/hingbert_mixed_v2_embeddings.npy \
  --embed-dim 768 --max-length 256 --batch-size 64 --chunk-size 10000 \
  2>&1 | tee logs/comments_hingbert_256.log

echo ""
echo "[4/5] HingRoBERTa-Mixed (l3cube-pune/hing-roberta-mixed, 768d, batch=64, max_length=256)"
$PYTHON $SCRIPT --model l3cube-pune/hing-roberta-mixed \
  --output $OUTDIR/hingroberta_mixed_embeddings.npy \
  --embed-dim 768 --max-length 256 --batch-size 64 --chunk-size 10000 \
  2>&1 | tee logs/comments_hingroberta_256.log

echo ""
echo "[5/5] BGE-m3 (BAAI/bge-m3, 1024d, batch=32, max_length=256)"
$PYTHON $SCRIPT --model BAAI/bge-m3 \
  --output $OUTDIR/bge_m3_embeddings.npy \
  --embed-dim 1024 --max-length 256 --batch-size 32 --chunk-size 10000 \
  2>&1 | tee logs/comments_bge_m3_256.log

echo ""
echo "========================================================================"
echo "ALL 5 COMMENT EMBEDDING MODELS COMPLETE (256 tokens)"
echo "========================================================================"
ls -lh $OUTDIR/*_embeddings.npy
