#!/bin/bash
# Run all 5 post embedding models with 256 tokens locally
# No timeout issues — just run and walk away
# Expected total runtime: ~4-5 hours on Apple M3

set -e

cd "$(dirname "$0")/../.."   # always run from the repo root

PYTHON="${PYTHON:-.venv/bin/python}"   # override with: PYTHON=python bash ...
SCRIPT="scripts/posts/generate_posts_embeddings.py"
OUTDIR="data/embeddings/posts"

mkdir -p "$OUTDIR"

echo "========================================================================"
echo "POSTS EMBEDDING GENERATION — 256 tokens, 70,425 posts"
echo "========================================================================"
echo ""
echo "This will run 5 models sequentially. Each model has checkpointing,"
echo "so if your machine sleeps or you Ctrl+C, just re-run this script."
echo ""
echo "Estimated runtime on Apple M3, 16GB RAM:"
echo "  - MuRIL, MentalBERT, HingBERT-Mixed-v2, HingRoBERTa-Mixed: ~30-40 min each"
echo "  - BGE-m3 (XLM-R large, 24 layers, 1024 dim): ~2-3 hours"
echo "  - Total: ~4-5 hours"
echo ""
echo "Starting in 5 seconds... (Ctrl+C to cancel)"
sleep 5

echo ""
echo "[1/5] MuRIL (google/muril-base-cased, 768 dim, batch=64, max_length=256)"
$PYTHON $SCRIPT --model google/muril-base-cased --short-name muril --dim 768 --batch-size 64 --max-length 256

echo ""
echo "[2/5] MentalBERT (mental/mental-bert-base-uncased, 768 dim, batch=64, max_length=256)"
$PYTHON $SCRIPT --model mental/mental-bert-base-uncased --short-name mentalbert --dim 768 --batch-size 64 --max-length 256

echo ""
echo "[3/5] HingBERT-Mixed-v2 (l3cube-pune/hingbert-mixed-v2, 768 dim, batch=64, max_length=256)"
$PYTHON $SCRIPT --model l3cube-pune/hingbert-mixed-v2 --short-name hingbert_mixed_v2 --dim 768 --batch-size 64 --max-length 256

echo ""
echo "[4/5] HingRoBERTa-Mixed (l3cube-pune/hing-roberta-mixed, 768 dim, batch=64, max_length=256)"
$PYTHON $SCRIPT --model l3cube-pune/hing-roberta-mixed --short-name hingroberta_mixed --dim 768 --batch-size 64 --max-length 256

echo ""
echo "[5/5] BGE-m3 (BAAI/bge-m3, 1024 dim, batch=32, max_length=256)"
$PYTHON $SCRIPT --model BAAI/bge-m3 --short-name bge_m3 --dim 1024 --batch-size 32 --max-length 256

echo ""
echo "========================================================================"
echo "ALL 5 POST EMBEDDING MODELS COMPLETE!"
echo "========================================================================"
echo ""
ls -lh "$OUTDIR"

echo ""
echo "Next steps after this finishes:"
echo "  1. Run BERTopic on each embedding set"
echo "  2. Compute coherence, stability, retrieval metrics"
echo "  3. Generate publication figures"
echo ""
