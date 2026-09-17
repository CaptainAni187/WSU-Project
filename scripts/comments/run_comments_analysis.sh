#!/bin/bash
# Full downstream comments analysis: BERTopic + coherence + retrieval + stability
# for all 5 models, then aggregate the benchmark. All steps skip-if-done (resumable).
set -e
cd "$(dirname "$0")/../.."   # always run from the repo root
PY="${PYTHON:-.venv/bin/python}"   # override with: PYTHON=python bash ...
E="data/embeddings/comments"

# model display-name : embedding-file-slug   (display name must match aggregate MODELS + retrieval MODEL_IDS)
PAIRS=(
  "MuRIL:muril"
  "MentalBERT:mentalbert"
  "HingBERT-Mixed-v2:hingbert_mixed_v2"
  "HingRoBERTa-Mixed:hingroberta_mixed"
  "BGE-m3:bge_m3"
)

for P in "${PAIRS[@]}"; do
  NAME="${P%%:*}"; SLUG="${P##*:}"
  EMB="$E/${SLUG}_embeddings.npy"
  echo "======================================================================"
  echo ">>> $NAME  ($EMB)"
  echo "======================================================================"
  echo "--- [1/4] BERTopic ---"
  $PY src/analysis/run_comments_bertopic.py --model-name "$NAME" --embedding-path "$EMB"
  echo "--- [2/4] Coherence ---"
  $PY src/analysis/comments_coherence.py --model "$NAME"
  echo "--- [3/4] Retrieval ---"
  $PY src/analysis/comments_retrieval.py --model "$NAME" --embedding-path "$EMB"
  echo "--- [4/4] Stability ---"
  $PY src/analysis/comments_stability.py --model "$NAME" --embedding-path "$EMB"
done

echo "======================================================================"
echo ">>> Aggregating benchmark"
echo "======================================================================"
$PY src/analysis/aggregate_comments_benchmark.py
echo "ALL DONE."
