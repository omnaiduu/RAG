#!/usr/bin/env bash
# Runs the small-model racers on both Kaggle T4s: each GPU takes the questions whose id % 2 == its index.
# Each process sees only its own GPU. Records already in results/raw/gemma-e2b-*.jsonl are reused (resume).
# Pass 1 = core60 subset, pass 2 = everything else.
set -u
MODEL=${MODEL:-google/gemma-4-E2B-it}
RACERS=${RACERS:-C2,D,C1,A,CB}
TAG=${TAG:-e2b}
OUT=/kaggle/working/results
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Kaggle's image ships TensorFlow/JAX; when a library imports them, XLA pre-grabs most of the GPU.
export USE_TF=0 USE_JAX=0 USE_FLAX=0 TRANSFORMERS_NO_TF=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform TF_FORCE_GPU_ALLOW_GROWTH=true
mkdir -p $OUT
nvidia-smi --query-gpu=name,memory.total --format=csv
for g in 0 1; do
  cat results/raw/gemma-$TAG-*.jsonl 2>/dev/null > $OUT/$TAG-shard$g.jsonl
  ( for subset in core60 ""; do
      CUDA_VISIBLE_DEVICES=$g python -m rag.run --backend gemma --model "$MODEL" --device cuda:0 --racers "$RACERS" --emb-cache data/emb_bge_small_f16.npy \
        --shard $g/2 ${subset:+--subset $subset} --out $OUT/$TAG-shard$g.jsonl
    done ) > $OUT/$TAG-shard$g.log 2>&1 &
done
wait
tail -3 $OUT/*.log
wc -l $OUT/*.jsonl
