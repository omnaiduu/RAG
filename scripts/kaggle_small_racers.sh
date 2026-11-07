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
mkdir -p $OUT
nvidia-smi --query-gpu=name,memory.total --format=csv
for g in 0 1; do
  cat results/raw/gemma-$TAG-*.jsonl 2>/dev/null > $OUT/$TAG-shard$g.jsonl
  ( for subset in core60 ""; do
      CUDA_VISIBLE_DEVICES=$g python -m rag.run --backend gemma --model "$MODEL" --device cuda:0 --racers "$RACERS" \
        --shard $g/2 ${subset:+--subset $subset} --out $OUT/$TAG-shard$g.jsonl
    done ) > $OUT/$TAG-shard$g.log 2>&1 &
done
wait
tail -3 $OUT/*.log
wc -l $OUT/*.jsonl
