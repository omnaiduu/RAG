#!/usr/bin/env bash
# Runs the small-model racers on both Kaggle T4s: each GPU takes half the questions.
# Pass 1 = core60 subset (so it finishes first), pass 2 = everything else.
set -u
MODEL=${MODEL:-google/gemma-4-E2B-it}
RACERS=${RACERS:-C2,C1,A,CB,D}
TAG=${TAG:-e2b}
OUT=/kaggle/working/results
mkdir -p $OUT
nvidia-smi --query-gpu=name,memory.total --format=csv
for g in 0 1; do
  ( for subset in core60 ""; do
      python -m rag.run --backend gemma --model "$MODEL" --device cuda:$g --racers "$RACERS" \
        --shard $g/2 ${subset:+--subset $subset} --out $OUT/$TAG-shard$g.jsonl
    done ) > $OUT/$TAG-shard$g.log 2>&1 &
done
wait
tail -3 $OUT/*.log
wc -l $OUT/*.jsonl
