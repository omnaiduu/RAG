#!/usr/bin/env bash
# Groq's free tier allows 200K tokens/day on a rolling window: keep resuming racer B until the subset is done.
SUBSET=${SUBSET:-core60}; RACERS=${RACERS:-B}; MODEL=${MODEL:-openai/gpt-oss-120b}
OUT=${OUT:-results/raw/B-gpt-oss-120b.jsonl}
target=$(python3 -c "import json;print(sum('$SUBSET' in json.loads(l).get('subset',[]) or '$SUBSET'=='' for l in open('data/questions.jsonl')))")
target=$((target * $(echo $RACERS | tr ',' '\n' | wc -l)))
while true; do
  HF_HUB_DISABLE_XET=1 python3 -m rag.run --backend groq --model $MODEL --racers $RACERS --device cpu \
    ${SUBSET:+--subset $SUBSET} --emb-cache data/cache/emb_bge_small.npy --out $OUT 2>&1 | grep -E "STOP|done in|/"
  n=$(wc -l < $OUT)
  echo "$(date -u +%H:%M) $n/$target records"
  [ "$n" -ge "$target" ] && break
  sleep 900
done
