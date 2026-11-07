"""Run racers over the question set and write one JSONL record per (racer, question).

  python -m rag.run --racers A,C1,C2,D,CB --backend gemma --model google/gemma-4-E2B-it \
      --device cuda:0 --out results/raw/gemma-e2b.jsonl

Resumable: questions already present in --out for a racer are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

from .racers import RACERS
from .tools import SearchTools, chunk_docs, load_corpus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.jsonl.gz")
    ap.add_argument("--questions", default="data/questions.jsonl")
    ap.add_argument("--racers", default="A,C1,C2,D,CB")
    ap.add_argument("--backend", choices=["gemma", "groq"], default="gemma")
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--device", default="cuda:0", help="device for Gemma and the retrieval models")
    ap.add_argument("--tools-device", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th question starting at i")
    ap.add_argument("--subset", default="", help="only questions whose 'subset' list contains this tag")
    ap.add_argument("--emb-cache", default="", help="npy file to load/save chunk embeddings")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # load the LLM first: it is the biggest allocation on the GPU
    if args.backend == "gemma":
        from .llm import GemmaLLM
        llm = GemmaLLM(args.model, device_map=args.device)
    else:
        from .llm import GroqLLM
        llm = GroqLLM(args.model)

    docs = load_corpus(args.corpus)
    chunks = chunk_docs(docs)
    t0 = time.time()
    emb = None
    if args.emb_cache and os.path.exists(args.emb_cache):
        import numpy as np
        emb = np.load(args.emb_cache)
        if emb.shape[0] != len(chunks):
            emb = None
    tools = SearchTools(chunks, device=args.tools_device or args.device, embeddings=emb)
    if args.emb_cache and emb is None:
        import numpy as np
        np.save(args.emb_cache, tools.emb.astype(np.float16))
    if args.backend == "gemma":
        import torch
        torch.cuda.empty_cache()
    print(f"indexed {len(docs)} docs / {len(chunks)} chunks in {time.time() - t0:.1f}s", flush=True)

    with open(args.questions) as f:
        questions = [json.loads(l) for l in f if l.strip()]
    if args.subset:
        questions = [q for q in questions if args.subset in q.get("subset", [])]
    if args.limit:
        questions = questions[: args.limit]
    si, sn = map(int, args.shard.split("/"))
    questions = [q for q in questions if int(q["qid"][1:]) % sn == si]  # by id, so subsets nest


    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["racer"], r["qid"]))

    with open(out, "a") as f:
        for name in args.racers.split(","):
            racer = RACERS[name](llm, tools)
            t_r = time.time()
            for n, q in enumerate(questions):
                if (name, q["qid"]) in done:
                    continue
                try:
                    rec = racer.solve(q["question"]).record()
                except Exception as exc:  # noqa: BLE001
                    if "daily limit" in str(exc):
                        print(f"STOP: {exc}", flush=True)
                        return
                    rec = {"answer": "unknown", "sources": [], "trace": [], "seen": [], "tool_calls": 0,
                           "tool_seconds": 0.0, "llm": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                                        "seconds": 0.0},
                           "flags": {"error": repr(exc), "tb": traceback.format_exc()[-800:]}}
                rec.update({"racer": name, "qid": q["qid"], "model": args.model, "gold": q["answer"]})
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if (n + 1) % 25 == 0:
                    print(f"{name}: {n + 1}/{len(questions)}  {time.time() - t_r:.0f}s", flush=True)
            print(f"{name}: done in {time.time() - t_r:.0f}s", flush=True)


if __name__ == "__main__":
    main()
