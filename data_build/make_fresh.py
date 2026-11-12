"""Build a FRESH held-out question set for an honest re-test of the tuned pipelines.

Same recipe as make_questions.py (same prompts, same checks, same closed-book filter), with three differences:
  * a different seed (1 instead of 0);
  * no article used anywhere in the original 200 questions, or in any candidate generated for them,
    may appear in a fresh question (so no page the pipeline was tuned on is involved);
  * 60 questions in the same mix as the original set (55% bridge, 25% bridge3, 20% comparison), ids f000..f059.

The file is written once and committed BEFORE any racer is run on it; nobody tunes anything on it.

  python data_build/make_fresh.py
"""
from __future__ import annotations

import gzip
import json
import queue
import random
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data_build.make_questions as mq  # noqa: E402
from rag.evaluate import normalize  # noqa: E402

SEED = 1
FINAL = {"bridge": 33, "bridge3": 15, "comparison": 12}
OUT = Path("data/fresh")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open("data/corpus.jsonl.gz", "rt") as f:
        docs = {d["title"]: d for d in map(json.loads, f)}
    with gzip.open("data/graph.json.gz", "rt") as f:
        graph = json.load(f)

    used = set()
    for line in open("data/questions.jsonl"):
        used.update(json.loads(line)["support_titles"])
    for line in open("data/candidates.jsonl"):
        used.update(json.loads(line)["titles"])
    print(f"excluding {len(used)} articles touched by the original set", flush=True)

    pool = {t: d for t, d in docs.items() if t not in used}
    graph = {a: [b for b in bs if b not in used] for a, bs in graph.items() if a not in used}
    cands = mq.candidates(pool, graph, random.Random(SEED), 90, 60, 0)
    print(f"{len(cands)} candidates", flush=True)

    gen_path = OUT / "candidates.jsonl"
    seen = {tuple(json.loads(l)["titles"]) for l in gen_path.read_text().splitlines()} if gen_path.exists() else set()
    jobs: queue.Queue = queue.Queue()
    for c in cands:
        if tuple(c["titles"]) not in seen:
            jobs.put(c)
    lock, stats = threading.Lock(), {}
    with open(gen_path, "a") as f:
        threads = [threading.Thread(target=mq.worker, args=(m, jobs, lock, f, stats)) for m in mq.GEN_MODELS]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    gens = [json.loads(l) for l in gen_path.read_text().splitlines()]
    valid = [g for g in gens if g["reject"] is None
             and normalize(mq.core(g["titles"][0])) in normalize(g["question"])]  # same anchor rule
    uniq, keys = [], set()
    for g in valid:
        if normalize(g["question"]) not in keys:
            keys.add(normalize(g["question"]))
            uniq.append(g)
    print(f"valid after checks: {len(uniq)} / {len(gens)}", flush=True)
    kept = mq.closed_book_filter(uniq, OUT / "closed_book_filter.jsonl")
    kept += mq.comparisons(pool, random.Random(SEED), 40)
    print(f"kept after closed-book filter: {len(kept)}", flush=True)

    rng = random.Random(SEED)
    final = []
    for kind, n in FINAL.items():
        p = [g for g in kept if g["kind"] == kind]
        rng.shuffle(p)
        if len(p) < n:
            print(f"WARNING: only {len(p)} {kind} questions available (wanted {n})", flush=True)
        final += p[:n]
    rng.shuffle(final)
    with open(OUT / "questions.jsonl", "w") as f:
        for i, g in enumerate(final):
            f.write(json.dumps({
                "qid": f"f{i:03d}", "question": g["question"], "answer": g["answer"], "aliases": [],
                "type": g["kind"], "hops": 3 if g["kind"] == "bridge3" else 2, "support_titles": g["titles"],
                "generator": g["generator"], "subset": ["fresh60"],
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(final)} fresh questions", flush=True)


if __name__ == "__main__":
    main()
