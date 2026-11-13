"""Build a LARGER locked exam: hard multi-step questions plus easy one-lookup questions.

Why: the fresh 60 swing by about +-12 points by chance, too much to see +3..+8 point changes; and every earlier
question needs 2-3 lookups, while real users mostly ask direct one-lookup questions.

Same recipe as make_questions.py / make_fresh.py (same Groq question writers, same checks, same closed-book filter),
with these differences:
  * seed 2;
  * no article used by the original 200, the fresh 60, or any candidate generated for them;
  * an article may start up to 3 bridge questions (each to a different second article): only 235 unused new
    articles remain, too few for one question each;
  * a new kind, "single": a direct question about one new article, answered from one passage (hops = 1).

Written once and committed BEFORE any setup is run on it; nobody tunes anything on it.

  python data_build/make_exam.py
"""
from __future__ import annotations

import gzip
import json
import queue
import random
import re
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data_build.make_questions as mq  # noqa: E402
from rag.evaluate import normalize  # noqa: E402

SEED = 2
OUT = Path("data/exam")
MAX_PER_A = 3
FINAL = {"bridge": 200, "bridge3": 60, "comparison": 40, "single": 100}  # caps; fewer if not enough survive

SINGLE_PROMPT = """You write questions that a real user might type into a search box, for testing a search system.

ARTICLE "{a}" excerpt:
{a_text}

Write ONE direct, natural question about a specific fact in this excerpt:
- Name the subject ("{a_core}") in the question, the way a user would.
- Ask for one fact that is stated in the excerpt: a name, place, date, number, or short phrase.
- Keep it short and plain, like "When did X open?" or "Who founded X?". No riddles, no chains.
- Never say "article", "document", "excerpt" or "passage".
- The answer must be copied exactly from the excerpt, 1 to 5 words.
If that is not possible with this excerpt, set "skip" to true.
Reply with only JSON: {{"question": "...", "answer": "...", "skip": false}}"""


def bridge_candidates(pool: dict, graph: dict, rng: random.Random) -> list[dict]:
    new = sorted(t for t, d in pool.items() if d["kind"] == "new")
    new_set = set(new)
    famous = lambda t: len(pool[t]["text"].split()) >= 2400  # noqa: E731
    pairs = [(a, b) for a in new for b in graph.get(a, []) if b in pool and b != a]
    rng.shuffle(pairs)
    pairs.sort(key=lambda p: (p[1] not in new_set, famous(p[1])))
    out, used_a = [], {}
    for a, b in pairs:
        if used_a.get(a, 0) >= MAX_PER_A:
            continue
        wa = mq.window(pool[a]["text"], b)
        if not wa:
            continue
        used_a[a] = used_a.get(a, 0) + 1
        out.append({"kind": "bridge", "titles": [a, b], "a_text": wa, "b_text": mq.lead(pool[b]["text"])})
    triples = [x for x in mq.candidates(pool, graph, random.Random(SEED), 0, 10_000, 0) if x["kind"] == "bridge3"]
    return out + triples


def single_candidates(pool: dict, rng: random.Random, per_article: int = 2) -> list[dict]:
    """Excerpts from different parts of each new article (not only the lead), so questions vary."""
    out = []
    for t in sorted(t for t, d in pool.items() if d["kind"] == "new"):
        words = mq.body_words(pool[t]["text"])
        if len(words) < 120:
            continue
        starts = sorted(rng.sample(range(0, max(1, len(words) - 150)), k=min(per_article, max(1, len(words) // 300))))
        for s in starts:
            out.append({"kind": "single", "titles": [t], "a_text": " ".join(words[s:s + 150])})
    rng.shuffle(out)
    return out


_prompt_for, _check = mq.prompt_for, mq.check


def prompt_for(c: dict) -> str:
    if c["kind"] == "single":
        return SINGLE_PROMPT.format(a=c["titles"][0], a_core=mq.core(c["titles"][0]), a_text=c["a_text"])
    return _prompt_for(c)


def check(c: dict, q: dict) -> str | None:
    if c["kind"] != "single":
        return _check(c, q)
    question, answer = str(q.get("question", "")).strip(), str(q.get("answer", "")).strip()
    if q.get("skip") in (True, "true") or not question or not answer:
        return "skipped"
    if not question.endswith("?") or len(question.split()) > 30:
        return "bad_question_shape"
    if re.search(r"\b(article|document|excerpt|passage)s?\b", question, re.I):
        return "meta_reference"
    if len(answer.split()) > 5:
        return "answer_too_long"
    na, nq = normalize(answer), normalize(question)
    if na not in normalize(c["a_text"]):
        return "answer_not_in_last_doc"
    if na in nq:
        return "answer_in_question"
    return None


mq.prompt_for, mq.check = prompt_for, check  # the shared worker looks these up at call time


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open("data/corpus.jsonl.gz", "rt") as f:
        docs = {d["title"]: d for d in map(json.loads, f)}
    with gzip.open("data/graph.json.gz", "rt") as f:
        graph = json.load(f)

    used = set()
    for path, key in [("data/questions.jsonl", "support_titles"), ("data/candidates.jsonl", "titles"),
                      ("data/fresh/questions.jsonl", "support_titles"), ("data/fresh/candidates.jsonl", "titles")]:
        for line in open(path):
            used.update(json.loads(line)[key])
    print(f"excluding {len(used)} articles touched by the original and fresh sets", flush=True)
    pool = {t: d for t, d in docs.items() if t not in used}
    graph = {a: [b for b in bs if b not in used] for a, bs in graph.items() if a not in used}

    rng = random.Random(SEED)
    cands = bridge_candidates(pool, graph, rng) + single_candidates(pool, rng)
    print(f"{len(cands)} candidates: " + ", ".join(
        f"{k}={sum(c['kind'] == k for c in cands)}" for k in ("bridge", "bridge3", "single")), flush=True)

    gen_path = OUT / "candidates.jsonl"
    seen = set()
    if gen_path.exists():
        seen = {(tuple(r["titles"]), r.get("a_text_key")) for r in map(json.loads, gen_path.read_text().splitlines())}
    jobs: queue.Queue = queue.Queue()
    for c in cands:
        c["a_text_key"] = c["a_text"][:60] if c["kind"] == "single" else None
        if (tuple(c["titles"]), c["a_text_key"]) not in seen:
            jobs.put(c)
    print(f"{jobs.qsize()} still to generate", flush=True)

    lock, stats = threading.Lock(), {}

    orig_worker = mq.worker

    def worker(model, jobs_, lock_, out_f, stats_):
        # wrap jobs so each record carries a_text_key: write via a per-job closure
        while True:
            try:
                c = jobs_.get_nowait()
            except queue.Empty:
                return
            one: queue.Queue = queue.Queue()
            one.put(c)
            buf = []

            class Buf:
                def write(self, s):
                    buf.append(s)

                def flush(self):
                    pass

            orig_worker(model, one, lock_, Buf(), stats_)
            if not buf:  # daily limit: the job was put back on `one`
                print(f"[{model}] stopping", flush=True)
                return
            with lock_:
                for s in buf:
                    r = json.loads(s)
                    r["a_text_key"] = c["a_text_key"]
                    r["a_text"] = c["a_text"] if c["kind"] == "single" else None
                    out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                out_f.flush()

    with open(gen_path, "a") as f:
        threads = [threading.Thread(target=worker, args=(m, jobs, lock, f, stats)) for m in mq.GEN_MODELS]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    gens = [json.loads(l) for l in gen_path.read_text().splitlines()]
    valid = [g for g in gens if g["reject"] is None and normalize(mq.core(g["titles"][0])) in normalize(g["question"])]
    uniq, keys = [], set()
    for g in valid:
        if normalize(g["question"]) not in keys:
            keys.add(normalize(g["question"]))
            uniq.append(g)
    print(f"valid after checks: {len(uniq)} / {len(gens)}", flush=True)
    kept = mq.closed_book_filter(uniq, OUT / "closed_book_filter.jsonl")
    kept += mq.comparisons(pool, random.Random(SEED), 200)
    print(f"kept after closed-book filter: {len(kept)}", flush=True)

    rng = random.Random(SEED)
    final = []
    for kind, n in FINAL.items():
        p = [g for g in kept if g["kind"] == kind]
        rng.shuffle(p)
        print(f"{kind}: {len(p)} available, taking {min(n, len(p))}", flush=True)
        final += p[:n]
    rng.shuffle(final)
    hops = {"single": 1, "bridge": 2, "comparison": 2, "bridge3": 3}
    with open(OUT / "questions.jsonl", "w") as f:
        for i, g in enumerate(final):
            f.write(json.dumps({
                "qid": f"x{i:03d}", "question": g["question"], "answer": g["answer"], "aliases": [],
                "type": g["kind"], "hops": hops[g["kind"]], "support_titles": g["titles"],
                "generator": g["generator"], "subset": ["exam"],
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(final)} exam questions", flush=True)


if __name__ == "__main__":
    main()
