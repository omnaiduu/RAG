"""Generate and verify multi-hop questions over the post-cutoff corpus.

Question kinds
  bridge      A (new article) mentions B; the question identifies B only through A and asks a fact from B.
  bridge3     A -> B -> C chain; the answer is a fact from C.
  comparison  two new articles; the answer is one of the two names (or yes/no).

Pipeline
  1. sample candidate pairs/chains from the link graph (seeded, deterministic)
  2. an LLM writes a question + short answer from short excerpts (Groq: qwen3.8-27b and gpt-oss-20b in parallel)
  3. deterministic checks: answer copied from the last-hop excerpt, bridge entity not named in the question,
     answer absent from the first excerpt, length limits
  4. closed-book filter: if either generator-family model answers correctly WITHOUT documents, drop it
     (removes questions answerable from memory; the evaluated models -- gpt-oss-120b and Gemma -- are
     different models, and we report their closed-book scores separately)
  5. pick the final set, stratified by kind, and tag a 60-question core subset

  python data_build/make_questions.py --corpus data/corpus.jsonl.gz --graph data/graph.json.gz --out data/
"""
from __future__ import annotations

import argparse
import gzip
import json
import queue
import random
import re
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.evaluate import correct, normalize  # noqa: E402
from rag.llm import GroqLLM, parse_json  # noqa: E402

GEN_MODELS = ["qwen/qwen3.8-27b", "openai/gpt-oss-20b"]


def opener(p):
    return gzip.open(p, "rt") if str(p).endswith(".gz") else open(p)


def core(title: str) -> str:
    return re.sub(r"\s*\(.*?\)$", "", title).strip()


def body_words(text: str) -> list[str]:
    return [w for line in text.split("\n") if not line.startswith("#") for w in line.split()]


def lead(text: str, n: int = 220) -> str:
    return " ".join(body_words(text)[:n])


def window(text: str, needle: str, n: int = 110) -> str | None:
    words = body_words(text)
    low = [w.lower() for w in words]
    target = core(needle).lower().split()
    for i in range(len(low) - len(target) + 1):
        if [re.sub(r"[^\w]", "", w) for w in low[i:i + len(target)]] == [re.sub(r"[^\w]", "", t) for t in target]:
            return " ".join(words[max(0, i - n): i + len(target) + n])
    return None


BRIDGE_PROMPT = """You write hard 2-hop quiz questions for testing a search system.

DOCUMENT A ("{a}") excerpt:
{a_text}

DOCUMENT B ("{b}") excerpt:
{b_text}

Write ONE question that requires BOTH documents:
- Refer to B only through how A describes it. Do NOT write B's name "{b_core}" in the question.
- Mention A clearly (its name) so a searcher can find A first.
- Ask for a fact stated in B's excerpt and NOT stated in A's excerpt.
- Write it as a natural, self-contained quiz question. Never say "document", "excerpt" or "passage".
- The answer must be copied exactly from B's excerpt, 1 to 5 words (a name, place, year, number...).
If that is not possible with these excerpts, set "skip" to true.
Reply with only JSON: {{"question": "...", "answer": "...", "skip": false}}"""

BRIDGE3_PROMPT = """You write hard 3-hop quiz questions for testing a search system.

DOCUMENT A ("{a}") excerpt:
{a_text}

DOCUMENT B ("{b}") excerpt:
{b_text}

DOCUMENT C ("{c}") excerpt:
{c_text}

A mentions B, and B mentions C. Write ONE question that requires all three documents:
- Mention A by name. Refer to B only through A, and to C only through B.
- Do NOT write "{b_core}" or "{c_core}" in the question.
- Ask for a fact stated in C's excerpt; the answer must be copied exactly from C's excerpt, 1 to 5 words.
- Write it as a natural, self-contained quiz question. Never say "document", "excerpt" or "passage".
If that is not possible with these excerpts, set "skip" to true.
Reply with only JSON: {{"question": "...", "answer": "...", "skip": false}}"""

COMPARE_PROMPT = """You write hard comparison quiz questions for testing a search system.

DOCUMENT A ("{a}") excerpt:
{a_text}

DOCUMENT B ("{b}") excerpt:
{b_text}

Write ONE question comparing A and B on a fact stated in BOTH excerpts (a date, number, size, count...),
such that answering needs both documents. Name both in the question.
The answer must be exactly "{a}" or exactly "{b}" (or "yes"/"no" for a same/different question).
If no fair comparison is possible, set "skip" to true.
Reply with only JSON: {{"question": "...", "answer": "...", "skip": false}}"""


def candidates(docs: dict[str, dict], graph: dict[str, list[str]], rng: random.Random,
               n_bridge: int, n_bridge3: int, n_compare: int) -> list[dict]:
    new = sorted(t for t, d in docs.items() if d["kind"] == "new")
    new_set = set(new)
    pairs, triples = [], []
    for a in new:
        for b in graph.get(a, []):
            if b in docs and b != a:
                pairs.append((a, b))
                for c in graph.get(b, []):
                    if c in docs and c not in (a, b):
                        triples.append((a, b, c))
    rng.shuffle(pairs)
    rng.shuffle(triples)
    famous = lambda t: len(docs[t]["text"].split()) >= 2400  # hit the fetch cap => long, well-known page
    # prefer bridges that are new articles (both hops unseen), then non-famous pages
    pairs.sort(key=lambda p: (p[1] not in new_set, famous(p[1])))
    triples = [t for t in triples if not famous(t[1]) and not famous(t[2])]
    out, used_a = [], {}
    for a, b in pairs:
        if len([x for x in out if x["kind"] == "bridge"]) >= n_bridge:
            break
        if used_a.get(a, 0) >= 1:
            continue
        wa = window(docs[a]["text"], b)
        if not wa:
            continue
        used_a[a] = used_a.get(a, 0) + 1
        out.append({"kind": "bridge", "titles": [a, b], "a_text": wa, "b_text": lead(docs[b]["text"])})
    n3 = 0
    for a, b, c in triples:
        if n3 >= n_bridge3:
            break
        if used_a.get(a, 0) >= 2:
            continue
        wa, wb = window(docs[a]["text"], b, 80), window(docs[b]["text"], c, 80)
        if not wa or not wb:
            continue
        used_a[a] = used_a.get(a, 0) + 1
        n3 += 1
        out.append({"kind": "bridge3", "titles": [a, b, c], "a_text": wa, "b_text": wb,
                    "c_text": lead(docs[c]["text"], 180)})
    return out


BORN = re.compile(r"\(([^()]{0,80})\)")


def birth_year(text: str) -> int | None:
    m = BORN.search(" ".join(body_words(text)[:40]))
    if not m:
        return None
    inner = m.group(1)
    years = re.findall(r"\b(1[5-9]\d\d|20[0-2]\d)\b", inner)
    if years and ("born" in inner.lower() or re.search(r"\d{4}\s*[–-]", inner)):
        return int(years[0])
    return None


def comparisons(docs: dict[str, dict], rng: random.Random, n: int) -> list[dict]:
    """People born in different years -> 'who was born earlier' questions with code-computed answers."""
    people = sorted((t, y) for t, d in docs.items() if d["kind"] == "new" and (y := birth_year(d["text"])))
    rng.shuffle(people)
    forms = ["Who was born earlier, {a} or {b}?", "Which of {a} and {b} is older?", "Was {a} born before {b}?"]
    out = []
    for i in range(0, len(people) - 1, 2):
        if len(out) >= n:
            break
        (a, ya), (b, yb) = people[i], people[i + 1]
        if ya == yb:
            continue
        form = forms[len(out) % 3]
        older = core(a) if ya < yb else core(b)
        answer = ("yes" if ya < yb else "no") if form.startswith("Was") else older
        out.append({"kind": "comparison", "titles": [a, b], "generator": "template",
                    "question": form.format(a=core(a), b=core(b)), "answer": answer, "reject": None,
                    "years": [ya, yb], "tokens": 0})
    return out


def prompt_for(c: dict) -> str:
    t = c["titles"]
    if c["kind"] == "bridge":
        return BRIDGE_PROMPT.format(a=t[0], b=t[1], b_core=core(t[1]), a_text=c["a_text"], b_text=c["b_text"])
    if c["kind"] == "bridge3":
        return BRIDGE3_PROMPT.format(a=t[0], b=t[1], c=t[2], b_core=core(t[1]), c_core=core(t[2]),
                                     a_text=c["a_text"], b_text=c["b_text"], c_text=c["c_text"])
    return COMPARE_PROMPT.format(a=t[0], b=t[1], a_text=c["a_text"], b_text=c["b_text"])


def check(c: dict, q: dict) -> str | None:
    """Deterministic validity checks. Returns a rejection reason or None."""
    question, answer = str(q.get("question", "")).strip(), str(q.get("answer", "")).strip()
    if q.get("skip") in (True, "true") or not question or not answer:
        return "skipped"
    limit = 60 if c["kind"] == "bridge3" else 45
    if not question.endswith("?") or len(question.split()) > limit:
        return "bad_question_shape"
    if re.search(r"\b(document|excerpt|passage)s?\b", question, re.I):
        return "meta_reference"
    na, nq = normalize(answer), normalize(question)
    if c["kind"] == "comparison":
        allowed = {normalize(c["titles"][0]), normalize(c["titles"][1]), "yes", "no"}
        return None if na in allowed else "answer_not_option"
    if len(answer.split()) > 5:
        return "answer_too_long"
    last = c["c_text"] if c["kind"] == "bridge3" else c["b_text"]
    if na not in normalize(last):
        return "answer_not_in_last_doc"
    if na in normalize(c["a_text"]):
        return "answer_in_first_doc"
    if na in nq:
        return "answer_in_question"
    for t in c["titles"][1:]:
        if normalize(core(t)) in nq:
            return "bridge_named"
    return None


def worker(model: str, jobs: "queue.Queue", lock: threading.Lock, out_f, stats: dict) -> None:
    llm = GroqLLM(model, reasoning_effort="none" if "qwen" in model else "low")
    while True:
        try:
            c = jobs.get_nowait()
        except queue.Empty:
            return
        try:
            text, usage = llm.chat([{"role": "user", "content": prompt_for(c)}], max_new_tokens=700, json_mode=True)
            q = parse_json(text) or {}
        except Exception as exc:  # noqa: BLE001
            if "daily limit" in str(exc):
                print(f"[{model}] daily limit reached; worker stops", flush=True)
                jobs.put(c)
                return
            q, usage = {}, None
        reason = check(c, q)
        rec = {**{k: c[k] for k in ("kind", "titles")}, "generator": model, "question": q.get("question"),
               "answer": q.get("answer"), "reject": reason,
               "tokens": (usage.prompt_tokens + usage.completion_tokens) if usage else 0}
        with lock:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            stats[reason or "ok"] = stats.get(reason or "ok", 0) + 1
            n = sum(stats.values())
            if n % 25 == 0:
                print(f"  generated {n}: {stats}", flush=True)


def closed_book_filter(items: list[dict], out_path: Path) -> list[dict]:
    """Ask each generator-family model WITHOUT documents; keep only questions none of them get right."""
    done = {}
    if out_path.exists():
        for l in out_path.read_text().splitlines():
            r = json.loads(l)
            if not any(str(a).startswith("ERROR") for a in r["closed_book"].values()):
                done[r["question"]] = r  # errored checks are redone
    llms = {m: GroqLLM(m, reasoning_effort="none" if "qwen" in m else "low") for m in GEN_MODELS}
    todo = [it for it in items if it["question"] not in done]
    for n, it in enumerate(todo):
        res = {}
        for m, llm in llms.items():
            try:
                text, _ = llm.chat([{"role": "user", "content":
                    'Answer from memory. Reply with only JSON {"answer": "<shortest answer>"}; '
                    f'use "unknown" if unsure.\n\nQUESTION: {it["question"]}'}], max_new_tokens=300, json_mode=True)
                res[m] = str((parse_json(text) or {}).get("answer", "unknown"))
            except Exception as exc:  # noqa: BLE001
                res[m] = f"ERROR {exc}"[:120]
        errored = any(a.startswith("ERROR") for a in res.values())
        rec = {"question": it["question"], "closed_book": res,
               # an unverified question counts as leaked, so it is dropped rather than kept unchecked
               "leaked": errored or any(correct(a, [it["answer"]]) for a in res.values())}
        done[it["question"]] = rec
        with open(out_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if (n + 1) % 50 == 0:
            print(f"  closed-book checked {n + 1}/{len(todo)}", flush=True)
    return [it for it in items if not done.get(it["question"], {}).get("leaked", True)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.jsonl.gz")
    ap.add_argument("--graph", default="data/graph.json.gz")
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-bridge", type=int, default=300)
    ap.add_argument("--n-bridge3", type=int, default=170)
    ap.add_argument("--n-compare", type=int, default=110)
    ap.add_argument("--final", default="bridge:110,bridge3:50,comparison:40")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    with opener(args.corpus) as f:
        docs = {d["title"]: d for d in map(json.loads, f)}
    with opener(args.graph) as f:
        graph = json.load(f)
    rng = random.Random(args.seed)
    cands = candidates(docs, graph, rng, args.n_bridge, args.n_bridge3, args.n_compare)
    print(f"{len(cands)} candidates: " + ", ".join(
        f"{k}={sum(c['kind'] == k for c in cands)}" for k in ("bridge", "bridge3", "comparison")), flush=True)

    gen_path = out / "candidates.jsonl"
    seen = set()
    if gen_path.exists():
        seen = {tuple(json.loads(l)["titles"]) for l in gen_path.read_text().splitlines()}
    jobs: queue.Queue = queue.Queue()
    for c in cands:
        if tuple(c["titles"]) not in seen:
            jobs.put(c)
    lock, stats = threading.Lock(), {}
    with open(gen_path, "a") as f:
        threads = [threading.Thread(target=worker, args=(m, jobs, lock, f, stats)) for m in GEN_MODELS]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    gens = [json.loads(l) for l in gen_path.read_text().splitlines()]
    valid = [g for g in gens if g["reject"] is None]
    # dedupe by normalized question
    uniq, keys = [], set()
    for g in valid:
        k = normalize(g["question"])
        if k not in keys:
            keys.add(k)
            uniq.append(g)
    print(f"valid after checks: {len(uniq)} / {len(gens)}", flush=True)

    # Binary comparison questions skip the closed-book filter: a 50% guess can't be told apart from knowledge.
    kept = closed_book_filter([g for g in uniq if g["kind"] != "comparison"], out / "closed_book_filter.jsonl")
    kept += comparisons(docs, random.Random(args.seed), args.n_compare)
    print(f"kept after closed-book filter: {len(kept)}", flush=True)

    quota = dict(x.split(":") for x in args.final.split(","))
    final = []
    for kind, n in quota.items():
        pool = [g for g in kept if g["kind"] == kind]
        rng.shuffle(pool)
        final += pool[: int(n)]
    rng.shuffle(final)
    core_ids = set()
    per_kind = {k: 0 for k in quota}
    core_quota = {"bridge": 30, "bridge3": 15, "comparison": 15}
    for i, g in enumerate(final):
        if per_kind[g["kind"]] < core_quota.get(g["kind"], 0):
            per_kind[g["kind"]] += 1
            core_ids.add(i)
    with open(out / "questions.jsonl", "w") as f:
        for i, g in enumerate(final):
            f.write(json.dumps({
                "qid": f"q{i:03d}", "question": g["question"], "answer": g["answer"], "aliases": [],
                "type": g["kind"], "hops": 3 if g["kind"] == "bridge3" else 2, "support_titles": g["titles"],
                "generator": g["generator"], "subset": ["core60"] if i in core_ids else [],
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(final)} questions ({len(core_ids)} in core60)", flush=True)


if __name__ == "__main__":
    main()
