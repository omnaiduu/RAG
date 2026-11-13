"""Router simulation on the exam, offline: combine a quick-path run (Q) and a full-chain run (C1v5) per question.

  python scripts/router_sim.py [quick_tag] [full_tag]     # defaults: q-e4b v5-e4b; writes results/router_sim.md

Rules compared (accuracy, and cost as model calls / prompt tokens per question):
  full only            every question takes the full step-by-step chain
  quick only           every question takes one search + one answer
  quick, then full     quick path first; if it flags itself unsure (unknown / not in the passages), run the full chain
                       too (cost = both)
  perfect router       easy questions -> quick, all others -> full (an upper bound: needs knowing the type)
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")
from rag.evaluate import correct, golds  # noqa: E402


def load(tag: str) -> dict:
    return {r["qid"]: r for r in map(json.loads, open(f"results/raw/vllm-{tag}-exam.jsonl"))}


def main() -> None:
    qt, ft = (sys.argv[1:3] + ["q-e4b", "v5-e4b"][len(sys.argv[1:3]):])[:2]
    gold = {q["qid"]: q for q in map(json.loads, open("data/exam/questions.jsonl"))}
    quick, full = load(qt), load(ft)
    ids = [q for q in gold if q in quick and q in full]
    ok = lambda r: correct(r["answer"], golds(gold[r["qid"]]))  # noqa: E731
    calls = lambda r: r["llm"].get("calls", 0)  # noqa: E731
    ptok = lambda r: r["llm"].get("prompt_tokens", 0)  # noqa: E731
    unsure = lambda r: bool((r.get("flags") or {}).get("quick_unsure"))  # noqa: E731

    def rule(name, pick):
        rows = {}
        for q in ids:
            right, c, t = pick(q)
            typ = "easy" if gold[q]["type"] == "single" else "hard"
            for k in (typ, "all"):
                a = rows.setdefault(k, [0, 0, 0, 0])
                a[0] += right; a[1] += 1; a[2] += c; a[3] += t
        return name, rows

    def quick_then_full(q):
        if unsure(quick[q]):
            return ok(full[q]), calls(quick[q]) + calls(full[q]), ptok(quick[q]) + ptok(full[q])
        return ok(quick[q]), calls(quick[q]), ptok(quick[q])

    results = [
        rule("Full chain only", lambda q: (ok(full[q]), calls(full[q]), ptok(full[q]))),
        rule("Quick path only", lambda q: (ok(quick[q]), calls(quick[q]), ptok(quick[q]))),
        rule("Quick first, full chain if unsure", quick_then_full),
        rule("Perfect router (easy -> quick, hard -> full)",
             lambda q: (ok(quick[q]), calls(quick[q]), ptok(quick[q])) if gold[q]["type"] == "single"
             else (ok(full[q]), calls(full[q]), ptok(full[q]))),
    ]
    n_unsure = sum(unsure(quick[q]) for q in ids)
    out = ["| Rule | Easy (one lookup) | Hard (chains) | All | Model calls / question | Prompt tokens / question |",
           "|---|---|---|---|---|---|"]
    for name, rows in results:
        e, h, a = rows.get("easy", [0, 1, 0, 0]), rows.get("hard", [0, 1, 0, 0]), rows["all"]
        out.append(f"| {name} | {e[0]}/{e[1]} ({e[0] / e[1]:.0%}) | {h[0]}/{h[1]} ({h[0] / h[1]:.0%}) "
                   f"| **{a[0]}/{a[1]} ({a[0] / a[1]:.0%})** | {a[2] / a[1]:.1f} | {a[3] / a[1]:.0f} |")
    out.append(f"\nQuick path flagged itself unsure on {n_unsure}/{len(ids)} questions "
               f"({sum(unsure(quick[q]) for q in ids if gold[q]['type'] == 'single')} easy, "
               f"{sum(unsure(quick[q]) for q in ids if gold[q]['type'] != 'single')} hard).")
    by_type = {}
    for q in ids:
        t = gold[q]["type"]
        a = by_type.setdefault(t, [0, 0, 0])
        a[0] += ok(quick[q]); a[1] += ok(full[q]); a[2] += 1
    out.append("\n| Question type | Quick path | Full chain |\n|---|---|---|")
    for t, (a, b, n) in sorted(by_type.items()):
        out.append(f"| {t} ({n}) | {a} ({a / n:.0%}) | {b} ({b / n:.0%}) |")
    text = "\n".join(out)
    print(text)
    open("results/router_sim.md", "w").write("# Router simulation on the exam\n\n" + __doc__.split("\n\n", 1)[1] + "\n" + text + "\n")


if __name__ == "__main__":
    main()
