"""Analyses quoted in docs/ENGINEERING_NOTES.md, reproducible from results/raw/.

  python scripts/analyze.py            # prints all sections, writes results/analysis.txt

Sections
  1. head-to-head on the questions racer B (gpt-oss-120b) has answered
  2. 2B vs 4B with identical code (C1 v3): paired fixes/breaks, sign test, behaviour counts
  3. confidence flags and simulated routing (small model first, flagged questions to the 120B)
"""
from __future__ import annotations

import collections
import glob
import json
import math
import sys

sys.path.insert(0, ".")
from rag.evaluate import correct  # noqa: E402
from rag.racers import GIVE_UP  # noqa: E402
from rag.tools import chunk_docs, load_corpus  # noqa: E402

UNSURE_FLAGS = ("ungrounded_final", "bridge_rejected", "ungrounded_step", "shape_retry", "forced_answer")


def main() -> None:
    qs = {q["qid"]: q for q in map(json.loads, open("data/questions.jsonl"))}
    runs: dict[tuple[str, str], dict[str, dict]] = collections.defaultdict(dict)
    for f in glob.glob("results/raw/*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            if not r.get("error"):
                runs[(r["racer"], r["model"].split("/")[-1])][r["qid"]] = r

    def ok(r: dict) -> bool:
        q = qs[r["qid"]]
        return correct(r["answer"], [q["answer"], *q.get("aliases", [])])

    out: list[str] = []
    p = out.append

    # 1. head-to-head --------------------------------------------------------
    b = runs[("B", "gpt-oss-120b")]
    bq = sorted(b)
    p(f"## 1. Head-to-head on the {len(bq)} questions racer B has answered\n")
    for key in sorted(runs):
        if set(bq) <= set(runs[key]):
            n = sum(ok(runs[key][i]) for i in bq)
            p(f"{key[0]:12s} {key[1]:16s} {n:3d}/{len(bq)} = {n / len(bq):.1%}")

    # 2. 2B vs 4B, same code -------------------------------------------------
    a, c = runs[("C1v3@lance", "gemma-4-E2B-it")], runs[("C1v3@lance", "gemma-4-E4B-it")]
    fixes = [i for i in a if ok(c[i]) and not ok(a[i])]
    breaks = [i for i in a if ok(a[i]) and not ok(c[i])]
    n, k = len(fixes) + len(breaks), len(fixes)
    pval = min(1.0, 2 * sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n)
    p("\n## 2. C1 v3 with 2B vs 4B (identical code)\n")
    p(f"4B fixes {len(fixes)}, 4B breaks {len(breaks)}, both right {sum(ok(a[i]) and ok(c[i]) for i in a)}, "
      f"both wrong {sum(not ok(a[i]) and not ok(c[i]) for i in a)}; two-sided sign test p = {pval:.3f}")
    title = {ch.id: ch.title for ch in chunk_docs(load_corpus("data/corpus.jsonl.gz"))}

    def behaviour(rs: dict[str, dict]) -> collections.Counter:
        st: collections.Counter = collections.Counter()
        for qid, r in rs.items():
            plan = next((t["out"] for t in r["trace"] if t.get("step") == "plan"), {})
            steps = plan.get("steps", []) if isinstance(plan, dict) else []
            st["plans linking steps with #1"] += any("#" in str(s) for s in steps)
            st["step answers 'unknown'"] += sum(1 for t in r["trace"] if str(t.get("step", "")).startswith("step")
                                                and str(t["out"].get("answer", "")).lower() == "unknown")
            seen = {title.get(x) for x in r["seen"]}
            sup = qs[qid]["support_titles"]
            st["found first needed page"] += bool(sup) and sup[0] in seen
            st["found every needed page"] += all(t in seen for t in sup)
            for f in ("shape_retry", "bridge_rejected"):
                st[f"flag {f}"] += f in r["flags"]
        return st

    sa, sc = behaviour(a), behaviour(c)
    for key in sorted(sa | sc):
        p(f"  {key:30s} 2B {sa[key]:4d}   4B {sc[key]:4d}")

    # 3. routing -------------------------------------------------------------
    p("\n## 3. Confidence flags and simulated routing (flagged -> 120B)\n")
    for key, label in [(("C1v4@lance", "gemma-4-E2B-it"), "2B C1 v4"), (("C1v3@lance", "gemma-4-E4B-it"), "4B C1 v3")]:
        rs = runs[key]

        def unsure(r: dict) -> bool:
            return any(f in r["flags"] for f in UNSURE_FLAGS) or bool(GIVE_UP.search(r["answer"]))

        u = [i for i in rs if unsure(rs[i])]
        s = [i for i in rs if not unsure(rs[i])]
        p(f"{label}: no flag {len(s)} q -> {sum(ok(rs[i]) for i in s) / len(s):.0%} right; "
          f"flagged {len(u)} q -> {sum(ok(rs[i]) for i in u) / len(u):.0%} right")
        routed = sum(ok(b[i]) if unsure(rs[i]) else ok(rs[i]) for i in bq)
        sent = sum(unsure(rs[i]) for i in bq)
        p(f"   on B's {len(bq)}: small alone {sum(ok(rs[i]) for i in bq)}, routed {routed} "
          f"(sent {sent} to the 120B), 120B alone {sum(ok(b[i]) for i in bq)}")

    text = "\n".join(out)
    print(text)
    open("results/analysis.txt", "w").write(text + "\n")


if __name__ == "__main__":
    main()
