"""Old (tuned-on) questions vs fresh held-out questions, per setup, with plain names.

  python scripts/fresh_report.py      # prints the table and writes results/fresh_test.md

"Old" = the 200 questions the pipeline was improved on. "Fresh" = data/fresh/questions.jsonl: 60 new questions on
articles no old question touched, frozen (committed) before any setup was run on them. Nothing was tuned on them.
"""
from __future__ import annotations

import collections
import glob
import json
import math
import sys

sys.path.insert(0, ".")
from rag.evaluate import correct  # noqa: E402

SETUPS = [  # (racer, model, plain name, tuned on the old questions?)
    ("CB@lance", "gemma-4-E2B-it", "No search, memory only (2B)", "no"),
    ("A@lance", "gemma-4-E2B-it", "Search once, then answer (2B)", "no"),
    ("C1@lance", "gemma-4-E2B-it", "Step-by-step plan run by code, first version (2B)", "no"),
    ("C1v3@lance", "gemma-4-E2B-it", "Step-by-step plan + answer-shape rule (2B)", "yes"),
    ("C1v4@lance", "gemma-4-E2B-it", "Step-by-step plan + answer-shape rule + retry (2B)", "yes"),
    ("C1@lance", "gemma-4-E4B-it", "Step-by-step plan run by code, first version (4B)", "no"),
    ("C1v3@lance", "gemma-4-E4B-it", "Step-by-step plan + answer-shape rule (4B)", "yes"),
    ("C2v2@lance", "gemma-4-E4B-it", "Model searches on its own, with checks (4B)", "yes"),
    ("B", "gpt-oss-120b", "Big 120B model searches on its own", "no"),
]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main() -> None:
    gold = {}
    for path in ("data/questions.jsonl", "data/fresh/questions.jsonl"):
        for q in map(json.loads, open(path)):
            gold[q["qid"]] = q
    runs: dict[tuple[str, str], dict[str, bool]] = collections.defaultdict(dict)
    for f in glob.glob("results/raw/*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            if r.get("error") or r.get("flags", {}).get("error") or r["qid"] not in gold:
                continue
            q = gold[r["qid"]]
            runs[(r["racer"], r["model"].split("/")[-1])][r["qid"]] = correct(r["answer"], [q["answer"], *q["aliases"]])

    fresh_ids = [q for q in gold if q.startswith("f")]
    rows = ["| Setup | Tuned on old questions? | Old questions | Fresh questions (95% range) | Change |",
            "|---|---|---|---|---|"]
    for racer, model, name, tuned in SETUPS:
        rs = runs.get((racer, model), {})
        old_rs = rs if any(k.startswith("q") for k in rs) else runs.get((racer.split("@")[0], model), {})
        old = [v for k, v in old_rs.items() if k.startswith("q")]
        new = [rs[k] for k in fresh_ids if k in rs]
        if not new:
            continue
        o = sum(old) / len(old) if old else float("nan")
        lo, hi = wilson(sum(new), len(new))
        n = sum(new) / len(new)
        rows.append(f"| {name} | {tuned} | {o:.0%} ({len(old)} q) | **{n:.0%}** ({sum(new)}/{len(new)}; {lo:.0%}-{hi:.0%}) "
                    f"| {100 * (n - o):+.0f} pts |")
    text = "\n".join(rows)
    print(text)
    open("results/fresh_test.md", "w").write("# Fresh held-out test\n\n" + __doc__.split("\n\n", 1)[1] + "\n" + text + "\n")


if __name__ == "__main__":
    main()
