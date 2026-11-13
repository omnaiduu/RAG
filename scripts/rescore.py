"""Every run, scored with the old grader and with the fixed one (shorter core forms + data/answer_fixes.json).

  python scripts/rescore.py      # prints the table and writes results/rescore.md
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, ".")
from rag.evaluate import correct, golds  # noqa: E402


def main() -> None:
    gold = {q["qid"]: q for p in ("data/questions.jsonl", "data/fresh/questions.jsonl") for q in map(json.loads, open(p))}
    rows = ["| Run | Racer | Model | Questions | Old grader | Fixed grader | Change |", "|---|---|---|---|---|---|---|"]
    for f in sorted(glob.glob("results/raw/*.jsonl")):
        recs = [json.loads(l) for l in open(f) if l.strip()]
        recs = [r for r in recs if r.get("qid") in gold and not r.get("error") and not (r.get("flags") or {}).get("error")]
        groups: dict = {}
        for r in recs:
            groups.setdefault((r["racer"], r["model"].split("/")[-1]), []).append(r)
        for (racer, model), rs in sorted(groups.items()):
            if len(rs) < 50:
                continue
            q = gold[rs[0]["qid"]]
            old = sum(correct(r["answer"], [gold[r["qid"]]["answer"], *gold[r["qid"]]["aliases"]], strict=True) for r in rs)
            new = sum(correct(r["answer"], golds(gold[r["qid"]])) for r in rs)
            rows.append(f"| {os.path.basename(f)[:-6]} | {racer} | {model} | {len(rs)} | {old} ({old / len(rs):.0%}) "
                        f"| {new} ({new / len(rs):.0%}) | +{new - old} |")
    text = "\n".join(rows)
    print(text)
    open("results/rescore.md", "w").write("# Old vs fixed grader\n\n" + __doc__.split("\n\n")[0] + "\n\n" + text + "\n")


if __name__ == "__main__":
    main()
