"""Scoring: answer accuracy, evidence recall, cost and latency per racer."""
from __future__ import annotations

import json
import re
import statistics
import string
from collections import Counter, defaultdict


def normalize(s: str) -> str:
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def correct(pred: str, golds: list[str]) -> bool:
    """Lenient match: exact after normalization, gold contained in a short prediction, or F1 >= 0.8."""
    np_ = normalize(pred)
    if not np_ or np_ == "unknown":
        return False
    for g in golds:
        ng = normalize(g)
        if not ng:
            continue
        if np_ == ng or f1(pred, g) >= 0.8:
            return True
        if re.search(r"(?<!\w)" + re.escape(ng) + r"(?!\w)", np_) and len(np_.split()) <= len(ng.split()) + 6:
            return True
    return False


def score(results: list[dict], questions: dict[str, dict], chunk_title: dict[str, str]) -> dict:
    rows = []
    for r in results:
        q = questions[r["qid"]]
        golds = [q["answer"], *q.get("aliases", [])]
        seen_titles = {chunk_title.get(c) for c in r["seen"]}
        support = q.get("support_titles", [])
        rows.append({
            "type": q.get("type", "?"),
            "correct": correct(r["answer"], golds),
            "em": max(normalize(r["answer"]) == normalize(g) for g in golds),
            "f1": max(f1(r["answer"], g) for g in golds),
            "support_recall": (sum(t in seen_titles for t in support) / len(support)) if support else None,
            "all_support": all(t in seen_titles for t in support) if support else None,
            "calls": r["llm"]["calls"], "ptok": r["llm"]["prompt_tokens"], "ctok": r["llm"]["completion_tokens"],
            "llm_s": r["llm"]["seconds"], "tool_s": r["tool_seconds"], "tool_calls": r["tool_calls"],
            "total_s": r["llm"]["seconds"] + r["tool_seconds"],
        })

    def agg(rs: list[dict]) -> dict:
        if not rs:
            return {}
        mean = lambda k: statistics.fmean(x[k] for x in rs)
        sup = [x["all_support"] for x in rs if x["all_support"] is not None]
        tot = sorted(x["total_s"] for x in rs)
        return {
            "n": len(rs), "accuracy": round(mean("correct"), 4), "em": round(mean("em"), 4), "f1": round(mean("f1"), 4),
            "all_support_found": round(statistics.fmean(sup), 4) if sup else None,
            "llm_calls": round(mean("calls"), 2), "tool_calls": round(mean("tool_calls"), 2),
            "prompt_tokens": round(mean("ptok"), 1), "completion_tokens": round(mean("ctok"), 1),
            "llm_seconds": round(mean("llm_s"), 3), "tool_seconds": round(mean("tool_s"), 3),
            "p50_seconds": round(tot[len(tot) // 2], 3), "p95_seconds": round(tot[min(len(tot) - 1, int(len(tot) * 0.95))], 3),
        }

    by_type = defaultdict(list)
    for x in rows:
        by_type[x["type"]].append(x)
    return {"overall": agg(rows), "by_type": {t: agg(v) for t, v in sorted(by_type.items())}}


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]
