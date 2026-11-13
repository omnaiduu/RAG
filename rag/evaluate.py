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


NUM = re.compile(r"^\d[\d,.]*(?:[\u2013-]\d[\d,.]*)?$")
MONTHS = re.compile(r"\b(january|february|march|april|may|june|july|august|september|october|november|december|"
                    r"jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)\b", re.I)
QUALIFIER = re.compile(r"^(approximately|nearly|about|around|almost|roughly|some|over|more than)\s+", re.I)
GENERIC = {"river", "stream", "lake", "neighborhood", "neighbourhood", "district", "city", "town", "village",
           "county", "province", "island"}


def core_forms(gold: str) -> set[str]:
    """Shorter forms of an expected answer that mean the same thing. Rules are general and apply to every question:
    a number without its unit word ("16 teams" -> "16"), a place without its region ("Monterrey, Nuevo León" ->
    "Monterrey"; no digits), a name without a generic word ("river Cua" -> "Cua"). Not dates ("7 April 2024" stays whole)
    and not lists ("France, Italy and Spain" stays whole). Matched exactly, never by containment."""
    out = set()
    g = QUALIFIER.sub("", gold.strip())
    out.add(g)
    words = g.split()
    unit = words[1:]
    if (len(words) >= 2 and NUM.match(words[0]) and not MONTHS.search(g) and len(unit) <= 4
            and all(w[0].islower() for w in unit) and not any(c.isdigit() for c in " ".join(unit))):
        if words[1].lower() in {"million", "billion", "thousand", "trillion"}:
            out.add(" ".join(words[:2]))
        else:
            out.add(words[0])
    if (g.count(",") == 1 and " and " not in g and " or " not in g and not any(c.isdigit() for c in g)
            and not MONTHS.search(g)):
        out.add(g.split(",")[0])
    if len(words) >= 2 and words[0].lower() in GENERIC:
        out.add(" ".join(words[1:]))
    if len(words) >= 2 and words[-1].lower() in GENERIC:
        out.add(" ".join(words[:-1]))
    return {normalize(x) for x in out if normalize(x)}


_FIXES: dict | None = None


def golds(q: dict) -> list[str]:
    """Expected answer + aliases + hand corrections from data/answer_fixes.json (if present)."""
    global _FIXES
    if _FIXES is None:
        try:
            with open("data/answer_fixes.json") as f:
                _FIXES = json.load(f)
        except OSError:
            _FIXES = {}
    return [q["answer"], *q.get("aliases", []), *_FIXES.get(q.get("qid", ""), {}).get("add", [])]


def correct(pred: str, golds: list[str], strict: bool = False) -> bool:
    """Lenient match: exact after normalization, gold contained in a short prediction, or F1 >= 0.8.
    Unless strict, a prediction equal to a shorter core form of the gold (see core_forms) also counts."""
    np_ = normalize(pred)
    if not np_ or np_ == "unknown":
        return False
    if not strict and any(np_ in core_forms(g) for g in golds):
        return True
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
        golds_ = golds(q)
        seen_titles = {chunk_title.get(c) for c in r["seen"]}
        support = q.get("support_titles", [])
        rows.append({
            "type": q.get("type", "?"),
            "correct": correct(r["answer"], golds_),
            "em": max(normalize(r["answer"]) == normalize(g) for g in golds_),
            "f1": max(f1(r["answer"], g) for g in golds_),
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
