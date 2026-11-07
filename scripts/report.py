"""Score all raw results and write results/summary.json, results/summary.md and charts.

Cost model (edit if prices change):
  - Groq models: list price per million tokens (input, output) below.
  - Gemma on a T4: GPU rental per hour x measured LLM seconds (single stream, no batching,
    so this is an upper bound for the small model).
Tool time (search + rerank) is reported separately and not priced: it is the same code for every racer.
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")
from rag.evaluate import load_jsonl, score  # noqa: E402
from rag.tools import chunk_docs, load_corpus  # noqa: E402

PRICE_PER_M = {"openai/gpt-oss-120b": (0.15, 0.75), "openai/gpt-oss-20b": (0.075, 0.30),
               "qwen/qwen3.8-27b": (0.29, 0.59)}
T4_PER_HOUR = 0.35

LABELS = {"A": "A: basic RAG", "B": "B: big agent", "C1": "C1: fixed steps", "C2": "C2: guardrailed agent",
          "D": "D: free small agent", "CB": "closed book",
          "C2v1": "C2 v1 (guardrails logged, not enforced)", "Dv1": "D v1 (strict parser)"}


def cost_per_1k(row: dict, model: str) -> float:
    if model in PRICE_PER_M:
        pin, pout = PRICE_PER_M[model]
        return 1000 * (row["prompt_tokens"] * pin + row["completion_tokens"] * pout) / 1e6
    return 1000 * row["llm_seconds"] / 3600 * T4_PER_HOUR


def main() -> None:
    questions = {q["qid"]: q for q in load_jsonl("data/questions.jsonl")}
    chunk_title = {c.id: c.title for c in chunk_docs(load_corpus("data/corpus.jsonl.gz"))}
    groups = defaultdict(list)
    for path in sorted(glob.glob("results/raw/*.jsonl")):
        for r in load_jsonl(path):
            if r["qid"] in questions:
                groups[(r["racer"], r["model"])].append(r)

    core = {q for q, v in questions.items() if "core60" in v.get("subset", [])}
    summary = {}
    for (racer, model), rows in sorted(groups.items()):
        # one record per question (later files win)
        rows = list({r["qid"]: r for r in rows}.values())
        key = f"{racer}|{model}"
        full = score(rows, questions, chunk_title)
        core_rows = [r for r in rows if r["qid"] in core]
        summary[key] = {"racer": racer, "model": model, "label": LABELS.get(racer, racer),
                        "all": full, "core60": score(core_rows, questions, chunk_title) if core_rows else None,
                        "errors": sum(1 for r in rows if r.get("flags", {}).get("error"))}
        for part in ("all", "core60"):
            if summary[key][part]:
                ov = summary[key][part]["overall"]
                ov["cost_per_1k_usd"] = round(cost_per_1k(ov, model), 4)

    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    lines = []
    for part, title in (("core60", "Core 60 (every racer, including B)"), ("all", "All questions")):
        lines += [f"### {title}", "",
                  "| Racer | Model | n | Accuracy | F1 | All evidence found | LLM calls | Tool calls | "
                  "Prompt tok | Output tok | LLM s/q | Tool s/q | $ / 1K q |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for key, s in summary.items():
            if not s[part]:
                continue
            o = s[part]["overall"]
            ev = "-" if o["all_support_found"] is None or s["racer"] == "CB" else f"{o['all_support_found']:.0%}"
            lines.append(f"| {s['label']} | `{s['model'].split('/')[-1]}` | {o['n']} | **{o['accuracy']:.1%}** | "
                         f"{o['f1']:.2f} | {ev} | {o['llm_calls']} | {o['tool_calls']} | {o['prompt_tokens']:.0f} | "
                         f"{o['completion_tokens']:.0f} | {o['llm_seconds']:.1f} | {o['tool_seconds']:.1f} | "
                         f"{o['cost_per_1k_usd']:.3f} |")
        lines.append("")
        types = sorted({t for s in summary.values() if s[part] for t in s[part]["by_type"]})
        lines += [f"Accuracy by question type ({part}):", "",
                  "| Racer | Model | " + " | ".join(types) + " |", "|---|---|" + "---|" * len(types)]
        for key, s in summary.items():
            if not s[part]:
                continue
            cells = [f"{s[part]['by_type'].get(t, {}).get('accuracy', 0):.0%} (n={s[part]['by_type'].get(t, {}).get('n', 0)})"
                     for t in types]
            lines.append(f"| {s['label']} | `{s['model'].split('/')[-1]}` | " + " | ".join(cells) + " |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for key, s in summary.items():
        if not s["core60"] or s["racer"] == "CB":
            continue
        o = s["core60"]["overall"]
        ax.scatter(o["cost_per_1k_usd"], o["accuracy"], s=70)
        ax.annotate(f"{s['label']}\n({s['model'].split('/')[-1]})", (o["cost_per_1k_usd"], o["accuracy"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("LLM cost per 1,000 questions (USD, log scale)")
    ax.set_ylabel("Accuracy (core 60)")
    ax.set_title("Accuracy vs. cost on post-cutoff multi-hop questions")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "accuracy_vs_cost.png")
    print("wrote results/accuracy_vs_cost.png")


if __name__ == "__main__":
    main()
