"""The racers.

A   basic RAG: one vector search, small model answers.
B   big-model free agent: native tool calling, full history (Groq gpt-oss-120b).
C1  small model, fixed pipeline: plan form -> hybrid search -> gap-check form (loop, bounded) -> answer form.
C2  small model, guardrailed agent: picks one tool per step via a JSON form over a compact memory;
    code enforces a step budget, blocks repeated searches, flags empty results.
D   small model, free agent: ReAct text loop with full history, no forms, no guardrails.
CB  closed book: no retrieval at all (contamination check).

Every racer returns the same record shape so the scorer treats them identically.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from .llm import Usage, parse_json
from .tools import SearchTools

TOOLS = ("grep", "bm25", "vector", "hybrid")
TOOL_HELP = (
    "grep(phrase): exact phrase match, best for names, titles, numbers, IDs.\n"
    "bm25(query): keyword search.\n"
    "vector(query): meaning-based search, works with paraphrases.\n"
    "hybrid(query): bm25 + vector combined and reranked; the strongest general search."
)
OBS_WORDS = 90  # words per passage shown to every agent
K = 5


@dataclass
class Run:
    answer: str = ""
    sources: list[str] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    llm: Usage = field(default_factory=Usage)
    tool_seconds: float = 0.0
    tool_calls: int = 0
    seen: list[str] = field(default_factory=list)  # chunk ids shown to the model, in order
    flags: dict = field(default_factory=dict)

    def record(self) -> dict:
        return {"answer": self.answer, "sources": self.sources, "trace": self.trace,
                "llm": self.llm.as_dict(), "tool_seconds": round(self.tool_seconds, 3),
                "tool_calls": self.tool_calls, "seen": self.seen, "flags": self.flags}


class Base:
    name = "base"

    def __init__(self, llm, tools: SearchTools):
        self.llm = llm
        self.tools = tools

    # ---- shared plumbing ------------------------------------------------
    def ask(self, run: Run, messages: list[dict], max_new_tokens: int = 200) -> str:
        text, usage = self.llm.chat(messages, max_new_tokens=max_new_tokens)
        run.llm.add(usage)
        return text

    def ask_json(self, run: Run, prompt: str, max_new_tokens: int = 200) -> dict:
        text = self.ask(run, [{"role": "user", "content": prompt}], max_new_tokens)
        obj = parse_json(text)
        if obj is None:
            run.flags["json_retries"] = run.flags.get("json_retries", 0) + 1
            text = self.ask(run, [{"role": "user", "content": prompt},
                                  {"role": "assistant", "content": text},
                                  {"role": "user", "content": "That was not valid JSON. Reply with only the JSON object."}],
                            max_new_tokens)
            obj = parse_json(text)
        if obj is None:
            run.flags["json_failures"] = run.flags.get("json_failures", 0) + 1
            return {}
        return obj

    def search(self, run: Run, tool: str, query: str, **kw) -> list[int]:
        t0 = time.time()
        if tool == "hybrid" and kw.get("exact_terms"):
            idxs = self.tools.hybrid(query, K, exact_terms=kw["exact_terms"])
        else:
            idxs = self.tools.run(tool, query, K)
        run.tool_seconds += time.time() - t0
        run.tool_calls += 1
        for i in idxs:
            cid = self.tools.chunks[i].id
            if cid not in run.seen:
                run.seen.append(cid)
        return idxs

    def fmt(self, idxs: list[int]) -> str:
        return self.tools.format(idxs, OBS_WORDS)

    def answer_form(self, run: Run, question: str, passages: str, facts: list[str] | None = None) -> None:
        fact_txt = ("FACTS FOUND:\n" + "\n".join(f"- {f}" for f in facts) + "\n\n") if facts else ""
        obj = self.ask_json(run, (
            "Answer the question using only the passages and facts below.\n"
            'Reply with only JSON: {"answer": "<shortest possible answer: a name, date, number or few words>", '
            '"sources": ["<passage ids used>"]}\n'
            'If the answer is not there, use "answer": "unknown".\n\n'
            f"QUESTION: {question}\n\n{fact_txt}PASSAGES:\n{passages}"
        ), max_new_tokens=120)
        run.answer = str(obj.get("answer", "unknown") or "unknown")
        run.sources = [str(s) for s in obj.get("sources", []) if isinstance(s, (str, int))]
        run.trace.append({"step": "answer", "out": obj})


# ---------------------------------------------------------------------------
class ClosedBook(Base):
    name = "CB"

    def solve(self, question: str) -> Run:
        run = Run()
        obj = self.ask_json(run, (
            "Answer from your own knowledge. Reply with only JSON: "
            '{"answer": "<shortest possible answer>"}. If you do not know, use "unknown".\n\n'
            f"QUESTION: {question}"), max_new_tokens=80)
        run.answer = str(obj.get("answer", "unknown") or "unknown")
        return run


class RacerA(Base):
    name = "A"

    def solve(self, question: str) -> Run:
        run = Run()
        idxs = self.search(run, "vector", question)
        run.trace.append({"step": "search", "tool": "vector", "query": question,
                          "hits": [self.tools.chunks[i].id for i in idxs]})
        self.answer_form(run, question, self.fmt(idxs))
        return run


class RacerC1(Base):
    """Fixed steps. The model never chooses the control flow; it only fills forms."""
    name = "C1"
    max_searches = 3

    def solve(self, question: str) -> Run:
        run = Run()
        plan = self.ask_json(run, (
            "You turn a question into a search plan. Reply with only JSON:\n"
            '{"query": "<short search query for the FIRST thing to look up>", '
            '"exact_terms": ["<names or titles copied exactly from the question>"]}\n\n'
            f"QUESTION: {question}"), max_new_tokens=100)
        query = str(plan.get("query") or question)
        exact = [str(t) for t in plan.get("exact_terms", []) if isinstance(t, str) and t.strip()][:3]
        run.trace.append({"step": "plan", "out": plan})

        kept: list[int] = []
        found: list[str] = []
        for round_no in range(1, self.max_searches + 1):
            idxs = self.search(run, "hybrid", query, exact_terms=exact if round_no == 1 else None)
            kept += [i for i in idxs if i not in kept]
            run.trace.append({"step": "search", "round": round_no, "query": query,
                              "hits": [self.tools.chunks[i].id for i in idxs]})
            if round_no == self.max_searches:
                break
            gap = self.ask_json(run, (
                "Decide if the passages contain enough to answer the question. Reply with only JSON:\n"
                '{"can_answer": true or false, "found": "<key fact learned so far, e.g. a name>", '
                '"next_query": "<if can_answer is false: short search query for the missing fact>"}\n\n'
                f"QUESTION: {question}\n"
                + ("FACTS FOUND:\n" + "\n".join(f"- {f}" for f in found) + "\n" if found else "")
                + f"\nPASSAGES:\n{self.fmt(kept[-10:])}"), max_new_tokens=120)
            run.trace.append({"step": "gap_check", "round": round_no, "out": gap})
            if gap.get("found"):
                found.append(str(gap["found"]))
            nxt = str(gap.get("next_query") or "").strip()
            if gap.get("can_answer") in (True, "true", "True", "yes") or not nxt:
                break
            query = nxt
        self.answer_form(run, question, self.fmt(kept[-10:]), found)
        return run


class RacerC2(Base):
    """Guardrailed agent: model picks the tool each step; code enforces the rules."""
    name = "C2"
    max_steps = 5

    def solve(self, question: str) -> Run:
        run = Run()
        facts: list[str] = []
        used: list[tuple[str, str, int]] = []
        last: list[int] = []
        note = ""
        for step in range(self.max_steps):
            steps_left = self.max_steps - step
            tools_used = "\n".join(f"{n + 1}. {t} \"{q}\" -> {c} results" for n, (t, q, c) in enumerate(used)) or "(none yet)"
            prompt = (
                "You are a search agent. Each turn you pick ONE action. Tools:\n" + TOOL_HELP +
                "\nanswer: give the final answer when the facts and results are enough.\n\n"
                "Reply with only JSON:\n"
                '{"reason": "<one short sentence>", "tool": "grep|bm25|vector|hybrid|answer", '
                '"query": "<search text; empty if tool is answer>", '
                '"answer": "<only if tool is answer: the shortest final answer, e.g. a name or year>", '
                '"new_fact": "<a fact from LAST RESULTS that helps, or empty>", '
                '"sources": ["<passage ids supporting the answer, only when answering>"]}\n\n'
                f"QUESTION: {question}\n\n"
                "FACTS SO FAR:\n" + ("\n".join(f"- {f}" for f in facts) or "(none yet)") + "\n\n"
                f"TOOLS USED:\n{tools_used}\n\n"
                f"LAST RESULTS:\n{self.fmt(last) if used else '(no search yet)'}\n\n"
                + (f"NOTE: {note}\n" if note else "")
                + f"STEPS LEFT: {steps_left}"
            )
            act = self.ask_json(run, prompt, max_new_tokens=180)
            run.trace.append({"step": step + 1, "out": act})
            nf = str(act.get("new_fact") or "").strip()
            if nf and nf.lower() not in {f.lower() for f in facts}:
                facts.append(nf)
            tool = str(act.get("tool") or "").lower().strip()
            query = str(act.get("query") or "").strip()
            note = ""
            final = str(act.get("answer") or "").strip() or query
            if tool == "answer" and final:
                if not act.get("sources") and run.seen:
                    run.flags["answer_without_sources"] = True
                run.answer = final
                run.sources = [str(s) for s in act.get("sources", []) if isinstance(s, (str, int))]
                return run
            if tool not in TOOLS:
                tool, run.flags["bad_tool"] = "hybrid", run.flags.get("bad_tool", 0) + 1
            if not query:
                query = question
            if any(t == tool and q.lower() == query.lower() for t, q, _ in used):
                run.flags["blocked_repeat"] = run.flags.get("blocked_repeat", 0) + 1
                note = f'You already ran {tool} "{query}". Use a different query or tool.'
                used.append((tool, query + " [BLOCKED]", 0))
                continue
            last = self.search(run, tool, query)
            used.append((tool, query, len(last)))
            if not last:
                note = "The last search returned no results. Try a different tool or a shorter query."
        # Budget exhausted: code forces an answer from what was gathered.
        run.flags["forced_answer"] = True
        self.answer_form(run, question, self.fmt([self.tools.by_id[c] for c in run.seen[-10:]]), facts)
        return run


class RacerD(Base):
    """Free ReAct agent with full history and no guardrails (besides a hard step cap)."""
    name = "D"
    max_steps = 8
    ACTION = re.compile(r"Action:\s*(\w+)\s*\n+\s*Action Input:\s*(.+?)(?:\n|$)", re.S)

    def solve(self, question: str) -> Run:
        run = Run()
        messages = [{"role": "user", "content": (
            "Answer the question by searching a document collection. Tools:\n" + TOOL_HELP +
            "\nanswer(final answer): finish with a short answer.\n\n"
            "Use exactly this format each turn:\nThought: <your reasoning>\nAction: <tool name>\n"
            "Action Input: <input>\n\nYou will then get an Observation. Repeat until you can answer.\n\n"
            f"Question: {question}")}]
        for step in range(self.max_steps):
            text = self.ask(run, messages, max_new_tokens=200)
            m = self.ACTION.search(text)
            run.trace.append({"step": step + 1, "out": text[:600]})
            if not m:
                run.flags["unparsed"] = True
                run.answer = text.strip().splitlines()[-1] if text.strip() else "unknown"
                return run
            tool, arg = m.group(1).lower(), m.group(2).strip().strip('"')
            if tool == "answer":
                run.answer = arg
                return run
            idxs = self.search(run, tool if tool in TOOLS else "hybrid", arg)
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": f"Observation:\n{self.fmt(idxs)}"}]
        run.flags["step_limit"] = True
        text = self.ask(run, messages + [{"role": "user", "content": "Step limit reached. Reply with only: Final Answer: <short answer>"}], 60)
        run.answer = text.split("Final Answer:")[-1].strip() or "unknown"
        return run


class RacerB(Base):
    """Big model with native tool calling and the full conversation history."""
    name = "B"
    max_steps = 8

    def solve(self, question: str) -> Run:
        run = Run()
        spec = lambda n, d, p: {"type": "function", "function": {"name": n, "description": d, "parameters": {
            "type": "object", "properties": {p: {"type": "string"}}, "required": [p]}}}
        tools = [spec("grep", "Exact phrase match; best for names, titles, numbers.", "phrase"),
                 spec("bm25", "Keyword search.", "query"),
                 spec("vector", "Meaning-based search.", "query"),
                 spec("hybrid", "bm25 + vector, reranked. Strongest general search.", "query"),
                 {"type": "function", "function": {"name": "answer", "description": "Give the final answer.",
                  "parameters": {"type": "object", "properties": {
                      "answer": {"type": "string", "description": "Shortest possible answer."},
                      "sources": {"type": "array", "items": {"type": "string"}}},
                      "required": ["answer"]}}}]
        messages = [{"role": "system", "content": (
            "Answer questions by searching a document collection with the tools. The documents are the only "
            "source of truth; do not rely on memory. Search as many times as needed (several hops are common), "
            "then call answer with the shortest possible answer and the passage ids you used.")},
            {"role": "user", "content": question}]
        for step in range(self.max_steps + 1):
            if step == self.max_steps:
                messages.append({"role": "user", "content": "Step limit reached. Call answer now."})
            msg, usage = self.llm.raw(messages, max_tokens=1024, tools=tools)
            run.llm.add(usage)
            calls = msg.get("tool_calls") or []
            if not calls:
                run.trace.append({"step": step + 1, "text": (msg.get("content") or "")[:400]})
                run.answer = (msg.get("content") or "unknown").strip()
                run.flags["no_tool_answer"] = True
                return run
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
            for call in calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                run.trace.append({"step": step + 1, "tool": name, "args": args})
                if name == "answer":
                    run.answer = str(args.get("answer", "unknown"))
                    run.sources = [str(s) for s in args.get("sources", [])]
                    return run
                arg = str(args.get("phrase") or args.get("query") or question)
                idxs = self.search(run, name if name in TOOLS else "hybrid", arg)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": self.fmt(idxs)})
        run.answer = run.answer or "unknown"
        return run


RACERS = {"A": RacerA, "B": RacerB, "C1": RacerC1, "C2": RacerC2, "D": RacerD, "CB": ClosedBook}
