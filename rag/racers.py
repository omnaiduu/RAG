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

from .llm import ToolCallError, Usage, parse_json
from .tools import SearchTools

TOOLS = ("grep", "bm25", "vector", "hybrid")
TOOL_HELP = (
    "grep(phrase): exact phrase match, best for names, titles, numbers, IDs.\n"
    "bm25(query): keyword search.\n"
    "vector(query): meaning-based search, works with paraphrases.\n"
    "hybrid(query): bm25 + vector combined and reranked; the strongest general search."
)
GIVE_UP = re.compile(r"\b(unknown|not found|no information|n/?a|none|could not|cannot|not (?:mentioned|available|specified|provided))\b", re.I)
ANAPHOR = re.compile(r"\b(that|this|these|those|it|its|he|she|his|her|they|their|the same)\b", re.I)
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
        k = kw.get("k", K)
        if tool == "hybrid" and kw.get("exact_terms"):
            idxs = self.tools.hybrid(query, k, exact_terms=kw["exact_terms"])
        else:
            idxs = self.tools.run(tool, query, k)
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
                "You are a search agent. Each turn you pick ONE action. Questions often need several hops: "
                "find the thing the question describes, then search for it by name. Tools:\n" + TOOL_HELP +
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
                sources = [str(s) for s in act.get("sources", []) if isinstance(s, (str, int)) and str(s).strip()]
                give_up = bool(GIVE_UP.search(final))
                # Guardrail: while steps remain, a give-up or an uncited answer is rejected by the code.
                if steps_left > 1 and (give_up or not sources):
                    run.flags["rejected_answer"] = run.flags.get("rejected_answer", 0) + 1
                    note = ("You may not give up while steps remain. " if give_up else
                            "An answer must cite passage ids in sources. ") + \
                           "Search for the missing fact: first find the person/thing the question describes, " \
                           "then search for it by name."
                    continue
                run.answer = final
                run.sources = sources
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
    @staticmethod
    def parse_action(text: str) -> tuple[str, str] | None:
        """Accept 'Action: tool\\nAction Input: x' and 'Action: tool(x)' (markdown stripped)."""
        t = text.replace("**", "").replace("`", "")
        m = re.search(r"Action:\s*(\w+)\s*(?:\((.*?)\))?\s*(?:\n|$)", t)
        if not m:
            return None
        tool, inline = m.group(1).lower(), (m.group(2) or "").strip().strip('"\'')
        m2 = re.search(r"Action Input:\s*(.+?)(?:\n|$)", t[m.end() - 1:])
        arg = m2.group(1).strip().strip('"\'') if m2 else ""
        if inline and inline.lower() not in {"query", "phrase", "input", "final answer", "answer"}:
            arg = arg or inline
            arg = inline if not m2 else arg
        return (tool, arg) if arg else None

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
            run.trace.append({"step": step + 1, "out": text[:600]})
            parsed = self.parse_action(text)
            if not parsed:
                run.flags["unparsed"] = True
                fa = re.search(r"Final Answer:\s*(.+)", text)
                run.answer = fa.group(1).strip() if fa else (text.strip().splitlines()[-1] if text.strip() else "unknown")
                return run
            tool, arg = parsed
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
            try:
                msg, usage = self.llm.raw(messages, max_tokens=1024, tools=tools)
            except ToolCallError as err:
                run.flags["tool_json_errors"] = run.flags.get("tool_json_errors", 0) + 1
                run.llm.calls += 1
                if run.flags["tool_json_errors"] <= 1:
                    messages.append({"role": "user", "content": "Your last tool call had invalid JSON arguments. "
                                     "Call the tool again with valid JSON arguments."})
                    continue
                # second failure: salvage the answer text from the rejected generation
                gen = err.failed_generation
                m = re.search(r'"answer"\s*:\s*"([^"]+)"', gen) or re.search(r'"arguments"\s*:\s*([^【{}"]+)', gen)
                run.answer = (m.group(1).strip() if m else "") or "unknown"
                run.flags["salvaged_answer"] = True
                return run
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


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(s).lower())).strip()


class RacerC1v2(Base):
    """C1 with two code-enforced fixes, both general (nothing about this benchmark's question types):

    1. Decompose: the model writes the lookup steps once ("Where was Messi born?" -> "What is the capital of #1?").
       Code runs them in order, filling #1 with step 1's answer, so it cannot stop half-way. A final answer that
       equals an intermediate answer is rejected once.
    2. Grounded answers: every answer form also returns the sentence it copied from. Code checks that the
       answer text really occurs in the retrieved passages (or facts/question); if not, it searches again once.
    """
    name = "C1v2"
    max_steps = 3

    PLAN = (
        "Break the question into the smallest lookup steps, in order (1 to 3 steps). Write each step as a short "
        "question. If a step needs the answer of an earlier step, write #1 (or #2) in its place.\n"
        'Example: "What is the capital of the country where Messi was born?" -> '
        '{"steps": ["Where was Messi born?", "What is the capital of #1?"], "exact_terms": ["Messi"]}\n'
        'Example: "Which is taller, the Eiffel Tower or Big Ben?" -> '
        '{"steps": ["How tall is the Eiffel Tower?", "How tall is Big Ben?"], "exact_terms": ["Eiffel Tower", "Big Ben"]}\n'
        "exact_terms: names or titles copied exactly from the question.\n"
        "Reply with only JSON.\n\nQUESTION: {q}")

    STEP = (
        "Answer the sub-question using only the passages. Reply with only JSON:\n"
        '{{"quote": "<copy the one sentence from the passages that contains the answer>", '
        '"answer": "<shortest answer, copied word for word from that sentence>"}}\n'
        'If the passages do not contain it, use "answer": "unknown".\n\n'
        "SUB-QUESTION: {q}\n\nPASSAGES:\n{p}")

    FINAL = (
        "Answer the question using the facts and passages. Reply with only JSON:\n"
        '{{"quote": "<the sentence or fact the answer comes from>", '
        '"answer": "<shortest possible answer: a name, date, number or few words>", "sources": ["<passage ids>"]}}\n'
        'If the answer is not there, use "answer": "unknown".{note}\n\n'
        "QUESTION: {q}\n\nFACTS FOUND:\n{f}\n\nPASSAGES:\n{p}")

    def _grounded(self, answer: str, idxs: list[int], extra: str = "") -> bool:
        a = _norm(answer)
        if not a or GIVE_UP.search(answer):
            return False
        hay = [_norm(self.tools.texts[i]) for i in idxs] + [_norm(extra)]
        return any(a in h for h in hay)

    def _step(self, run: Run, n: int, q: str, question: str, exact: list[str], kept: list[int]) -> tuple[str, str]:
        obj: dict = {}
        for attempt in (0, 1):
            query = q if attempt == 0 else f"{q} {question}"  # second try: widen with the full question
            idxs = self.search(run, "hybrid", query, exact_terms=exact or None)
            kept += [i for i in idxs if i not in kept]
            # full 140-word passages: with a 90-word cut, 20 words of every chunk are never shown
            obj = self.ask_json(run, self.STEP.format(q=q, p=self.tools.format(idxs, 140)), max_new_tokens=160)
            ans = str(obj.get("answer") or "unknown")
            ok = self._grounded(ans, idxs)
            run.trace.append({"step": f"step{n + 1}", "attempt": attempt, "query": query,
                              "hits": [self.tools.chunks[i].id for i in idxs], "out": obj, "grounded": ok})
            if ok:
                return ans, str(obj.get("quote", ""))
            run.flags["ungrounded_step"] = run.flags.get("ungrounded_step", 0) + 1
        return str(obj.get("answer") or "unknown"), str(obj.get("quote", ""))

    def solve(self, question: str) -> Run:
        run = Run()
        plan = self.ask_json(run, self.PLAN.replace("{q}", question), max_new_tokens=200)
        run.trace.append({"step": "plan", "out": plan})
        steps = [str(s) for s in plan.get("steps", []) if isinstance(s, str) and s.strip()][: self.max_steps]
        steps = steps or [question]
        exact = [str(t) for t in plan.get("exact_terms", []) if isinstance(t, str) and t.strip()][:3]

        answers: list[str] = []
        facts: list[str] = []
        kept: list[int] = []
        used_as_input: set[str] = set()
        for n, step in enumerate(steps):
            known = [a if a and not GIVE_UP.search(a) else "" for a in answers]
            refs = [int(m) for m in re.findall(r"#(\d)", step)]
            if any(not (0 < r <= len(known)) or not known[r - 1] for r in refs):
                q = question  # an earlier step failed: fall back to the whole question
            else:
                q = re.sub(r"#(\d)", lambda m: known[int(m.group(1)) - 1], step)
                used_as_input.update(known[r - 1] for r in refs)
                if n > 0 and not refs and known and known[-1] and ANAPHOR.search(step):
                    q = f"{step} ({known[-1]})"  # "that series" -> keep the name found in the previous step
                    used_as_input.add(known[-1])
            terms = [t for t in exact if t.lower() in q.lower()]
            terms = [a for a in used_as_input if a and a.lower() in q.lower()] + terms
            ans, quote = self._step(run, n, q, question, terms[:3], kept)
            answers.append(ans)
            facts.append(f"{q} -> {ans}" + (f' (from: "{quote[:200]}")' if quote else ""))

        facts_txt = "\n".join(f"- {f}" for f in facts)
        note = ""
        for attempt in (0, 1):
            obj = self.ask_json(run, self.FINAL.format(q=question, f=facts_txt, p=self.fmt(kept[-10:]), note=note),
                                max_new_tokens=160)
            ans = str(obj.get("answer") or "unknown")
            run.trace.append({"step": "answer", "attempt": attempt, "out": obj})
            if _norm(ans) in {_norm(a) for a in used_as_input} and attempt == 0:
                run.flags["bridge_rejected"] = run.flags.get("bridge_rejected", 0) + 1
                note = (f'\nNote: "{ans}" is only an intermediate fact (the answer to an earlier step), '
                        "not what the question asks for.")
                continue
            if not self._grounded(ans, kept, facts_txt + " " + question) and attempt == 0:
                run.flags["ungrounded_final"] = run.flags.get("ungrounded_final", 0) + 1
                note = "\nNote: your last answer was not found in the facts or passages. Copy it from them."
                continue
            break
        run.answer = ans
        run.sources = [str(s) for s in obj.get("sources", []) if isinstance(s, (str, int))]
        return run


YESNO_Q = re.compile(r"^\s*(is|was|were|are|did|do|does|has|have|had|can|could|will|would)\b", re.I)
# "Who was born earlier, X or Y?" / "Which of X and Y is older?": a choice between named options
CHOICE_Q = re.compile(r"^\s*which of\b|^\s*(who|which)\b[^?]*,[^,?]*\bor\b[^,?]*\?\s*$", re.I)


def answer_shape_problem(question: str, answer: str) -> str:
    """General check that the answer has the form the question asks for (no dataset rules)."""
    a = _norm(answer)
    if YESNO_Q.search(question) and not CHOICE_Q.search(question):
        return "" if a in {"yes", "no"} else 'This is a yes/no question: the answer must be "yes" or "no".'
    if CHOICE_Q.search(question) and a and a not in _norm(question):
        return "The question asks you to choose one of the options it names: answer with that option, copied from the question."
    return ""


class RacerC1v3(RacerC1v2):
    """C1 v2 plus an answer-shape check on the final answer (yes/no, or one of the named options)."""
    name = "C1v3"

    def solve(self, question: str) -> Run:
        return self._shape_fix(super().solve(question), question)

    def _shape_fix(self, run: Run, question: str) -> Run:
        problem = answer_shape_problem(question, run.answer)
        if not problem:
            return run
        run.flags["shape_retry"] = 1
        facts = [t for t in run.trace if t.get("step", "").startswith("step") and t.get("grounded") is not None]
        facts_txt = "\n".join(f'- {t["query"]} -> {t["out"].get("answer", "unknown")}' for t in facts)
        obj = self.ask_json(run, (
            "Answer the question from the facts. First write the values you compare, then the answer.\n"
            'Reply with only JSON: {"reasoning": "<one short sentence>", "answer": "<the answer>"}\n'
            f"Note: {problem}\n\nQUESTION: {question}\n\nFACTS FOUND:\n{facts_txt}"), max_new_tokens=160)
        run.trace.append({"step": "shape_retry", "out": obj})
        ans = str(obj.get("answer") or "")
        if ans and not answer_shape_problem(question, ans):
            run.answer = ans
        return run


class RacerC1v4(RacerC1v3):
    """C1 v3 plus four general changes, all aimed at the first lookup going wrong:

    1. Backtrack: when a step that uses an earlier answer finds nothing, that earlier answer was probably wrong.
       Redo the earlier step once, asking for a different answer, then retry the step.
    2. Anchor: a step that names nothing from the question (no #1, no exact term) gets the question's names added
       to its search, so "What film shares its universe with the novel?" still searches for the novel's title.
    3. Facts first: the final form writes the facts it uses (and compares) before the answer.
    4. Plan rules: every step names what it is about; the first step starts from the most specific clue.
    """
    name = "C1v4"

    PLAN = (
        "Break the question into the smallest lookup steps, in order (1 to 3 steps). Write each step as a short "
        "question. If a step needs the answer of an earlier step, write #1 (or #2) in its place.\n"
        "Rules: start from the most specific name or clue in the question. Every step must say what it is about: "
        "copy the name from the question or use #1. Never write a vague step like \"that character\" or "
        "\"the description\".\n"
        'Example: "What is the capital of the country where Messi was born?" -> '
        '{"steps": ["Where was Messi born?", "What is the capital of #1?"], "exact_terms": ["Messi"]}\n'
        'Example: "According to the Inception article, who played the dream thief, and where was that actor born?" -> '
        '{"steps": ["Who played the dream thief in Inception?", "Where was #1 born?"], "exact_terms": ["Inception"]}\n'
        'Example: "Which is taller, the Eiffel Tower or Big Ben?" -> '
        '{"steps": ["How tall is the Eiffel Tower?", "How tall is Big Ben?"], "exact_terms": ["Eiffel Tower", "Big Ben"]}\n'
        "exact_terms: names or titles copied exactly from the question.\n"
        "Reply with only JSON.\n\nQUESTION: {q}")

    STEP4 = (
        "Answer the sub-question using only the passages. Reply with only JSON:\n"
        '{{"quote": "<copy the one sentence from the passages that contains the answer>", '
        '"answer": "<shortest answer, copied word for word from that sentence>"}}\n'
        'If the passages do not contain it, use "answer": "unknown".{note}\n\n'
        "SUB-QUESTION: {q}\n\nPASSAGES:\n{p}")

    FINAL = (
        "Answer the question using the facts and passages. Reply with only JSON:\n"
        '{{"quote": "<the sentence or fact the answer comes from>", '
        '"reasoning": "<one short sentence: the facts you use; if the question compares, write both values>", '
        '"answer": "<shortest possible answer: a name, date, number or few words>", "sources": ["<passage ids>"]}}\n'
        'If the answer is not there, use "answer": "unknown".{note}\n\n'
        "QUESTION: {q}\n\nFACTS FOUND:\n{f}\n\nPASSAGES:\n{p}")

    def _step4(self, run: Run, n: int, q: str, question: str, exact: list[str], kept: list[int],
               avoid: str = "") -> tuple[str, str, bool]:
        note = (f'\nThe answer is not "{avoid}" (that led nowhere): give a different one that fits.' if avoid else "")
        obj: dict = {}
        for attempt in (0, 1):
            query = q if attempt == 0 else f"{q} {question}"
            idxs = self.search(run, "hybrid", query, exact_terms=exact or None, k=8 if avoid else 5)
            kept += [i for i in idxs if i not in kept]
            obj = self.ask_json(run, self.STEP4.format(q=q, p=self.tools.format(idxs, 140), note=note), max_new_tokens=160)
            ans = str(obj.get("answer") or "unknown")
            ok = self._grounded(ans, idxs) and not (avoid and _norm(ans) == _norm(avoid))
            run.trace.append({"step": f"step{n + 1}", "attempt": attempt, "query": query, "avoid": avoid or None,
                              "hits": [self.tools.chunks[i].id for i in idxs], "out": obj, "grounded": ok})
            if ok:
                return ans, str(obj.get("quote", "")), True
            run.flags["ungrounded_step"] = run.flags.get("ungrounded_step", 0) + 1
        return str(obj.get("answer") or "unknown"), str(obj.get("quote", "")), False

    def _fill(self, n: int, step: str, question: str, answers: list[str], exact: list[str]) -> tuple[str, set[str]]:
        """The step's search text with #k filled in, and the earlier answers it used."""
        known = [a if a and not GIVE_UP.search(a) else "" for a in answers]
        refs = [int(m) for m in re.findall(r"#(\d)", step)]
        if any(not (0 < r <= len(known)) or not known[r - 1] for r in refs):
            return question, set()
        q = re.sub(r"#(\d)", lambda m: known[int(m.group(1)) - 1], step)
        used = {known[r - 1] for r in refs}
        if n > 0 and not refs and known and known[-1] and ANAPHOR.search(step):
            q = f"{step} ({known[-1]})"
            used.add(known[-1])
        if not used and exact and not any(t.lower() in q.lower() for t in exact):
            q = f"{q} ({'; '.join(exact)})"  # anchor a step that names nothing from the question
        return q, used

    def solve(self, question: str) -> Run:
        run = Run()
        plan = self.ask_json(run, self.PLAN.replace("{q}", question), max_new_tokens=200)
        run.trace.append({"step": "plan", "out": plan})
        steps = [str(s) for s in plan.get("steps", []) if isinstance(s, str) and s.strip()][: self.max_steps]
        steps = steps or [question]
        exact = [str(t) for t in plan.get("exact_terms", []) if isinstance(t, str) and t.strip()][:3]

        answers: list[str] = []
        queries: list[str] = []
        quotes: list[str] = []
        kept: list[int] = []
        used_as_input: set[str] = set()
        backtracked = False
        n = 0
        while n < len(steps):
            q, used = self._fill(n, steps[n], question, answers, exact)
            terms = [a for a in used if a and a.lower() in q.lower()] + [t for t in exact if t.lower() in q.lower()]
            ans, quote, ok = self._step4(run, n, q, question, terms[:3], kept)
            prev = n - 1
            if not ok and used and not backtracked and prev >= 0 and answers[prev] in used:
                # the earlier answer led nowhere: ask that step again for a different answer, then retry this one
                backtracked = True
                run.flags["backtrack"] = 1
                pq, pused = self._fill(prev, steps[prev], question, answers[:prev], exact)
                pterms = [t for t in exact if t.lower() in pq.lower()]
                pans, pquote, pok = self._step4(run, prev, pq, question, pterms[:3], kept, avoid=answers[prev])
                if pok:
                    answers[prev], queries[prev], quotes[prev] = pans, pq, pquote
                    continue  # redo step n with the new answer
            used_as_input |= used
            answers.append(ans); queries.append(q); quotes.append(quote)
            n += 1

        facts = [f"{q} -> {a}" + (f' (from: "{qt[:200]}")' if qt else "") for q, a, qt in zip(queries, answers, quotes)]
        facts_txt = "\n".join(f"- {f}" for f in facts)
        note = ""
        for attempt in (0, 1):
            obj = self.ask_json(run, self.FINAL.format(q=question, f=facts_txt, p=self.fmt(kept[-10:]), note=note),
                                max_new_tokens=220)
            ans = str(obj.get("answer") or "unknown")
            run.trace.append({"step": "answer", "attempt": attempt, "out": obj})
            if _norm(ans) in {_norm(a) for a in used_as_input} and attempt == 0:
                run.flags["bridge_rejected"] = run.flags.get("bridge_rejected", 0) + 1
                note = (f'\nNote: "{ans}" is only an intermediate fact (the answer to an earlier step), '
                        "not what the question asks for.")
                continue
            if not self._grounded(ans, kept, facts_txt + " " + question) and attempt == 0:
                run.flags["ungrounded_final"] = run.flags.get("ungrounded_final", 0) + 1
                note = "\nNote: your last answer was not found in the facts or passages. Copy it from them."
                continue
            break
        run.answer = ans
        run.sources = [str(x) for x in obj.get("sources", []) if isinstance(x, (str, int))]
        return self._shape_fix(run, question)


class RacerC1v5(RacerC1v4):
    """C1 v4 plus "read wider": when a step's answer is missing or not in the passages, the retry shows the
    passages just before and after the best hits (same article), together with a wider search, instead of
    only a wider search. Aimed at a general failure: the right article was found but the answer sat in a
    neighbouring passage. Designed from failures on the original 200 questions; nothing benchmark-specific."""
    name = "C1v5"

    def _neighbours(self, idxs: list[int], top: int = 3) -> list[int]:
        out: list[int] = []
        n = len(self.tools.chunks)
        for i in idxs[:top]:
            for j in (i - 1, i, i + 1):
                if 0 <= j < n and self.tools.chunks[j].title == self.tools.chunks[i].title and j not in out:
                    out.append(j)
        return out

    def _step4(self, run: Run, n: int, q: str, question: str, exact: list[str], kept: list[int],
               avoid: str = "") -> tuple[str, str, bool]:
        note = (f'\nThe answer is not "{avoid}" (that led nowhere): give a different one that fits.' if avoid else "")
        obj: dict = {}
        first: list[int] = []
        for attempt in (0, 1):
            if attempt == 0:
                query = q
                idxs = self.search(run, "hybrid", query, exact_terms=exact or None, k=8 if avoid else 5)
                first = idxs
            else:  # read wider: neighbours of the best first-round hits + top hits of a wider search
                query = f"{q} {question}"
                wide = self.search(run, "hybrid", query, exact_terms=exact or None, k=5)
                idxs = self._neighbours(first)
                idxs += [i for i in wide if i not in idxs][: max(0, 10 - len(idxs))]
                run.flags["read_wider"] = run.flags.get("read_wider", 0) + 1
            kept += [i for i in idxs if i not in kept]
            obj = self.ask_json(run, self.STEP4.format(q=q, p=self.tools.format(idxs, 140), note=note), max_new_tokens=160)
            ans = str(obj.get("answer") or "unknown")
            ok = self._grounded(ans, idxs) and not (avoid and _norm(ans) == _norm(avoid))
            run.trace.append({"step": f"step{n + 1}", "attempt": attempt, "query": query, "avoid": avoid or None,
                              "hits": [self.tools.chunks[i].id for i in idxs], "out": obj, "grounded": ok})
            if ok:
                return ans, str(obj.get("quote", "")), True
            run.flags["ungrounded_step"] = run.flags.get("ungrounded_step", 0) + 1
        return str(obj.get("answer") or "unknown"), str(obj.get("quote", "")), False


class RacerC2v2(RacerC2):
    """C2 with the general checks that helped C1. The model still picks the tool and query every turn.

    - whole 140-word passages (the 90-word cut hid the answer in some chunks)
    - an answer must occur in a passage it saw or in its saved facts (grounded), else rejected while turns remain
    - an answer must have the form the question asks for (yes/no, or one of the named options)
    - tool advice: hybrid by default, grep for exact names (bm25 was its favourite and its weakest first tool)
    - facts are saved with the sentence they came from, so earlier results are not lost
    """
    name = "C2v2"
    max_steps = 5
    PASSAGE_WORDS = 140

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
                "You are a search agent. Each turn you pick ONE action. Questions often need several hops: "
                "find the thing the question describes, then search for it by name. Tools:\n" + TOOL_HELP +
                "\nanswer: give the final answer when the facts and results are enough.\n"
                "Advice: use hybrid unless you have an exact name or title, then grep. Put the most specific "
                "names from the question in your query.\n\n"
                "Reply with only JSON:\n"
                '{"reason": "<one short sentence>", "tool": "grep|bm25|vector|hybrid|answer", '
                '"query": "<search text; empty if tool is answer>", '
                '"new_fact": "<a fact from LAST RESULTS that helps, with the words copied from the passage, or empty>", '
                '"answer": "<only if tool is answer: the shortest final answer, copied from the passages>", '
                '"sources": ["<passage ids supporting the answer, only when answering>"]}\n\n'
                f"QUESTION: {question}\n\n"
                "FACTS SO FAR:\n" + ("\n".join(f"- {f}" for f in facts) or "(none yet)") + "\n\n"
                f"TOOLS USED:\n{tools_used}\n\n"
                f"LAST RESULTS:\n{self.tools.format(last, self.PASSAGE_WORDS) if used else '(no search yet)'}\n\n"
                + (f"NOTE: {note}\n" if note else "")
                + f"STEPS LEFT: {steps_left}"
            )
            act = self.ask_json(run, prompt, max_new_tokens=200)
            run.trace.append({"step": step + 1, "out": act})
            nf = str(act.get("new_fact") or "").strip()
            if nf and nf.lower() not in {f.lower() for f in facts}:
                facts.append(nf)
            tool = str(act.get("tool") or "").lower().strip()
            query = str(act.get("query") or "").strip()
            note = ""
            final = str(act.get("answer") or "").strip() or query
            if tool == "answer" and final:
                sources = [str(x) for x in act.get("sources", []) if isinstance(x, (str, int)) and str(x).strip()]
                problem = ""
                if GIVE_UP.search(final):
                    problem = "You may not give up while steps remain."
                elif not sources:
                    problem = "An answer must cite passage ids in sources."
                elif not self._grounded(final, run, facts, question):
                    run.flags["ungrounded_answer"] = run.flags.get("ungrounded_answer", 0) + 1
                    problem = f'"{final}" does not appear in any passage you saw. Copy the answer from a passage.'
                elif answer_shape_problem(question, final):
                    run.flags["shape_rejected"] = run.flags.get("shape_rejected", 0) + 1
                    problem = answer_shape_problem(question, final)
                if steps_left > 1 and problem:
                    run.flags["rejected_answer"] = run.flags.get("rejected_answer", 0) + 1
                    note = problem + " If a fact is missing, search for it: first find the person/thing the " \
                                     "question describes, then search for it by name."
                    continue
                run.answer, run.sources = final, sources
                return self._shape_fix2(run, question, facts)
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
        run.flags["forced_answer"] = True
        self.answer_form(run, question, self.tools.format([self.tools.by_id[c] for c in run.seen[-8:]],
                                                          self.PASSAGE_WORDS), facts)
        return self._shape_fix2(run, question, facts)

    def _grounded(self, answer: str, run: Run, facts: list[str], question: str) -> bool:
        a = _norm(answer)
        if not a:
            return False
        hay = [_norm(self.tools.texts[self.tools.by_id[c]]) for c in run.seen] + [_norm(" ".join(facts)), _norm(question)]
        return any(a in h for h in hay)

    def _shape_fix2(self, run: Run, question: str, facts: list[str]) -> Run:
        problem = answer_shape_problem(question, run.answer)
        if not problem:
            return run
        run.flags["shape_retry"] = 1
        obj = self.ask_json(run, (
            "Answer the question from the facts. First write the values you compare, then the answer.\n"
            'Reply with only JSON: {"reasoning": "<one short sentence>", "answer": "<the answer>"}\n'
            f"Note: {problem}\n\nQUESTION: {question}\n\nFACTS FOUND:\n" + "\n".join(f"- {f}" for f in facts)),
            max_new_tokens=160)
        run.trace.append({"step": "shape_retry", "out": obj})
        ans = str(obj.get("answer") or "")
        if ans and not answer_shape_problem(question, ans):
            run.answer = ans
        return run


RACERS = {"A": RacerA, "B": RacerB, "C1": RacerC1, "C2": RacerC2, "D": RacerD, "CB": ClosedBook, "C1v2": RacerC1v2, "C1v3": RacerC1v3, "C1v4": RacerC1v4, "C1v5": RacerC1v5, "C2v2": RacerC2v2}
