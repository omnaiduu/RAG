"""Run every racer end-to-end on a tiny corpus slice with a scripted fake LLM (no GPU, no API)."""
import itertools
import json
import sys

sys.path.insert(0, ".")
from rag.evaluate import score  # noqa: E402
from rag.llm import Usage  # noqa: E402
from rag.racers import RACERS  # noqa: E402
from rag.tools import SearchTools, chunk_docs, load_corpus  # noqa: E402


class FakeLLM:
    """Cycles through canned replies covering each racer's expected formats."""
    def __init__(self):
        self.replies = itertools.cycle([
            '{"query": "Tommy Grogan boxer", "exact_terms": ["Tommy Grogan"]}',
            '{"can_answer": false, "found": "x", "next_query": "Goldie Hess title"}',
            '{"reason": "r", "tool": "grep", "query": "Goldie Hess", "answer": "", "new_fact": "f"}',
            'Thought: t\nAction: bm25\nAction Input: Goldie Hess',
            '{"answer": "Pacific Coast lightweight title", "sources": ["c00001"]}',
            '{"reason": "r", "tool": "answer", "query": "", "answer": "Pacific Coast lightweight title"}',
            'Thought: done\nAction: answer\nAction Input: Pacific Coast lightweight title',
        ])

    def chat(self, messages, max_new_tokens=200):
        return next(self.replies), Usage(1, 100, 20, 0.01)


docs = [d for d in load_corpus("data/corpus.jsonl.gz")][:300]
chunks = chunk_docs(docs)
tools = SearchTools(chunks, device="cpu")
print(len(docs), "docs", len(chunks), "chunks")
for name in ["grep", "bm25", "vector", "hybrid"]:
    print(name, [chunks[i].title for i in tools.run(name, docs[5]["title"], 3)])
q = {"qid": "t0", "question": "What title did the boxer who beat Tommy Grogan hold?",
     "answer": "Pacific Coast lightweight title", "type": "bridge", "support_titles": ["Tommy Grogan"]}
results = []
for name, cls in RACERS.items():
    if name == "B":
        continue
    rec = cls(FakeLLM(), tools).solve(q["question"]).record()
    rec.update(qid="t0", racer=name)
    results.append(rec)
    print(name, "->", rec["answer"], rec["llm"], rec["flags"])
print(json.dumps(score(results, {"t0": q}, {c.id: c.title for c in chunks})["overall"], indent=1))
