"""Corpus chunking and the four search tools every racer shares.

grep    exact (case-insensitive) phrase match; ranks by match count. No index, no model.
bm25    keyword ranking (bm25s).
vector  dense similarity (bge-small-en-v1.5, cosine).
hybrid  RRF fusion of bm25 + vector (+ grep for exact terms), then a cross-encoder
        reranker (bge-reranker-base) over the top candidates.

All ranking is deterministic: ties break on chunk index.
"""
from __future__ import annotations

import json
import re
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-base")  # "none" = no reranker
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


QWEN_RERANK_PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and '
                      'the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
QWEN_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def make_reranker(device: str):
    if RERANK_MODEL == "none":
        return None
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL, max_length=512 if "Qwen3" in RERANK_MODEL else 384, device=device)


def rerank_pairs(query: str, docs: list[str]) -> list[tuple[str, str]]:
    if "Qwen3-Reranker" not in RERANK_MODEL:
        return [(query, d) for d in docs]
    q = (QWEN_RERANK_PREFIX + "<Instruct>: Given a question, find passages that help answer it\n"
         f"<Query>: {query}\n")
    return [(q, f"<Document>: {d}{QWEN_RERANK_SUFFIX}") for d in docs]


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    section: str
    text: str

    def header(self) -> str:
        return f"{self.title} ({self.section})" if self.section else self.title

    def indexed_text(self) -> str:
        # Title/section header on every chunk: a cheap, deterministic form of "contextual" chunks.
        return f"{self.header()}: {self.text}"


def load_corpus(path: str | Path) -> list[dict]:
    import gzip
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return [json.loads(line) for line in f]


def chunk_docs(docs: list[dict], size: int = 140, overlap: int = 30) -> list[Chunk]:
    chunks: list[Chunk] = []
    for d in docs:
        section, buf = "", []
        blocks: list[tuple[str, list[str]]] = []
        for line in d["text"].split("\n"):
            m = re.match(r"^#{2,3} (.*)$", line.strip())
            if m:
                if buf:
                    blocks.append((section, buf))
                section, buf = m.group(1), []
            else:
                buf.extend(line.split())
        if buf:
            blocks.append((section, buf))
        for sec, words in blocks:
            step = size - overlap
            for start in range(0, max(len(words) - overlap, 1), step):
                piece = words[start:start + size]
                if len(piece) < 15 and start > 0:
                    continue
                chunks.append(Chunk(f"c{len(chunks):05d}", d["id"], d["title"], sec, " ".join(piece)))
    return chunks


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda i: (-scores[i], i))


class SearchTools:
    def __init__(self, chunks: list[Chunk], device: str = "cpu", embeddings: np.ndarray | None = None,
                 batch_size: int = 64):
        import bm25s
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self.chunks = chunks
        self.by_id = {c.id: i for i, c in enumerate(chunks)}
        self.texts = [c.indexed_text() for c in chunks]
        self.lower = [t.lower() for t in self.texts]

        self.bm25 = bm25s.BM25()
        self.bm25.index(bm25s.tokenize(self.texts, stopwords="en", show_progress=False), show_progress=False)
        self._bm25s = bm25s

        self.embedder = SentenceTransformer(EMBED_MODEL, device=device)
        if embeddings is None:
            embeddings = self.embedder.encode(self.texts, batch_size=batch_size, normalize_embeddings=True,
                                              convert_to_numpy=True, show_progress_bar=False)
        self.emb = embeddings.astype(np.float32)
        self.reranker = make_reranker(device)

    # ---- individual tools -------------------------------------------------
    def grep(self, phrase: str, k: int = 5) -> list[int]:
        phrase = phrase.strip().strip('"').lower()
        if not phrase:
            return []
        pat = re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)")
        hits = [(len(pat.findall(t)), i) for i, t in enumerate(self.lower) if phrase in t]
        hits = [(n, i) for n, i in hits if n > 0]
        hits.sort(key=lambda x: (-x[0], x[1]))
        return [i for _, i in hits[:k]]

    def bm25_search(self, query: str, k: int = 5) -> list[int]:
        toks = self._bm25s.tokenize([query], stopwords="en", show_progress=False)
        if not toks.vocab:
            return []
        res, scores = self.bm25.retrieve(toks, k=min(k, len(self.chunks)), show_progress=False)
        return [int(i) for i, s in zip(res[0], scores[0]) if s > 0]

    def vector_search(self, query: str, k: int = 5) -> list[int]:
        q = self.embedder.encode([QUERY_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False)[0]
        sims = self.emb @ q
        top = np.argpartition(-sims, min(k, len(sims) - 1))[: k * 2]
        return [int(i) for i in sorted(top, key=lambda i: (-sims[i], i))[:k]]

    def rerank(self, query: str, cands: list[int], k: int = 5) -> list[int]:
        if not cands:
            return []
        k = int(os.environ.get("SHOW_K", k))  # experiment: show more passages to the model
        if self.reranker is None:  # experiment: no reranker, keep the database's fused order
            return cands[:k]
        scores = self.reranker.predict(rerank_pairs(query, [self.texts[i] for i in cands]), show_progress_bar=False)
        order = sorted(range(len(cands)), key=lambda j: (-float(scores[j]), cands[j]))
        return [cands[j] for j in order[:k]]

    def hybrid(self, query: str, k: int = 5, exact_terms: list[str] | None = None, pool: int = 30) -> list[int]:
        rankings = [self.bm25_search(query, 50), self.vector_search(query, 50)]
        for term in exact_terms or []:
            rankings.append(self.grep(term, 20))
        cands = rrf(rankings)[:pool]
        return self.rerank(query, cands, k)

    # ---- helpers ------------------------------------------------------------
    def run(self, tool: str, query: str, k: int = 5) -> list[int]:
        tool = (tool or "").lower().strip()
        if tool == "grep":
            return self.grep(query, k)
        if tool == "bm25":
            return self.bm25_search(query, k)
        if tool == "vector":
            return self.vector_search(query, k)
        return self.hybrid(query, k)

    def format(self, idxs: list[int], max_words: int = 110) -> str:
        if not idxs:
            return "(no results)"
        lines = []
        for i in idxs:
            c = self.chunks[i]
            words = c.text.split()
            body = " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")
            lines.append(f"[{c.id}] {c.header()}: {body}")
        return "\n".join(lines)
