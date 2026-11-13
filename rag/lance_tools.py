"""The same four search tools, served by one LanceDB table on disk instead of separate Python structures.

One table holds every chunk: its text, a BM25 full-text index (with word positions, so exact
phrases work) and its 384-d embedding. That replaces bm25s, the numpy matrix and the Python
grep loop:

  grep    -> full-text phrase query  ("exact phrase" with positions)
  bm25    -> full-text query
  vector  -> nearest-neighbour search on the embedding column (cosine)
  hybrid  -> one LanceDB hybrid query (full-text + vector, fused with RRF) plus phrase hits,
             then the same cross-encoder reranker as before

The interface is identical to rag.tools.SearchTools, so every racer runs unchanged.
"""
from __future__ import annotations

import os
import re

import numpy as np

from .tools import EMBED_MODEL, QUERY_PREFIX, Chunk, SearchTools, make_reranker, rrf

TABLE = "chunks"


class LanceSearchTools(SearchTools):
    def __init__(self, chunks: list[Chunk], device: str = "cpu", embeddings: np.ndarray | None = None,
                 batch_size: int = 64, db_path: str = "data/cache/lancedb"):
        import lancedb
        import pyarrow as pa
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self.chunks = chunks
        self.by_id = {c.id: i for i, c in enumerate(chunks)}
        self.texts = [c.indexed_text() for c in chunks]
        self.embedder = SentenceTransformer(EMBED_MODEL, device=device)
        self.reranker = make_reranker(device)

        self.db = lancedb.connect(db_path)
        if TABLE in self.db.table_names() and self.db.open_table(TABLE).count_rows() == len(chunks):
            self.tbl = self.db.open_table(TABLE)
        else:
            if embeddings is None:
                embeddings = self.embedder.encode(self.texts, batch_size=batch_size, normalize_embeddings=True,
                                                  convert_to_numpy=True, show_progress_bar=False)
            emb = embeddings.astype(np.float32)
            data = pa.table({
                "idx": pa.array(range(len(chunks)), pa.int32()),
                "text": pa.array(self.texts),
                "vector": pa.FixedSizeListArray.from_arrays(pa.array(emb.reshape(-1)), emb.shape[1]),
            })
            self.tbl = self.db.create_table(TABLE, data, mode="overwrite")
            # English stemming + stopwords, like the bm25s setup; positions enable phrase queries
            self.tbl.create_fts_index("text", with_position=True, stem=True, remove_stop_words=True,
                                      replace=True)
        self.emb = embeddings.astype(np.float32) if embeddings is not None else None

    # ---- helpers ----------------------------------------------------------
    def _qvec(self, query: str) -> np.ndarray:
        return self.embedder.encode([QUERY_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True,
                                    show_progress_bar=False)[0].astype(np.float32)

    @staticmethod
    def _clean(text: str) -> str:
        # the full-text parser treats some characters as syntax; search words only
        return re.sub(r"[^\w\s'-]", " ", text).strip()

    def _ids(self, q, k: int) -> list[int]:
        return [int(i) for i in q.limit(k).select(["idx"]).to_arrow()["idx"].to_pylist()]

    # ---- the four tools ----------------------------------------------------
    def grep(self, phrase: str, k: int = 5) -> list[int]:
        phrase = self._clean(phrase.strip().strip('"'))
        if not phrase:
            return []
        try:
            return self._ids(self.tbl.search(f'"{phrase}"', query_type="fts"), k)
        except Exception:  # noqa: BLE001 - a phrase made only of stopwords, etc.
            return []

    def bm25_search(self, query: str, k: int = 5) -> list[int]:
        query = self._clean(query)
        if not query:
            return []
        try:
            return self._ids(self.tbl.search(query, query_type="fts"), k)
        except Exception:  # noqa: BLE001
            return []

    def vector_search(self, query: str, k: int = 5) -> list[int]:
        return self._ids(self.tbl.search(self._qvec(query), vector_column_name="vector").distance_type("cosine"), k)

    def hybrid(self, query: str, k: int = 5, exact_terms: list[str] | None = None, pool: int = 30) -> list[int]:
        from lancedb.rerankers import RRFReranker

        text = self._clean(query)
        cands: list[int] = []
        if text:
            try:
                q = (self.tbl.search(query_type="hybrid", vector_column_name="vector")
                     .vector(self._qvec(query)).text(text).rerank(RRFReranker()))
                cands = self._ids(q, 50)
            except Exception:  # noqa: BLE001
                cands = []
        if not cands:
            cands = self.vector_search(query, 50)
        rankings = [cands] + [self.grep(t, 20) for t in exact_terms or []]
        return self.rerank(query, rrf(rankings)[:pool], k)


def make_tools(kind: str, chunks, device, embeddings):
    if kind == "lance":
        return LanceSearchTools(chunks, device=device, embeddings=embeddings,
                                db_path=os.environ.get("LANCEDB_PATH", "data/cache/lancedb"))
    return SearchTools(chunks, device=device, embeddings=embeddings)
