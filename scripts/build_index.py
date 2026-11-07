"""Precompute chunk embeddings once (CPU or GPU) into a cache file."""
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from rag.tools import SearchTools, chunk_docs, load_corpus  # noqa: E402

device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
out = sys.argv[2] if len(sys.argv) > 2 else "data/cache/emb_bge_small.npy"
chunks = chunk_docs(load_corpus("data/corpus.jsonl.gz"))
t0 = time.time()
tools = SearchTools(chunks, device=device, batch_size=128)
np.save(out, tools.emb.astype(np.float16))
print(f"{len(chunks)} chunks embedded on {device} in {time.time() - t0:.0f}s -> {out}")
