# RAG

Research project: a fast, deterministic small-model RAG pipeline (grep + BM25 + dense retrieval + reranking), benchmarked against naive RAG and frontier-model agentic RAG on accuracy, latency and cost.

## Compute

GPU jobs run on [Modal](https://modal.com). From a local machine with the Modal CLI set up:

```bash
modal run modal_jobs/gpu_check.py
```

Or from GitHub: **Actions → Modal GPU check → Run workflow** (needs the `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` repository secrets).
