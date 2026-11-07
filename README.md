# RAG

Research project: a fast, deterministic small-model RAG pipeline (grep + BM25 + dense retrieval + reranking), benchmarked against naive RAG and frontier-model agentic RAG on accuracy, latency and cost.

## Compute

GPU jobs run on **Kaggle** (free, 2× Tesla T4, 16 GB each, ~30 GPU-hours/week). Each job is a folder under `kaggle_jobs/` with a `kernel-metadata.json`:

```bash
kaggle kernels push -p kaggle_jobs/gpu_check          # submit
kaggle kernels status ommnaidu/rag-gpu-check          # poll
kaggle kernels output ommnaidu/rag-gpu-check -p out/  # fetch results
```

Needs a Kaggle API token in `KAGGLE_API_TOKEN` or `~/.kaggle/access_token` (never commit it).

`modal_jobs/` holds the same smoke test for [Modal](https://modal.com) (needs a payment method on the Modal account): `modal run modal_jobs/gpu_check.py`, or **Actions → Modal GPU check** on GitHub.
