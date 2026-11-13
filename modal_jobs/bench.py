"""Batched serving benchmark: accuracy AND cost per 1,000 answers, on one GPU, with the step-by-step pipeline.

vLLM serves the model; all questions run at the same time (one thread per question), so the GPU batches the
requests the way a real multi-user service would. Search (LanceDB + bge reranker) runs in the same container.

  modal run modal_jobs/bench.py --model google/gemma-4-E4B-it-qat-w4a16-ct --tag e4b-w4
  modal run modal_jobs/bench.py --model cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4 --tag 26b-a4b-w4

Writes results/raw/vllm-<tag>-fresh.jsonl (one record per question, same format as rag.run)
and results/bench/<tag>.json (throughput, latency, memory, cost).
"""
from __future__ import annotations

import json
import os
import pathlib

import modal

GPU = os.environ.get("GPU_TYPE", "L4")
ROOT = pathlib.Path(__file__).resolve().parents[1]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.33.0", "bm25s", "lancedb==0.41.0", "sentence-transformers", "requests", "pyarrow")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(ROOT / "rag", "/app/rag", ignore=["__pycache__"])
    .add_local_file(ROOT / "data/corpus.jsonl.gz", "/app/data/corpus.jsonl.gz")
    .add_local_file(ROOT / "data/emb_bge_small_f16.npy", "/app/data/emb_bge_small_f16.npy")
    .add_local_file(ROOT / "data/fresh/questions.jsonl", "/app/data/fresh/questions.jsonl")
)
hf_cache = modal.Volume.from_name("rag-hf-cache", create_if_missing=True)
app = modal.App("rag-bench", image=image)


@app.function(gpu=GPU, timeout=5400, volumes={"/root/.cache/huggingface": hf_cache}, cpu=8, memory=32768)
def bench(model: str, racer: str = "C1v3", max_len: int = 6144, mem_util: float = 0.85,
          concurrency: int = 60) -> dict:
    import subprocess
    import sys
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    import requests

    os.chdir("/app")
    sys.path.insert(0, "/app")
    t_start = time.time()
    log = open("/tmp/vllm.log", "w")
    server = subprocess.Popen(
        ["vllm", "serve", model, "--port", "8000", "--max-model-len", str(max_len),
         "--gpu-memory-utilization", str(mem_util), "--served-model-name", "m", "--disable-log-requests"],
        stdout=log, stderr=subprocess.STDOUT)
    for _ in range(900):
        try:
            if requests.get("http://127.0.0.1:8000/health", timeout=2).status_code == 200:
                break
        except Exception:  # noqa: BLE001
            pass
        if server.poll() is not None:
            print(open("/tmp/vllm.log").read()[-6000:])
            raise RuntimeError("vLLM exited")
        time.sleep(2)
    else:
        print(open("/tmp/vllm.log").read()[-6000:])
        raise RuntimeError("vLLM did not start")
    t_ready = time.time()
    hf_cache.commit()
    vlog = open("/tmp/vllm.log").read()
    kv_lines = [l for l in vlog.splitlines() if "KV cache" in l or "Maximum concurrency" in l]

    from rag.lance_tools import make_tools
    from rag.llm import GroqLLM
    from rag.racers import RACERS
    from rag.tools import chunk_docs, load_corpus
    import numpy as np

    chunks = chunk_docs(load_corpus("data/corpus.jsonl.gz"))
    os.environ["LANCEDB_PATH"] = "/tmp/lancedb"
    tools = make_tools("lance", chunks, "cuda", np.load("data/emb_bge_small_f16.npy"))
    lock = threading.Lock()

    class Locked:  # search models are shared; one search at a time, LLM calls run concurrently
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            attr = getattr(self._inner, name)
            if not callable(attr):
                return attr

            def call(*a, **k):
                with lock:
                    return attr(*a, **k)
            return call

    class LocalLLM(GroqLLM):
        URL = "http://127.0.0.1:8000/v1/chat/completions"

        def __init__(self):
            super().__init__("m", reasoning_effort=None)
            self.key = "none"

    llm = LocalLLM()
    shared_tools = Locked(tools)
    questions = [json.loads(l) for l in open("data/fresh/questions.jsonl")]

    def metric(name: str) -> float:
        txt = requests.get("http://127.0.0.1:8000/metrics", timeout=10).text
        return sum(float(l.split()[-1]) for l in txt.splitlines() if l.startswith(name) and not l.startswith("#"))

    def one(q: dict) -> dict:
        t0 = time.time()
        try:
            rec = RACERS[racer](llm, shared_tools).solve(q["question"]).record()
        except Exception as exc:  # noqa: BLE001
            rec = {"answer": "unknown", "sources": [], "trace": [], "seen": [], "tool_calls": 0,
                   "tool_seconds": 0.0, "llm": {}, "flags": {"error": repr(exc)}}
        rec.update({"racer": f"{racer}@lance", "qid": q["qid"], "model": model, "gold": q["answer"],
                    "latency_s": round(time.time() - t0, 2)})
        return rec

    gen0, prompt0 = metric("vllm:generation_tokens_total"), metric("vllm:prompt_tokens_total")
    peak = {"mb": 0}
    stop = threading.Event()

    def watch_mem():
        while not stop.is_set():
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout.strip()
            if out:
                peak["mb"] = max(peak["mb"], int(out.splitlines()[0]))
            time.sleep(1)

    threading.Thread(target=watch_mem, daemon=True).start()
    t0 = time.time()
    with ThreadPoolExecutor(concurrency) as ex:
        records = list(ex.map(one, questions))
    wall = time.time() - t0
    stop.set()
    gen, prompt = metric("vllm:generation_tokens_total") - gen0, metric("vllm:prompt_tokens_total") - prompt0
    lat = sorted(r["latency_s"] for r in records)
    gpu_name = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    stats = {
        "model": model, "racer": racer, "gpu": gpu_name, "questions": len(records), "concurrency": concurrency,
        "startup_s": round(t_ready - t_start, 1), "wall_s": round(wall, 1),
        "answers_per_min": round(60 * len(records) / wall, 2),
        "latency_p50_s": lat[len(lat) // 2], "latency_max_s": lat[-1],
        "prompt_tokens": int(prompt), "generated_tokens": int(gen),
        "gen_tokens_per_s": round(gen / wall, 1), "peak_gpu_mem_mb": peak["mb"],
        "errors": sum(bool(r["flags"].get("error")) for r in records),
        "vllm_kv_log": kv_lines[-4:],
    }
    server.terminate()
    return {"stats": stats, "records": records}


@app.local_entrypoint()
def main(model: str, tag: str, racer: str = "C1v3", max_len: int = 6144, mem_util: float = 0.85,
         concurrency: int = 60):
    out = bench.remote(model, racer, max_len, mem_util, concurrency)
    raw = ROOT / f"results/raw/vllm-{tag}-fresh.jsonl"
    raw.parent.mkdir(parents=True, exist_ok=True)
    with open(raw, "w") as f:
        for r in out["records"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (ROOT / "results/bench").mkdir(parents=True, exist_ok=True)
    (ROOT / f"results/bench/{tag}.json").write_text(json.dumps(out["stats"], indent=2))
    print(json.dumps(out["stats"], indent=2))
