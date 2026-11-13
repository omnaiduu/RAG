"""Finish the exam build (data_build/make_exam.py) on Modal, with the SAME two question-writing models that Groq
serves (gpt-oss-20b and Qwen3.8-27B), self-hosted with vLLM on one GPU because Groq's free daily limit ran out.
Same prompts, same checks, same closed-book filter, temperature 0. Resumes from data/exam/candidates.jsonl.

  modal run modal_jobs/make_exam_modal.py::download     # CPU only: put both models in the cache volume
  modal run --detach modal_jobs/make_exam_modal.py::run  # GPU: build; files saved to volume rag-exam as it goes
  modal volume get --force rag-exam / data/exam/        # fetch candidates/closed_book_filter/questions.jsonl
"""
from __future__ import annotations

import pathlib

import modal

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS = {  # served name (as on Groq) -> (Hugging Face id, port, GPU memory share)
    "openai/gpt-oss-20b": ("openai/gpt-oss-20b", 8001, 0.30),
    "qwen/qwen3.8-27b": ("Qwen/Qwen3.8-27B-FP8", 8002, 0.55),
}
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.33.0", "requests", "huggingface_hub[hf_transfer]")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(ROOT / "rag", "/app/rag", ignore=["__pycache__"])
    .add_local_dir(ROOT / "data_build", "/app/data_build", ignore=["__pycache__"])
    .add_local_file(ROOT / "data/corpus.jsonl.gz", "/app/data/corpus.jsonl.gz")
    .add_local_file(ROOT / "data/graph.json.gz", "/app/data/graph.json.gz")
    .add_local_file(ROOT / "data/questions.jsonl", "/app/data/questions.jsonl")
    .add_local_file(ROOT / "data/candidates.jsonl", "/app/data/candidates.jsonl")
    .add_local_dir(ROOT / "data/fresh", "/app/data/fresh")
    .add_local_file(ROOT / "data/exam/candidates.jsonl", "/app/data/exam_seed/candidates.jsonl")
)
hf_cache = modal.Volume.from_name("rag-hf-cache", create_if_missing=True)
exam_vol = modal.Volume.from_name("rag-exam", create_if_missing=True)
app = modal.App("rag-make-exam", image=image)


@app.function(timeout=3600, volumes={"/root/.cache/huggingface": hf_cache}, cpu=4, memory=8192)
def download() -> None:
    from huggingface_hub import snapshot_download
    for hf, _, _ in MODELS.values():
        print("downloading", hf, flush=True)
        snapshot_download(hf, max_workers=16)
        hf_cache.commit()
        print("done", hf, flush=True)


@app.function(gpu="A100-80GB", timeout=3600, volumes={"/root/.cache/huggingface": hf_cache, "/out": exam_vol},
              cpu=8, memory=32768)
def build() -> dict:
    import os
    import shutil
    import subprocess
    import sys
    import time

    import requests

    os.chdir("/app")
    sys.path.insert(0, "/app")
    os.makedirs("data/exam", exist_ok=True)
    shutil.copy("data/exam_seed/candidates.jsonl", "data/exam/candidates.jsonl")
    for name, (hf, port, share) in MODELS.items():  # one after the other: each checks free memory at start
        log = open(f"/tmp/vllm-{port}.log", "w")
        p = subprocess.Popen(["vllm", "serve", hf, "--port", str(port), "--served-model-name", name,
                              "--max-model-len", "4096", "--gpu-memory-utilization", str(share),
                              "--max-num-seqs", "64"],  # we send <= 32 at once; Qwen3.8's Mamba cache fits ~198
                             stdout=log, stderr=subprocess.STDOUT)
        for _ in range(900):
            try:
                if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                    break
            except Exception:  # noqa: BLE001
                pass
            if p.poll() is not None:
                print(open(f"/tmp/vllm-{port}.log").read()[-5000:])
                raise RuntimeError(f"vLLM for {name} exited")
            time.sleep(2)
            if _ % 30 == 0:
                tail = open(f"/tmp/vllm-{port}.log").read().strip().splitlines()[-1:] or [""]
                print(f"[{name} loading] {tail[0][-160:]}", flush=True)
        print(f"{name} ready", flush=True)
    hf_cache.commit()

    import rag.llm as llm_mod

    class LocalLLM(llm_mod.GroqLLM):
        def raw(self, messages, max_tokens=1024, tools=None, temperature=0.0, json_mode=False):
            port = MODELS[self.model][1]
            body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                    "temperature": temperature, "seed": 0}
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            if "gpt-oss" in self.model and self.reasoning_effort:
                body["reasoning_effort"] = self.reasoning_effort  # "low", as on Groq
            if "qwen" in self.model:
                body["chat_template_kwargs"] = {"enable_thinking": False}  # Groq: reasoning_effort "none"
            t0 = time.time()
            r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=body, timeout=300)
            if r.status_code != 200:
                raise RuntimeError(f"vLLM {r.status_code}: {r.text[:300]}")
            data = r.json()
            u = data.get("usage", {})
            return data["choices"][0]["message"], llm_mod.Usage(1, u.get("prompt_tokens", 0),
                                                                u.get("completion_tokens", 0), time.time() - t0)

    import threading

    def save():
        for f in ("candidates.jsonl", "closed_book_filter.jsonl", "questions.jsonl"):
            if os.path.exists(f"data/exam/{f}"):
                shutil.copy(f"data/exam/{f}", f"/out/{f}")
        exam_vol.commit()

    def saver():
        while True:
            time.sleep(60)
            try:
                save()
            except Exception as exc:  # noqa: BLE001
                print("save failed", exc, flush=True)

    threading.Thread(target=saver, daemon=True).start()

    import data_build.make_questions as mq
    mq.GroqLLM = LocalLLM
    os.environ["WORKERS_PER_MODEL"] = "16"
    os.environ["CB_PARALLEL"] = "16"
    import data_build.make_exam as me
    me.main()
    save()
    print("EXAM BUILD DONE", flush=True)
    out = {}
    for f in ("candidates.jsonl", "closed_book_filter.jsonl", "questions.jsonl"):
        out[f] = open(f"data/exam/{f}").read()
    return out


@app.local_entrypoint()
def run():
    call = build.spawn()  # not tied to this local process: the build keeps running if it exits
    print("spawned", call.object_id, flush=True)


def main():  # not used with --detach; kept for interactive runs
    out = build.remote()
    for name, text in out.items():
        (ROOT / "data/exam" / name).write_text(text)
        print(name, len(text.splitlines()), "lines")
