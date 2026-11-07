"""Smoke test: start a GPU container on Modal and confirm CUDA works.

Run locally:   modal run modal_jobs/gpu_check.py
Pick a GPU:    GPU_TYPE=L4 modal run modal_jobs/gpu_check.py
"""
import os

import modal

GPU_TYPE = os.environ.get("GPU_TYPE", "T4")

app = modal.App("rag-gpu-check")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch")


@app.function(gpu=GPU_TYPE, image=image, timeout=600)
def check() -> dict:
    import subprocess
    import time

    import torch

    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda")
    a = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    b = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(50):
        a @ b
    torch.cuda.synchronize()
    elapsed = time.time() - start
    tflops = 50 * 2 * 4096**3 / elapsed / 1e12

    return {
        "gpu": torch.cuda.get_device_name(0),
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "fp16_matmul_tflops": round(tflops, 1),
    }


@app.local_entrypoint()
def main():
    result = check.remote()
    print("GPU CHECK OK:", result)
