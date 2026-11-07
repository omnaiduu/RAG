"""Smoke test: confirm the Kaggle GPU works and measure fp16 matmul speed."""
import json
import subprocess
import time

import torch

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

assert torch.cuda.is_available(), "CUDA not available"
results = []
for i in range(torch.cuda.device_count()):
    device = torch.device(f"cuda:{i}")
    a = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    b = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    a @ b  # warmup
    torch.cuda.synchronize(device)
    start = time.time()
    for _ in range(50):
        a @ b
    torch.cuda.synchronize(device)
    elapsed = time.time() - start
    results.append({
        "index": i,
        "gpu": torch.cuda.get_device_name(i),
        "vram_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1),
        "fp16_matmul_tflops": round(50 * 2 * 4096**3 / elapsed / 1e12, 1),
    })

summary = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpus": results}
print("GPU CHECK OK:", json.dumps(summary, indent=2))
with open("gpu_check.json", "w") as f:
    json.dump(summary, f, indent=2)
