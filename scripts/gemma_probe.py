"""Probe Gemma 4 E2B/E4B on a Kaggle T4: does it load, is fp16 sane, how fast, does it follow a JSON form."""
import json
import sys
import time

import torch

sys.path.insert(0, ".")
from rag.llm import GemmaLLM, parse_json  # noqa: E402

CONTEXT = " ".join(["The Delhaven Tramway opened in 1911 and was extended twice before closing in 1957."] * 60)
FORM = (
    "You are a search planner. Read the state and fill this JSON form, nothing else:\n"
    '{"reason": "<one short sentence>", "tool": "grep|bm25|vector|hybrid|answer", '
    '"query": "<search text or final answer>", "new_fact": "<fact learned from last results, or empty>"}\n\n'
    "QUESTION: In what year did the tramway in the hometown of the architect of Veltrix Tower close?\n"
    "FACTS SO FAR:\n- Veltrix Tower was designed by Mara Okonjo.\n"
    "LAST RESULTS:\n[1] Mara Okonjo (born 1961 in Delhaven) is a Nigerian-British architect.\n"
    f"[2] {CONTEXT}\nSTEPS LEFT: 3"
)

results = {"torch": torch.__version__, "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
import transformers  # noqa: E402
results["transformers"] = transformers.__version__

for model_id, device_map in [("google/gemma-4-E2B-it", "cuda:0"), ("google/gemma-4-E4B-it", "auto")]:
    for dtype in ["float16"]:
        key = f"{model_id}:{dtype}"
        try:
            t0 = time.time()
            llm = GemmaLLM(model_id, device_map=device_map, dtype=dtype)
            load_s = time.time() - t0
            mem = [round(torch.cuda.memory_allocated(i) / 1e9, 2) for i in range(torch.cuda.device_count())]
            llm.chat([{"role": "user", "content": "Say OK."}], max_new_tokens=5)  # warmup
            runs = []
            for _ in range(3):
                text, u = llm.chat([{"role": "user", "content": FORM}], max_new_tokens=120)
                runs.append(u.as_dict())
            results[key] = {"ok": True, "load_s": round(load_s, 1), "mem_gb": mem, "runs": runs,
                            "sample": text, "parsed": parse_json(text),
                            "gen_tok_per_s": round(runs[-1]["completion_tokens"] / runs[-1]["seconds"], 1)}
            del llm
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            import traceback
            results[key] = {"ok": False, "error": repr(exc), "tb": traceback.format_exc()[-2000:]}
        print(json.dumps({key: results[key]}, indent=1, default=str), flush=True)

with open("/kaggle/working/gemma_probe.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
