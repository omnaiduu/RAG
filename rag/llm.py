"""LLM backends with uniform usage accounting.

Every call returns (text, Usage). Usage counts prompt/completion tokens and
wall-clock seconds so racers can be compared on cost, not just accuracy.

- GemmaLLM: local Hugging Face model (greedy decoding => deterministic).
- GroqLLM:  OpenAI-compatible Groq API (big-model baselines, question generation).
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0

    def add(self, other: "Usage") -> None:
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.seconds += other.seconds

    def as_dict(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "seconds": round(self.seconds, 3)}


def parse_json(text: str) -> dict | None:
    """Pull the first JSON object out of model output; tolerate code fences and trailing commas."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start:i + 1]
                    for cand in (chunk, re.sub(r",\s*([}\]])", r"\1", chunk)):
                        try:
                            obj = json.loads(cand)
                            if isinstance(obj, dict):
                                return obj
                        except json.JSONDecodeError:
                            pass
                    break
        start = text.find("{", start + 1)
    return None


class GemmaLLM:
    """Greedy-decoding wrapper around a local instruction-tuned Gemma checkpoint."""

    def __init__(self, model_id: str = "google/gemma-4-E2B-it", device_map="cuda:0", dtype: str = "float16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        try:  # the loader's warm-up grabs the whole model size in one block; skip it to load next to other models
            import transformers.modeling_utils as _mu
            _mu.caching_allocator_warmup = lambda *a, **k: None
        except Exception:  # noqa: BLE001
            pass
        self.tok = AutoTokenizer.from_pretrained(model_id)
        torch_dtype = getattr(torch, dtype)
        kw = {"dtype": torch_dtype, "device_map": device_map}
        # GEMMA_QUANT=8bit: int8 weights (bitsandbytes), so a ~8B-parameter model fits on one 16 GB T4
        self.quant = os.environ.get("GEMMA_QUANT", "")
        if self.quant == "8bit":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        except (ValueError, KeyError):
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kw)
        self.model.eval()

    def chat(self, messages: list[dict], max_new_tokens: int = 256) -> tuple[str, Usage]:
        torch = self.torch
        t0 = time.time()
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True)
        ids = {k: v.to(self.model.device) for k, v in ids.items()}
        with torch.inference_mode():
            out = self.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        n_in = ids["input_ids"].shape[1]
        new = out[0, n_in:]
        text = self.tok.decode(new, skip_special_tokens=True).strip()
        return text, Usage(1, int(n_in), int(new.shape[0]), time.time() - t0)


class ToolCallError(RuntimeError):
    """Groq rejected the model's tool call because its arguments were not valid JSON."""

    def __init__(self, failed_generation: str):
        super().__init__("tool_use_failed")
        self.failed_generation = failed_generation


class GroqLLM:
    """Groq chat completions. The key is injected by the sandbox proxy, so a placeholder is fine there."""

    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, model: str = "openai/gpt-oss-120b", reasoning_effort: str | None = "low"):
        import requests

        self.requests = requests
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.key = os.environ.get("GROQ_API_KEY", "proxy-injected")

    def raw(self, messages: list[dict], max_tokens: int = 1024, tools: list | None = None,
            temperature: float = 0.0, json_mode: bool = False) -> tuple[dict, Usage]:
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "seed": 0}
        if tools:
            body["tools"] = tools
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.reasoning_effort and ("gpt-oss" in self.model or "qwen" in self.model):
            body["reasoning_effort"] = self.reasoning_effort
        t0 = time.time()
        for attempt in range(12):
            r = self.requests.post(self.URL, json=body, timeout=120,
                                   headers={"Authorization": f"Bearer {self.key}"})
            if r.status_code == 429:
                err = r.text
                if "per day" in err or "TPD" in err or "RPD" in err:
                    raise RuntimeError(f"Groq daily limit hit for {self.model}: {err[:300]}")
                wait = float(r.headers.get("retry-after", 2 ** min(attempt, 5)))
                time.sleep(min(wait + 0.5, 65))
                continue
            if r.status_code >= 500:
                time.sleep(2 ** min(attempt, 5))
                continue
            if r.status_code == 400 and ("tool_use_failed" in r.text or "failed_generation" in r.text):
                try:
                    gen = r.json()["error"].get("failed_generation", "")
                except Exception:  # noqa: BLE001
                    gen = ""
                raise ToolCallError(gen)
            if r.status_code != 200:
                raise RuntimeError(f"Groq {r.status_code}: {r.text[:500]}")
            data = r.json()
            u = data.get("usage", {})
            usage = Usage(1, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), time.time() - t0)
            return data["choices"][0]["message"], usage
        raise RuntimeError("Groq kept failing")

    def chat(self, messages: list[dict], max_new_tokens: int = 1024, json_mode: bool = False) -> tuple[str, Usage]:
        msg, usage = self.raw(messages, max_tokens=max_new_tokens, json_mode=json_mode)
        return (msg.get("content") or "").strip(), usage
