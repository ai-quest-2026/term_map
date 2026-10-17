"""Optional LLM adjudication for low-confidence terms (off unless LLM_PROVIDER is set).

The LLM only chooses among the engine's top candidates (so answers stay inside the dictionaries),
runs at temperature 0, and every response is cached on disk so a rerun reproduces the same output.

Providers: gemini (GEMINI_API_KEY) | openai_compat (any OpenAI-style server, e.g. vLLM/Ollama on the
team GPU box). NOTE: sending chart text to a hosted API may violate the dataset terms if the provider
retains data; use hosted providers only for small preliminary experiments.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time

import requests

from . import config

_CACHE = config.CACHE_DIR / "llm"

SYSTEM_PROMPT = """You are an expert medical coder. For each task you get a clinical term extracted from a chart,
the text around it, a target coding system and a numbered list of candidate codes. Pick the ONE candidate
a professional coder would assign for this mention.
Rules:
- Choose only from the listed candidates.
- Match meaning, not wording: abbreviations and misspellings are common.
- Use the context to resolve ambiguous abbreviations and laterality.
- Do not invent specificity: if the text gives no more detail, prefer the unspecified / general code;
  if it states site, side, type, acuity or stage, prefer the code that encodes it.
- For SNOMED CT, prefer the clinical finding/disorder concept for the condition mentioned; choose a
  negative concept ("No ...", "... absent") only if the text clearly states the finding is absent.
- Some tasks show how our own coders coded similar terms. Their conventions (how specific to be, which
  concept they prefer, how they treat negated findings) take priority over your general preference when the
  situation matches. Still choose only from that task's candidates.
Answer with JSON only: [{"task": <task number>, "choice": <candidate number>}, ...]"""


def provider() -> str:
    return os.environ.get("LLM_PROVIDER", "none").strip().lower()


def enabled() -> bool:
    return provider() in ("gemini", "openai_compat")


def _call_gemini(prompt: str) -> str:
    key = os.environ["GEMINI_API_KEY"]
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {"systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
    r = requests.post(url, params={"key": key}, json=body, timeout=120)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def _call_openai_compat(prompt: str) -> str:
    base = os.environ.get("OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1").rstrip("/")
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_COMPAT_API_KEY', 'none')}"}
    body = {"model": os.environ.get("OPENAI_COMPAT_MODEL", ""), "temperature": 0, "seed": config.SEED,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]}
    r = requests.post(f"{base}/chat/completions", json=body, headers=headers, timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _complete(prompt: str) -> str:
    model = os.environ.get("GEMINI_MODEL") if provider() == "gemini" else os.environ.get("OPENAI_COMPAT_MODEL")
    key = hashlib.sha1(f"{provider()}|{model}|{SYSTEM_PROMPT}|{prompt}".encode()).hexdigest()
    path = _CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())["text"]
    call = _call_gemini if provider() == "gemini" else _call_openai_compat
    for attempt in range(6):
        try:
            text = call(prompt)
            break
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code in (429, 500, 503) and attempt < 5:
                time.sleep(min(60, 4 * 2 ** attempt))
                continue
            raise
    _CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"prompt": prompt, "text": text}))
    return text


def _parse(text: str) -> dict[int, int]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return {}
    try:
        return {int(d["task"]): int(d["choice"]) for d in json.loads(m.group(0))}
    except (ValueError, KeyError, TypeError):
        return {}


def adjudicate(tasks: list[dict], batch: int | None = None) -> list[str | None]:
    """tasks: {entity, context, system, candidates: [(code, description), ...]} -> chosen code (or None).
    Terms per request: TERMMAP_LLM_BATCH (default 10; smaller models do better with fewer)."""
    batch = batch or int(os.environ.get("TERMMAP_LLM_BATCH", "10"))
    out: list[str | None] = [None] * len(tasks)
    for s in range(0, len(tasks), batch):
        chunk = tasks[s:s + batch]
        lines = []
        for i, t in enumerate(chunk, 1):
            lines.append(f"### Task {i}\nSystem: {t['system']}\nTerm: {t['entity']}\nContext: {t['context']}")
            if t.get("examples"):
                lines.append("Similar terms coded by our coders:")
                lines += [f"  - '{e}' -> {c} {d}" for e, c, d in t["examples"]]
                lines.append("Candidates:")
            lines += [f"  {j}. {code} - {desc}" for j, (code, desc) in enumerate(t["candidates"], 1)]
        picks = _parse(_complete("\n".join(lines)))
        for i, t in enumerate(chunk, 1):
            j = picks.get(i)
            if j and 1 <= j <= len(t["candidates"]):
                out[s + i - 1] = t["candidates"][j - 1][0]
    return out
