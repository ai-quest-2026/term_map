"""LLM selector (BeLink-style): a LoRA-tuned instruction model picks the right code among the ranker's top-k.

Why fine-tuned and not prompted: zero-shot Qwen-14B/32B were less accurate than our ranker on the terms they
were sent (docs/experiments.md L2-L5) because they do not know the local coding conventions; a selector trained
on the labelled terms learns them (BeLink, SIGIR 2026: +3.2 points over a fine-tuned cross-encoder on average).

Input: the term marked in its context, the coding system and k lettered candidates (official wording + one
synonym), plus a "none of the above" option. Output: one letter. At inference the letter probabilities
(renormalised over the offered letters) become the `sel_p` feature of the second-level ranker; the selector
never generates a code, so answers stay inside the dictionaries. Scoring is a single deterministic forward pass
(no sampling). Training: LoRA r=16 on all linear layers, loss on the answer token only, shuffled option order
(LLMs favour positions) — see scripts/train_selector.py.
"""
from __future__ import annotations

import hashlib
import random
import string

import numpy as np
import pandas as pd

from . import config

LETTERS = list(string.ascii_uppercase)
K_SEL = 20
SYSTEM_PROMPT = ("You are an expert medical coder. Pick the single code a professional coder would assign to the "
                 "marked clinical term, following the coding conventions you were trained on. Answer with one letter.")
_NAMES = {config.ICD: "ICD-10-CM", config.SNOMED: "SNOMED CT"}


def _context(entity: str, chunk: str, width: int = 150) -> str:
    e, c = str(entity).strip(), " ".join(str(chunk).split())
    i = c.lower().find(e.lower()) if e else -1
    if i < 0:
        return f"<<{e}>>"
    return f"{c[max(0, i - width): i]}<<{e}>>{c[i + len(e): i + len(e) + width]}".strip()


def option_text(code: str, cs) -> str:
    text = cs.pref.get(code, "")
    syn = cs.synonyms(code, 1, exclude=text) if hasattr(cs, "synonyms") else []
    if syn:
        text += f" | {syn[0]}"
    if cs.tag.get(code):
        text += f" ({cs.tag[code]})"
    return f"{code} {text}"


def build_prompt(entity: str, chunk: str, system: str, options: list[str]) -> str:
    lines = [f"Coding system: {_NAMES.get(system, system)}", f"Context: {_context(entity, chunk)}",
             f"Term: {entity}", "Options:"]
    lines += [f"{LETTERS[i]}. {o}" for i, o in enumerate(options)]
    lines.append(f"{LETTERS[len(options)]}. none of the above")
    lines.append("Answer:")
    return "\n".join(lines)


def make_examples(frame: pd.DataFrame, texts: dict, cs, system: str, score: str, k: int = K_SEL,
                  shuffle: bool = True, with_answer: bool = True) -> list[dict]:
    """frame: item, code, label, <score>; texts: item -> (entity, chunk). One example per item; in training the
    option order is shuffled (seeded per term) and a term whose gold is not among the options answers 'none'."""
    out = []
    top = frame.sort_values(["item", score, "code"], ascending=[True, False, True]).groupby("item").head(k)
    for item, g in top.groupby("item", sort=True):
        codes = list(g.code)
        ent, chunk = texts[item]
        if shuffle:
            rnd = random.Random(int(hashlib.md5(f"{system}|{ent}".encode()).hexdigest(), 16) % 2**32)
            rnd.shuffle(codes)
        ex = {"item": item, "codes": codes,
              "prompt": build_prompt(ent, chunk, system, [option_text(c, cs) for c in codes])}
        if with_answer and "label" in g.columns:
            gold = g.loc[g.label == 1, "code"]
            ex["answer"] = LETTERS[codes.index(gold.iloc[0])] if len(gold) else LETTERS[len(codes)]
        out.append(ex)
    return out


def chat(tok, prompt: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def letter_ids(tok, n: int) -> list[int]:
    ids = []
    for L in LETTERS[:n]:
        t = tok.encode(L, add_special_tokens=False)
        if len(t) != 1:
            raise ValueError(f"letter {L!r} is not a single token for this tokenizer")
        ids.append(t[0])
    return ids


class Selector:
    """Loads base model + LoRA adapter; `score` returns letter probabilities per (item, code)."""

    def __init__(self, adapter: str, base: str | None = None, batch: int = 8):
        import json
        import os

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(config.SEED)
        meta = json.loads(open(os.path.join(adapter, "selector.json")).read())
        base = base or meta["base"]
        self.k = meta.get("k", K_SEL)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(adapter)
        self.tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda"
                                                     if torch.cuda.is_available() else None)
        self.model = PeftModel.from_pretrained(model, adapter).eval()
        self.batch = batch

    def score_examples(self, examples: list[dict]) -> pd.DataFrame:
        torch, rows = self.torch, []
        order = sorted(range(len(examples)), key=lambda i: len(examples[i]["prompt"]))
        for s in range(0, len(order), self.batch):
            exs = [examples[i] for i in order[s:s + self.batch]]
            enc = self.tok([chat(self.tok, e["prompt"]) for e in exs], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.model.device)
            with torch.inference_mode():
                logits = self.model(**enc).logits[:, -1, :].float()
            for e, lg in zip(exs, logits):
                ids = letter_ids(self.tok, len(e["codes"]) + 1)
                p = torch.softmax(lg[ids], dim=-1).cpu().numpy()
                rows += [{"item": e["item"], "code": c, "sel_p": float(p[j])} for j, c in enumerate(e["codes"])]
        return pd.DataFrame(rows, columns=["item", "code", "sel_p"])

    def score(self, items: pd.DataFrame, frame: pd.DataFrame, cs, system: str, score: str = "prob") -> pd.DataFrame:
        texts = {i: (items.entity.iloc[i], items.chunk_text.iloc[i]) for i in range(len(items))}
        return self.score_examples(make_examples(frame, texts, cs, system, score, self.k, shuffle=False,
                                                 with_answer=False))
