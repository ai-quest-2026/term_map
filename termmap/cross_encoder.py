"""Cross-encoder reranker: reads (term + context sentence, candidate description) together and scores the pair.

Trained listwise: for each training term, a softmax over its top-K candidates (gold + hard negatives from
our own retriever/ranker), cross-entropy on the gold. Initialised from SapBERT (PubMedBERT weights), so it
starts with biomedical vocabulary. Deterministic for a fixed input (seeded, no dropout at inference).
"""
from __future__ import annotations

import random

import numpy as np

from . import config
from .embed import SAPBERT_MODEL
from .normalize import context_window


def query_text(entity: str, chunk: str) -> str:
    return f"{entity} ; {context_window(entity, chunk, 120)}"


def candidate_text(code: str, cs) -> str:
    text = cs.pref.get(code, "")
    if cs.tag.get(code):
        text += f" ({cs.tag[code]})"
    if cs.system == config.ICD and code[:3] in cs.pref and code != code[:3]:
        text += f" [{cs.pref[code[:3]]}]"
    return text


class CrossEncoder:
    def __init__(self, model: str = SAPBERT_MODEL, max_length: int = 128):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.manual_seed(config.SEED)
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=1).to(self.device)
        self.max_length = max_length

    def _logits(self, queries: list[str], cands: list[str]):
        enc = self.tok(queries, cands, padding=True, truncation=True, max_length=self.max_length,
                       return_tensors="pt").to(self.device)
        with self.torch.autocast(self.device, dtype=self.torch.bfloat16, enabled=self.device == "cuda"):
            return self.model(**enc).logits.squeeze(-1).float()

    def fit(self, groups: list[tuple[str, list[str], int]], epochs: int = 2, lr: float = 2e-5,
            groups_per_batch: int = 8, log_every: int = 50):
        """groups: (query, candidate texts, index of the gold candidate)."""
        torch = self.torch
        rng = random.Random(config.SEED)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        steps = epochs * ((len(groups) + groups_per_batch - 1) // groups_per_batch)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.06 * steps)) * max(0.0, 1 - s / steps))
        self.model.train()
        step = 0
        for ep in range(epochs):
            order = list(range(len(groups)))
            rng.shuffle(order)
            for b in range(0, len(order), groups_per_batch):
                batch = [groups[i] for i in order[b:b + groups_per_batch]]
                qs = [q for q, cands, _ in batch for _ in cands]
                cs = [c for _, cands, _ in batch for c in cands]
                logits = self._logits(qs, cs)
                loss, pos = 0.0, 0
                for q, cands, gold in batch:
                    seg = logits[pos:pos + len(cands)]
                    loss = loss + torch.nn.functional.cross_entropy(seg[None], torch.tensor([gold], device=self.device))
                    pos += len(cands)
                loss = loss / len(batch)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                step += 1
                if log_every and step % log_every == 0:
                    print(f"  ce epoch {ep + 1} step {step}/{steps} loss {loss.item():.4f}", flush=True)
        self.model.eval()
        return self

    def score(self, queries: list[str], cands: list[str], batch: int = 256) -> np.ndarray:
        out = []
        self.model.eval()
        with self.torch.inference_mode():
            for s in range(0, len(queries), batch):
                out.append(self._logits(queries[s:s + batch], cands[s:s + batch]).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def save(self, path):
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)
