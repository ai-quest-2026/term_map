"""Cross-encoder reranker: reads (term in its context, candidate description) together and scores the pair.

Trained listwise: for each training term, a softmax over its top-K candidates (gold + hard negatives taken
from the OUT-OF-FOLD ranker, i.e. the same kind of list it will see at test time), cross-entropy on the gold.
The mention is marked inside its local context ("... [START] lad [END] ...", as in BLINK / xMEN), and the
candidate side carries the full official wording plus synonyms. Initialised from SapBERT (PubMedBERT weights)
unless TERMMAP_CE_INIT names another checkpoint. Deterministic for a fixed input (seeded, deterministic
kernels, eager attention, no dropout at inference).
"""
from __future__ import annotations

import os
import random

import numpy as np

from . import config
from .embed import SAPBERT_MODEL
from .normalize import basic

MARKERS = ["[START]", "[END]"]
CTX_CHARS = 128


def query_text(entity: str, chunk: str, width: int = CTX_CHARS) -> str:
    """The term marked inside its local context; the term alone when the chunk does not contain it."""
    e, c = str(entity).strip(), " ".join(str(chunk).split())
    i = c.lower().find(e.lower()) if e else -1
    if i < 0:
        return f"[START] {e} [END]"
    left, right = c[max(0, i - width): i], c[i + len(e): i + len(e) + width]
    return f"{left} [START] {e} [END] {right}".strip()


def candidate_text(code: str, cs, n_syn: int = 2) -> str:
    """Official wording | up to n_syn other descriptions (tag) [ICD category title]."""
    pref = cs.pref.get(code, "")
    text = pref
    syns = cs.synonyms(code, n_syn, exclude=pref) if hasattr(cs, "synonyms") else []
    if syns:
        text += " | " + " | ".join(syns)
    if cs.tag.get(code):
        text += f" ({cs.tag[code]})"
    if cs.system == config.ICD and code[:3] in cs.pref and code != code[:3]:
        text += f" [{cs.pref[code[:3]]}]"
    return text


def ce_init() -> str:
    return os.environ.get("TERMMAP_CE_INIT", "").strip() or SAPBERT_MODEL


class CrossEncoder:
    def __init__(self, model: str | None = None, max_length: int = 192):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # Training on GPU must give the same weights every run: deterministic kernels, eager attention
        # (fused attention backward uses atomics), cuBLAS workspace pinned in config.
        torch.manual_seed(config.SEED)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        model = model or ce_init()
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model)
        if any(m not in self.tok.get_vocab() for m in MARKERS):
            self.tok.add_special_tokens({"additional_special_tokens": MARKERS})
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model, num_labels=1, attn_implementation="eager")
        self.model.resize_token_embeddings(len(self.tok))
        self.model.to(self.device)
        self.max_length = max_length
        self.history: list[dict] = []

    def _logits(self, queries: list[str], cands: list[str]):
        enc = self.tok(queries, cands, padding=True, truncation="longest_first", max_length=self.max_length,
                       return_tensors="pt").to(self.device)
        with self.torch.autocast(self.device, dtype=self.torch.bfloat16, enabled=self.device == "cuda"):
            return self.model(**enc).logits.squeeze(-1).float()

    def accuracy(self, groups: list[tuple[str, list[str], int]]) -> float:
        """Share of groups whose gold candidate gets the top score."""
        if not groups:
            return float("nan")
        q = [g[0] for g in groups for _ in g[1]]
        c = [x for g in groups for x in g[1]]
        s, pos, hit = self.score(q, c), 0, 0
        for _, cands, gold in groups:
            hit += int(np.argmax(s[pos:pos + len(cands)]) == gold)
            pos += len(cands)
        return hit / len(groups)

    def fit(self, groups: list[tuple[str, list[str], int]], epochs: int = 4, lr: float = 2e-5,
            groups_per_batch: int = 8, dev: list | None = None, log_every: int = 100,
            min_dev_after_first: float | None = None, tag: str = ""):
        """groups: (query, candidate texts, index of the gold candidate). With `dev`, accuracy is measured
        after every epoch and the best epoch's weights are kept (early stopping). `min_dev_after_first`
        aborts training (raises RuntimeError) if the first epoch is below it: a failing run shows in minutes."""
        torch = self.torch
        rng = random.Random(config.SEED)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        steps = epochs * ((len(groups) + groups_per_batch - 1) // groups_per_batch)
        warm = max(1, int(0.06 * steps))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / steps))
        best, best_state, step = -1.0, None, 0
        for ep in range(epochs):
            self.model.train()
            order = list(range(len(groups)))
            rng.shuffle(order)
            run_loss = 0.0
            for b in range(0, len(order), groups_per_batch):
                batch = [groups[i] for i in order[b:b + groups_per_batch]]
                logits = self._logits([q for q, cands, _ in batch for _ in cands],
                                      [c for _, cands, _ in batch for c in cands])
                loss, pos = 0.0, 0
                for _, cands, gold in batch:
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
                run_loss += loss.item()
                if log_every and step % log_every == 0:
                    print(f"  [ce{tag}] epoch {ep + 1} step {step}/{steps} loss {run_loss / log_every:.4f}", flush=True)
                    run_loss = 0.0
            self.model.eval()
            rec = {"epoch": ep + 1, "train_acc": self.accuracy(groups[:400])}
            if dev:
                rec["dev_acc"] = self.accuracy(dev)
            self.history.append(rec)
            print(f"  [ce{tag}] epoch {ep + 1}: " + " ".join(f"{k}={v:.4f}" for k, v in rec.items() if k != "epoch"),
                  flush=True)
            if dev:
                if ep == 0 and min_dev_after_first is not None and rec["dev_acc"] < min_dev_after_first:
                    raise RuntimeError(f"cross-encoder gate failed: dev acc {rec['dev_acc']:.3f} after epoch 1 "
                                       f"< {min_dev_after_first}")
                if rec["dev_acc"] > best:
                    best = rec["dev_acc"]
                    best_state = {k: v.detach().to("cpu", copy=True) for k, v in self.model.state_dict().items()}
        if best_state is not None:
            self.model.load_state_dict(best_state)
        # Release training state before scoring (an optimizer + grads left on the GPU preceded a cuBLAS
        # internal error when scoring right after training on the WSL box).
        self.model.zero_grad(set_to_none=True)
        del opt, sched, best_state
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        self.model.eval()
        return self

    def score(self, queries: list[str], cands: list[str], batch: int = 256) -> np.ndarray:
        try:
            return self._score(queries, cands, batch)
        except RuntimeError as e:  # transient CUDA/cuBLAS failure: free the cache, retry once with small batches
            if self.device != "cuda":
                raise
            print(f"  [ce] scoring failed ({str(e).splitlines()[0][:120]}); retrying with batch {batch // 4}", flush=True)
            self.torch.cuda.synchronize()
            self.torch.cuda.empty_cache()
            return self._score(queries, cands, max(16, batch // 4))

    def _score(self, queries: list[str], cands: list[str], batch: int) -> np.ndarray:
        out = []
        self.model.eval()
        with self.torch.inference_mode():
            for s in range(0, len(queries), batch):
                out.append(self._logits(queries[s:s + batch], cands[s:s + batch]).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def save(self, path):
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)


def synonym_groups(cs, n: int, k: int = 15) -> list[tuple[str, list[str], int]]:
    """Distant supervision for the cross-encoder: open-ontology / CDC synonyms ("hyperpiesia" -> I10) as
    queries, our own dictionary search results as candidates, the synonym's code as the answer.
    No labelled terms are used, so this adds no leakage."""
    import pandas as pd

    rng = random.Random(config.SEED)
    al = []
    for f in sorted(config.EXTERNAL_DIR.glob(f"{cs.system}_aliases*.parquet")):
        a = pd.read_parquet(f)
        al += [(basic(t), c) for t, c in zip(a.description, a.code)
               if c in cs.pref and basic(t) and basic(t) != basic(cs.pref[c])]
    al = sorted(set(al))
    rng.shuffle(al)
    al = al[:n]
    res = cs.retriever.search([t for t, _ in al], 30)
    groups = []
    for (t, code), (ch, wd, de, sp) in zip(al, res):
        cands = []
        for h in (sp or []) + ch + wd:  # SapBERT hits first: the hardest negatives
            if h.code not in cands:
                cands.append(h.code)
        cands = cands[:k]
        if code not in cands:
            cands[-1:] = [code]
        cands = sorted(cands, key=lambda c: rng.random())  # gold position must not leak
        groups.append((query_text(t, ""), [candidate_text(c, cs) for c in cands], cands.index(code)))
    return groups
