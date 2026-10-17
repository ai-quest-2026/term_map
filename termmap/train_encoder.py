"""Contrastive fine-tuning of SapBERT on our own mapping pairs (term -> gold code description).

Loss: InfoNCE over the batch (every other pair's description is a negative; pairs with the same code are
masked out so synonyms are not pushed apart) plus one hard negative per term, mined from the base model's
own retrieval (the most similar wrong code). Open-ontology synonym pairs are mixed in so the model keeps
its general synonym knowledge instead of overfitting to the training terms.
"""
from __future__ import annotations

import random

import numpy as np

from . import config
from .embed import SAPBERT_MODEL, SapbertEncoder


def finetune(pairs: list[tuple[str, str, str]], hard_negs: dict[int, str], out_dir, epochs: int = 2,
             lr: float = 2e-5, batch: int = 128, scale: float = 20.0, log_every: int = 20) -> SapbertEncoder:
    """pairs: (term, positive description, code id); hard_negs: pair index -> hard negative description."""
    enc = SapbertEncoder(SAPBERT_MODEL)
    torch, model, tok = enc.torch, enc.model, enc.tok
    dev = model.device
    rng = random.Random(config.SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(pairs) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.06 * steps)) * max(0.0, 1 - s / steps))

    def embed(texts):
        t = tok(texts, padding=True, truncation=True, max_length=enc.max_length, return_tensors="pt").to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            cls = model(**t).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(cls.float(), dim=-1)

    model.train()
    step = 0
    for ep in range(epochs):
        order = list(range(len(pairs)))
        rng.shuffle(order)
        for b in range(0, len(order), batch):
            idx = order[b:b + batch]
            terms = [pairs[i][0] for i in idx]
            docs = [pairs[i][1] for i in idx]
            codes = [pairs[i][2] for i in idx]
            negs = [hard_negs[i] for i in idx if i in hard_negs]
            q = embed(terms)
            d = embed(docs + negs)
            sim = scale * q @ d.T                                         # (B, B + H)
            same = torch.tensor([[ci == cj for cj in codes] for ci in codes], device=dev)
            mask = same & ~torch.eye(len(idx), dtype=torch.bool, device=dev)  # other pairs of the same code
            sim[:, :len(idx)] = sim[:, :len(idx)].masked_fill(mask, -1e4)
            loss = torch.nn.functional.cross_entropy(sim, torch.arange(len(idx), device=dev))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if log_every and step % log_every == 0:
                print(f"  sapbert-ft epoch {ep + 1} step {step}/{steps} loss {loss.item():.4f}", flush=True)
    model.eval()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    enc.cache.clear()
    return enc


def search(enc: SapbertEncoder, mat: np.ndarray, codes_of: list[list[str]], queries: list[str], k: int = 30,
           batch: int = 512) -> list[list[str]]:
    """Top-k distinct codes per query by cosine against a pre-encoded description matrix."""
    import torch

    M = torch.from_numpy(mat.astype(np.float32))
    if torch.cuda.is_available():
        M = M.cuda()
    out = []
    for s in range(0, len(queries), batch):
        q = torch.from_numpy(enc.encode(queries[s:s + batch]).astype(np.float32)).to(M.device)
        top = torch.topk(q @ M.T, min(k * 4, M.shape[0]), dim=1).indices.cpu().numpy()
        for row in top:
            seen, hits = set(), []
            for d in row:
                for c in codes_of[d]:
                    if c not in seen:
                        seen.add(c)
                        hits.append(c)
                if len(hits) >= k:
                    break
            out.append(hits[:k])
    return out
