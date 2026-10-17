"""Does fine-tuning SapBERT on our pairs improve retrieval? Clean test on the 20% hold-out (GPU).

    python -m scripts.finetune_sapbert [--epochs 2] [--aliases 15000]

1. Training pairs come only from the 80% train side: term -> gold display (and preferred term), plus a
   sample of open-ontology synonym pairs (alias -> preferred term) to keep general synonym knowledge.
2. Hard negatives: the base model's most similar wrong code for each training term.
3. Fine-tune, re-encode both dictionaries, and compare base vs fine-tuned retrieval on held-out terms:
   accuracy@1 and recall@5/15/30, split by codes seen / unseen in training.
The model is saved to data/cache/<name>/. A synonyms-only model (--no-train-terms) never sees labelled terms,
so it can replace the base encoder in the full pipeline without leakage:
    TERMMAP_SAPBERT_MODEL=data/cache/sapbert_ft_syn python -m scripts.evaluate --cv 5
"""
from __future__ import annotations

import argparse
import random
import time

import numpy as np
import pandas as pd

from scripts.evaluate import split
from termmap import config, data
from termmap.embed import sapbert
from termmap.engine import CodeSystem
from termmap.normalize import basic
from termmap.train_encoder import finetune, search


def report(name: str, hits: list[list[str]], gold: list[str], seen: np.ndarray):
    rank = np.array([h.index(g) if g in h else 999 for h, g in zip(hits, gold)])
    line = f"  {name:12s} acc@1={np.mean(rank < 1):.4f}"
    for k in (5, 15, 30):
        line += f" r@{k}={np.mean(rank < k):.4f}"
    line += f" | seen acc@1={np.mean(rank[seen] < 1):.4f} unseen acc@1={np.mean(rank[~seen] < 1):.4f}"
    line += f" unseen r@15={np.mean(rank[~seen] < 15):.4f}"
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--aliases", type=int, default=15000, help="open-ontology synonym pairs per system")
    ap.add_argument("--no-train-terms", action="store_true",
                    help="train on open-ontology synonym pairs only (no labelled terms -> no leakage into the ranker)")
    ap.add_argument("--name", default=None, help="model name under data/cache/ (default: ft_split or ft_syn)")
    args = ap.parse_args()
    name = args.name or ("sapbert_ft_syn" if args.no_train_terms else "sapbert_ft_split")
    rng = random.Random(config.SEED)

    train = data.load_train()
    is_val = split(train, 0.2)
    tr, va = train[~is_val], train[is_val]
    base = sapbert()
    systems = {s: CodeSystem.load(s) for s in config.SYSTEMS}

    pairs, hard_negs = [], {}
    for system, cs in systems.items():
        idx = cs.retriever.sap
        rows = tr[(tr.coding_system == system) & (tr.subconcept == "DiagnosisCode")]
        if args.no_train_terms:
            rows = rows.iloc[:0]
        terms = [basic(e) for e in rows.entity]
        base_hits = search(base, idx.mat, idx.codes_of, terms, k=5)
        for term, display, code, hits in zip(terms, rows.display, rows.code, base_hits):
            cid = f"{system}:{code}"
            wrong = next((c for c in hits if c != code), None)
            if wrong:
                hard_negs[len(pairs)] = basic(cs.pref.get(wrong, ""))
            pairs.append((term, basic(display), cid))
            pref = basic(cs.pref.get(code, ""))
            if pref and pref != basic(display):
                pairs.append((term, pref, cid))
        al = []
        for f in sorted(config.EXTERNAL_DIR.glob(f"{system}_aliases*.parquet")):
            a = pd.read_parquet(f)
            al += [(basic(t), basic(cs.pref.get(c, "")), f"{system}:{c}") for t, c in zip(a.description, a.code)
                   if c in cs.pref and basic(t) != basic(cs.pref.get(c, ""))]
        rng.shuffle(al)
        al = al[:args.aliases]
        if args.no_train_terms:  # hard negatives for synonym pairs too: the base model's nearest wrong code
            hits = search(base, idx.mat, idx.codes_of, [t for t, _, _ in al], k=5)
            for (t, d, cid), h in zip(al, hits):
                wrong = next((c for c in h if f"{system}:{c}" != cid), None)
                if wrong:
                    hard_negs[len(pairs)] = basic(cs.pref.get(wrong, ""))
                pairs.append((t, d, cid))
        else:
            pairs += al
        print(f"[{system}] {len(rows)} training terms, {min(len(al), args.aliases)} synonym pairs")
    print(f"fine-tuning on {len(pairs)} pairs ({len(hard_negs)} with hard negatives), {args.epochs} epochs")
    t0 = time.time()
    ft = finetune(pairs, hard_negs, config.CACHE_DIR / name, epochs=args.epochs)
    print(f"fine-tune took {time.time() - t0:.0f}s")

    for system, cs in systems.items():
        idx = cs.retriever.sap
        t0 = time.time()
        ft_mat = ft.encode(list(idx.texts), batch=512)
        rows = va[(va.coding_system == system) & (va.subconcept == "DiagnosisCode")]
        terms = [basic(e) for e in rows.entity]
        seen = rows.code.isin(set(tr[tr.coding_system == system].code)).to_numpy()
        print(f"[{system}] held-out terms n={len(rows)} (dictionary re-encoded in {time.time() - t0:.0f}s)")
        report("base", search(base, idx.mat, idx.codes_of, terms), list(rows.code), seen)
        report("fine-tuned", search(ft, ft_mat, idx.codes_of, terms), list(rows.code), seen)


if __name__ == "__main__":
    main()
