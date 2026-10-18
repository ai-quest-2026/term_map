"""Train and score the partner ICD engine (ai-quest-2026/icd10-map-experiments) on OUR fold-0 split.

    python -m scripts.partner_f0 --partner ~/icd10-map-experiments [--work ~/partner_f0]
    python -m scripts.partner_f0 --partner ~/icd10-map-experiments --work ~/partner_b2 --their-split   # fast path

Both engines are then compared on the same held-out terms (scripts/partner_merge.py). The partner checkout is
not modified: it is copied to --work without its own B2 caches (their semantic memory, adapters, models and
cross-encoder caches were built on their split and would leak), our fold-0 split is written in place of their B2
split files, and their Phase-9 trainer runs there (`run_phase6_reranker("b2")`, offline: no Gemini calls).
Writes data/cache/partner/f0_val_scores.parquet: per held-out ICD row, the partner's top-10 codes and scores.
--their-split keeps their B2 split and caches instead (minutes) and writes b2_val_scores.parquet; compare it
with our pooled 5-fold out-of-fold predictions (partner_merge fit --fold all), which never saw those rows either.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from termmap import config
from scripts.pipeline import cv_folds

OUT = config.CACHE_DIR / "partner" / "f0_val_scores.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partner", required=True, help="checkout of icd10-map-experiments")
    ap.add_argument("--work", default=str(Path.home() / "partner_f0"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--their-split", action="store_true",
                    help="fast path: keep THEIR B2 split and caches (minutes); compare against our 5-fold OOF predictions")
    args = ap.parse_args()
    src, work = Path(args.partner).expanduser().resolve(), Path(args.work).expanduser().resolve()

    if not work.exists():
        skip = shutil.ignore_patterns(*(() if args.their_split else ("*_b2*",)), ".git", "__pycache__")
        shutil.copytree(src, work, ignore=skip)
    if args.their_split:
        va = pd.read_parquet(work / "artifacts" / "val_icd_chunk_only.parquet")
        out = OUT.with_name("b2_val_scores.parquet")
    else:
        for f in list((work / "artifacts").rglob("*_b2*")):  # never reuse anything built on their split
            f.unlink()
        out = OUT

    if not args.their_split:
        va = write_our_split(work, args.fold)
    _run(work, va, out)


def write_our_split(work: Path, fold: int) -> pd.DataFrame:
    base = pd.read_parquet(work / "artifacts" / "cv_splits_chunk_only.parquet")
    base["fold"] = cv_folds(base)  # our md5 chunk folds (same rule as evaluate --cv 5 / pipeline)
    tr = base[base.fold != fold].copy().reset_index(drop=True)
    va = base[base.fold == fold].copy().reset_index(drop=True)
    tr["is_seen_entity"] = True
    va["is_seen_entity"] = va.norm_entity.isin(set(tr.norm_entity))
    cnt = Counter(tr.code)
    va["code_count_in_train"] = [cnt.get(c, 0) for c in va.code]
    va["diagnostic_slice"] = ["Slice_A_Head_Code" if n >= 8 else "Slice_B_Mid_Code" if n >= 2 else
                              "Slice_C_Singleton_Code" if n == 1 else "Slice_D_Zero_Shot_Code"
                              for n in va.code_count_in_train]
    tr.to_parquet(work / "artifacts" / "train_icd_chunk_only.parquet", index=False)
    va.to_parquet(work / "artifacts" / "val_icd_chunk_only.parquet", index=False)
    print(f"partner fold {fold}: train {len(tr)} / val {len(va)} ICD rows", flush=True)
    return va


def _run(work: Path, va: pd.DataFrame, out: Path):
    os.chdir(work)
    sys.path.insert(0, str(work))
    os.environ.pop("GEMINI_API_KEY", None)  # offline: chart text must not go to a retaining API
    import src.phase6_reranker as P

    stash = {}
    build, predict = P.build_reranker_dataset, P.TermMapReranker.predict_scores

    def build_spy(*a, **k):
        out = build(*a, **k)
        if not k.get("is_training", True):
            stash["cands"] = out[3]
            stash["groups"] = out[2]
        return out

    def predict_spy(self, X, groups=None):
        s = predict(self, X, groups=groups)
        stash["scores"] = s
        return s

    P.build_reranker_dataset, P.TermMapReranker.predict_scores = build_spy, predict_spy
    P.run_phase6_reranker("b2")

    rows, off = [], 0
    for i, g in enumerate(stash["groups"]):
        sc, cands = stash["scores"][off:off + g], stash["cands"][i]
        off += g
        order = sorted(range(len(cands)), key=lambda j: -sc[j])[:10]
        rows.append({"entity": va.entity[i], "chunk_text": va.chunk_text[i], "gold": va.code[i],
                     "codes": [cands[j] for j in order], "scores": [float(sc[j]) for j in order]})
    df = pd.DataFrame(rows)
    df["pred"] = df.codes.str[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"partner exact={(df.pred == df.gold).mean():.4f} n={len(df)} -> {out}")


if __name__ == "__main__":
    main()
