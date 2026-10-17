"""Slice-level error analysis of an evaluate run (no chart text is printed, only entities and codes).

    python -m scripts.error_report [reports/val_predictions.csv] [--examples 15]

Slices follow the problem statement's failure modes: unseen codes (long tail), abbreviations, negated /
historical mentions, laterality, specificity ("unspecified" gold), noisy tokens, version drift (non-leaf
gold), plus where the error comes from: retrieval (gold not in candidates) vs decision (gold ranked lower),
and for ICD whether the 3-character category was still right.
"""
from __future__ import annotations

import argparse

import pandas as pd

from termmap import config, data
from termmap.normalize import ABBREVIATIONS, basic, is_history, is_negated, laterality


def slices(df: pd.DataFrame, icd_leaf: dict, unspec: set, vocab: set) -> dict[str, pd.Series]:
    ent = df.entity.astype(str)
    toks = ent.map(lambda e: basic(e).split())
    return {
        "all": pd.Series(True, index=df.index),
        "seen code": df.seen,
        "unseen code (long tail)": ~df.seen,
        "abbreviation (<=5 chars / known abbrev)": ent.map(lambda e: basic(e) in ABBREVIATIONS or (len(basic(e)) <= 5 and " " not in basic(e))),
        "negated mention": df.apply(lambda r: is_negated(r.entity, r.chunk_text), axis=1),
        "historical mention": df.apply(lambda r: is_history(r.entity, r.chunk_text), axis=1),
        "laterality in term": ent.map(lambda e: bool(laterality(e))),
        "gold is 'unspecified'": df.code.isin(unspec),
        "noisy token (not in dictionary vocab)": toks.map(lambda ts: any(t not in vocab and not t.isdigit() for t in ts)),
        "long term (>=5 words)": toks.map(len) >= 5,
        "gold non-leaf (version drift)": df.code.map(lambda c: icd_leaf.get(c) is False),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=str(config.ROOT / "reports" / "val_predictions.csv"))
    ap.add_argument("--examples", type=int, default=15)
    args = ap.parse_args()

    df = pd.read_csv(args.path, dtype=str, keep_default_na=False)
    for c in ("seen", "in_candidates"):
        df[c] = df[c].isin(["True", "1", "1.0"])
    df["confidence"] = pd.to_numeric(df.confidence, errors="coerce").fillna(0)
    df["correct"] = df.pred == df.code
    train = data.load_train()
    ctx = dict(zip(zip(train.entity, train.coding_system), train.chunk_text))
    df["chunk_text"] = [ctx.get((e, s), "") for e, s in zip(df.entity, df.system)]

    icd = data.icd_index_rows()
    icd_leaf = dict(zip(icd.code, icd.is_leaf))
    pref = data.preferred_terms()
    unspec = set(icd.loc[icd.description.str.lower().str.contains("unspecified"), "code"])
    vocab = set()
    for d in (icd.description, data.snomed_index_rows().description):
        for t in d.map(basic):
            vocab.update(t.split())

    for system, d in df.groupby("system", sort=False):
        print(f"\n=== {system}  n={len(d)}  exact={d.correct.mean():.4f}")
        err = d[~d.correct]
        print(f"errors: {len(err)}  retrieval misses (gold not in candidates): {(~err.in_candidates).sum()}  "
              f"decision misses (gold ranked lower): {err.in_candidates.sum()}")
        if system == config.ICD:
            same_cat = (err.pred.str[:3] == err.code.str[:3])
            print(f"ICD errors with the right 3-char category (specificity errors): {same_cat.sum()} "
                  f"({same_cat.mean():.1%} of errors)")
            pred_unspec = err.pred.isin(unspec)
            print(f"  of those: predicted unspecified but gold specific: {(same_cat & pred_unspec & ~err.code.isin(unspec)).sum()}, "
                  f"predicted specific but gold unspecified: {(same_cat & ~pred_unspec & err.code.isin(unspec)).sum()}")
        print(f"{'slice':42s} {'n':>6s} {'share':>6s} {'exact':>7s}")
        for name, mask in slices(d, icd_leaf if system == config.ICD else {}, unspec if system == config.ICD else set(),
                                 vocab).items():
            mask = mask.reindex(d.index, fill_value=False).astype(bool)
            if mask.sum() == 0:
                continue
            print(f"{name:42s} {mask.sum():6d} {mask.mean():6.1%} {d[mask].correct.mean():7.4f}")
        conf = pd.cut(d.confidence, [0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0001], right=False)
        print("confidence bucket -> accuracy (calibration):")
        print(d.groupby(conf, observed=True).correct.agg(["size", "mean"]).round(3).to_string())
        print(f"sample decision errors (entity | gold | predicted):")
        for _, r in err[err.in_candidates].head(args.examples).iterrows():
            print(f"  {r.entity[:40]:40s} | {r.code} {pref[system].get(r.code, '')[:35]:35s} | "
                  f"{r.pred} {pref[system].get(r.pred, '')[:35]}")
        print(f"sample retrieval misses:")
        for _, r in err[~err.in_candidates].head(max(5, args.examples // 2)).iterrows():
            print(f"  {r.entity[:40]:40s} | {r.code} {pref[system].get(r.code, '')[:35]}")


if __name__ == "__main__":
    main()
