"""Paired comparison of two runs on the same terms: is B really better than A?

    python -m scripts.compare A.csv B.csv [--slices]

A / B are per-row prediction files with columns system, entity, code, pred (scripts/evaluate.py output or
reports/stages/<run>/predictions.csv). Rows are matched on (system, entity) — entity strings are unique per
system in the labelled file — restricted to the rows both runs scored (e.g. a fold-0 smoke run against a 5-fold
baseline). Reports per system: accuracy of A and B, the difference with a paired bootstrap 95% CI (10,000
resamples), and the exact McNemar p-value on the discordant pairs. With ~1,000 terms per system only
differences of >= ~3 points are detectable; pooled 5-fold (~2,500) detects ~1.5 points.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from termmap import config


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["correct"] = (df.pred == df.code).astype(int)
    keep = ["system", "entity", "code", "correct"] + [c for c in ("seen",) if c in df.columns]
    return df[keep]


def paired(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = config.SEED) -> dict:
    rng = np.random.default_rng(seed)
    d = b - a
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    b_only, a_only = int(((b == 1) & (a == 0)).sum()), int(((a == 1) & (b == 0)).sum())
    p = binomtest(b_only, b_only + a_only, 0.5).pvalue if b_only + a_only else 1.0
    return {"n": len(d), "acc_a": a.mean(), "acc_b": b.mean(), "diff": d.mean(),
            "ci_lo": np.percentile(boots, 2.5), "ci_hi": np.percentile(boots, 97.5),
            "b_only": b_only, "a_only": a_only, "p_mcnemar": p}


def fmt(name: str, r: dict) -> str:
    sig = "SIGNIFICANT" if r["p_mcnemar"] < 0.05 else "not significant"
    return (f"{name:28s} n={r['n']:5d}  A={r['acc_a']:.4f}  B={r['acc_b']:.4f}  diff={100 * r['diff']:+.2f} pt "
            f"[{100 * r['ci_lo']:+.2f}, {100 * r['ci_hi']:+.2f}]  B-only={r['b_only']} A-only={r['a_only']}  "
            f"McNemar p={r['p_mcnemar']:.4f} ({sig})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--slices", action="store_true", help="also seen / unseen codes")
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    M = A.merge(B, on=["system", "entity", "code"], suffixes=("_a", "_b"))
    print(f"matched rows: {len(M)} (A has {len(A)}, B has {len(B)})")
    heads = {}
    for system, d in M.groupby("system", sort=False):
        r = paired(d.correct_a.to_numpy(), d.correct_b.to_numpy())
        heads[system] = r
        print(fmt(system, r))
        if args.slices:
            seen_col = "seen_b" if "seen_b" in d.columns else ("seen_a" if "seen_a" in d.columns else None)
            if seen_col:
                s = d[seen_col].isin(["True", "1", "1.0"])
                for name, m in (("  seen codes", s), ("  unseen codes", ~s)):
                    if m.any():
                        print(fmt(name, paired(d.correct_a[m].to_numpy(), d.correct_b[m].to_numpy())))
    if heads:
        print(f"headline (min of systems): A={min(r['acc_a'] for r in heads.values()):.4f}  "
              f"B={min(r['acc_b'] for r in heads.values()):.4f}")


if __name__ == "__main__":
    main()
