"""Grouped hold-out evaluation on the labelled training file.

    python -m scripts.evaluate [--val-frac 0.2]

Splits DiagnosisCode rows by chunk (no chunk appears on both sides), fits the engine on the train
side and reports, per coding system: exact-match accuracy, candidate recall, accuracy on codes seen /
unseen in training (long tail), ICD category-level partial credit and accuracy at reduced coverage.
Writes per-row results to reports/val_predictions.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import time

import numpy as np
import pandas as pd

from termmap import config, data
from termmap.engine import TermMapEngine


def split(train: pd.DataFrame, frac: float) -> np.ndarray:
    buckets = train.chunk_text.astype(str).map(lambda t: int(hashlib.md5(("val:" + t).encode()).hexdigest(), 16) % 1000)
    return (buckets < frac * 1000).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--llm", action="store_true", help="adjudicate low-confidence terms with the LLM in .env")
    ap.add_argument("--llm-threshold", type=float, default=0.6)
    ap.add_argument("--out", default=str(config.ROOT / "reports" / "val_predictions.csv"))
    args = ap.parse_args()

    train = data.load_train()
    is_val = split(train, args.val_frac)
    tr, va = train[~is_val], train[is_val]
    t0 = time.time()
    eng = TermMapEngine().fit(tr)
    print(f"fit: {time.time() - t0:.1f}s")

    diag_tr = tr[tr.subconcept == "DiagnosisCode"]
    diag_va = va[va.subconcept == "DiagnosisCode"]
    results, summary = [], {}
    for system in config.SYSTEMS:
        items = diag_va[diag_va.coding_system == system].reset_index(drop=True)
        t0 = time.time()
        R = eng.rank(items, system, use_llm=args.llm, llm_threshold=args.llm_threshold)
        dt = (time.time() - t0) / max(len(items), 1)
        top = R.groupby("item").head(1).set_index("item")
        pred = items.index.map(lambda i: top.code.get(i, ""))
        conf = items.index.map(lambda i: float(top.prob.get(i, 0.0)))
        recall = R.groupby("item").label.max().reindex(items.index, fill_value=0)
        seen_codes = set(diag_tr[diag_tr.coding_system == system].code)
        df = items.assign(system=system, pred=list(pred), confidence=list(conf), in_candidates=recall.values,
                          seen=items.code.isin(seen_codes))
        df["correct"] = df.pred == df.code
        if system == config.ICD:
            df["category_correct"] = df.pred.str[:3] == df.code.str[:3]
        results.append(df)

        acc = df.correct.mean()
        line = (f"[{system}] n={len(df)} exact={acc:.4f} candidate_recall={df.in_candidates.mean():.4f} "
                f"seen={df[df.seen].correct.mean():.4f} (n={df.seen.sum()}) "
                f"unseen={df[~df.seen].correct.mean():.4f} (n={(~df.seen).sum()}) {1000 * dt:.1f} ms/term")
        if system == config.ICD:
            line += f" category={df.category_correct.mean():.4f}"
        print(line)
        for cov in (0.97, 0.9, 0.8):
            k = int(round(cov * len(df)))
            kept = df.sort_values("confidence", ascending=False).head(k)
            print(f"    accuracy at {int(cov * 100)}% coverage: {kept.correct.mean():.4f}")
        summary[system] = acc
    print(f"HEADLINE (min of systems): {min(summary.values()):.4f}")
    pd.concat(results).drop(columns=["chunk_text"]).to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
