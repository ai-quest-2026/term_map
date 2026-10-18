"""Grouped hold-out evaluation on the labelled training file.

    python -m scripts.evaluate [--val-frac 0.2 | --cv 5] [--no-context] [--ce auto|on|off] [--llm]

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


def evaluate_fold(tr: pd.DataFrame, va: pd.DataFrame, args) -> pd.DataFrame:
    t0 = time.time()
    use_ce = {"on": True, "off": False}.get(args.ce)  # None -> auto (on when a GPU is available)
    eng = TermMapEngine().fit(tr, use_ce=use_ce)
    print(f"fit: {time.time() - t0:.1f}s")
    diag_tr = tr[tr.subconcept == "DiagnosisCode"]
    diag_va = va[va.subconcept == "DiagnosisCode"]
    results = []
    for system in config.SYSTEMS:
        items = diag_va[diag_va.coding_system == system].reset_index(drop=True)
        if args.no_context:  # the PS eval schema is chunk_id + clinical_entity: maybe no chunk text at all
            items = items.assign(chunk_text="")
        t0 = time.time()
        R = eng.rank(items, system, use_llm=args.llm, llm_threshold=args.llm_threshold, llm_k=args.llm_k)
        top = R.groupby("item").head(1).set_index("item")
        recall = R.groupby("item").label.max().reindex(items.index, fill_value=0)
        df = items.assign(system=system, pred=[top.code.get(i, "") for i in items.index],
                          confidence=[float(top.conf.get(i, 0.0)) for i in items.index],
                          llm=[int(top.llm.get(i, 0)) for i in items.index], in_candidates=recall.values,
                          seen=items.code.isin(set(diag_tr[diag_tr.coding_system == system].code)),
                          ms_per_term=1000 * (time.time() - t0) / max(len(items), 1))
        results.append(df)
    return pd.concat(results)


def report(df: pd.DataFrame) -> None:
    summary = {}
    for system, d in df.groupby("system", sort=False):
        d = d.assign(correct=d.pred == d.code)
        line = (f"[{system}] n={len(d)} exact={d.correct.mean():.4f} candidate_recall={d.in_candidates.mean():.4f} "
                f"seen={d[d.seen].correct.mean():.4f} (n={d.seen.sum()}) "
                f"unseen={d[~d.seen].correct.mean():.4f} (n={(~d.seen).sum()}) {d.ms_per_term.mean():.1f} ms/term")
        if system == config.ICD:
            line += f" category={(d.pred.str[:3] == d.code.str[:3]).mean():.4f}"
        if d.llm.any():
            line += f" llm_changed={d.llm.sum()}"
        print(line)
        for cov in (0.97, 0.9, 0.8):
            kept = d.sort_values("confidence", ascending=False).head(int(round(cov * len(d))))
            print(f"    accuracy at {int(cov * 100)}% coverage: {kept.correct.mean():.4f}")
        summary[system] = d.correct.mean()
    print(f"HEADLINE (min of systems): {min(summary.values()):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--cv", type=int, default=0, help="k-fold cross-validation over all chunks instead of one split")
    ap.add_argument("--llm", action="store_true", help="adjudicate low-confidence terms with the LLM in .env")
    ap.add_argument("--llm-threshold", type=float, default=0.8)
    ap.add_argument("--llm-k", type=int, default=15)
    ap.add_argument("--ce", choices=["auto", "on", "off"], default="auto",
                    help="cross-encoder reranking (auto = on when a GPU is available)")
    ap.add_argument("--no-context", action="store_true",
                    help="blank chunk_text for the evaluated terms (training keeps it): robustness to an eval file without context")
    ap.add_argument("--out", default=str(config.ROOT / "reports" / "val_predictions.csv"))
    args = ap.parse_args()

    train = data.load_train()
    if args.cv:
        fold = train.chunk_text.astype(str).map(
            lambda t: int(hashlib.md5(("cv:" + t).encode()).hexdigest(), 16) % args.cv).to_numpy()
        parts = []
        for f in range(args.cv):
            print(f"--- fold {f + 1}/{args.cv}")
            parts.append(evaluate_fold(train[fold != f], train[fold == f], args).assign(fold=f))
        df = pd.concat(parts)
    else:
        is_val = split(train, args.val_frac)
        df = evaluate_fold(train[~is_val], train[is_val], args)
    report(df)
    if args.no_context and args.out == ap.get_default("out"):
        args.out = args.out.replace(".csv", "_nocontext.csv")
    df.drop(columns=["chunk_text"]).to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
