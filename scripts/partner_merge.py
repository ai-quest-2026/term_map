"""Combine our ICD predictions with the partner ICD engine (ai-quest-2026/icd10-map-experiments).

Fit on fold 0 (both engines trained on folds 1-4 only):
    python -m scripts.partner_merge fit --run v2 [--fold 0]
  reads reports/stages/<run>/predictions.csv (our cross-fitted s5 output, ICD rows of the fold) and
  data/cache/partner/f0_val_scores.parquet (scripts/partner_f0.py). Reports both engines, their agreement, the
  oracle, a paired test, and a switch rule "take the partner's code when our confidence < t and the partner's
  top score >= u". The rule's accuracy is estimated by two-way cross-fitting inside the fold (thresholds chosen
  on one half, scored on the other); the thresholds refitted on the whole fold go to data/cache/partner/rule.json,
  but only if the cross-fitted rule beats our engine alone (otherwise the rule is "never switch").

Apply to a submission:
    python -m scripts.partner_merge apply --details details.csv --partner partner_pred.csv --output predictions.csv
  details.csv = our scripts.predict --details output; partner_pred.csv = the partner's
  `predict.py --no-llm` output on the same input file (row order preserved by both). SNOMED is always ours.
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from termmap import config

RULE = config.CACHE_DIR / "partner" / "rule.json"


def _switch(ours_ok, part_ok, oc, pc, t, u):
    take = (oc < t) & (pc >= u)
    return np.where(take, part_ok, ours_ok).mean(), take.mean()


def _best(ours_ok, part_ok, oc, pc):
    ts = np.unique(np.quantile(oc, np.linspace(0, 1, 21)))
    us = np.unique(np.quantile(pc, np.linspace(0, 1, 21)))
    best = (ours_ok.mean(), -1.0, np.inf)  # never switch
    for t, u in itertools.product(ts, us):
        acc, _ = _switch(ours_ok, part_ok, oc, pc, t, u)
        if acc > best[0] + 1e-12:
            best = (acc, float(t), float(u))
    return best


def fit(args):
    stages = config.ROOT / "reports" / "stages" / args.run
    P = pd.read_csv(stages / "predictions.csv", dtype={"code": str, "pred": str}, keep_default_na=False)
    folds = sorted(P.fold.unique()) if args.fold == "all" else [int(args.fold)]
    parts = []
    for f in folds:  # predictions.csv rows follow test_items order within each fold and system
        Pf = P[(P.fold == f) & (P.system == config.ICD)].reset_index(drop=True)
        items = pd.read_parquet(config.CACHE_DIR / "stages" / args.run / f"f{f}_{config.ICD}_test_items.parquet")
        parts.append(Pf.assign(chunk_text=items.chunk_text.values))
    P = pd.concat(parts, ignore_index=True)
    Q = pd.read_parquet(args.partner_file or config.CACHE_DIR / "partner" / f"f{args.fold}_val_scores.parquet")
    Q["p1"] = Q.scores.str[0]
    Q = Q.drop_duplicates(["entity", "chunk_text"])
    M = P.merge(Q[["entity", "chunk_text", "pred", "p1", "codes"]].rename(columns={"pred": "ppred"}),
                on=["entity", "chunk_text"], how="inner")
    print(f"matched {len(M)} / {len(P)} of our fold-{args.fold} ICD terms ({len(Q)} partner rows)")
    o, p = (M.pred == M.code).to_numpy(), (M.ppred == M.code).to_numpy()
    oc, pc = M.confidence.astype(float).to_numpy(), M.p1.astype(float).to_numpy()
    b, c = int((o & ~p).sum()), int((~o & p).sum())
    rep = {"n": len(M), "ours": round(o.mean(), 4), "partner": round(p.mean(), 4),
           "agree": round(float((M.pred == M.ppred).mean()), 4), "oracle": round(float((o | p).mean()), 4),
           "ours_only_right": b, "partner_only_right": c,
           "mcnemar_p": round(binomtest(min(b, c), b + c).pvalue, 4) if b + c else 1.0,
           "partner_gold_in_top10": round(float(np.mean([g in set(cs) for g, cs in zip(M.code, M.codes)])), 4),
           "ours_seen": round(o[M.seen.astype(str) == "True"].mean(), 4),
           "partner_seen": round(p[M.seen.astype(str) == "True"].mean(), 4),
           "ours_unseen": round(o[M.seen.astype(str) != "True"].mean(), 4),
           "partner_unseen": round(p[M.seen.astype(str) != "True"].mean(), 4)}
    half = (np.arange(len(M)) % 2).astype(bool)
    hits = np.zeros(len(M), dtype=bool)
    for h in (False, True):
        _, t, u = _best(o[half != h], p[half != h], oc[half != h], pc[half != h])
        sel = half == h
        take = (oc[sel] < t) & (pc[sel] >= u)
        hits[sel] = np.where(take, p[sel], o[sel])
    rep["switch_rule_crossfit"] = round(hits.mean(), 4)
    acc, t, u = _best(o, p, oc, pc)
    use = bool(hits.mean() > o.mean())
    rule = {"t_ours_below": t if use else -1.0, "u_partner_at_least": u if use else float("inf"),
            "fitted_on": f"{args.run} fold {args.fold}", "in_sample_acc": round(acc, 4), "used": use}
    rep["rule"] = rule
    RULE.parent.mkdir(parents=True, exist_ok=True)
    RULE.write_text(json.dumps(rule, indent=1))
    out = config.ROOT / "reports" / "stages" / f"partner_merge_{args.run}_f{args.fold}.json"
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    print(f"MERGE HEADLINE ICD: ours {rep['ours']:.4f} | partner {rep['partner']:.4f} | "
          f"switch rule (cross-fitted) {rep['switch_rule_crossfit']:.4f} | oracle {rep['oracle']:.4f} -> {RULE}")


def apply(args):
    rule = json.loads(RULE.read_text())
    D = pd.read_csv(args.details, dtype=str, keep_default_na=False)
    Q = pd.read_csv(args.partner, dtype=str, keep_default_na=False)
    if len(D) != len(Q):
        raise SystemExit(f"row count mismatch: ours {len(D)} vs partner {len(Q)}")
    oc = pd.to_numeric(D.icd10cm_confidence, errors="coerce").fillna(0).to_numpy()
    pc = pd.to_numeric(Q.confidence_score, errors="coerce").fillna(0).to_numpy()
    take = (oc < rule["t_ours_below"]) & (pc >= rule["u_partner_at_least"]) & (Q.icd10cm_code.str.len() > 0).to_numpy()
    icd = np.where(take, Q.icd10cm_code.to_numpy(), D.icd10cm_code.to_numpy())
    term = "clinical_entity" if "clinical_entity" in D.columns else "entity"
    out = pd.DataFrame({"chunk_id": D.get("chunk_id", pd.Series([""] * len(D))), "clinical_entity": D[term],
                        "icd10cm_code": icd, "snomed_code": D.snomed_code})
    out.to_csv(args.output, index=False)
    print(f"partner ICD code taken for {int(take.sum())} / {len(D)} rows (rule {rule}) -> {args.output}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--run", default="v2")
    f.add_argument("--fold", default="0", help="fold number, or 'all' (pooled out-of-fold predictions)")
    f.add_argument("--partner-file", default=None, help="default data/cache/partner/f<fold>_val_scores.parquet")
    a = sub.add_parser("apply")
    a.add_argument("--details", required=True)
    a.add_argument("--partner", required=True)
    a.add_argument("--output", required=True)
    args = ap.parse_args()
    fit(args) if args.cmd == "fit" else apply(args)


if __name__ == "__main__":
    main()
