"""Blind test: a dress rehearsal of the judges' evaluation, through the exact submission command.

    python -m scripts.blind_test --fold 0 [--selector data/cache/stages/v2/selector_f0] [--no-context] [--reuse]

1. Splits the labelled file with the same chunk-grouped folds as scripts/evaluate.py --cv 5 / scripts/pipeline.py.
   Training side: every row of the other folds. Test side: the DiagnosisCode rows of --fold.
2. Writes the test side in the problem-statement input schema (chunk_id, clinical_entity [, chunk_text]) with the
   answer columns removed, and keeps the gold codes in a separate file.
3. Runs `python -m scripts.predict` on it as a separate process, with an engine trained only on the training
   side (--train / --engine, so the real engine.joblib is untouched).
4. Scores the predictions against the hidden gold per coding system: exact match, codes seen / unseen in the
   training side, ICD category, accuracy at 80/90% coverage by confidence, and checks the output schema.

The terms are never in the engine's training data, so the training-term lookup cannot inflate the score.
Use the selector trained on the same training side (selector_f<fold>), never one trained on the test fold.
Caveat: the exported stacker (data/cache/stacker_*.joblib) is fitted on fold 0's own out-of-fold features, so
for --fold 0 its few weights have seen these terms; the pipeline's cross-fitted s5 numbers are the clean ones.
Outputs go to data/cache/blind/f<fold>/ (gitignored: they contain chart text).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import pandas as pd

from termmap import config, data
from scripts.pipeline import cv_folds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--selector", default=None, help="selector adapter trained WITHOUT this fold")
    ap.add_argument("--no-context", action="store_true", help="omit chunk_text, as the PS input schema does")
    ap.add_argument("--reuse", action="store_true", help="reuse the engine trained on this split if it exists")
    args = ap.parse_args()

    out = config.CACHE_DIR / "blind" / f"f{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    train = data.load_train()
    folds = cv_folds(train)
    train[folds != args.fold].to_csv(out / "train.csv", index=False)

    te = train[(folds == args.fold) & (train.subconcept == "DiagnosisCode")]
    gold = te.pivot_table(index=["entity", "chunk_text"], columns="coding_system", values="code",
                          aggfunc="first").reset_index()
    gold.insert(0, "chunk_id", [hashlib.md5(c.encode()).hexdigest()[:12] for c in gold.chunk_text])
    gold = gold.rename(columns={"entity": "clinical_entity"})
    cols = ["chunk_id", "clinical_entity"] + ([] if args.no_context else ["chunk_text"])
    gold[cols].to_csv(out / "input.csv", index=False)           # what the engine sees: no answers
    gold.drop(columns=["chunk_text"]).to_csv(out / "gold.csv", index=False)
    print(f"blind fold {args.fold}: {len(gold)} terms (input columns: {', '.join(cols)}); "
          f"training side {int((folds != args.fold).sum())} rows", flush=True)

    tag = "noctx" if args.no_context else "ctx"
    pred_path = out / f"predictions_{tag}.csv"
    cmd = [sys.executable, "-m", "scripts.predict", "--input", str(out / "input.csv"), "--output", str(pred_path),
           "--details", str(out / f"details_{tag}.csv"), "--train", str(out / "train.csv"),
           "--engine", str(out / "engine.joblib")]
    if not (args.reuse and (out / "engine.joblib").exists()):
        cmd.append("--refit")
    if args.selector:
        cmd += ["--selector", args.selector]
    print("running:", " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=config.ROOT, env=os.environ.copy(), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    map_line = ""
    for line in proc.stdout:
        print(line, end="", flush=True)
        if line.startswith("mapped "):
            map_line = line.strip()
    if proc.wait():
        sys.exit(f"predict failed with exit code {proc.returncode}")
    elapsed = time.time() - t0

    pred = pd.read_csv(pred_path, dtype=str, keep_default_na=False)
    det = pd.read_csv(out / f"details_{tag}.csv", dtype=str, keep_default_na=False)
    schema_ok = list(pred.columns) == ["chunk_id", "clinical_entity", "icd10cm_code", "snomed_code"] and len(pred) == len(gold)
    print(f"output schema {'OK' if schema_ok else 'MISMATCH'}: columns {list(pred.columns)}, rows {len(pred)} / {len(gold)}")

    tr = train[(folds != args.fold) & (train.subconcept == "DiagnosisCode")]
    res = {"fold": args.fold, "context": not args.no_context, "selector": args.selector, "terms": len(gold),
           "schema_ok": schema_ok, "wall_seconds": round(elapsed, 1), "mapping": map_line}
    for system, pcol, ccol in ((config.ICD, "icd10cm_code", "icd10cm_confidence"),
                               (config.SNOMED, "snomed_code", "snomed_confidence")):
        m = gold[system].notna() & (gold[system] != "")
        g, p = gold.loc[m, system].values, pred.loc[m, pcol].values
        ok = pd.Series(g == p)
        seen = pd.Series(gold.loc[m, system].isin(set(tr[tr.coding_system == system].code)).values)
        conf = pd.to_numeric(det.loc[m, ccol], errors="coerce").fillna(0).reset_index(drop=True)
        r = {"n": int(m.sum()), "exact": round(ok.mean(), 4), "seen": round(ok[seen].mean(), 4),
             "unseen": round(ok[~seen].mean(), 4), "n_unseen": int((~seen).sum())}
        for cov in (0.9, 0.8):
            keep = conf.sort_values(ascending=False).index[:int(round(cov * len(conf)))]
            r[f"acc_at_{int(cov * 100)}pct"] = round(ok[keep].mean(), 4)
        if system == config.ICD:
            r["category"] = round(float((pd.Series(g).str[:3] == pd.Series(p).str[:3]).mean()), 4)
        res[system] = r
        print(f"[{system}] n={r['n']} exact={r['exact']:.4f} seen={r['seen']:.4f} unseen={r['unseen']:.4f} "
              f"(n={r['n_unseen']}) acc@90%={r['acc_at_90pct']:.4f} acc@80%={r['acc_at_80pct']:.4f}"
              + (f" category={r['category']:.4f}" if "category" in r else ""))
    res["headline"] = min(res[config.ICD]["exact"], res[config.SNOMED]["exact"])
    print(f"BLIND HEADLINE (min of systems) = {res['headline']:.4f}   ({map_line}; wall time incl. any training "
          f"{elapsed / 60:.1f} min)")
    rep = config.ROOT / "reports" / "stages"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / f"blind_f{args.fold}_{tag}{'_sel' if args.selector else ''}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
