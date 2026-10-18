"""Staged, cached, gated evaluation: see a failing change in minutes, not after a full 5-fold run.

    python -m scripts.pipeline --run v2 --folds 0 --stages s2,s3,s5            # smoke: fold 0 only (~20% of data)
    python -m scripts.pipeline --run v2 --folds all --stages s2,s3,s5          # full 5-fold CV
    python -m scripts.pipeline --run v2 --folds all --stages s5 --export-stacker  # second level only, from caches
    ... --baseline-csv reports/val_predictions.csv   (per-row output of scripts/evaluate.py --cv 5, same folds)
    ... --baseline-run v1                            (an earlier pipeline run)

Folds are the md5("cv:" + chunk_text) % 5 folds of `scripts/evaluate.py --cv 5`, so fold-0 smoke numbers are a
subset of the CV numbers and can be compared row by row with an earlier evaluate run.

Stages (each fold's outputs are cached under data/cache/stages/<run>/, metrics under reports/stages/<run>/):
  s2  retrieval (S1) + ranker (S2): engine fit on the other folds, candidates for the test fold.
      Metrics: recall@1/5/15/30/all overall/seen/unseen, ranker top-1, in-sample vs OOF top-1, candidates/term.
      Gate: recall@all >= baseline - 1 pt (both systems).
  s3  cross-encoder trained on the OUT-OF-FOLD ranker lists of the training side, scored on the test top-k.
      Metrics: dev accuracy per epoch, CE-alone top-1, ranker+CE blends. Gates: dev accuracy after epoch 1
      >= --ce-gate (training aborts), best blend >= ranker + --s3-margin.
  s4  (scripts/train_selector.py writes s4 caches: LLM selector letter probabilities)
  s5  second-level ranker over ranker / cross-encoder / selector evidence, fitted leave-one-fold-out
      (two-way cross-fit inside the fold when only one fold is available) + training-term lookup.
      Writes per-row predictions (for scripts/compare.py) and the final metrics; --export-stacker fits it on
      all cached folds and saves data/cache/stacker_<system>.joblib for scripts/predict.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from termmap import config, data
from termmap.engine import (CE_K, STACK_FEATURES, STACKER_PARAMS, CodeSystem, TermMapEngine, stack_frame,
                            stacker_path)
from termmap.normalize import expansions
from termmap.rank import Ranker

N_FOLDS = 5
TOPN = 50  # candidates kept per term in the stage caches
KS = (1, 5, 15, 30, 50)


def cv_folds(train: pd.DataFrame, k: int = N_FOLDS) -> np.ndarray:
    return train.chunk_text.astype(str).map(
        lambda t: int(hashlib.md5(("cv:" + t).encode()).hexdigest(), 16) % k).to_numpy()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


S2_KINDS = {"train", "train_items", "test", "test_items"}


class Run:
    def __init__(self, name: str, s2_from: str | None = None):
        self.name = name
        self.s2_from = config.CACHE_DIR / "stages" / s2_from if s2_from else None
        self.cache = config.CACHE_DIR / "stages" / name
        self.reports = config.ROOT / "reports" / "stages" / name
        self.cache.mkdir(parents=True, exist_ok=True)
        self.reports.mkdir(parents=True, exist_ok=True)

    def path(self, kind: str, fold: int, system: str):
        p = self.cache / f"f{fold}_{system}_{kind}.parquet"
        if self.s2_from is not None and kind in S2_KINDS and not p.exists():
            return self.s2_from / p.name  # candidates of another run: compare rerankers on identical lists
        return p

    def has(self, kind: str, fold: int, system: str) -> bool:
        return self.path(kind, fold, system).exists()

    def save(self, df: pd.DataFrame, kind: str, fold: int, system: str):
        df.to_parquet(self.path(kind, fold, system), index=False)

    def load(self, kind: str, fold: int, system: str) -> pd.DataFrame:
        return pd.read_parquet(self.path(kind, fold, system))

    def metrics(self, stage: str, fold: int) -> dict:
        p = self.reports / f"{stage}_f{fold}.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def write_metrics(self, stage: str, fold: int, m: dict):
        (self.reports / f"{stage}_f{fold}.json").write_text(json.dumps(m, indent=1, default=float))


# --------------------------------------------------------------------------- metrics helpers

def gold_rank(F: pd.DataFrame, score: str) -> pd.Series:
    """Rank (1-based) of the gold code per item under `score` (NaN when gold is not a candidate)."""
    F = F.sort_values(["item", score, "code"], ascending=[True, False, True])
    F = F.assign(_r=F.groupby("item").cumcount() + 1)
    return F[F.label == 1].groupby("item")._r.min()


def recall_report(ranks: pd.Series, items: pd.DataFrame) -> dict:
    r = ranks.reindex(items.index)
    out = {}
    for name, m in (("all", np.ones(len(items), bool)), ("seen", items.seen.values), ("unseen", ~items.seen.values)):
        rr = r[m]
        out[name] = {f"r@{k}": round(float((rr <= k).mean()), 4) for k in KS}
        out[name]["r@any"] = round(float(rr.notna().mean()), 4)
        out[name]["n"] = int(m.sum())
    return out


def top1_acc(F: pd.DataFrame, score: str, n_items: int) -> float:
    F = F.sort_values(["item", score, "code"], ascending=[True, False, True])
    return float(F.groupby("item").head(1).label.sum() / max(n_items, 1))


def gate(ok: bool, msg: str, args) -> None:
    log(("GATE PASS: " if ok else "GATE FAIL: ") + msg)
    if not ok and not args.no_gate:
        log("stopping (use --no-gate to continue anyway)")
        sys.exit(2)


# --------------------------------------------------------------------------- baselines

def baseline_fold(args, fold: int) -> dict:
    """{system: {"acc", "recall"}} for this fold from an earlier evaluate CSV or pipeline run."""
    if args.baseline_csv:
        b = pd.read_csv(args.baseline_csv, dtype=str, keep_default_na=False)
        if "fold" in b.columns:
            b = b[b.fold.astype(str) == str(fold)]
        out = {}
        for system, d in b.groupby("system"):
            out[system] = {"acc": float((d.pred == d.code).mean()),
                           "recall": float(d.in_candidates.isin(["True", "1", "1.0"]).mean()), "n": len(d)}
        return out
    if args.baseline_run:
        prev = Run(args.baseline_run)
        m2, m5 = prev.metrics("s2", fold), prev.metrics("s5", fold)
        out = {}
        for system in config.SYSTEMS:
            out[system] = {"recall": m2.get(system, {}).get("recall", {}).get("all", {}).get("r@any"),
                           "acc": m5.get(system, {}).get("final_acc")}
        return out
    return {}


# --------------------------------------------------------------------------- stages

def items_frame(rows: pd.DataFrame, seen_codes: set) -> pd.DataFrame:
    return pd.DataFrame({"entity": rows.entity.values, "chunk_text": rows.chunk_text.values,
                         "code": rows.code.values, "seen": rows.code.isin(seen_codes).values})


def stage_s2(run: Run, fold: int, train: pd.DataFrame, folds: np.ndarray, args) -> None:
    if all(run.has("test", fold, s) for s in config.SYSTEMS) and not args.force:
        log(f"s2 fold {fold}: cached")
        return
    tr, te = train[folds != fold], train[folds == fold]
    t0 = time.time()
    log(f"s2 fold {fold}: fitting engine on {len(tr)} rows (retrieval + OOF features + ranker, no CE)")
    eng = TermMapEngine().fit(tr, use_ce=False)
    log(f"s2 fold {fold}: fit {time.time() - t0:.0f}s")
    m = {"fold": fold, "fit_s": round(time.time() - t0, 1)}
    for system in config.SYSTEMS:
        diag_tr = tr[(tr.coding_system == system) & (tr.subconcept == "DiagnosisCode")]
        rows = te[(te.coding_system == system) & (te.subconcept == "DiagnosisCode")].reset_index(drop=True)
        items = items_frame(rows, set(diag_tr.code))
        t1 = time.time()
        R = eng.rank(rows, system)
        ms = 1000 * (time.time() - t1) / max(len(rows), 1)
        R["score"] = R.key  # ranker order (+ lookup); no CE in this stage
        test = R.sort_values(["item", "score", "code"], ascending=[True, False, True]).groupby("item").head(TOPN)
        run.save(test[["item", "code", "label", "rk", "prob", "lookup", "score"]], "test", fold, system)
        run.save(R.drop(columns=["key", "conf", "llm", "score"], errors="ignore"), "test_feats", fold, system)
        run.save(items, "test_items", fold, system)
        # Training side: OOF ranker scores and texts (cross-encoder / selector training lists).
        X = eng.oof[system]
        trn = X.sort_values(["item", "oof_rk", "code"], ascending=[True, False, True]).groupby("item").head(TOPN)
        run.save(trn[["item", "code", "label", "oof_rk"]], "train", fold, system)
        run.save(X, "train_feats", fold, system)  # full OOF features: ranker ablations in seconds (ranker_lab-style)
        texts = eng.oof_items(tr, system)
        run.save(pd.DataFrame({"item": list(texts), "entity": [t[0] for t in texts.values()],
                               "chunk_text": [t[1] for t in texts.values()]}), "train_items", fold, system)
        insample = X.assign(ins=eng.ranker[system].score(X))
        n_tr = X.item.nunique()
        m[system] = {
            "n": len(rows), "ms_per_term": round(ms, 1), "cands_per_term": float(R.groupby("item").size().median()),
            "recall": recall_report(gold_rank(R, "score"), items),
            "ranker_top1": round(top1_acc(R, "score", len(rows)), 4),
            "ranker_top1_seen": round(top1_acc(R[R.item.isin(items.index[items.seen])], "score", int(items.seen.sum())), 4),
            "ranker_top1_unseen": round(top1_acc(R[R.item.isin(items.index[~items.seen])], "score", int((~items.seen).sum())), 4),
            "train_top1_oof": round(top1_acc(X, "oof_rk", n_tr), 4),
            "train_top1_insample": round(top1_acc(insample, "ins", n_tr), 4),
            "lookup_hits": int(R.groupby("item").lookup.max().sum()),
        }
        r = m[system]["recall"]["all"]
        log(f"s2 fold {fold} [{system}] n={len(rows)} ranker_top1={m[system]['ranker_top1']:.4f} "
            f"r@5={r['r@5']:.3f} r@15={r['r@15']:.3f} r@30={r['r@30']:.3f} r@any={r['r@any']:.3f} "
            f"unseen r@any={m[system]['recall']['unseen']['r@any']:.3f} "
            f"train top1 oof/in-sample={m[system]['train_top1_oof']:.3f}/{m[system]['train_top1_insample']:.3f} "
            f"{ms:.0f} ms/term")
    run.write_metrics("s2", fold, m)
    base = baseline_fold(args, fold)
    for system in config.SYSTEMS:
        b = base.get(system, {}).get("recall")
        if b:
            new = m[system]["recall"]["all"]["r@any"]
            gate(new >= b - 0.01, f"s2 fold {fold} {system} candidate recall {new:.4f} vs baseline {b:.4f} (allow -1 pt)", args)


def stage_s3(run: Run, fold: int, args) -> None:
    from termmap.cross_encoder import CrossEncoder, candidate_text, query_text

    if all(run.has("ce", fold, s) for s in config.SYSTEMS) and not args.force:
        log(f"s3 fold {fold}: cached")
        return
    m = {"fold": fold, "k": args.ce_k}
    for system in config.SYSTEMS:
        cs = CodeSystem.load(system)
        trn, ti = run.load("train", fold, system), run.load("train_items", fold, system).set_index("item")
        test, items = run.load("test", fold, system), run.load("test_items", fold, system)
        top = trn.sort_values(["item", "oof_rk", "code"], ascending=[True, False, True]).groupby("item").head(args.ce_k)
        tr_g, dev_g = [], []
        for item, g in top.groupby("item", sort=True):
            if g.label.max():
                grp = (query_text(ti.entity[item], ti.chunk_text[item]), [candidate_text(c, cs) for c in g.code],
                       int(np.argmax(g.label.values)))
                (dev_g if item % 10 == 0 else tr_g).append(grp)
        log(f"s3 fold {fold} [{system}] cross-encoder: {len(tr_g)} train / {len(dev_g)} dev groups, k={args.ce_k}, "
            f"{args.ce_epochs} epochs")
        t0 = time.time()
        ce = CrossEncoder().fit(tr_g, epochs=args.ce_epochs, dev=dev_g, min_dev_after_first=args.ce_gate,
                                tag=f" f{fold} {system}")
        tt = test.sort_values(["item", "score", "code"], ascending=[True, False, True]).groupby("item").head(args.ce_k)
        qs = {i: query_text(items.entity[i], items.chunk_text[i]) for i in tt.item.unique()}
        tt = tt.assign(ce=ce.score([qs[i] for i in tt.item], [candidate_text(c, cs) for c in tt.code]))
        run.save(tt[["item", "code", "ce"]], "ce", fold, system)
        if args.save_models:
            ce.save(run.cache / f"ce_f{fold}_{system}")
        n = len(items)
        F = stack_frame(tt.assign(prob=tt.prob))
        res = {"train_s": round(time.time() - t0, 1), "history": ce.history,
               "gold_in_topk": round(float(tt.groupby("item").label.max().sum() / n), 4),
               "ranker_top1": round(top1_acc(tt, "score", n), 4), "ce_top1": round(top1_acc(tt, "ce", n), 4)}
        for w in (0.2, 0.3, 0.5, 0.7):
            res[f"blend_{w}"] = round(top1_acc(F.assign(b=(1 - w) * F.lp_z + w * F.ce_z), "b", n), 4)
        m[system] = res
        best = max(res[f"blend_{w}"] for w in (0.2, 0.3, 0.5, 0.7))
        log(f"s3 fold {fold} [{system}] gold@{args.ce_k}={res['gold_in_topk']:.3f} ranker={res['ranker_top1']:.4f} "
            f"ce={res['ce_top1']:.4f} blend0.3={res['blend_0.3']:.4f} best_blend={best:.4f} ({res['train_s']:.0f}s)")
    run.write_metrics("s3", fold, m)
    for system in config.SYSTEMS:
        res = m[system]
        best = max(res[f"blend_{w}"] for w in (0.2, 0.3, 0.5, 0.7))
        gate(best >= res["ranker_top1"] + args.s3_margin,
             f"s3 fold {fold} {system} best ranker+CE blend {best:.4f} vs ranker {res['ranker_top1']:.4f} "
             f"(need +{args.s3_margin:.3f})", args)


def stack_data(run: Run, fold: int, system: str, k: int, with_sel: bool) -> pd.DataFrame | None:
    if not run.has("ce", fold, system) or (with_sel and not run.has("sel", fold, system)):
        return None
    test = run.load("test", fold, system)
    tt = test.sort_values(["item", "score", "code"], ascending=[True, False, True]).groupby("item").head(k)
    tt = tt.merge(run.load("ce", fold, system), on=["item", "code"], how="left")
    if with_sel:
        tt = tt.merge(run.load("sel", fold, system)[["item", "code", "sel_p"]], on=["item", "code"], how="left")
    F = stack_frame(tt.reset_index(drop=True))
    F["fold"] = fold
    F["gitem"] = fold * 1_000_000 + F.item
    return F


def fit_stacker(F: pd.DataFrame) -> Ranker:
    return Ranker(STACK_FEATURES, **STACKER_PARAMS).fit(F.assign(item=F.gitem))


def stage_s5(run: Run, folds_eval: list[int], train: pd.DataFrame, args) -> None:
    out_rows, m = [], {}
    for system in config.SYSTEMS:
        # With --with-selector only folds the selector scored are used (it is trained per fold and is slow);
        # without it the selector feature is absent everywhere, so the stacker is consistent across folds.
        frames = {f: stack_data(run, f, system, args.ce_k, args.with_selector) for f in range(N_FOLDS)}
        frames = {f: F for f, F in frames.items() if F is not None}
        if not frames:
            log(f"s5 [{system}]: no s3 caches; run s3 first")
            continue
        has_sel = any("sel_p" in F.columns and F.sel_p.notna().any() for F in frames.values())
        for f in [f for f in folds_eval if f in frames]:
            F = frames[f]
            with_sel = lambda X: bool("sel_p" in X.columns and X.sel_p.notna().any())
            # Folds scored by the selector only stack with folds that have it too (else cross-fit in-fold).
            others = [frames[g] for g in frames if g != f and with_sel(frames[g]) == with_sel(F)]
            if others:   # leave-one-fold-out: the stacker never saw this fold
                F = F.assign(st=fit_stacker(pd.concat(others)).score(F))
            else:        # a single cached fold: two-way cross-fit inside it
                half = (F.item % 2).to_numpy()
                st = np.zeros(len(F))
                for h in (0, 1):
                    st[half == h] = fit_stacker(F[half != h]).score(F[half == h])
                F = F.assign(st=st)
            items = run.load("test_items", f, system)
            n = len(items)
            pick = F.sort_values(["item", "st", "code"], ascending=[True, False, True]).groupby("item").head(1)
            pred = pick.set_index("item").code.reindex(items.index).fillna("")
            look = F[F.lookup == 1].groupby("item").code.first()
            pred.loc[look.index] = look
            correct = pred.values == items.code.values
            res = {"n": n, "final_acc": round(float(correct.mean()), 4),
                   "final_seen": round(float(correct[items.seen.values].mean()), 4),
                   "final_unseen": round(float(correct[~items.seen.values].mean()), 4),
                   "ranker_top1": round(top1_acc(F, "score", n), 4), "ce_top1": round(top1_acc(F, "ce", n), 4),
                   "blend_0.3": round(top1_acc(F.assign(b=0.7 * F.lp_z + 0.3 * F.ce_z), "b", n), 4),
                   "stacker": round(top1_acc(F, "st", n), 4), "uses_selector": bool(has_sel)}
            if has_sel:
                res["selector_top1"] = round(top1_acc(F.assign(s=F.sel_p.fillna(-1)), "s", n), 4)
            if system == config.ICD:
                res["category_acc"] = round(float((pred.str[:3].values == items.code.str[:3].values).mean()), 4)
            m.setdefault(f, {})[system] = res
            log(f"s5 fold {f} [{system}] FINAL={res['final_acc']:.4f} (seen {res['final_seen']:.3f} / unseen "
                f"{res['final_unseen']:.3f}) ranker={res['ranker_top1']:.4f} ce={res['ce_top1']:.4f} "
                f"blend0.3={res['blend_0.3']:.4f} stacker={res['stacker']:.4f}"
                + (f" selector={res['selector_top1']:.4f}" if has_sel else ""))
            conf = F.assign(p=np.exp(F.st - F.groupby("item").st.transform("max")))
            conf = (conf.p / conf.groupby("item").p.transform("sum")).groupby(conf.item).max()
            out_rows.append(pd.DataFrame({"fold": f, "system": system, "entity": items.entity, "code": items.code,
                                          "pred": pred.values, "seen": items.seen,
                                          "confidence": conf.reindex(items.index).fillna(0).values,
                                          "in_candidates": items.index.isin(F[F.label == 1].item)}))
        if args.export_stacker:
            st = fit_stacker(pd.concat(frames.values()))
            import joblib
            joblib.dump(st, stacker_path(system))
            log(f"s5 [{system}]: stacker fitted on folds {sorted(frames)} -> {stacker_path(system)}")
    for f, mm in m.items():
        run.write_metrics("s5", f, {"fold": f, **mm})
    if out_rows:
        P = pd.concat(out_rows)
        P.to_csv(run.reports / "predictions.csv", index=False)
        for system, d in P.groupby("system", sort=False):
            log(f"s5 POOLED [{system}] folds={sorted(int(x) for x in d.fold.unique())} n={len(d)} exact={(d.pred == d.code).mean():.4f} "
                f"unseen={(d.pred == d.code)[~d.seen].mean():.4f}")
        accs = P.assign(c=P.pred == P.code).groupby("system").c.mean()
        log(f"s5 POOLED HEADLINE (min) = {accs.min():.4f}   -> {run.reports / 'predictions.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--folds", default="0", help="'0', '0,1' or 'all'")
    ap.add_argument("--stages", default="s2,s3,s5")
    ap.add_argument("--baseline-csv", default=None)
    ap.add_argument("--baseline-run", default=None)
    ap.add_argument("--ce-k", type=int, default=CE_K)
    ap.add_argument("--ce-epochs", type=int, default=int(os.environ.get("TERMMAP_CE_EPOCHS", "4")))
    ap.add_argument("--ce-gate", type=float, default=0.45, help="min cross-encoder dev accuracy after epoch 1")
    ap.add_argument("--s3-margin", type=float, default=0.0, help="required gain of the best blend over the ranker")
    ap.add_argument("--save-models", action="store_true")
    ap.add_argument("--with-selector", action="store_true", help="s5: use the LLM selector feature (folds with s4 caches)")
    ap.add_argument("--s2-from", default=None, help="read s2 caches (candidates) from another run")
    ap.add_argument("--export-stacker", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute cached stages")
    ap.add_argument("--no-gate", action="store_true")
    args = ap.parse_args()

    folds_eval = list(range(N_FOLDS)) if args.folds == "all" else [int(f) for f in args.folds.split(",")]
    stages = args.stages.split(",")
    run = Run(args.run, args.s2_from)
    train = data.load_train()
    folds = cv_folds(train)
    log(f"run={args.run} folds={folds_eval} stages={stages} cache={run.cache}")
    for f in folds_eval:
        if "s2" in stages:
            stage_s2(run, f, train, folds, args)
        if "s3" in stages:
            stage_s3(run, f, args)
    if "s5" in stages:
        stage_s5(run, folds_eval, train, args)


if __name__ == "__main__":
    main()
