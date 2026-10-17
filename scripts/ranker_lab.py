"""Fast ranker experiments on cached candidate features.

    python -m scripts.ranker_lab dump      # build raw features for the grouped split (slow, once per retrieval change)
    python -m scripts.ranker_lab run       # train/score ranker variants on the cache (seconds)
    python -m scripts.ranker_lab llm [threshold] [max_terms] [k]   # LLM adjudication gain on the val split
    python -m scripts.ranker_lab ce [k] [epochs]   # train a cross-encoder reranker on the train side, test on val
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from scripts.evaluate import split
from termmap import config, data, llm
from termmap.engine import FEATURES, Memory, TermMapEngine, build_features
from termmap.rank import DERIVED, Ranker, derive, softmax_by_group


def _path(kind: str, system: str):
    return config.CACHE_DIR / f"lab_{kind}_{system}.parquet"


def dump():
    train = data.load_train()
    is_val = split(train, 0.2)
    tr, va = train[~is_val], train[is_val]
    eng = TermMapEngine()
    for system in config.SYSTEMS:
        eng.oof_features(tr, system).to_parquet(_path("train", system))
        items = va[(va.coding_system == system) & (va.subconcept == "DiagnosisCode")].reset_index(drop=True)
        X = build_features(items, eng.systems[system], Memory(tr, system))
        X["seen"] = X.item.map(items.code.isin(set(tr[tr.coding_system == system].code)))
        X.to_parquet(_path("val", system))
        print(f"{system}: dumped")


def top1(X: pd.DataFrame, scores) -> pd.Series:
    X = X.assign(s=scores).sort_values(["item", "s", "code"], ascending=[True, False, True])
    return X.groupby("item").head(1).set_index("item").label


def run(variants: dict[str, dict] | None = None):
    variants = variants or {
        "lgbm_rank": {},
        "lgbm_rank_slow": {"n_estimators": 1200, "learning_rate": 0.015},
        "lgbm_rank_small": {"num_leaves": 15, "min_child_samples": 40},
        "lgbm_binary": {"objective": "binary"},
    }
    for system in config.SYSTEMS:
        Xtr = derive(pd.read_parquet(_path("train", system)))
        Xva = derive(pd.read_parquet(_path("val", system)))
        n_items = Xva.item.nunique()
        seen = Xva.groupby("item").seen.first()
        print(f"[{system}] val items={n_items} candidate recall={Xva.groupby('item').label.max().mean():.4f}")
        for name, params in variants.items():
            for feats_name, feats in (("all", FEATURES + DERIVED), ("raw", FEATURES)):
                if feats_name == "raw" and name != "lgbm_rank":
                    continue
                r = Ranker(feats, **params).fit(Xtr)
                hit = top1(Xva, r.score(Xva))
                hit = hit.reindex(seen.index, fill_value=0)
                print(f"  {name:16s} {feats_name:4s} acc={hit.sum() / n_items:.4f} "
                      f"seen={hit[seen].mean():.4f} unseen={hit[~seen].mean():.4f}")


def run_llm(threshold: float = 0.8, max_terms: int = 200, k: int = 15):
    """Accuracy before/after LLM adjudication of low-confidence val terms (first `max_terms` per system)."""
    if not llm.enabled():
        raise SystemExit("set LLM_PROVIDER and its key in .env first")
    train = data.load_train()
    va = train[split(train, 0.2)]
    eng = TermMapEngine()
    for system in config.SYSTEMS:
        items = va[(va.coding_system == system) & (va.subconcept == "DiagnosisCode")].reset_index(drop=True)
        Xva = derive(pd.read_parquet(_path("val", system)))
        Xva["prob"] = softmax_by_group(Xva, Ranker(FEATURES + DERIVED).fit(derive(pd.read_parquet(
            _path("train", system)))).score(Xva))
        Xva = Xva.sort_values(["item", "prob", "code"], ascending=[True, False, True])
        Xva = Xva[Xva.item < max_terms]
        before = Xva.groupby("item").head(1).label.sum()
        after_X = eng._adjudicate(items, Xva.copy(), system, threshold, k)
        after = after_X.groupby("item").head(1).label.sum()
        n = Xva.item.nunique()
        sent = (Xva.groupby("item").prob.max() < threshold).sum()
        print(f"[{system}] n={n} sent_to_llm={sent} before={before / n:.4f} after={after / n:.4f}")


def _items(train: pd.DataFrame, system: str):
    """Rebuild the term rows behind the cached feature tables (same order and ids as `dump`)."""
    is_val = split(train, 0.2)
    tr, va = train[~is_val], train[is_val]
    folds = TermMapEngine._folds(tr)
    target = (tr.coding_system == system) & (tr.subconcept == "DiagnosisCode")
    tr_items = {}
    for f in range(5):
        rows = tr[target & (folds == f)].reset_index(drop=True)
        for i, r in rows.iterrows():
            tr_items[f * 1_000_000 + i] = (r.entity, r.chunk_text)
    va_rows = va[(va.coding_system == system) & (va.subconcept == "DiagnosisCode")].reset_index(drop=True)
    return tr_items, {i: (r.entity, r.chunk_text) for i, r in va_rows.iterrows()}


def run_ce(k: int = 15, epochs: int = 2):
    """Cross-encoder over the ranker's top-k: trained on the train side's OOF candidates, tested on val.
    Reports ranker alone, cross-encoder alone and z-score blends of the two."""
    from termmap.cross_encoder import CrossEncoder, candidate_text, query_text
    from termmap.engine import CodeSystem

    train = data.load_train()
    for system in config.SYSTEMS:
        cs = CodeSystem.load(system)
        tr_items, va_items = _items(train, system)
        Xtr = derive(pd.read_parquet(_path("train", system)))
        Xva = derive(pd.read_parquet(_path("val", system)))
        ranker = Ranker(FEATURES + DERIVED).fit(Xtr)
        for X in (Xtr, Xva):
            X["rk"] = ranker.score(X)
        tops = {}
        for name, X in (("train", Xtr), ("val", Xva)):
            X = X.sort_values(["item", "rk", "code"], ascending=[True, False, True])
            tops[name] = X.groupby("item").head(k)
        groups = []
        for item, g in tops["train"].groupby("item"):
            if g.label.max() == 0:
                continue
            ent, chunk = tr_items[item]
            groups.append((query_text(ent, chunk), [candidate_text(c, cs) for c in g.code], int(np.argmax(g.label.values))))
        print(f"[{system}] training cross-encoder on {len(groups)} terms x {k} candidates")
        ce = CrossEncoder().fit(groups, epochs=epochs)
        ce.save(config.CACHE_DIR / f"ce_{system}")
        V = tops["val"].copy()
        V["q"] = [query_text(*va_items[i]) for i in V.item]
        V["ce"] = ce.score(V.q.tolist(), [candidate_text(c, cs) for c in V.code])
        n = Xva.item.nunique()
        z = lambda col: (V[col] - V.groupby("item")[col].transform("mean")) / (V.groupby("item")[col].transform("std") + 1e-6)
        V["zrk"], V["zce"] = z("rk"), z("ce")
        print(f"[{system}] val n={n}  gold in top-{k}: {V.groupby('item').label.max().sum() / n:.4f}")
        for name, col in [("ranker", V.rk), ("cross-encoder", V.ce)] + [
                (f"blend w_ce={w}", (1 - w) * V.zrk + w * V.zce) for w in (0.3, 0.5, 0.7)]:
            hit = V.assign(s=col).sort_values(["item", "s"], ascending=[True, False]).groupby("item").head(1).label.sum()
            print(f"  {name:16s} acc={hit / n:.4f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "ce":
        run_ce(int(sys.argv[2]) if len(sys.argv) > 2 else 15, int(sys.argv[3]) if len(sys.argv) > 3 else 2)
    elif cmd == "llm":
        run_llm(float(sys.argv[2]) if len(sys.argv) > 2 else 0.8, int(sys.argv[3]) if len(sys.argv) > 3 else 200,
                int(sys.argv[4]) if len(sys.argv) > 4 else 15)
    else:
        {"dump": dump, "run": run}[cmd]()
