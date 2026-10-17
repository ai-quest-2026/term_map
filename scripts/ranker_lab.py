"""Fast ranker experiments on cached candidate features.

    python -m scripts.ranker_lab dump      # build raw features for the grouped split (slow, once per retrieval change)
    python -m scripts.ranker_lab run       # train/score ranker variants on the cache (seconds)
    python -m scripts.ranker_lab llm [threshold] [max_terms] [k]   # LLM adjudication gain on the val split
"""
from __future__ import annotations

import sys

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


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "llm":
        run_llm(float(sys.argv[2]) if len(sys.argv) > 2 else 0.8, int(sys.argv[3]) if len(sys.argv) > 3 else 200,
                int(sys.argv[4]) if len(sys.argv) > 4 else 15)
    else:
        {"dump": dump, "run": run}[cmd]()
