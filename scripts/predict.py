"""End-to-end prediction from one command.

    python -m scripts.predict --input eval.csv --output predictions.csv [--refit] [--details details.csv] [--llm]

Input: CSV with `entity` + `chunk_text` (training schema) or `chunk_id` + `clinical_entity` (problem schema).
Output: one row per input row with icd10cm_code and snomed_code (codes always come from the supplied
dictionaries). --details writes confidences and top-5 alternatives for error analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import time

from termmap import config, data
from termmap.engine import TermMapEngine


def get_engine(refit: bool = False, path=None, train=None) -> TermMapEngine:
    path = path or config.CACHE_DIR / "engine.joblib"
    if os.path.exists(path) and not refit:
        return TermMapEngine.load(path)
    eng = TermMapEngine().fit(data.load_train(train))
    eng.save(path)
    return eng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--details", default=None)
    ap.add_argument("--selector", default=None,
                    help="LoRA selector adapter dir (scripts/train_selector.py); used with an exported stacker")
    ap.add_argument("--llm", action="store_true", help="experimental: adjudicate low-confidence terms with LLM_PROVIDER")
    ap.add_argument("--refit", action="store_true", help="retrain on the training file instead of using the cache")
    ap.add_argument("--engine", default=None, help="engine file to load / save (default data/cache/engine.joblib)")
    ap.add_argument("--train", default=None, help="labelled training file (default hackathon_train.csv)")
    args = ap.parse_args()

    items = data.read_input(args.input)
    if "chunk_id" not in items.columns:  # training-schema input: derive a stable id so the output keeps all 4 columns
        items["chunk_id"] = [hashlib.md5(str(c).encode()).hexdigest()[:12] for c in items.chunk_text]
    if args.selector:
        from termmap.engine import stacker_path
        missing = [str(p) for p in [os.path.join(args.selector, "selector.json")] + [stacker_path(s) for s in config.SYSTEMS]
                   if not os.path.exists(p)]
        if missing:
            raise SystemExit("--selector needs the trained LoRA adapter and the two stackers; missing: " + ", ".join(missing)
                             + ". Download them as described in the README (Hugging Face weights), or run without --selector.")
    eng = get_engine(args.refit, args.engine, args.train)
    if args.selector:
        from termmap.selector import Selector
        eng.selector_model = Selector(args.selector)
    t0 = time.time()
    pred = eng.predict(items, use_llm=args.llm)
    print(f"mapped {len(items)} terms in {time.time() - t0:.1f}s")

    out = items[[c for c in ("chunk_id",) if c in items.columns]].copy()
    out["clinical_entity"] = items.get("clinical_entity", items.entity)
    out["icd10cm_code"] = pred.icd10cm_code
    out["snomed_code"] = pred.snomed_code
    out.to_csv(args.output, index=False)
    print(f"wrote {args.output}")
    if args.details:
        pred.drop(columns=["chunk_text"], errors="ignore").to_csv(args.details, index=False)
        print(f"wrote {args.details}")


if __name__ == "__main__":
    main()
