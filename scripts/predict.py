"""End-to-end prediction from one command.

    python -m scripts.predict --input eval.csv --output predictions.csv [--refit] [--details details.csv] [--llm]

Input: CSV with `entity` + `chunk_text` (training schema) or `chunk_id` + `clinical_entity` (problem schema).
Output: one row per input row with icd10cm_code and snomed_code (codes always come from the supplied
dictionaries). --details writes confidences and top-5 alternatives for error analysis.
"""
from __future__ import annotations

import argparse
import time

from termmap import config, data
from termmap.engine import TermMapEngine


def get_engine(refit: bool = False) -> TermMapEngine:
    path = config.CACHE_DIR / "engine.joblib"
    if path.exists() and not refit:
        return TermMapEngine.load(path)
    eng = TermMapEngine().fit(data.load_train())
    eng.save(path)
    return eng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--details", default=None)
    ap.add_argument("--llm", action="store_true", help="experimental: adjudicate low-confidence terms with LLM_PROVIDER")
    ap.add_argument("--refit", action="store_true", help="retrain on the training file instead of using the cache")
    args = ap.parse_args()

    items = data.read_input(args.input)
    eng = get_engine(args.refit)
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
