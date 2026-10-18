"""Pretty-print a predict --details file for the demo video: both codes, confidence and the runner-up candidates.

    python -m scripts.predict --input docs/demo/hard_cases.csv --output /tmp/demo_pred.csv --details /tmp/demo_details.csv \
        --selector data/cache/stages/v2/selector_f0
    python -m scripts.demo_show /tmp/demo_details.csv [--top 3] [--review-below 0.5]
"""
from __future__ import annotations

import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("details")
    ap.add_argument("--top", type=int, default=3, help="alternatives shown per system")
    ap.add_argument("--review-below", type=float, default=0.5, help="flag predictions below this confidence")
    args = ap.parse_args()
    d = pd.read_csv(args.details, dtype=str, keep_default_na=False)
    term = "clinical_entity" if "clinical_entity" in d.columns else "entity"
    for _, r in d.iterrows():
        print(f"\n{r.get('chunk_id', '')}  \"{r[term]}\"")
        for col, name in (("icd10cm", "ICD-10-CM"), ("snomed", "SNOMED CT")):
            conf = float(r[f"{col}_confidence"] or 0)
            flag = "  <- low confidence: route to review" if conf < args.review_below else ""
            print(f"  {name:9} {r[f'{col}_code']:<10} {r[f'{col}_display'][:70]:<70} conf {conf:.2f}{flag}")
            for alt in [a for a in r[f"{col}_top5"].split(" | ") if a][1:args.top + 1]:
                code, *rest = alt.split(":")
                print(f"  {'':9}   also {code:<10} {':'.join(rest[:-1])[:62]:<62} {float(rest[-1]):.2f}")


if __name__ == "__main__":
    main()
