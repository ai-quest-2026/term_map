"""Download/convert public resources into data/external/ (run once; inference never calls the network).

    python -m scripts.ingest cdc --year 2026      # ICD-10-CM tabular + index aliases
    python -m scripts.ingest map                   # check the NLM SNOMED->ICD map TSV you downloaded
"""
from __future__ import annotations

import argparse

from termmap import config
from termmap.sources import cdc_icd10cm, snomed_icd_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["cdc", "map"])
    ap.add_argument("--year", type=int, default=2026)
    args = ap.parse_args()
    config.EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    if args.what == "cdc":
        aliases = cdc_icd10cm.build_aliases(args.year)
        path = config.EXTERNAL_DIR / f"{config.ICD}_aliases.parquet"
        aliases.to_parquet(path, index=False)
        print(f"wrote {len(aliases)} ICD-10-CM aliases to {path}")
        (config.CACHE_DIR / f"codesystem_{config.ICD}.joblib").unlink(missing_ok=True)
    else:
        m = snomed_icd_map.load_map()
        if not len(m):
            print(f"no map file found; put tls_Icd10cmHumanReadableMap_*.tsv in {config.EXTERNAL_DIR}")
        else:
            print(f"{len(m)} map rows, {m.snomed_code.nunique()} SNOMED concepts, "
                  f"{m.unconditional.mean():.1%} unconditional")


if __name__ == "__main__":
    main()
