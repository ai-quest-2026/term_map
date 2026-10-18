"""Download/convert public resources into data/external/ (run once; inference never calls the network).

    python -m scripts.ingest all                   # everything below
    python -m scripts.ingest cdc --year 2026       # CDC ICD-10-CM tabular + index aliases (public domain)
    python -m scripts.ingest obo [--refresh]       # DOID + Mondo + HPO -> local terminology store + aliases
    python -m scripts.ingest map                   # check the NLM SNOMED->ICD map TSV (needs UMLS licence)
"""
from __future__ import annotations

import argparse

from termmap import config
from termmap.sources import cdc_icd10cm, obo, snomed_icd_map


def _invalidate_indexes():
    for system in config.SYSTEMS:
        (config.CACHE_DIR / f"codesystem_{system}_{config.CACHE_VERSION}.joblib").unlink(missing_ok=True)
    (config.CACHE_DIR / "engine.joblib").unlink(missing_ok=True)


def ingest_cdc(year: int):
    aliases = cdc_icd10cm.build_aliases(year)
    aliases["source"] = f"cdc_fy{year}"
    path = config.EXTERNAL_DIR / f"{config.ICD}_aliases.parquet"
    aliases.to_parquet(path, index=False)
    print(f"wrote {len(aliases)} ICD-10-CM aliases to {path}")


def ingest_obo(refresh: bool):
    from termmap import terminology

    for source in obo.SOURCES:
        obo.fetch(source, refresh=refresh)
        print(f"{source}: {obo.version(source)}")
    tables = terminology.build()
    terminology.save(tables)
    al = tables["aliases"]
    print(f"terminology store: {len(tables['concepts'])} concepts, {len(tables['synonyms'])} synonyms, "
          f"{len(tables['crosswalk'])} SNOMED<->ICD crosswalk pairs")
    print(al.groupby(["system", "source"]).size().to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["all", "cdc", "obo", "map"])
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--refresh", action="store_true", help="re-download ontologies")
    args = ap.parse_args()
    config.EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    if args.what in ("all", "cdc"):
        ingest_cdc(args.year)
    if args.what in ("all", "obo"):
        ingest_obo(args.refresh)
    if args.what == "map":
        m = snomed_icd_map.load_map()
        if not len(m):
            print(f"no map file found; put tls_Icd10cmHumanReadableMap_*.tsv in {config.EXTERNAL_DIR}")
        else:
            print(f"{len(m)} map rows, {m.snomed_code.nunique()} SNOMED concepts, "
                  f"{m.unconditional.mean():.1%} unconditional")
    if args.what != "map":
        _invalidate_indexes()


if __name__ == "__main__":
    main()
