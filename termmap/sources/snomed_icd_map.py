"""NLM SNOMED CT to ICD-10-CM map (UMLS licence required), human-readable TSV:
tls_Icd10cmHumanReadableMap_US1000124_<date>.tsv from https://www.nlm.nih.gov/research/umls/mapping_projects/snomedct_to_icd10cm.html

Used only as a prior/feature (agreement between the independent ICD and SNOMED predictions), never as the
source of either prediction.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import config


def find_map_file() -> Path | None:
    hits = sorted(config.EXTERNAL_DIR.glob("*Icd10cmHumanReadableMap*.tsv"))
    return hits[-1] if hits else None


def load_map(path: Path | None = None) -> pd.DataFrame:
    """Columns: snomed_code, icd10cm_code, map_group, map_priority, map_rule, unconditional."""
    path = path or find_map_file()
    if path is None:
        return pd.DataFrame(columns=["snomed_code", "icd10cm_code", "map_group", "map_priority", "map_rule",
                                     "unconditional"])
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df = df[df.get("active", "1").astype(str) == "1"] if "active" in df.columns else df
    out = pd.DataFrame({
        "snomed_code": df.referencedComponentId,
        "icd10cm_code": df.mapTarget,
        "map_group": pd.to_numeric(df.mapGroup, errors="coerce"),
        "map_priority": pd.to_numeric(df.mapPriority, errors="coerce"),
        "map_rule": df.mapRule,
    })
    out["unconditional"] = out.map_rule.str.upper().isin(["TRUE", "OTHERWISE TRUE"])
    return out[out.icd10cm_code != ""].reset_index(drop=True)
