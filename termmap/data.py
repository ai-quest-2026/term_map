"""Loading the training file and the two code dictionaries (cached as parquet)."""
from __future__ import annotations

import bisect
import re
from functools import lru_cache

import pandas as pd

from . import config

# Semantic tag at the end of a SNOMED fully specified name, e.g. "Diabetes mellitus (disorder)".
_TAG_RE = re.compile(r"\s\(([a-z][a-z /\-]+)\)$", re.I)


def _cached(name: str, build) -> pd.DataFrame:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.CACHE_DIR / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = build()
    df.to_parquet(path, index=False)
    return df


def load_train(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or config.TRAIN_CSV, dtype=str, keep_default_na=False)
    return df


def read_input(path) -> pd.DataFrame:
    """Read an evaluation/inference file in either the training schema
    (entity, chunk_text[, ...]) or the problem-statement schema (chunk_id, clinical_entity)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "entity" not in df.columns and "clinical_entity" in df.columns:
        df["entity"] = df["clinical_entity"]
    if "chunk_text" not in df.columns:
        df["chunk_text"] = df.get("chunk", "")
    if "entity" not in df.columns:
        raise ValueError(f"input needs an 'entity' or 'clinical_entity' column, got {list(df.columns)}")
    return df


def diagnosis_rows(train: pd.DataFrame) -> pd.DataFrame:
    return train[train.subconcept == "DiagnosisCode"].copy()


@lru_cache(maxsize=1)
def load_icd() -> pd.DataFrame:
    """One row per (code, description). Adds is_range, is_leaf, category (3-char)."""

    def build():
        df = pd.read_excel(config.ICD_XLSX, dtype=str, engine="openpyxl")
        df.columns = ["code", "description"]
        df = df.dropna().drop_duplicates()
        df["code"] = df.code.str.strip()
        df["description"] = df.description.str.strip()
        df["is_range"] = df.code.str.contains("-")
        codes = sorted(set(df.loc[~df.is_range, "code"]))

        def is_leaf(c):
            i = bisect.bisect_right(codes, c)
            return not (i < len(codes) and codes[i].startswith(c))

        leaf = {c: is_leaf(c) for c in codes}
        df["is_leaf"] = df.code.map(leaf).fillna(False).astype(bool)
        df["category"] = df.code.str[:3]
        return df.reset_index(drop=True)

    return _cached("icd", build)


@lru_cache(maxsize=1)
def load_snomed() -> pd.DataFrame:
    """One row per (code, description) with the concept's semantic tag (from its FSN).
    FSN rows have the tag stripped so they index like synonyms."""

    def build():
        df = pd.read_excel(config.SNOMED_XLSX, sheet_name="SNOMED codes", dtype=str, engine="openpyxl")
        df.columns = ["code", "description", "order"]
        df = df.dropna(subset=["code", "description"])
        df["order"] = pd.to_numeric(df.order, errors="coerce").fillna(99).astype(int)
        tag = df.description.str.extract(_TAG_RE)[0].str.lower()
        # The FSN is the last description carrying a semantic tag.
        fsn_tag = tag.groupby(df.code).agg(lambda s: s.dropna().iloc[-1] if s.notna().any() else "")
        df["tag"] = df.code.map(fsn_tag).fillna("")
        is_fsn = tag.notna() & (tag == df.tag)
        df["is_fsn"] = is_fsn
        df.loc[is_fsn, "description"] = df.loc[is_fsn, "description"].str.replace(_TAG_RE, "", regex=True)
        df = df.drop_duplicates(["code", "description"])
        return df.reset_index(drop=True)

    return _cached("snomed", build)


def snomed_index_rows() -> pd.DataFrame:
    df = load_snomed()
    return df[~df.tag.isin(config.SNOMED_EXCLUDED_TAGS)].reset_index(drop=True)


def icd_index_rows() -> pd.DataFrame:
    df = load_icd()
    return df[~df.is_range].reset_index(drop=True)


@lru_cache(maxsize=1)
def preferred_terms() -> dict[str, dict[str, str]]:
    """code -> a human-readable display (first description) per system."""
    icd = load_icd()
    icd = icd[~icd.is_range].drop_duplicates("code")
    sn = load_snomed().sort_values(["code", "order"])
    sn_pref = sn[~sn.is_fsn].drop_duplicates("code")
    sn_pref = pd.concat([sn_pref, sn.drop_duplicates("code")]).drop_duplicates("code")
    return {
        config.ICD: dict(zip(icd.code, icd.description)),
        config.SNOMED: dict(zip(sn_pref.code, sn_pref.description)),
    }
