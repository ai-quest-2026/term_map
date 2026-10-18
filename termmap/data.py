"""Loading the training file and the two code dictionaries (cached as parquet)."""
from __future__ import annotations

import bisect
import re
from functools import lru_cache

import pandas as pd

from . import config

# Semantic tag at the end of a SNOMED fully specified name, e.g. "Diabetes mellitus (disorder)".
_TAG_RE = re.compile(r"\s\(([a-z][a-z /\-]+)\)$", re.I)
# SNOMED CT semantic tags. Only these count as an FSN tag: synonyms can also end in a parenthetical
# ("... (helicobacter pylori)", "... (combined site)"), which is not the concept's semantic tag.
SEMANTIC_TAGS = {
    "disorder", "finding", "procedure", "body structure", "organism", "substance", "physical object",
    "observable entity", "qualifier value", "medicinal product", "clinical drug", "medicinal product form",
    "situation", "morphologic abnormality", "occupation", "event", "regime/therapy", "specimen",
    "assessment scale", "product", "environment", "attribute", "cell", "geographic location",
    "navigational concept", "record artifact", "cell structure", "person", "dose form", "disposition",
    "namespace concept", "foundation metadata concept", "religion/philosophy", "physical force", "tumor staging",
    "role", "property", "contextual qualifier", "basic dose form", "core metadata concept", "unit of presentation",
    "staging scale", "intended site", "social concept", "administration method", "racial group", "ethnic group",
    "link assertion", "linkage concept", "transformation", "release characteristic", "state of matter",
    "owl metadata concept", "life style", "environment / location", "inactive concept", "supplier",
    "real clinical drug", "packaged clinical drug", "special concept", "metadata", "number",
}


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
        df = df.reset_index(drop=True)
        tag = df.description.str.extract(_TAG_RE)[0].str.lower()
        tag = tag.where(tag.isin(SEMANTIC_TAGS))
        # The FSN is the highest-order description ending in a real semantic tag.
        fsn = df[tag.notna()].assign(_tag=tag[tag.notna()]).sort_values(["code", "order"], kind="stable")
        fsn = fsn.groupby("code").tail(1)
        df["tag"] = df.code.map(dict(zip(fsn.code, fsn._tag))).fillna("")
        df["is_fsn"] = df.index.isin(fsn.index)
        df.loc[df.is_fsn, "description"] = df.loc[df.is_fsn, "description"].str.replace(_TAG_RE, "", regex=True)
        df = df.drop_duplicates(["code", "description"])
        return df.reset_index(drop=True)

    return _cached("snomed_v2", build)


def snomed_index_rows() -> pd.DataFrame:
    df = load_snomed()
    return df[~df.tag.isin(config.SNOMED_EXCLUDED_TAGS)].reset_index(drop=True)


def icd_index_rows() -> pd.DataFrame:
    df = load_icd()
    return df[~df.is_range].reset_index(drop=True)


# Short-form words of the CMS abbreviated descriptions ("Oth symptoms...", "Unsp fracture ..., init").
_ICD_SHORT_RE = re.compile(r"\b(?:oth|unsp|init|subs|sqla|encntr|nondisp|displ|w|w/o|fx|sys)\b", re.I)


def _icd_display(descs: list[str]) -> str:
    """The full official wording among a code's descriptions: not abbreviated, not an index 'NOS' alias."""
    return max(descs, key=lambda d: (not _ICD_SHORT_RE.search(d), not d.endswith(" NOS"), len(d)))


@lru_cache(maxsize=1)
def preferred_terms() -> dict[str, dict[str, str]]:
    """code -> a human-readable display per system. ICD: the CDC tabular long description when ingested
    (scripts/ingest.py cdc), else the full official wording among the dictionary rows (the first row is
    often the abbreviated short description). SNOMED: the first synonym, else the FSN."""
    icd = load_icd()
    icd = icd[~icd.is_range]
    icd_pref = icd.groupby("code").description.agg(lambda s: _icd_display(list(s))).to_dict()
    tab = config.EXTERNAL_DIR / "icd10cm_tabular_2026.parquet"
    if tab.exists():
        t = pd.read_parquet(tab)
        t = t[(t.kind == "desc") & t.code.isin(icd_pref)].drop_duplicates("code")
        icd_pref.update(zip(t.code, t.text))
    sn = load_snomed().sort_values(["code", "order"])
    sn_pref = sn[~sn.is_fsn].drop_duplicates("code")
    sn_pref = pd.concat([sn_pref, sn.drop_duplicates("code")]).drop_duplicates("code")
    return {
        config.ICD: icd_pref,
        config.SNOMED: dict(zip(sn_pref.code, sn_pref.description)),
    }
