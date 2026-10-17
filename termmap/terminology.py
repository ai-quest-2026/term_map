"""Local, licence-free terminology service built from open ontologies (DOID, Mondo, HPO) + CDC ICD-10-CM.

It stands in for the UMLS API: synonyms per code, a SNOMED<->ICD-10-CM crosswalk, and synonym search,
all served from parquet files in data/external/terminology/ (built offline by `scripts/ingest.py obo`).

Linking rules (conservative, so aliases stay precise):
- Mondo and DOID concepts are merged when Mondo asserts equivalence (DOID or UMLS cross-reference with
  `equivalentTo` provenance) or when they share a UMLS CUI.
- A merged concept is linked to a dictionary code through an ontology cross-reference, or through an
  exact (normalized) name match against the supplied dictionary when that name maps to <= 2 codes.
- Aliases are only emitted for concepts linked to <= 3 codes of a system (broad groupings are skipped).
"""
from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache

import pandas as pd

from . import config, data
from .normalize import basic
from .sources import obo

STORE = config.EXTERNAL_DIR / "terminology"
MAX_CODES_PER_CONCEPT = 3
MAX_CODES_PER_NAME = 2


class _UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _dictionary_names() -> dict[str, dict[str, set[str]]]:
    out = {}
    for system, df in ((config.ICD, data.icd_index_rows()), (config.SNOMED, data.snomed_index_rows())):
        names: dict[str, set[str]] = defaultdict(set)
        for code, desc in zip(df.code, df.description):
            names[basic(desc)].add(code)
        out[system] = names
    return out


def build(sources=("doid", "mondo", "hpo")) -> dict[str, pd.DataFrame]:
    terms = {s: obo.parse(s) for s in sources}
    uf = _UnionFind()
    by_cui: dict[str, list[str]] = defaultdict(list)
    for src, ts in terms.items():
        for t in ts:
            uf.find(t.id)
            for system, code, prov in t.xrefs:
                if system == "UMLS" and (src == "doid" or prov == "equivalent"):
                    by_cui[code].append(t.id)
                if src == "mondo" and system == "DOID" and prov == "equivalent":
                    uf.union(t.id, f"DOID:{code}")
    for ids in by_cui.values():
        for other in ids[1:]:
            uf.union(ids[0], other)

    valid = {config.ICD: set(data.icd_index_rows().code), config.SNOMED: set(data.load_snomed().code)}
    names = _dictionary_names()
    concepts, synonyms, xrefs = [], [], []
    for src, ts in terms.items():
        for t in ts:
            g = uf.find(t.id)
            concepts.append((g, t.id, src, t.name))
            synonyms.append((g, t.name, "LABEL", src))
            synonyms += [(g, text, scope, src) for text, scope in t.synonyms]
            for system, code, prov in t.xrefs:
                if system in valid and code in valid[system]:
                    xrefs.append((g, system, code, f"xref:{src}" + (":equivalent" if prov else "")))
    concepts = pd.DataFrame(concepts, columns=["concept", "source_id", "source", "label"])
    synonyms = pd.DataFrame(synonyms, columns=["concept", "text", "scope", "source"]).drop_duplicates()
    synonyms["norm"] = synonyms.text.map(basic)

    # Exact-name bridge to the supplied dictionaries (this is how HPO gets linked).
    exact = synonyms[synonyms.scope.isin(["LABEL", "EXACT"])]
    for g, norm in zip(exact.concept, exact.norm):
        for system in (config.ICD, config.SNOMED):
            codes = names[system].get(norm)
            if codes and len(codes) <= MAX_CODES_PER_NAME:
                xrefs += [(g, system, c, "name_match") for c in codes]
    xrefs = pd.DataFrame(xrefs, columns=["concept", "system", "code", "via"]).drop_duplicates()

    # Aliases: label + exact synonyms of concepts linked to few codes of a system.
    n_codes = xrefs.groupby(["concept", "system"]).code.nunique()
    linked = xrefs.join(n_codes.rename("n_codes"), on=["concept", "system"])
    linked = linked[linked.n_codes <= MAX_CODES_PER_CONCEPT]
    aliases = linked.merge(exact[["concept", "text", "source"]], on="concept")
    aliases = aliases[["system", "code", "text", "source", "via"]].drop_duplicates(["system", "code", "text"])

    # Open crosswalk: SNOMED and ICD codes linked to the same merged concept.
    sn = linked[linked.system == config.SNOMED][["concept", "code", "via"]]
    icd = linked[linked.system == config.ICD][["concept", "code", "via"]]
    xwalk = sn.merge(icd, on="concept", suffixes=("_snomed", "_icd"))
    xwalk = (xwalk.groupby(["code_snomed", "code_icd"]).agg(n_concepts=("concept", "nunique"),
                                                             via=("via_icd", lambda s: ",".join(sorted(set(s)))))
             .reset_index().rename(columns={"code_snomed": "snomed_code", "code_icd": "icd10cm_code"}))

    versions = pd.DataFrame([(s, obo.version(s)) for s in sources], columns=["source", "version"])
    return {"concepts": concepts, "synonyms": synonyms, "xrefs": xrefs, "aliases": aliases,
            "crosswalk": xwalk, "versions": versions}


def save(tables: dict[str, pd.DataFrame]):
    STORE.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(STORE / f"{name}.parquet", index=False)
    for system in config.SYSTEMS:
        al = tables["aliases"]
        al = al[al.system == system].rename(columns={"text": "description"})[["code", "description", "source"]]
        al.to_parquet(config.EXTERNAL_DIR / f"{system}_aliases_obo.parquet", index=False)


@lru_cache(maxsize=None)
def table(name: str) -> pd.DataFrame:
    path = STORE / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m scripts.ingest obo`")
    return pd.read_parquet(path)


def concept_info(system: str, code: str) -> dict:
    """Everything the open sources know about one code: linked concepts, synonyms, cross-system codes."""
    x = table("xrefs")
    concepts = x[(x.system == system) & (x.code == code)]
    ids = set(concepts.concept)
    syn = table("synonyms")
    syn = syn[syn.concept.isin(ids)]
    other = x[x.concept.isin(ids) & (x.system != system)]
    src = table("concepts")
    return {
        "system": system, "code": code,
        "concepts": src[src.concept.isin(ids)][["source_id", "source", "label"]].to_dict("records"),
        "links": concepts[["concept", "via"]].to_dict("records"),
        "synonyms": syn[["text", "scope", "source"]].drop_duplicates("text").to_dict("records"),
        "related_codes": other[["system", "code", "via"]].drop_duplicates().to_dict("records"),
    }


def crosswalk(system: str, code: str) -> list[dict]:
    x = table("crosswalk")
    if system == config.SNOMED:
        hits = x[x.snomed_code == code].rename(columns={"icd10cm_code": "code"})
    else:
        hits = x[x.icd10cm_code == code].rename(columns={"snomed_code": "code"})
    return hits.sort_values("n_concepts", ascending=False)[["code", "n_concepts", "via"]].to_dict("records")


def search(q: str, k: int = 20) -> list[dict]:
    syn = table("synonyms")
    norm = basic(q)
    hit = syn[syn.norm.str.contains(rf"\b{re.escape(norm)}\b", regex=True)].copy()
    hit["exact"] = hit.norm == norm
    hit = hit.sort_values(["exact", "norm"], ascending=[False, True], key=lambda s: s if s.dtype == bool else s.str.len())
    x = table("xrefs")
    out = []
    for g, grp in hit.groupby("concept", sort=False):
        codes = x[x.concept == g]
        out.append({"concept": g, "matched": grp.text.iloc[0],
                    "icd10cm": sorted(codes[codes.system == config.ICD].code.unique().tolist()),
                    "snomed": sorted(codes[codes.system == config.SNOMED].code.unique().tolist())})
        if len(out) >= k:
            break
    return out
