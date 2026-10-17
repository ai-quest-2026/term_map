"""TermMap HTTP API.

    uvicorn api.main:app --port 8000

Mapping:          POST /map, POST /map/batch
Dictionaries:     GET /search, GET /codes/{system}/{code}
Terminology:      GET /terminology/... (our own local service over open ontologies + CDC; no UMLS needed)
Public sources:   GET /external/... (thin proxies over the clients in termmap.sources; these send the
                  query to a third party, so never call them with chart data)
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from termmap import config, terminology
from termmap.normalize import basic
from termmap.sources import clinical_tables, umls

app = FastAPI(title="TermMap", version="0.1.0",
              description="Clinical term -> ICD-10-CM + SNOMED CT mapping engine")

_SYSTEM_ALIASES = {"icd10cm": config.ICD, "icd": config.ICD, "snomed": config.SNOMED, "snomedct_us": config.SNOMED}


@lru_cache(maxsize=1)
def engine():
    from scripts.predict import get_engine
    return get_engine()


class Term(BaseModel):
    entity: str = Field(..., examples=["chronickidney disease stag 3"])
    chunk_text: str = Field("", examples=["a/p: chronickidney disease stag 3, cr stable"])
    chunk_id: str | None = None


class Batch(BaseModel):
    terms: list[Term]


def _parse(row: pd.Series) -> dict:
    out = {"entity": row["entity"]}
    for col, key in (("icd10cm", "icd10cm"), ("snomed", "snomed")):
        alts = []
        for part in str(row[f"{col}_top5"]).split(" | "):
            if not part:
                continue
            code, *rest = part.split(":")
            alts.append({"code": code, "display": ":".join(rest[:-1]), "score": float(rest[-1])})
        out[key] = {"code": row[f"{col}_code"], "display": row[f"{col}_display"],
                    "confidence": row[f"{col}_confidence"], "margin": row[f"{col}_margin"], "candidates": alts}
    return out


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/map")
def map_term(term: Term):
    return map_batch(Batch(terms=[term]))[0]


@app.post("/map/batch")
def map_batch(batch: Batch):
    items = pd.DataFrame([t.model_dump() for t in batch.terms])
    pred = engine().predict(items)
    out = [_parse(r) for _, r in pred.iterrows()]
    for o, t in zip(out, batch.terms):
        o["chunk_id"] = t.chunk_id
    return out


@app.get("/search")
def search(q: str, system: str = "icd10cm", k: int = 10):
    sysname = _SYSTEM_ALIASES.get(system.lower())
    if not sysname:
        raise HTTPException(400, f"unknown system {system}")
    cs = engine().systems[sysname]
    hits = cs.retriever.search([basic(q)], k)[0][0]
    return [{"code": h.code, "matched": h.text, "display": cs.pref.get(h.code, ""), "score": round(h.score, 4)}
            for h in hits]


@app.get("/codes/{system}/{code}")
def code_info(system: str, code: str):
    sysname = _SYSTEM_ALIASES.get(system.lower())
    if not sysname:
        raise HTTPException(400, f"unknown system {system}")
    cs = engine().systems[sysname]
    if code not in cs.valid:
        raise HTTPException(404, f"{code} not in the {sysname} dictionary")
    info = {"code": code, "system": sysname, "display": cs.pref.get(code, ""),
            "training_frequency": engine().memory[sysname].freq.get(code, 0)}
    if sysname == config.ICD:
        info.update(is_billable_leaf=bool(cs.is_leaf.get(code)), category=code[:3])
    else:
        info.update(semantic_tag=cs.tag.get(code, ""))
    return info


# --------------------------------------------------------------------------- local terminology service

def _system(system: str) -> str:
    sysname = _SYSTEM_ALIASES.get(system.lower())
    if not sysname:
        raise HTTPException(400, f"unknown system {system}")
    return sysname


@app.get("/terminology/concept/{system}/{code}")
def term_concept(system: str, code: str):
    """Dictionary descriptions + open-ontology synonyms + linked codes for one code (UMLS-like lookup)."""
    sysname = _system(system)
    info = terminology.concept_info(sysname, code)
    cs = engine().systems[sysname]
    info["dictionary_display"] = cs.pref.get(code, "")
    return info


@app.get("/terminology/crosswalk/{system}/{code}")
def term_crosswalk(system: str, code: str):
    """Codes in the other system linked through shared open-ontology concepts (not ground truth)."""
    return terminology.crosswalk(_system(system), code)


@app.get("/terminology/search")
def term_search(q: str, k: int = 20):
    """Synonym search across DOID, Mondo and HPO with the dictionary codes each concept links to."""
    return terminology.search(q, k)


@app.get("/terminology/versions")
def term_versions():
    return terminology.table("versions").to_dict("records")


# --------------------------------------------------------------------------- public sources

@app.get("/external/clinicaltables/icd10cm")
def ext_ct_icd(q: str, k: int = 10):
    return clinical_tables.search_icd10cm(q, k)


@app.get("/external/clinicaltables/conditions")
def ext_ct_conditions(q: str, k: int = 10):
    return clinical_tables.search_conditions(q, k)


@app.get("/external/umls/search")
def ext_umls_search(q: str, sabs: str = "SNOMEDCT_US,ICD10CM", k: int = 10):
    try:
        return umls.search(q, sabs=sabs, k=k)
    except umls.MissingKey as e:
        raise HTTPException(503, str(e))


@app.get("/external/umls/crosswalk/{source}/{code}")
def ext_umls_crosswalk(source: str, code: str, target: str = "ICD10CM"):
    try:
        return umls.crosswalk(source, code, target)
    except umls.MissingKey as e:
        raise HTTPException(503, str(e))
