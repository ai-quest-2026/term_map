"""NLM Clinical Table Search Service (keyless): https://clinicaltables.nlm.nih.gov/

Used offline to harvest consumer-friendly aliases for ICD-10-CM codes. Never send chart text here.
"""
from __future__ import annotations

from .http import get_json

BASE = "https://clinicaltables.nlm.nih.gov/api"


def search_icd10cm(terms: str, k: int = 10) -> list[dict]:
    """Search ICD-10-CM codes by code or name. Response: [total, codes, extra, [[code, name], ...]]."""
    res = get_json(f"{BASE}/icd10cm/v3/search", {"sf": "code,name", "terms": terms, "maxList": k})
    return [{"code": c, "name": n} for c, n in res[3]]


def search_conditions(terms: str, k: int = 10) -> list[dict]:
    """Problem-list conditions with consumer names and their ICD-10-CM codes."""
    res = get_json(f"{BASE}/conditions/v3/search",
                   {"terms": terms, "maxList": k, "df": "primary_name,consumer_name", "ef": "icd10cm_codes,synonyms"})
    extra = res[2] or {}
    out = []
    for i, (primary, consumer) in enumerate(res[3]):
        out.append({"primary_name": primary, "consumer_name": consumer,
                    "icd10cm_codes": (extra.get("icd10cm_codes") or [""] * len(res[3]))[i],
                    "synonyms": (extra.get("synonyms") or [[]] * len(res[3]))[i]})
    return out
