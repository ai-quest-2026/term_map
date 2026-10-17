"""UMLS Terminology Services REST API (needs a free UMLS licence; key in UMLS_API_KEY).

Docs: https://documentation.uts.nlm.nih.gov/rest/home.html
"""
from __future__ import annotations

import os

from .http import get_json

BASE = "https://uts-ws.nlm.nih.gov/rest"


class MissingKey(RuntimeError):
    pass


def _key() -> str:
    key = os.environ.get("UMLS_API_KEY", "").strip()
    if not key:
        raise MissingKey("UMLS_API_KEY is not set (see .env.example and README: 'Getting a UMLS key')")
    return key


def search(term: str, sabs: str = "SNOMEDCT_US,ICD10CM", k: int = 10, search_type: str = "words") -> list[dict]:
    """Source-asserted codes whose names match `term`."""
    res = get_json(f"{BASE}/search/current", {"string": term, "sabs": sabs, "returnIdType": "code",
                                              "searchType": search_type, "pageSize": k, "apiKey": _key()})
    return [{"code": r["ui"], "source": r["rootSource"], "name": r["name"]}
            for r in res["result"]["results"] if r["ui"] != "NONE"]


def crosswalk(source: str, code: str, target: str = "ICD10CM") -> list[dict]:
    """Codes in `target` sharing a UMLS concept with `source`/`code` (e.g. SNOMEDCT_US -> ICD10CM)."""
    try:
        res = get_json(f"{BASE}/crosswalk/current/source/{source}/{code}",
                       {"targetSource": target, "apiKey": _key()})
    except Exception as e:  # 404 means no crosswalk entry
        if "404" in str(e):
            return []
        raise
    return [{"code": r["ui"], "name": r["name"], "source": r["rootSource"]} for r in res.get("result", [])]


def atoms(source: str, code: str) -> list[str]:
    """All synonyms (atom strings) of a source code, e.g. extra aliases for a SNOMED concept."""
    res = get_json(f"{BASE}/content/current/source/{source}/{code}/atoms",
                   {"pageSize": 100, "apiKey": _key()})
    return sorted({a["name"] for a in res.get("result", [])})
