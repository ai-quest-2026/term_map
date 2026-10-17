"""Paths and constants. Everything overridable through environment variables (.env supported)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()


def _path(env: str, default: str) -> Path:
    p = Path(os.environ.get(env, default))
    return p if p.is_absolute() else ROOT / p


TRAIN_CSV = _path("TERMMAP_TRAIN", "hackathon_train.csv")
ICD_XLSX = _path("TERMMAP_ICD_XLSX", "icd10cm_codes.xlsx")
SNOMED_XLSX = _path("TERMMAP_SNOMED_XLSX", "SNOMEDCT_US_codes_and_descriptions.xlsx")
CACHE_DIR = _path("TERMMAP_CACHE", "data/cache")
EXTERNAL_DIR = _path("TERMMAP_EXTERNAL", "data/external")

ICD = "ICD10CM"
SNOMED = "SNOMEDCT_US"
SYSTEMS = (ICD, SNOMED)

SEED = 13

# SNOMED semantic tags that never hold a diagnosis/condition answer; dropped from the index
# to cut the candidate space roughly in half.
SNOMED_EXCLUDED_TAGS = {
    "medicinal product", "clinical drug", "medicinal product form", "product",
    "real clinical drug", "packaged clinical drug", "cell", "cell structure",
    "geographic location", "navigational concept", "namespace concept", "linkage concept",
    "attribute", "core metadata concept", "foundation metadata concept", "OWL metadata concept",
    "specimen", "occupation", "person", "ethnic group", "racial group", "religion/philosophy",
    "social concept", "life style", "environment", "environment / location", "inactive concept",
    "record artifact", "staging scale", "tumor staging", "unit of presentation", "basic dose form",
    "dose form", "role", "state of matter", "supplier", "transformation", "intended site",
    "release characteristic", "administration method", "disposition", "number", "product name",
}
