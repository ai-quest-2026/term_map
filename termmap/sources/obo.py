"""Open biomedical ontologies in OBO format (no UMLS licence needed).

- Disease Ontology (DOID), CC0 1.0: synonyms + ICD10CM / SNOMEDCT_US / UMLS_CUI cross-references
- Mondo Disease Ontology, CC BY 4.0: synonyms + ICD10CM / SCTID (SNOMED) / UMLS cross-references
- Human Phenotype Ontology (HPO), HPO licence (free with attribution): signs/symptoms + synonyms
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .. import config
from .http import download

SOURCES = {
    "doid": "https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/main/src/ontology/doid.obo",
    "mondo": "https://raw.githubusercontent.com/monarch-initiative/mondo/master/src/ontology/mondo-edit.obo",
    "hpo": "https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/master/hp.obo",
}
FILES = {"doid": "doid.obo", "mondo": "mondo-edit.obo", "hpo": "hp.obo"}

_SYN_RE = re.compile(r'^synonym: "((?:[^"\\]|\\.)*)" (EXACT|RELATED|NARROW|BROAD)')
_XREF_RE = re.compile(r"^xref: ([A-Za-z0-9_.\-]+):([^\s{]+)(.*)$")

# Cross-reference prefix -> canonical system.
PREFIX = {"ICD10CM": config.ICD, "SCTID": config.SNOMED, "SNOMEDCT_US": config.SNOMED, "SNOMEDCT": config.SNOMED,
          "UMLS": "UMLS", "UMLS_CUI": "UMLS", "DOID": "DOID", "MONDO": "MONDO", "HP": "HPO"}


@dataclass
class Term:
    id: str
    name: str = ""
    obsolete: bool = False
    synonyms: list[tuple[str, str]] = field(default_factory=list)  # (text, scope)
    xrefs: list[tuple[str, str, str]] = field(default_factory=list)  # (system, code, provenance)


def path(source: str):
    return config.EXTERNAL_DIR / "obo" / FILES[source]


def fetch(source: str, refresh: bool = False):
    p = path(source)
    p.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not p.exists():
        download(SOURCES[source], p)
    return p


def version(source: str) -> str:
    with open(path(source), encoding="utf-8") as f:
        for line in f:
            if line.startswith("data-version:"):
                return line.split(":", 1)[1].strip()
            if line.startswith("[Term]"):
                break
    # mondo-edit.obo carries no data-version: record when our copy was downloaded instead.
    import datetime
    return "downloaded " + datetime.date.fromtimestamp(path(source).stat().st_mtime).isoformat()


def _system(prefix: str) -> str | None:
    if prefix.startswith("SNOMEDCT_US"):  # DOID uses dated prefixes, e.g. SNOMEDCT_US_2025_09_01
        return config.SNOMED
    return PREFIX.get(prefix)


def parse(source: str) -> list[Term]:
    terms, cur = [], None
    with open(path(source), encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("["):
                if cur is not None:
                    terms.append(cur)
                cur = Term("") if line == "[Term]" else None
                continue
            if cur is None or not line:
                continue
            if line.startswith("id: "):
                cur.id = line[4:].strip()
            elif line.startswith("name: "):
                cur.name = line[6:].strip()
            elif line.startswith("is_obsolete: true"):
                cur.obsolete = True
            elif line.startswith("synonym: "):
                m = _SYN_RE.match(line)
                if m:
                    cur.synonyms.append((m.group(1).replace('\\"', '"'), m.group(2)))
            elif line.startswith("xref: "):
                m = _XREF_RE.match(line)
                if m:
                    system = _system(m.group(1))
                    if system:
                        prov = "equivalent" if "equivalentTo" in m.group(3) else ""
                        cur.xrefs.append((system, m.group(2).strip(), prov))
    if cur is not None:
        terms.append(cur)
    return [t for t in terms if t.id and not t.obsolete]
