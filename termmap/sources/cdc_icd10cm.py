"""CDC/NCHS ICD-10-CM release files (public domain):
https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/<fiscal year>/

From the tabular XML we take inclusion terms and Excludes1/2 notes; from the alphabetic index we take
index paths ("Hypertension, essential" -> I10). Both become extra retrieval aliases.
"""
from __future__ import annotations

import io
import re
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd

from .. import config
from .http import download, get_text

BASE = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM"


def fetch_release(year: int) -> zipfile.ZipFile:
    """Download the 'table and index' zip for a fiscal year (file names vary by year, so list the folder)."""
    listing = get_text(f"{BASE}/{year}/")
    names = re.findall(r'href="([^"]+\.zip)"', listing, re.I)
    pick = [n for n in names if re.search(r"tab", n, re.I) and re.search(r"index", n, re.I)]
    if not pick:
        raise RuntimeError(f"no table+index zip under {BASE}/{year}/; found {names}")
    name = pick[0].rsplit("/", 1)[-1]
    dest = config.EXTERNAL_DIR / name
    config.EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        download(f"{BASE}/{year}/{name}", dest)
    return zipfile.ZipFile(dest)


def _member(z: zipfile.ZipFile, pattern: str) -> bytes:
    for n in z.namelist():
        if re.search(pattern, n, re.I) and n.lower().endswith(".xml"):
            return z.read(n)
    raise KeyError(pattern)


def parse_tabular(xml: bytes) -> pd.DataFrame:
    """Rows: code, kind (inclusion|excludes1|excludes2|desc), text."""
    rows = []
    root = ET.parse(io.BytesIO(xml)).getroot()
    for diag in root.iter("diag"):
        code = (diag.findtext("name") or "").strip()
        rows.append((code, "desc", (diag.findtext("desc") or "").strip()))
        for kind in ("inclusionTerm", "excludes1", "excludes2"):
            for el in diag.findall(kind):
                for note in el.findall("note"):
                    rows.append((code, kind.lower().replace("term", ""), (note.text or "").strip()))
    return pd.DataFrame(rows, columns=["code", "kind", "text"])


def parse_index(xml: bytes) -> pd.DataFrame:
    """Flatten the alphabetic index into (path text, code) aliases."""
    rows = []
    root = ET.parse(io.BytesIO(xml)).getroot()

    def walk(el, path):
        title = " ".join("".join(el.find("title").itertext()).split()) if el.find("title") is not None else ""
        here = path + [title] if title else path
        code = (el.findtext("code") or "").strip()
        if code and here:
            rows.append((code.replace("-", ""), " ".join(reversed(here[1:])) + " " + here[0] if len(here) > 1 else here[0]))
        for child in el.findall("term"):
            walk(child, here)

    for main in root.iter("mainTerm"):
        walk(main, [])
    return pd.DataFrame(rows, columns=["code", "text"])


def build_aliases(year: int) -> pd.DataFrame:
    z = fetch_release(year)
    tab = parse_tabular(_member(z, r"tabular"))
    idx = parse_index(_member(z, r"index(?!.*(drug|neoplasm|eindex))"))
    tab.to_parquet(config.EXTERNAL_DIR / f"icd10cm_tabular_{year}.parquet", index=False)
    aliases = pd.concat([
        tab[tab.kind.isin(["desc", "inclusion"])][["code", "text"]],
        idx,
    ]).rename(columns={"text": "description"})
    return aliases.drop_duplicates()
