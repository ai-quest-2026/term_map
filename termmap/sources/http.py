"""Shared HTTP session with an on-disk JSON cache so repeated lookups are deterministic and offline-friendly."""
from __future__ import annotations

import hashlib
import json

import requests

from .. import config

_CACHE = config.EXTERNAL_DIR / "http_cache"
_session = requests.Session()
_session.headers["User-Agent"] = "termmap/0.1 (hackathon research)"


def get_json(url: str, params: dict | None = None, cache: bool = True, timeout: int = 30):
    params = dict(params or {})
    public = {k: v for k, v in params.items() if k.lower() != "apikey"}
    key = hashlib.sha1((url + json.dumps(public, sort_keys=True)).encode()).hexdigest()
    path = _CACHE / f"{key}.json"
    if cache and path.exists():
        return json.loads(path.read_text())
    r = _session.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if cache:
        _CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    return data


def download(url: str, dest, timeout: int = 300):
    with _session.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return dest


def get_text(url: str, timeout: int = 60) -> str:
    r = _session.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text
