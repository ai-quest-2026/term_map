"""Dense signal from biomedical word vectors (scispacy en_core_sci_lg 0.5.4, CC BY-SA 3.0).

Texts are embedded as the IDF-weighted mean of their word vectors. Weak next to a sentence encoder like
SapBERT, but it knows that "nose"~"nasal" and "discomfort"~"pain", which character n-grams cannot.
"""
from __future__ import annotations

import glob
import json
import math
from collections import Counter
from functools import lru_cache

import numpy as np

from . import config
from .normalize import basic
from .retrieve import Hit

SCISPACY_URL = "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz"


def _extract_from_scispacy() -> tuple[list[str], np.ndarray]:
    from spacy.vocab import Vocab  # only needed once, to convert the model's vectors

    found = glob.glob(str(config.EXTERNAL_DIR / "scispacy" / "en_core_sci_lg-*" / "en_core_sci_lg" / "en_core_sci_lg-*" / "vocab"))
    if not found:
        raise FileNotFoundError(f"download and untar {SCISPACY_URL} into {config.EXTERNAL_DIR / 'scispacy'}")
    vocab = Vocab().from_disk(found[0])
    vecs = vocab.vectors
    words, rows = [], []
    for s in vocab.strings:
        key = vocab.strings[s]
        if key in vecs.key2row and s == s.lower() and len(s) < 40:
            words.append(s)
            rows.append(vecs.key2row[key])
    return words, np.asarray(vecs.data[rows], dtype=np.float32)


class WordVectors:
    def __init__(self):
        npy, js = config.CACHE_DIR / "wordvec.npy", config.CACHE_DIR / "wordvec.json"
        if not npy.exists():
            words, mat = _extract_from_scispacy()
            mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
            np.save(npy, mat.astype(np.float16))
            js.write_text(json.dumps(words))
        self.mat = np.load(npy)  # float16 rows, unit-normalized
        self.row = {w: i for i, w in enumerate(json.loads(js.read_text()))}
        self.idf: dict[str, float] = {}

    def fit_idf(self, texts: list[str]):
        df = Counter()
        for t in texts:
            df.update(set(basic(t).split()))
        n = len(texts)
        self.idf = {w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items()}
        self.default_idf = math.log(n + 1) + 1
        return self

    def vec(self, word: str):
        i = self.row.get(word)
        return None if i is None else self.mat[i].astype(np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.mat.shape[1]), dtype=np.float32)
        for j, t in enumerate(texts):
            acc, wsum = None, 0.0
            for w in basic(t).split():
                i = self.row.get(w)
                if i is None:
                    continue
                wt = self.idf.get(w, self.default_idf)
                v = self.mat[i].astype(np.float32) * wt
                acc = v if acc is None else acc + v
                wsum += wt
            if acc is not None:
                out[j] = acc / (np.linalg.norm(acc) + 1e-8)
        return out

    def soft_coverage(self, a: list[str], b: list[str]) -> float:
        """Mean over words of `a` of their best cosine to a word of `b` (exact match counts 1)."""
        if not a or not b:
            return 0.0
        bv = [(w, self.vec(w)) for w in b]
        tot = 0.0
        for w in a:
            if w in b:
                tot += 1.0
                continue
            v = self.vec(w)
            if v is None:
                continue
            tot += max((float(v @ u) for _, u in bv if u is not None), default=0.0)
        return tot / len(a)


@lru_cache(maxsize=1)
def word_vectors() -> WordVectors:
    from . import data
    wv = WordVectors()
    texts = data.snomed_index_rows().description.tolist() + data.icd_index_rows().description.tolist()
    return wv.fit_idf(texts)


class DenseIndex:
    def __init__(self, texts: list[str], codes: list[str]):
        self.texts = np.array([basic(t) for t in texts], dtype=object)
        self.codes = np.array(codes, dtype=object)
        self.mat = word_vectors().embed(list(self.texts))

    def search(self, queries: list[str], k: int = 30, batch: int = 128) -> list[list[Hit]]:
        out = []
        for s in range(0, len(queries), batch):
            sims = word_vectors().embed(queries[s:s + batch]) @ self.mat.T
            for row in sims:
                if not row.any():
                    out.append([])
                    continue
                take = min(len(row), k * 6)
                idx = np.argpartition(-row, take - 1)[:take]
                idx = idx[np.argsort(-row[idx], kind="stable")]
                hits, seen = [], set()
                for d in idx:
                    c = self.codes[d]
                    if c in seen:
                        continue
                    seen.add(c)
                    hits.append(Hit(c, float(row[d]), len(hits), self.texts[d]))
                    if len(hits) >= k:
                        break
                out.append(hits)
        return out
