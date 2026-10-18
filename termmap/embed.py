"""Dense signal from biomedical word vectors (scispacy en_core_sci_lg 0.5.4, CC BY-SA 3.0).

Texts are embedded as the IDF-weighted mean of their word vectors. Weak next to a sentence encoder like
SapBERT, but it knows that "nose"~"nasal" and "discomfort"~"pain", which character n-grams cannot.
"""
from __future__ import annotations

import glob
import json
import os
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


# --------------------------------------------------------------------------- SapBERT (optional)

SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"  # Apache-2.0


def sapbert_model() -> str:
    """Encoder in use: TERMMAP_SAPBERT_MODEL (a fine-tuned copy, e.g. data/cache/sapbert_ft_syn) or the public
    SapBERT. Public is the default: the synonyms-only fine-tune was neutral in 5-fold CV (docs/experiments.md #9)."""
    env = os.environ.get("TERMMAP_SAPBERT_MODEL", "").strip()
    return SAPBERT_MODEL if env in ("", "base") else env


def sapbert_tag() -> str:
    """Suffix for cached dictionary vectors, so each encoder gets its own cache."""
    m = sapbert_model()
    return "" if m == SAPBERT_MODEL else "_" + os.path.basename(os.path.normpath(m))
# SNOMED tags covering ~95% of the gold diagnosis concepts; the rest stay lexical-only to save encode time.
SAPBERT_SNOMED_TAGS = {"disorder", "finding", "morphologic abnormality", "situation", "observable entity", "event"}


def sapbert_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


class SapbertEncoder:
    def __init__(self, model: str | None = None, max_length: int = 32):
        model = model or sapbert_model()
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.manual_seed(config.SEED)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).eval()
        self.max_length = max_length
        self.cache: dict[str, np.ndarray] = {}
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def encode_cached(self, texts: list[str]) -> np.ndarray:
        """Encode with an in-process cache (queries and memory aliases recur across folds)."""
        todo = sorted({t for t in texts if t not in self.cache})
        if todo:
            for t, v in zip(todo, self.encode(todo)):
                self.cache[t] = v
        return np.stack([self.cache[t] for t in texts]) if texts else np.zeros((0, 768), dtype=np.float16)

    def encode(self, texts: list[str], batch: int = 256, log_every: int = 0) -> np.ndarray:
        if not texts:
            return np.zeros((0, 768), dtype=np.float16)
        # Batch texts of similar length together (far less padding), then restore the input order.
        order = sorted(range(len(texts)), key=lambda i: (len(texts[i]), texts[i]))
        out = np.zeros((len(texts), 768), dtype=np.float16)
        with self.torch.inference_mode():
            for b, s in enumerate(range(0, len(order), batch)):
                idx = order[s:s + batch]
                enc = self.tok([texts[i] for i in idx], padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt").to(self.model.device)
                cls = self.model(**enc).last_hidden_state[:, 0]
                out[idx] = self.torch.nn.functional.normalize(cls, dim=-1).cpu().numpy().astype(np.float16)
                if log_every and b % log_every == 0:
                    print(f"  sapbert {s + len(idx)}/{len(texts)}", flush=True)
        return out


@lru_cache(maxsize=1)
def sapbert() -> SapbertEncoder:
    return SapbertEncoder()


class SapbertIndex:
    """Dense retrieval with SapBERT [CLS] embeddings; vectors cached per unique text in data/cache."""

    def __init__(self, texts: list[str], codes: list[str], name: str | None = None):
        norm = [basic(t) for t in texts]
        uniq = sorted(set(norm))
        path = config.CACHE_DIR / f"sapbert_{name}.npy"
        names = config.CACHE_DIR / f"sapbert_{name}.json"
        if name is None:
            mat = sapbert().encode_cached(uniq)
        else:
            old = json.loads(names.read_text()) if path.exists() and names.exists() else []
            if old == uniq:
                mat = np.load(path)
            else:
                # Reuse vectors of texts encoded before (dictionary fixes change only a few texts).
                prev = dict(zip(old, np.load(path))) if old else {}
                todo = [t for t in uniq if t not in prev]
                print(f"encoding {len(todo)} new of {len(uniq)} {name} texts with SapBERT...", flush=True)
                new = dict(zip(todo, sapbert().encode(todo, log_every=200)))
                mat = np.stack([prev[t] if t in prev else new[t] for t in uniq]).astype(np.float16)
                np.save(path, mat)
                names.write_text(json.dumps(uniq))
        self.name = name
        self.texts = np.array(uniq, dtype=object)
        self.mat = mat.astype(np.float32)  # float32 for BLAS matmul; float16 on disk
        pos = {t: i for i, t in enumerate(uniq)}
        self.codes_of: list[list[str]] = [[] for _ in uniq]
        for t, c in zip(norm, codes):
            self.codes_of[pos[t]].append(c)

    def __getstate__(self):
        # Named indexes keep their vectors in data/cache/sapbert_<name>.npy; don't duplicate them in pickles.
        state = dict(self.__dict__)
        if self.name is not None:
            state["mat"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.mat is None:
            self.mat = np.load(config.CACHE_DIR / f"sapbert_{self.name}.npy").astype(np.float32)

    def search(self, queries: list[str], k: int = 30, batch: int = 256) -> list[list[Hit]]:
        out = []
        q = sapbert().encode_cached([basic(x) for x in queries]).astype(np.float32)
        for s in range(0, len(q), batch):
            sims = q[s:s + batch] @ self.mat.T
            for row in sims:
                take = min(len(row), k * 3)
                idx = np.argpartition(-row, take - 1)[:take]
                idx = idx[np.argsort(-row[idx], kind="stable")]
                hits, seen = [], set()
                for d in idx:
                    for c in self.codes_of[d]:
                        if c in seen:
                            continue
                        seen.add(c)
                        hits.append(Hit(c, float(row[d]), len(hits), self.texts[d]))
                    if len(hits) >= k:
                        break
                out.append(hits[:k])
        return out
