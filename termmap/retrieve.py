"""Lexical retrieval: TF-IDF over character n-grams (robust to typos/OCR) and words (exact terms)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .normalize import basic


@dataclass
class Hit:
    code: str
    score: float
    rank: int
    text: str


class LexicalIndex:
    def __init__(self, kind: str = "char", min_df: int = 1):
        if kind == "char":
            self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=min_df,
                                       sublinear_tf=True, dtype=np.float32)
        elif kind == "word":
            self.vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, token_pattern=r"[a-z0-9.+%]+",
                                       sublinear_tf=True, dtype=np.float32)
        else:
            raise ValueError(kind)
        self.kind = kind

    def fit(self, texts: list[str], codes: list[str]) -> "LexicalIndex":
        self.texts = np.array([basic(t) for t in texts], dtype=object)
        self.codes = np.array(codes, dtype=object)
        self.matrix_t = self.vec.fit_transform(self.texts).T.tocsr()  # (V, N) for fast q @ M.T
        return self

    def search(self, queries: list[str], k: int = 30, batch: int = 64) -> list[list[Hit]]:
        """Top-k distinct codes per query (a code's score is its best description's cosine)."""
        out: list[list[Hit]] = []
        for s in range(0, len(queries), batch):
            q = self.vec.transform([basic(x) for x in queries[s:s + batch]])
            sims = (q @ self.matrix_t).tocsr()
            for i in range(sims.shape[0]):
                row = sims.getrow(i)
                if row.nnz == 0:
                    out.append([])
                    continue
                take = min(row.nnz, k * 6)
                idx = np.argpartition(-row.data, take - 1)[:take]
                order = idx[np.argsort(-row.data[idx], kind="stable")]
                hits, seen = [], set()
                for j in order:
                    doc = row.indices[j]
                    code = self.codes[doc]
                    if code in seen:
                        continue
                    seen.add(code)
                    hits.append(Hit(code, float(row.data[j]), len(hits), self.texts[doc]))
                    if len(hits) >= k:
                        break
                out.append(hits)
        return out


class DictionaryRetriever:
    """Char + word indexes over one code system's dictionary descriptions."""

    def __init__(self, texts: list[str], codes: list[str]):
        self.char = LexicalIndex("char", min_df=2).fit(texts, codes)
        self.word = LexicalIndex("word").fit(texts, codes)
        self.exact: dict[str, set[str]] = {}
        for t, c in zip(self.char.texts, codes):
            self.exact.setdefault(t, set()).add(c)
        self._cache: dict[tuple[str, int], tuple[list[Hit], list[Hit]]] = {}

    def search(self, queries: list[str], k: int = 30) -> list[tuple[list[Hit], list[Hit]]]:
        todo = sorted({q for q in queries if (q, k) not in self._cache})
        if todo:
            ch = self.char.search(todo, k)
            wd = self.word.search(todo, k)
            for q, a, b in zip(todo, ch, wd):
                self._cache[(q, k)] = (a, b)
        return [self._cache[(q, k)] for q in queries]
