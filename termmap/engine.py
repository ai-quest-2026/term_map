"""TermMap engine: normalize -> retrieve candidates (dictionary + training memory) -> learned rerank.

Each coding system is predicted independently with its own retrievers and ranker; the two systems
only share the input text, so neither prediction is derived from the other through a crosswalk.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config, data
from .normalize import (ABBREVIATIONS, basic, context_window, expansions, icd_codes_in_text, is_history,
                        is_negated, laterality)
from .embed import DenseIndex, word_vectors
from .retrieve import DictionaryRetriever, LexicalIndex

K_DICT = 30
K_MEM = 15
N_FOLDS = 5

_NEG_WORDS = {"no", "absent", "negative", "without", "not"}
_HIST_WORDS = {"history", "h", "o", "past"}
_SN_TAGS = ["disorder", "finding", "morphologic abnormality", "situation", "observable entity", "procedure",
            "body structure", "qualifier value", "event", "regime/therapy"]


# --------------------------------------------------------------------------- dictionary side

class CodeSystem:
    """Dictionary retriever plus per-code metadata for one coding system."""

    def __init__(self, system: str):
        self.system = system
        if system == config.ICD:
            df = data.icd_index_rows()
            self.is_leaf = dict(zip(df.code, df.is_leaf))
            unspec = df[df.description.str.lower().str.contains("unspecified") & df.is_leaf]
            self.unspec_by_cat: dict[str, list[str]] = unspec.groupby("category").code.agg(
                lambda s: sorted(set(s))).to_dict()
            self.valid = set(df.code)
            self.tag = {}
        else:
            df = data.snomed_index_rows()
            self.is_leaf = {}
            self.unspec_by_cat = {}
            self.valid = set(data.load_snomed().code)
            self.tag = dict(zip(df.code, df.tag))
        self.n_desc = df.code.value_counts().to_dict()
        self.unspecified = set(df.loc[df.description.str.lower().str.contains("unspecified"), "code"])
        self.pref = data.preferred_terms()[system]
        texts, codes = df.description.tolist(), df.code.tolist()
        # Optional aliases ingested from public sources (scripts/ingest.py); restricted to the
        # supplied dictionary so predictions stay inside the legal answer space.
        ext = config.EXTERNAL_DIR / f"{system}_aliases.parquet"
        if ext.exists():
            al = pd.read_parquet(ext)
            al = al[al.code.isin(self.valid)]
            texts += al.description.tolist()
            codes += al.code.tolist()
        self.retriever = DictionaryRetriever(texts, codes, dense_factory=DenseIndex)

    @classmethod
    def load(cls, system: str) -> "CodeSystem":
        path = config.CACHE_DIR / f"codesystem_{system}.joblib"
        if path.exists():
            return joblib.load(path)
        cs = cls(system)
        cs.retriever._cache.clear()
        joblib.dump(cs, path, compress=0)
        return cs


# --------------------------------------------------------------------------- training memory

class Memory:
    """What the training labels teach: aliases (entity strings and gold displays) -> code, code priors."""

    def __init__(self, rows: pd.DataFrame, system: str):
        rows = rows[rows.coding_system == system]
        texts = [basic(t) for t in rows.entity] + [basic(t) for t in rows.display]
        codes = list(rows.code) + list(rows.code)
        self.exact: dict[str, Counter] = defaultdict(Counter)
        for t, c in zip(texts, codes):
            self.exact[t][c] += 1
        self.index = LexicalIndex("char").fit(texts, codes) if texts else None
        self.dense = DenseIndex(texts, codes) if texts else None
        diag = rows[rows.subconcept == "DiagnosisCode"]
        self.freq = Counter(diag.code)
        self.cat_freq = Counter(c[:3] for c in diag.code) if system == config.ICD else Counter()
        self.n = max(len(diag), 1)

    def search(self, queries: list[str], k: int = K_MEM):
        if self.index is None:
            return [([], []) for _ in queries]
        return list(zip(self.index.search(queries, k), self.dense.search(queries, k)))


# --------------------------------------------------------------------------- features

FEATURES = [
    "d_char", "d_char_rank", "d_word", "d_word_rank", "d_exact", "m_char", "m_rank", "m_votes", "m_exact",
    "from_expansion", "code_in_text", "hier", "freq", "cat_freq", "is_leaf", "unspec", "code_len", "n_desc",
    "jacc", "q_cov", "d_cov", "tsr", "ratio", "lat_match", "lat_extra", "lat_missing", "ctx_overlap",
    "q_neg", "d_neg", "q_hist", "d_hist", "q_ntok", "q_abbrev", "rank_in_group", "score_gap",
    "d_dense", "d_dense_rank", "m_dense", "soft_q", "soft_d", "ctx_dense",
] + [f"tag_{t}" for t in _SN_TAGS]


@dataclass
class Cand:
    code: str
    d_char: float = 0.0
    d_char_rank: int = 99
    d_word: float = 0.0
    d_word_rank: int = 99
    d_exact: int = 0
    m_char: float = 0.0
    m_rank: int = 99
    m_votes: float = 0.0
    m_exact: int = 0
    d_dense: float = 0.0
    d_dense_rank: int = 99
    m_dense: float = 0.0
    from_expansion: int = 1
    code_in_text: int = 0
    hier: int = 0
    text: str = ""
    sources: set = field(default_factory=set)


def _gather(entity: str, chunk: str, cs: CodeSystem, mem: Memory, dict_hits, mem_hits, queries) -> dict[str, Cand]:
    cands: dict[str, Cand] = {}

    def get(code):
        if code not in cands:
            cands[code] = Cand(code)
        return cands[code]

    for qi, (q, (ch, wd, de), (mh, mdh)) in enumerate(zip(queries, dict_hits, mem_hits)):
        orig = int(qi > 0)
        for h in ch:
            c = get(h.code)
            if h.score > c.d_char:
                c.d_char, c.text = h.score, h.text
            c.d_char_rank = min(c.d_char_rank, h.rank)
            c.from_expansion = min(c.from_expansion, orig)
            c.sources.add("dict")
        for h in wd:
            c = get(h.code)
            c.d_word = max(c.d_word, h.score)
            c.d_word_rank = min(c.d_word_rank, h.rank)
            c.from_expansion = min(c.from_expansion, orig)
            if not c.text:
                c.text = h.text
            c.sources.add("dict")
        for h in de:
            c = get(h.code)
            c.d_dense = max(c.d_dense, h.score)
            c.d_dense_rank = min(c.d_dense_rank, h.rank)
            c.from_expansion = min(c.from_expansion, orig)
            if not c.text:
                c.text = h.text
        for h in mdh[:5]:
            c = get(h.code)
            c.m_dense = max(c.m_dense, h.score)
        for code in cs.retriever.exact.get(q, ()):
            get(code).d_exact = 1
            if not cands[code].text:
                cands[code].text = q
        for h in mh:
            c = get(h.code)
            c.m_char = max(c.m_char, h.score)
            c.m_rank = min(c.m_rank, h.rank)
            c.m_votes += h.score if h.rank < 10 else 0.0
            c.from_expansion = min(c.from_expansion, orig)
            c.sources.add("mem")
        for code, n in mem.exact.get(q, {}).items():
            get(code).m_exact += n

    if cs.system == config.ICD:
        for code in icd_codes_in_text(entity):
            if code in cs.valid:
                get(code).code_in_text = 1
        # Specificity calibration: offer the "unspecified" leaves of the top categories.
        top = sorted(cands.values(), key=lambda c: -(c.d_char + c.m_char))[:5]
        for c in top:
            for sib in cs.unspec_by_cat.get(c.code[:3], [])[:4]:
                get(sib).hier = 1
    for c in cands.values():
        if not c.text:
            c.text = basic(cs.pref.get(c.code, ""))
    return cands


def _featurize(entity: str, chunk: str, cands: dict[str, Cand], cs: CodeSystem, mem: Memory) -> pd.DataFrame:
    qn = basic(entity)
    qt = set(qn.split())
    q_exp = set(" ".join(expansions(entity)).split())
    # Context words not already in the entity: lets "ms" + "mitral valve" favour mitral stenosis.
    ctx = set(basic(context_window(entity, chunk)).split()) - qt
    q_lat = laterality(entity)
    q_neg = int(is_negated(entity, chunk))
    q_hist = int(is_history(entity, chunk))
    q_abbrev = int(qn in ABBREVIATIONS or (len(qn) <= 5 and " " not in qn))
    wv = word_vectors()
    exp_toks = [q.split() for q in expansions(entity)]
    clist = list(cands.values())
    ctx_vec = wv.embed([" ".join(sorted(ctx))])[0]
    ctx_sim = wv.embed([c.text for c in clist]) @ ctx_vec if clist else []
    rows = []
    for ci, c in enumerate(clist):
        dt = set(c.text.split())
        inter = len(q_exp & dt)
        d_lat = laterality(c.text)
        extra = dt - qt
        tag = cs.tag.get(c.code, "")
        dl = c.text.split()
        rows.append({
            "code": c.code, "d_char": c.d_char, "d_char_rank": c.d_char_rank, "d_word": c.d_word,
            "d_word_rank": c.d_word_rank, "d_exact": c.d_exact, "m_char": c.m_char, "m_rank": c.m_rank,
            "m_votes": c.m_votes, "m_exact": c.m_exact, "from_expansion": c.from_expansion,
            "code_in_text": c.code_in_text, "hier": c.hier,
            "freq": math.log1p(mem.freq.get(c.code, 0)), "cat_freq": math.log1p(mem.cat_freq.get(c.code[:3], 0)),
            "is_leaf": int(cs.is_leaf.get(c.code, True)), "unspec": int(c.code in cs.unspecified),
            "code_len": len(c.code), "n_desc": math.log1p(cs.n_desc.get(c.code, 0)),
            "jacc": inter / max(len(q_exp | dt), 1), "q_cov": len(qt & dt) / max(len(qt), 1),
            "d_cov": inter / max(len(dt), 1), "tsr": fuzz.token_set_ratio(qn, c.text) / 100,
            "ratio": fuzz.ratio(qn, c.text) / 100,
            "lat_match": int(bool(q_lat) and q_lat == d_lat), "lat_extra": int(bool(d_lat) and not q_lat),
            "lat_missing": int(bool(q_lat) and not d_lat),
            "ctx_overlap": len(extra & ctx) / max(len(extra), 1) if extra else 1.0,
            "q_neg": q_neg, "d_neg": int(bool(dt & _NEG_WORDS)), "q_hist": q_hist, "d_hist": int(bool(dt & _HIST_WORDS)),
            "q_ntok": len(qt), "q_abbrev": q_abbrev,
            "d_dense": c.d_dense, "d_dense_rank": c.d_dense_rank, "m_dense": c.m_dense,
            "soft_q": max(wv.soft_coverage(e, dl) for e in exp_toks),
            "soft_d": max(wv.soft_coverage(dl, e) for e in exp_toks),
            "ctx_dense": float(ctx_sim[ci]),
            **{f"tag_{t}": int(tag == t) for t in _SN_TAGS},
        })
    df = pd.DataFrame(rows)
    if len(df):
        s = df.d_char + df.m_char + df.d_word
        df["rank_in_group"] = s.rank(ascending=False, method="first")
        df["score_gap"] = s.max() - s
    return df


def build_features(items: pd.DataFrame, cs: CodeSystem, mem: Memory) -> pd.DataFrame:
    """items: DataFrame with entity, chunk_text (and optional gold `code`). Returns one row per candidate."""
    queries = [expansions(e) for e in items.entity]
    flat = sorted({q for qs in queries for q in qs})
    dres = dict(zip(flat, cs.retriever.search(flat, K_DICT)))
    mres = dict(zip(flat, mem.search(flat, K_MEM)))
    parts = []
    for i, (ent, chunk, qs) in enumerate(zip(items.entity, items.chunk_text, queries)):
        cands = _gather(ent, chunk, cs, mem, [dres[q] for q in qs], [mres[q] for q in qs], qs)
        f = _featurize(ent, chunk, cands, cs, mem)
        f.insert(0, "item", i)
        parts.append(f)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["item", "code"] + FEATURES)
    if "code" in items.columns:
        gold = items.code.reset_index(drop=True)
        out["label"] = (out.code.values == gold.values[out.item.values]).astype(int)
    return out


# --------------------------------------------------------------------------- engine

class TermMapEngine:
    def __init__(self):
        self.systems = {s: CodeSystem.load(s) for s in config.SYSTEMS}
        self.memory: dict[str, Memory] = {}
        self.ranker: dict[str, HistGradientBoostingClassifier] = {}

    @staticmethod
    def _folds(train: pd.DataFrame, n: int = N_FOLDS) -> np.ndarray:
        # Deterministic group-by-chunk folds (python's str hash is salted, so use a stable digest).
        return train.chunk_text.astype(str).map(
            lambda t: int(hashlib.md5(t.encode()).hexdigest(), 16) % n).to_numpy()

    def fit(self, train: pd.DataFrame, verbose: bool = True) -> "TermMapEngine":
        folds = self._folds(train)
        for system, cs in self.systems.items():
            target = (train.coding_system == system) & (train.subconcept == "DiagnosisCode")
            feats = []
            for f in range(N_FOLDS):
                mem = Memory(train[folds != f], system)
                items = train[target & (folds == f)].reset_index(drop=True)
                ff = build_features(items, cs, mem)
                ff["item"] += f * 1_000_000
                feats.append(ff)
            X = pd.concat(feats, ignore_index=True)
            groups_with_gold = X.groupby("item").label.max()
            if verbose:
                print(f"[{system}] OOF candidate recall: {groups_with_gold.mean():.3f} "
                      f"({len(groups_with_gold)} items, {len(X)} candidate rows)")
            model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                                   l2_regularization=1.0, early_stopping=False,
                                                   random_state=config.SEED)
            model.fit(X[FEATURES], X.label)
            self.ranker[system] = model
            self.memory[system] = Memory(train, system)
        return self

    def rank(self, items: pd.DataFrame, system: str) -> pd.DataFrame:
        X = build_features(items, self.systems[system], self.memory[system])
        if not len(X):
            return X
        X["prob"] = self.ranker[system].predict_proba(X[FEATURES])[:, 1]
        return X.sort_values(["item", "prob", "code"], ascending=[True, False, True])

    def predict(self, items: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        items = items.reset_index(drop=True)
        out = items[[c for c in items.columns if c in ("chunk_id", "entity", "clinical_entity", "chunk_text")]].copy()
        for system in config.SYSTEMS:
            col = "icd10cm" if system == config.ICD else "snomed"
            R = self.rank(items, system)
            pref = self.systems[system].pref
            best, conf, margin, alts = [], [], [], []
            groups = dict(tuple(R.groupby("item"))) if len(R) else {}
            for i in range(len(items)):
                g = groups.get(i)
                if g is None or not len(g):
                    best.append(""); conf.append(0.0); margin.append(0.0); alts.append("")
                    continue
                p = g.prob.to_numpy()
                best.append(g.code.iloc[0])
                conf.append(round(float(p[0]), 4))
                margin.append(round(float(p[0] - (p[1] if len(p) > 1 else 0.0)), 4))
                alts.append(" | ".join(f"{c}:{pref.get(c, '')}:{q:.3f}" for c, q in
                                       zip(g.code.iloc[:top_n], g.prob.iloc[:top_n])))
            out[f"{col}_code"] = best
            out[f"{col}_display"] = [pref.get(c, "") for c in best]
            out[f"{col}_confidence"] = conf
            out[f"{col}_margin"] = margin
            out[f"{col}_top{top_n}"] = alts
        return out

    def save(self, path=None):
        path = path or config.CACHE_DIR / "engine.joblib"
        joblib.dump({"memory": self.memory, "ranker": self.ranker}, path)

    @classmethod
    def load(cls, path=None) -> "TermMapEngine":
        eng = cls()
        state = joblib.load(path or config.CACHE_DIR / "engine.joblib")
        eng.memory, eng.ranker = state["memory"], state["ranker"]
        return eng
