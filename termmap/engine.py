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

from . import config, data, llm
from .normalize import (ABBREVIATIONS, basic, context_window, expansions, icd_codes_in_text, is_history,
                        is_negated, laterality, polarity)
import os

from .embed import SAPBERT_SNOMED_TAGS, DenseIndex, SapbertIndex, sapbert_available, word_vectors
from .rank import DERIVED, Ranker, derive, softmax_by_group
from .retrieve import DictionaryRetriever, LexicalIndex

K_DICT = 30
K_MEM = 15
N_FOLDS = 5

_NEG_WORDS = {"no", "absent", "negative", "without", "not"}
_HIST_WORDS = {"history", "h", "o", "past"}
_SN_TAGS = ["disorder", "finding", "morphologic abnormality", "situation", "observable entity", "procedure",
            "body structure", "qualifier value", "event", "regime/therapy"]


# --------------------------------------------------------------------------- dictionary side

def use_sapbert() -> bool:
    """SapBERT retrieval is on when torch/transformers are installed (TERMMAP_SAPBERT=0 disables it)."""
    return os.environ.get("TERMMAP_SAPBERT", "1") == "1" and sapbert_available()


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
        self.ext_pairs: set[tuple[str, str]] = set()
        for ext in sorted(config.EXTERNAL_DIR.glob(f"{system}_aliases*.parquet")):
            al = pd.read_parquet(ext)
            al = al[al.code.isin(self.valid)]
            texts += al.description.tolist()
            codes += al.code.tolist()
            self.ext_pairs |= {(c, basic(t)) for c, t in zip(al.code, al.description)}
        self.retriever = DictionaryRetriever(texts, codes, dense_factory=DenseIndex)
        self.attach_sapbert()

    def attach_sapbert(self) -> bool:
        """Add the SapBERT index to the retriever (also to a cached CodeSystem built before torch was installed)."""
        r = self.retriever
        if not use_sapbert() or getattr(r, "sap", None) is not None:
            return False
        keep = [i for i, c in enumerate(r.char.codes) if self.system == config.ICD or self.tag.get(c) in SAPBERT_SNOMED_TAGS]
        r.sap = SapbertIndex([r.char.texts[i] for i in keep], [r.char.codes[i] for i in keep], name=self.system)
        r._cache.clear()
        return True

    @classmethod
    def load(cls, system: str) -> "CodeSystem":
        path = config.CACHE_DIR / f"codesystem_{system}.joblib"
        if path.exists():
            cs = joblib.load(path)
            if cs.attach_sapbert():
                joblib.dump(cs, path, compress=0)
            return cs
        cs = cls(system)
        cs.retriever._cache.clear()
        joblib.dump(cs, path, compress=0)
        return cs


# --------------------------------------------------------------------------- training memory

class Memory:
    """What the training labels teach: aliases (entity strings and gold displays) -> code, code priors."""

    def __init__(self, rows: pd.DataFrame, system: str):
        other = rows[(rows.coding_system != system) & (rows.subconcept == "DiagnosisCode")]
        rows = rows[rows.coding_system == system]
        # The other system's gold display for the same mention is one more way coders worded it.
        paired = rows.merge(other[["entity", "chunk_text", "display"]], on=["entity", "chunk_text"],
                            suffixes=("", "_other"))
        texts = ([basic(t) for t in rows.entity] + [basic(t) for t in rows.display]
                 + [basic(t) for t in paired.display_other])
        codes = list(rows.code) + list(rows.code) + list(paired.code)
        self.exact: dict[str, Counter] = defaultdict(Counter)
        for t, c in zip(texts, codes):
            self.exact[t][c] += 1
        self.index = LexicalIndex("char").fit(texts, codes) if texts else None
        self.dense = DenseIndex(texts, codes) if texts else None
        self.sap = SapbertIndex(texts, codes) if texts and use_sapbert() else None
        diag = rows[rows.subconcept == "DiagnosisCode"]
        self.freq = Counter(diag.code)
        self.cat_freq = Counter(c[:3] for c in diag.code) if system == config.ICD else Counter()
        self.n = max(len(diag), 1)
        self.diag_codes = Counter(diag.code)
        self._unspec_rate: dict[str, float] | None = None

    def unspec_rate(self, cs: "CodeSystem") -> dict[str, float]:
        """Per ICD category: share of training golds that are an 'unspecified' code (coder convention)."""
        if self._unspec_rate is None:
            tot, uns = Counter(), Counter()
            for code, n in self.diag_codes.items():
                tot[code[:3]] += n
                uns[code[:3]] += n * (code in cs.unspecified)
            self._unspec_rate = {c: uns[c] / tot[c] for c in tot}
        return self._unspec_rate

    def search(self, queries: list[str], k: int = K_MEM):
        if self.index is None:
            return [([], [], []) for _ in queries]
        sap = self.sap.search(queries, k) if getattr(self, "sap", None) is not None else [[] for _ in queries]
        return list(zip(self.index.search(queries, k), self.dense.search(queries, k), sap))


# --------------------------------------------------------------------------- features

FEATURES = [
    "d_char", "d_char_rank", "d_word", "d_word_rank", "d_exact", "m_char", "m_rank", "m_votes", "m_exact",
    "from_expansion", "code_in_text", "hier", "freq", "cat_freq", "is_leaf", "unspec", "code_len", "n_desc",
    "jacc", "q_cov", "d_cov", "tsr", "ratio", "lat_match", "lat_extra", "lat_missing", "ctx_overlap",
    "q_neg", "d_neg", "q_hist", "d_hist", "q_ntok", "q_abbrev", "rank_in_group", "score_gap",
    "d_dense", "d_dense_rank", "m_dense", "soft_q", "soft_d", "ctx_dense", "head_match", "cat_unspec_rate",
    "neg_query", "ext_alias", "d_sap", "d_sap_rank", "m_sap", "m_sap_votes",
    "drift_parent", "polarity_conflict", "polarity_match",
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
    neg_query: int = 0
    ext_alias: int = 0
    drift_parent: int = 0
    d_sap: float = 0.0
    d_sap_rank: int = 99
    m_sap: float = 0.0
    m_sap_votes: float = 0.0
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

    for qi, (q, (ch, wd, de, sp), (mh, mdh, msp)) in enumerate(zip(queries, dict_hits, mem_hits)):
        orig = int(qi > 0)
        negq = int(q.startswith("no ") or q.endswith(" absent"))
        for h in ch[:10] if negq else ():
            get(h.code).neg_query = 1
        for h in de[:10] if negq else ():
            get(h.code).neg_query = 1
        for h in ch:
            c = get(h.code)
            if (h.code, h.text) in cs.ext_pairs:
                c.ext_alias = max(c.ext_alias, 1)
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
        for h in sp:
            c = get(h.code)
            c.d_sap = max(c.d_sap, h.score)
            c.d_sap_rank = min(c.d_sap_rank, h.rank)
            c.from_expansion = min(c.from_expansion, orig)
            if not c.text:
                c.text = h.text
        for h in msp[:10]:
            c = get(h.code)
            c.m_sap = max(c.m_sap, h.score)
            c.m_sap_votes += h.score
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
        # Specificity calibration: offer the "unspecified" leaves of the top categories, and the
        # initial-encounter (7th character A) variant of injury codes retrieved as D/S encounters.
        top = sorted(cands.values(), key=lambda c: -(c.d_char + c.m_char + c.d_dense))[:5]
        for c in top:
            for sib in cs.unspec_by_cat.get(c.code[:3], [])[:4]:
                get(sib).hier = 1
        # Version drift: gold labels sometimes use a parent code that a recent release split into
        # children (M51.36 -> M51.360/...). Offer the non-leaf parent of the top candidates.
        for c in top[:3]:
            parent = c.code[:-1].rstrip(".")
            if len(parent) >= 4 and parent in cs.valid and not cs.is_leaf.get(parent, True):
                get(parent).drift_parent = 1
        for code in list(cands):
            if code[0] in "ST" and len(code) == 8 and code[-1] in "DS":
                a = code[:-1] + "A"
                if a in cs.valid:
                    get(a).hier = 1
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
    head = qn.split()[-1] if qn else ""
    q_negw, q_posw = polarity(entity)
    unspec_rate = mem.unspec_rate(cs) if cs.system == config.ICD else {}
    rows = []
    for ci, c in enumerate(clist):
        dt = set(c.text.split())
        inter = len(q_exp & dt)
        d_lat = laterality(c.text)
        extra = dt - qt
        tag = cs.tag.get(c.code, "")
        dl = c.text.split()
        d_negw, d_posw = polarity(c.text)
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
            "head_match": int(head in dt),
            "cat_unspec_rate": unspec_rate.get(c.code[:3], -1.0),
            "neg_query": c.neg_query, "ext_alias": c.ext_alias, "d_sap": c.d_sap, "d_sap_rank": c.d_sap_rank,
            "m_sap": c.m_sap, "m_sap_votes": c.m_sap_votes, "drift_parent": c.drift_parent,
            # Opposite meaning: the term negates a word the candidate asserts, or vice versa.
            "polarity_conflict": len(q_negw & d_posw) + len(d_negw & q_posw),
            "polarity_match": len(q_negw & d_negw),
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
    # Negated mentions also query the negative concept ("no nausea", "nausea absent"); SNOMED encodes
    # some negations as their own concepts, the ranker decides with the q_neg/d_neg features.
    queries = [expansions(e) + ([f"no {basic(e)}", f"{basic(e)} absent"] if is_negated(e, c) else [])
               for e, c in zip(items.entity, items.chunk_text)]
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
        self.ranker: dict[str, Ranker] = {}

    @staticmethod
    def _folds(train: pd.DataFrame, n: int = N_FOLDS) -> np.ndarray:
        # Deterministic group-by-chunk folds (python's str hash is salted, so use a stable digest).
        return train.chunk_text.astype(str).map(
            lambda t: int(hashlib.md5(t.encode()).hexdigest(), 16) % n).to_numpy()

    def oof_features(self, train: pd.DataFrame, system: str) -> pd.DataFrame:
        """Candidate features for every training diagnosis row, with the training memory built from the
        other folds only, so the ranker learns how memory behaves on unseen terms."""
        folds = self._folds(train)
        target = (train.coding_system == system) & (train.subconcept == "DiagnosisCode")
        feats = []
        for f in range(N_FOLDS):
            items = train[target & (folds == f)].reset_index(drop=True)
            ff = build_features(items, self.systems[system], Memory(train[folds != f], system))
            ff["item"] += f * 1_000_000
            feats.append(ff)
        return pd.concat(feats, ignore_index=True)

    def fit(self, train: pd.DataFrame, verbose: bool = True) -> "TermMapEngine":
        for system in self.systems:
            X = derive(self.oof_features(train, system))
            if verbose:
                rec = X.groupby("item").label.max()
                print(f"[{system}] OOF candidate recall: {rec.mean():.3f} ({len(rec)} items, {len(X)} rows)")
            self.ranker[system] = Ranker(FEATURES + DERIVED).fit(X)
            self.memory[system] = Memory(train, system)
        return self

    def rank(self, items: pd.DataFrame, system: str, use_llm: bool = False,
             llm_threshold: float = 0.6, llm_k: int = 8) -> pd.DataFrame:
        X = derive(build_features(items, self.systems[system], self.memory[system]))
        if not len(X):
            return X
        X["prob"] = softmax_by_group(X, self.ranker[system].score(X))
        X = X.sort_values(["item", "prob", "code"], ascending=[True, False, True])
        X["llm"] = 0
        if use_llm and llm.enabled():
            X = self._adjudicate(items, X, system, llm_threshold, llm_k)
        return X

    def _adjudicate(self, items, X, system, threshold, k) -> pd.DataFrame:
        """Send low-confidence terms with their top-k candidates to the LLM; its pick moves to the top."""
        cs = self.systems[system]
        top = X.groupby("item").head(k)
        low = top.groupby("item").prob.max()
        low = low[low < threshold].index.tolist()
        tasks = []
        for i in low:
            g = top[top.item == i]
            cands = [(c, cs.pref.get(c, "") + (f" ({cs.tag[c]})" if cs.tag.get(c) else "")
                      + (f" [{c[:3]}: {cs.pref[c[:3]]}]" if system == config.ICD and c[:3] in cs.pref and c != c[:3] else ""))
                     for c in g.code]
            tasks.append({"entity": items.entity.iloc[i], "system": system,
                          "context": context_window(items.entity.iloc[i], items.chunk_text.iloc[i], 150),
                          "candidates": cands})
        picks = llm.adjudicate(tasks)
        for i, code in zip(low, picks):
            if code:
                m = (X.item == i) & (X.code == code)
                X.loc[m, "prob"] = X.loc[m, "prob"] + 1.0
                X.loc[m, "llm"] = 1
        return X.sort_values(["item", "prob", "code"], ascending=[True, False, True])

    def predict(self, items: pd.DataFrame, top_n: int = 5, use_llm: bool | None = None) -> pd.DataFrame:
        use_llm = llm.enabled() if use_llm is None else use_llm
        items = items.reset_index(drop=True)
        out = items[[c for c in items.columns if c in ("chunk_id", "entity", "clinical_entity", "chunk_text")]].copy()
        for system in config.SYSTEMS:
            col = "icd10cm" if system == config.ICD else "snomed"
            R = self.rank(items, system, use_llm=use_llm)
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
