"""TermMap engine: normalize -> retrieve candidates (dictionary + training memory) -> learned rerank.

Each coding system is predicted independently with its own retrievers and ranker; the two systems
only share the input text, so neither prediction is derived from the other through a crosswalk.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from . import config, data, llm
from .normalize import (ABBREVIATIONS, basic, context_window, expansions, icd_codes_in_text, is_history,
                        is_negated, laterality)
import os

from .embed import SAPBERT_SNOMED_TAGS, DenseIndex, SapbertIndex, sapbert_available, sapbert_tag, word_vectors
from .rank import DERIVED, Ranker, derive, softmax_by_group
from .retrieve import DictionaryRetriever, LexicalIndex

K_DICT = 30
K_MEM = 15
N_FOLDS = 5

_NEG_WORDS = {"no", "absent", "negative", "without", "not"}
_HIST_WORDS = {"history", "h", "o", "past"}
# Most general billable codes under an ICD prefix: what a coder picks when the text supports nothing more.
_GENERAL_RE = re.compile(r"unspecified|\bnos\b|without complication|^other\b", re.I)
_SN_TAGS = ["disorder", "finding", "morphologic abnormality", "situation", "observable entity", "procedure",
            "body structure", "qualifier value", "event", "regime/therapy"]


# --------------------------------------------------------------------------- dictionary side

CE_K = int(os.environ.get("TERMMAP_CE_K", "30"))              # candidates the cross-encoder rescores
CE_EPOCHS = int(os.environ.get("TERMMAP_CE_EPOCHS", "4"))      # best epoch kept by inner-dev early stopping
# Ranker capacity per system. ICD: 600 deep trees overfit (in-sample top-1 0.86 vs OOF 0.57); the regularised
# ranker is +0.9 (split) / +1.5 (CV fold 0) points. SNOMED: the same regularisation cost 1.9 points on fold 0
# (split: +0.4), so SNOMED keeps the original settings (docs/experiments.md, v2 ablations).
RANKER_PARAMS = {
    config.ICD: dict(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=50,
                     colsample_bytree=0.5, reg_lambda=10.0, subsample=0.8, subsample_freq=1),
    config.SNOMED: {},
}
# Second-level features over each term's top-K: ranker and cross-encoder evidence (and the LLM selector's).
STACK_FEATURES = ["lp_z", "ce_z", "lp_gap", "ce_gap", "rk_rank", "ce_rank", "lp", "sel_p", "sel_rank"]
CE_WEIGHT = float(os.environ.get("TERMMAP_CE_WEIGHT", "0.3"))  # chosen on the split: 0.3 best for both systems
LLM_GATE = float(os.environ.get("TERMMAP_LLM_GATE", "0.3"))    # LLM may only pick candidates with prob >= gate x top


def ce_enabled() -> bool:
    """Cross-encoder reranking: TERMMAP_CE=1/0; default on only when a GPU is available (CPU training is slow)."""
    flag = os.environ.get("TERMMAP_CE", "auto")
    if flag != "auto":
        return flag == "1"
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


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

    def synonyms(self, code: str, n: int = 2, exclude: str = "") -> list[str]:
        """Up to n other dictionary descriptions of a code (no abbreviated CMS short forms)."""
        syn = getattr(self, "_syn", None)
        if syn is None:
            if self.system == config.ICD:
                df = data.icd_index_rows()
                df = df[~df.description.str.contains(data._ICD_SHORT_RE)]
            else:
                df = data.snomed_index_rows().sort_values(["code", "order"], kind="stable")
            syn = df.groupby("code", sort=False).description.agg(list).to_dict()
            self._syn = syn
        seen, out = {basic(exclude)}, []
        for d in syn.get(code, []):
            b = basic(d)
            if b not in seen:
                seen.add(b)
                out.append(d)
                if len(out) == n:
                    break
        return out

    def general_codes(self, prefix: str) -> list[str]:
        """Most general leaves under an ICD prefix (category "K76" or subcategory "I71.0"), shortest code first:
        wording "unspecified" / "NOS" / "without complications" / "Other ...", or a .9 / .x9 code."""
        gen = getattr(self, "_general", None)
        if gen is None:
            df = data.icd_index_rows()
            leaves = df[df.is_leaf]
            ok = leaves.description.str.contains(_GENERAL_RE) | (leaves.code.str[-1].eq("9") & leaves.code.str.len().le(6))
            gen = defaultdict(list)
            for c in sorted(set(leaves.code[ok]), key=lambda c: (len(c), c)):
                gen[c[:3]].append(c)
                if len(c) > 5:
                    gen[c[:5]].append(c)
            self._general = gen
        return gen.get(prefix, [])

    def attach_sapbert(self) -> bool:
        """Add the SapBERT index to the retriever (also to a cached CodeSystem built before torch was installed)."""
        r = self.retriever
        name = self.system + sapbert_tag()
        if not use_sapbert() or getattr(getattr(r, "sap", None), "name", None) == name:
            return False
        keep = [i for i, c in enumerate(r.char.codes) if self.system == config.ICD or self.tag.get(c) in SAPBERT_SNOMED_TAGS]
        r.sap = SapbertIndex([r.char.texts[i] for i in keep], [r.char.codes[i] for i in keep], name=name)
        r._cache.clear()
        return True

    @classmethod
    def load(cls, system: str) -> "CodeSystem":
        path = config.CACHE_DIR / f"codesystem_{system}_{config.CACHE_VERSION}.joblib"
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
        # S0: a labelled term seen again verbatim (after normalisation) gets its gold code, when unambiguous.
        seen: dict[str, Counter] = defaultdict(Counter)
        for e, c in zip(diag.entity, diag.code):
            seen[expansions(e)[0]][c] += 1
        self.lookup = {k: next(iter(v)) for k, v in seen.items() if len(v) == 1}
        self._unspec_rate: dict[str, float] | None = None
        # Labelled diagnosis terms, used as few-shot examples of local coding convention for the LLM.
        self._ex = (list(diag.entity), list(diag.code))
        self._ex_index = None

    def examples(self, queries: list[str], k: int = 5) -> list[list[tuple[str, str]]]:
        """Most similar labelled training terms (entity, gold code) per query: SapBERT if available, else char."""
        if not self._ex[0]:
            return [[] for _ in queries]
        if getattr(self, "_ex_index", None) is None:
            ents, codes = self._ex
            ids = [str(i) for i in range(len(ents))]
            self._ex_index = (SapbertIndex(ents, ids) if use_sapbert() else LexicalIndex("char").fit(ents, ids))
        ents, codes = self._ex
        hits = self._ex_index.search([basic(q) for q in queries], k)
        return [[(ents[int(h.code)], codes[int(h.code)]) for h in hs] for hs in hits]

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
    "drift_parent",
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
        # Specificity calibration: offer the general codes ("unspecified", "NOS", ".9", "Other ...") of the
        # categories and subcategories every source ranks highest, and the initial-encounter (7th character A)
        # variant of injury codes retrieved as D/S encounters.
        def best_rank(c):
            return min(c.d_char_rank, c.d_word_rank, c.d_dense_rank, c.d_sap_rank, c.m_rank)

        top = sorted(cands.values(), key=lambda c: (best_rank(c), -(c.d_char + c.m_char + c.d_dense + c.d_sap)))[:8]
        for cat in dict.fromkeys(c.code[:3] for c in top):
            for sib in cs.unspec_by_cat.get(cat, [])[:4] + cs.general_codes(cat)[:5]:
                get(sib).hier = 1
        for c in top[:5]:
            if len(c.code) > 5:
                for sib in cs.general_codes(c.code[:5])[:3]:
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
    unspec_rate = mem.unspec_rate(cs) if cs.system == config.ICD else {}
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
            # per-1000 rates: OOF memories hold 4/5 of the data, the final memory all of it
            "freq": math.log1p(1000 * mem.freq.get(c.code, 0) / mem.n),
            "cat_freq": math.log1p(1000 * mem.cat_freq.get(c.code[:3], 0) / mem.n),
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
            **{f"tag_{t}": int(tag == t) for t in _SN_TAGS},
        })
    df = pd.DataFrame(rows)
    if len(df):
        s = df.d_char + df.m_char + df.d_word + df.d_sap + df.m_sap
        df["rank_in_group"] = s.rank(ascending=False, method="first")
        df["score_gap"] = s.max() - s
    return df


def build_features(items: pd.DataFrame, cs: CodeSystem, mem: Memory) -> pd.DataFrame:
    """items: DataFrame with entity, chunk_text (and optional gold `code`). Returns one row per candidate."""
    # Negated mentions also query the negative concept ("no nausea", "nausea absent"); SNOMED encodes
    # some negations as their own concepts, the ranker decides with the q_neg/d_neg features.
    neg_ok = cs.system == config.SNOMED  # ICD-10-CM has no negative concepts
    queries = [expansions(e) + ([f"no {basic(e)}", f"{basic(e)} absent"] if neg_ok and is_negated(e, c) else [])
               for e, c in zip(items.entity, items.chunk_text)]
    flat = sorted({q for qs in queries for q in qs})
    dres = dict(zip(flat, cs.retriever.search(flat, K_DICT)))
    mres = dict(zip(flat, mem.search(flat, K_MEM)))
    parts = []
    beat = int(os.environ.get("TERMMAP_PROGRESS", "0"))
    for i, (ent, chunk, qs) in enumerate(zip(items.entity, items.chunk_text, queries)):
        if beat and i and i % beat == 0:
            print(f"    [{cs.system}] features {i}/{len(items)}", flush=True)
        cands = _gather(ent, chunk, cs, mem, [dres[q] for q in qs], [mres[q] for q in qs], qs)
        f = _featurize(ent, chunk, cands, cs, mem)
        f.insert(0, "item", i)
        parts.append(f)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["item", "code"] + FEATURES)
    if "code" in items.columns:
        gold = items.code.reset_index(drop=True)
        out["label"] = (out.code.values == gold.values[out.item.values]).astype(int)
    return out


# --------------------------------------------------------------------------- second level

def stack_frame(top: pd.DataFrame) -> pd.DataFrame:
    """Features for the second-level ranker over each term's top-k (`prob`, `ce`, optional `sel_p`)."""
    T = top.copy()
    T["lp"] = np.log(T.prob.clip(lower=1e-12))
    g = T.groupby("item", sort=False)
    z = lambda c: (T[c] - g[c].transform("mean")) / (g[c].transform("std").fillna(0) + 1e-6)
    T["lp_z"], T["ce_z"] = z("lp"), z("ce")
    T["lp_gap"], T["ce_gap"] = T.lp - g.lp.transform("max"), T.ce - g.ce.transform("max")
    T["rk_rank"] = g.lp.rank(ascending=False, method="first")
    T["ce_rank"] = g.ce.rank(ascending=False, method="first")
    if "sel_p" not in T.columns:
        T["sel_p"] = np.nan
    T["sel_rank"] = T.groupby("item", sort=False).sel_p.rank(ascending=False, method="first")
    return T


STACKER_PARAMS = dict(n_estimators=150, learning_rate=0.05, num_leaves=7, min_child_samples=30,
                      colsample_bytree=1.0, reg_lambda=5.0)


def stacker_path(system: str):
    return config.CACHE_DIR / f"stacker_{system}.joblib"


def load_stackers() -> dict:
    """Second-level rankers exported by `scripts/pipeline.py --export-stacker` (TERMMAP_STACKER=0 ignores them)."""
    if os.environ.get("TERMMAP_STACKER", "1") == "0":
        return {}
    return {s: joblib.load(stacker_path(s)) for s in config.SYSTEMS if stacker_path(s).exists()}


# --------------------------------------------------------------------------- engine

class TermMapEngine:
    def __init__(self):
        self.systems = {s: CodeSystem.load(s) for s in config.SYSTEMS}
        self.memory: dict[str, Memory] = {}
        self.ranker: dict[str, Ranker] = {}
        self.ce: dict = {}
        self.oof: dict[str, pd.DataFrame] = {}    # OOF training features + OOF ranker scores (pipeline caches)
        self.stacker: dict[str, Ranker] = load_stackers()

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

    def oof_items(self, train: pd.DataFrame, system: str) -> dict[int, tuple[str, str]]:
        """(entity, chunk_text) for each OOF item id produced by `oof_features`."""
        folds = self._folds(train)
        target = (train.coding_system == system) & (train.subconcept == "DiagnosisCode")
        out = {}
        for f in range(N_FOLDS):
            rows = train[target & (folds == f)].reset_index(drop=True)
            out.update({f * 1_000_000 + i: (e, c) for i, (e, c) in enumerate(zip(rows.entity, rows.chunk_text))})
        return out

    def fit(self, train: pd.DataFrame, verbose: bool = True, use_ce: bool | None = None,
            ce_gate: float | None = None) -> "TermMapEngine":
        use_ce = ce_enabled() if use_ce is None else use_ce
        for system in self.systems:
            X = derive(self.oof_features(train, system))
            if verbose:
                rec = X.groupby("item").label.max()
                print(f"[{system}] OOF candidate recall: {rec.mean():.3f} ({len(rec)} items, {len(X)} rows)", flush=True)
            self.ranker[system] = Ranker(FEATURES + DERIVED, **RANKER_PARAMS[system]).fit(X)
            # Cross-fitted ranker scores: what the ranker says about a training term it was not trained on.
            # Cross-encoder training lists come from these (the in-sample ranker puts gold first ~95% of
            # the time, so a cross-encoder trained on its lists never sees the confusers it meets at test).
            X["oof_rk"] = self._oof_ranker_scores(X, RANKER_PARAMS[system])
            self.oof[system] = X
            self.memory[system] = Memory(train, system)
            if use_ce:
                self.ce[system] = self._train_ce(train, system, X, gate=ce_gate)
        return self

    @staticmethod
    def _oof_ranker_scores(X: pd.DataFrame, params: dict) -> np.ndarray:
        fold = (X.item // 1_000_000).to_numpy()
        out = np.zeros(len(X))
        for f in np.unique(fold):
            m = fold == f
            out[m] = Ranker(FEATURES + DERIVED, **params).fit(X[~m]).score(X[m])
        return out

    def ce_groups(self, train, system, X, k: int = CE_K):
        """(query, candidate texts, gold index) for training terms whose gold is in the OOF ranker's top-k;
        every 10th term goes to an inner dev set used for early stopping."""
        from .cross_encoder import candidate_text, query_text

        cs, texts = self.systems[system], self.oof_items(train, system)
        top = X.sort_values(["item", "oof_rk", "code"], ascending=[True, False, True]).groupby("item").head(k)
        tr, dev = [], []
        for item, g in top.groupby("item", sort=True):
            if g.label.max():
                grp = (query_text(*texts[item]), [candidate_text(c, cs) for c in g.code], int(np.argmax(g.label.values)))
                (dev if item % 10 == 0 else tr).append(grp)
        return tr, dev

    def _train_ce(self, train, system, X, gate: float | None = None):
        """Cross-encoder on the OOF ranker's top-k candidates (gold + hard negatives) of the training terms."""
        from .cross_encoder import CrossEncoder, synonym_groups

        tr, dev = self.ce_groups(train, system, X)
        n_syn = int(os.environ.get("TERMMAP_CE_SYN", "0"))
        if n_syn:  # distant supervision from open synonyms (no labelled terms, no leakage)
            tr += synonym_groups(self.systems[system], n_syn, CE_K)
        print(f"[{system}] training cross-encoder on {len(tr)} groups (+{len(dev)} dev, {n_syn} synonym groups, "
              f"k={CE_K}, {CE_EPOCHS} epochs)", flush=True)
        return CrossEncoder().fit(tr, epochs=CE_EPOCHS, dev=dev, min_dev_after_first=gate, tag=f" {system}")

    def rank(self, items: pd.DataFrame, system: str, use_llm: bool = False,
             llm_threshold: float = 0.8, llm_k: int = 15, selector=None) -> pd.DataFrame:
        """Candidates for each term, best first. `prob` is the ranker's probability (LLM routing), `key` the
        final order (cross-encoder / stacker, LLM, training-term lookup) and `conf` the final confidence."""
        X = derive(build_features(items, self.systems[system], self.memory[system]))
        if not len(X):
            return X
        X["rk"] = self.ranker[system].score(X)
        X["prob"] = softmax_by_group(X, X.rk.to_numpy())
        X = X.sort_values(["item", "prob", "code"], ascending=[True, False, True])
        X["key"], X["llm"], X["conf"] = X.prob, 0, X.prob
        if selector is None and getattr(self, "selector_model", None) is not None:
            sm = self.selector_model
            selector = sm.score(items, X.groupby("item").head(sm.k), self.systems[system], system, score="prob")
        if self.ce.get(system) is not None:
            X = self._ce_rerank(items, X, system, selector)
        if use_llm and llm.enabled():
            X = self._adjudicate(items, X, system, llm_threshold, llm_k)
        X = self._apply_lookup(items, X, system)
        return X.sort_values(["item", "key", "code"], ascending=[True, False, True])

    def _ce_rerank(self, items, X, system, selector=None) -> pd.DataFrame:
        """Rescore each term's top-k with the cross-encoder; combine with the ranker through the learned
        stacker when one is available (scripts/pipeline.py s5), else (1-w)*z(ranker) + w*z(cross-encoder)."""
        from .cross_encoder import candidate_text, query_text

        cs = self.systems[system]
        top = X.groupby("item").head(CE_K).copy()
        qs = {i: query_text(items.entity.iloc[i], items.chunk_text.iloc[i]) for i in top.item.unique()}
        top["ce"] = self.ce[system].score([qs[i] for i in top.item], [candidate_text(c, cs) for c in top.code])
        if selector is not None:
            top = top.merge(selector, on=["item", "code"], how="left")
            top.index = X.groupby("item").head(CE_K).index
        F = stack_frame(top)
        st = self.stacker.get(system)
        key = st.score(F) if st is not None else (1 - CE_WEIGHT) * F.lp_z + CE_WEIGHT * F.ce_z
        X.loc[top.index, "ce"] = top.ce
        X.loc[top.index, "key"] = 10 + np.asarray(key)
        X.loc[top.index, "conf"] = softmax_by_group(top, np.asarray(key))
        X.loc[~X.index.isin(top.index), "conf"] = 0.0
        return X.sort_values(["item", "key", "code"], ascending=[True, False, True])

    def _apply_lookup(self, items, X, system) -> pd.DataFrame:
        """S0: a term seen verbatim (after normalisation) in the training labels gets its gold code."""
        lk = getattr(self.memory[system], "lookup", {})
        hits = [(i, lk[k]) for i, k in enumerate(expansions(e)[0] for e in items.entity) if k in lk]
        X["lookup"] = 0
        if not hits:
            return X
        H = pd.DataFrame(hits, columns=["item", "code"])
        m = X.set_index(["item", "code"]).index.isin(pd.MultiIndex.from_frame(H))
        X.loc[m, ["key", "conf", "lookup"]] = [1e9, 1.0, 1]
        new = H[~pd.MultiIndex.from_frame(H).isin(X.set_index(["item", "code"]).index)]
        if len(new):
            add = new.assign(key=1e9, conf=1.0, lookup=1, prob=0.0, llm=0)
            if "label" in X.columns and "code" in items.columns:
                add["label"] = (add.code.values == items.code.values[add.item.values]).astype(int)
            X = pd.concat([X, add], ignore_index=True)
        return X

    def _adjudicate(self, items, X, system, threshold, k) -> pd.DataFrame:
        """Send low-confidence terms with their top-k candidates to the LLM. Its pick goes to the top only
        when the ranker also found it plausible (pick prob >= LLM_GATE x top prob): the LLM settles close
        calls but does not overrule a clear ranker decision (measured: unrestricted overrides hurt)."""
        cs = self.systems[system]
        if "key" not in X.columns:
            X = X.assign(key=X.prob, llm=0)
        X = X.sort_values(["item", "key", "code"], ascending=[True, False, True])
        top = X.groupby("item").head(k)
        conf = X.groupby("item").prob.max()
        low = conf[conf < threshold].index.tolist()
        tasks = []
        for i in low:
            g = top[top.item == i]
            cands = [(c, cs.pref.get(c, "") + (f" ({cs.tag[c]})" if cs.tag.get(c) else "")
                      + (f" [{c[:3]}: {cs.pref[c[:3]]}]" if system == config.ICD and c[:3] in cs.pref and c != c[:3] else ""))
                     for c in g.code]
            tasks.append({"entity": items.entity.iloc[i], "system": system,
                          "context": context_window(items.entity.iloc[i], items.chunk_text.iloc[i], 150),
                          "candidates": cands})
        mem, n_ex = self.memory.get(system), int(os.environ.get("TERMMAP_LLM_EXAMPLES", "5"))
        if mem is not None and n_ex > 0 and tasks:
            for t, ex in zip(tasks, mem.examples([t["entity"] for t in tasks], n_ex)):
                t["examples"] = [(e, c, cs.pref.get(c, "")) for e, c in ex]
        picks = llm.adjudicate(tasks)
        X.attrs["llm_picks"] = dict(zip(low, picks))
        for i, code in zip(low, picks):
            if not code:
                continue
            m = (X.item == i) & (X.code == code)
            if m.any() and X.loc[m, "prob"].iloc[0] >= LLM_GATE * conf[i]:
                X.loc[m, "key"] = 1e6
                X.loc[m, "llm"] = 1
        return X.sort_values(["item", "key", "code"], ascending=[True, False, True])

    def predict(self, items: pd.DataFrame, top_n: int = 5, use_llm: bool = False) -> pd.DataFrame:
        """LLM adjudication only when asked for explicitly (and LLM_PROVIDER is set): the shipped
        configuration is LLM-off, and a provider left in .env must not change predictions silently."""
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
                p = g.conf.fillna(0).to_numpy()
                best.append(g.code.iloc[0])
                conf.append(round(float(p[0]), 4))
                margin.append(round(float(p[0] - (p[1] if len(p) > 1 else 0.0)), 4))
                alts.append(" | ".join(f"{c}:{pref.get(c, '')}:{q:.3f}" for c, q in
                                       zip(g.code.iloc[:top_n], p[:top_n])))
            out[f"{col}_code"] = best
            out[f"{col}_display"] = [pref.get(c, "") for c in best]
            out[f"{col}_confidence"] = conf
            out[f"{col}_margin"] = margin
            out[f"{col}_top{top_n}"] = alts
        return out

    def save(self, path=None):
        path = path or config.CACHE_DIR / "engine.joblib"
        joblib.dump({"memory": self.memory, "ranker": self.ranker, "ce_systems": sorted(self.ce)}, path)
        for system, ce in self.ce.items():
            ce.save(config.CACHE_DIR / f"engine_ce_{system}")

    @classmethod
    def load(cls, path=None) -> "TermMapEngine":
        from .cross_encoder import CrossEncoder

        eng = cls()
        state = joblib.load(path or config.CACHE_DIR / "engine.joblib")
        eng.memory, eng.ranker = state["memory"], state["ranker"]
        for system in state.get("ce_systems", []):
            eng.ce[system] = CrossEncoder(str(config.CACHE_DIR / f"engine_ce_{system}"))
        return eng
