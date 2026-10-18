"""TermMap demo and test front end.

    pip install streamlit
    streamlit run app/streamlit_app.py

Tabs:
  1. Map a term: one mention + chart text, live (the demo's hard cases are presets). Shows both codes, the
     confidence, the margin, a "route to review" flag and the top-5 alternatives.
  2. Batch + evaluate: upload a CSV in the problem schema (chunk_id, clinical_entity[, chunk_text]) or the
     training schema. Download predictions in the submission schema. If gold codes are present, it scores them
     with the problem statement's metrics: exact match per system, headline = the lower of the two, ICD
     3-character category, long tail (codes rare or absent in training), accuracy at coverage, and latency.
  3. Blind test: answers hidden for one cross-validation fold, mapped by an engine trained without that fold
     (scripts/blind_test.py builds it), then scored against the hidden gold.

Runs fully offline, like scripts/predict.py. Nothing typed here leaves the machine.
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from termmap import config, data  # noqa: E402

ICD, SNOMED = config.ICD, config.SNOMED
COLS = {ICD: "icd10cm", SNOMED: "snomed"}
NAMES = {ICD: "ICD-10-CM", SNOMED: "SNOMED CT"}
DEFAULT_SELECTOR = config.CACHE_DIR / "stages" / "v2" / "selector_f0"

st.set_page_config(page_title="TermMap", layout="wide")


# --------------------------------------------------------------------------- engine
@st.cache_resource(show_spinner="Loading engine…")
def load_engine(path: str):
    from termmap.engine import TermMapEngine
    return TermMapEngine.load(path)


@st.cache_resource(show_spinner="Loading the LLM selector (Qwen2.5-7B + LoRA)…")
def load_selector(adapter: str):
    from termmap.selector import Selector
    return Selector(adapter)


def engine_for(path: str, use_selector: bool, adapter: str):
    eng = load_engine(path)
    eng.selector_model = load_selector(adapter) if use_selector else None
    return eng


def run(eng, items: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    t0 = time.time()
    pred = eng.predict(items.reset_index(drop=True))
    return pred, time.time() - t0


@st.cache_data
def train_code_counts() -> dict[str, pd.Series]:
    tr = data.load_train()
    tr = tr[tr.subconcept == "DiagnosisCode"]
    return {s: tr[tr.coding_system == s].code.value_counts() for s in (ICD, SNOMED)}


# --------------------------------------------------------------------------- inputs and gold
def split_gold(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Items to map plus gold codes (None if the file has none). Accepts the long training schema
    (entity, chunk_text, coding_system, code) or a wide file with gold_icd10cm / gold_snomed or
    icd10cm_code / snomed_code columns."""
    df = df.fillna("").astype(str)
    if "entity" not in df.columns and "clinical_entity" in df.columns:
        df["entity"] = df["clinical_entity"]
    if "chunk_text" not in df.columns:
        df["chunk_text"] = ""
    if {"coding_system", "code"} <= set(df.columns):  # training schema: one row per (mention, system)
        if "subconcept" in df.columns:
            df = df[df.subconcept == "DiagnosisCode"]
        g = df.pivot_table(index=["entity", "chunk_text"], columns="coding_system", values="code",
                           aggfunc="first").reset_index()
        g.insert(0, "chunk_id", [hashlib.md5(c.encode()).hexdigest()[:12] for c in g.chunk_text])
        items = g[["chunk_id", "entity", "chunk_text"]].assign(clinical_entity=g.entity)
        gold = pd.DataFrame({s: g.get(s, pd.Series([""] * len(g))).fillna("").values for s in (ICD, SNOMED)})
        return items, gold
    for gi, gs in (("gold_icd10cm", "gold_snomed"), ("icd10cm_code", "snomed_code")):
        if gi in df.columns or gs in df.columns:
            gold = pd.DataFrame({ICD: df.get(gi, ""), SNOMED: df.get(gs, "")}).reset_index(drop=True)
            return df.drop(columns=[c for c in (gi, gs) if c in df.columns]).reset_index(drop=True), gold
    return df.reset_index(drop=True), None


def submission_frame(items: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"chunk_id": items.get("chunk_id", pd.Series([""] * len(items))).values,
                        "clinical_entity": items.get("clinical_entity", items.entity).values})
    out["icd10cm_code"], out["snomed_code"] = pred.icd10cm_code.values, pred.snomed_code.values
    return out


# --------------------------------------------------------------------------- metrics (problem statement)
def evaluate(pred: pd.DataFrame, gold: pd.DataFrame, counts: dict[str, pd.Series], seconds: float,
             review_below: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, cov_rows, errors = [], [], []
    for s in (ICD, SNOMED):
        c = COLS[s]
        m = gold[s].astype(str).str.len() > 0
        if not m.any():
            continue
        g, p = gold.loc[m, s].values, pred.loc[m, f"{c}_code"].values
        conf = pd.to_numeric(pred.loc[m, f"{c}_confidence"], errors="coerce").fillna(0).values
        ok = g == p
        n_train = counts[s].reindex(g).fillna(0).values
        r = {"system": NAMES[s], "n": int(m.sum()), "exact match": ok.mean(),
             "seen codes": ok[n_train > 0].mean() if (n_train > 0).any() else np.nan,
             "rare codes (1–2 in training)": ok[(n_train > 0) & (n_train <= 2)].mean()
             if ((n_train > 0) & (n_train <= 2)).any() else np.nan,
             "unseen codes (long tail)": ok[n_train == 0].mean() if (n_train == 0).any() else np.nan,
             "share unseen": (n_train == 0).mean()}
        if s == ICD:
            r["3-char category"] = (pd.Series(g).str[:3].values == pd.Series(p).str[:3].values).mean()
        keep = conf >= review_below
        r[f"answered (conf ≥ {review_below:.2f})"] = keep.mean()
        r["accuracy when answered"] = ok[keep].mean() if keep.any() else np.nan
        rows.append(r)
        order = np.argsort(-conf, kind="stable")
        for cov in (1.0, 0.9, 0.8, 0.7, 0.5):
            k = max(1, int(round(cov * len(order))))
            cov_rows.append({"system": NAMES[s], "coverage": f"{int(cov * 100)}%", "accuracy": ok[order[:k]].mean(),
                             "min confidence": conf[order[k - 1]]})
        top = pred.loc[m, f"{c}_top5"].values
        for i in np.where(~ok)[0]:
            alts = [a.split(":")[0] for a in str(top[i]).split(" | ") if a]
            errors.append({"system": NAMES[s], "entity": pred.loc[m].iloc[i].get("entity", ""), "gold": g[i],
                           "predicted": p[i], "confidence": round(float(conf[i]), 3),
                           "gold in top 5": g[i] in alts,
                           "same 3-char category": s == ICD and g[i][:3] == p[i][:3],
                           "gold seen in training": bool(n_train[i] > 0)})
    summary = pd.DataFrame(rows)
    if len(summary):
        summary.attrs["headline"] = summary["exact match"].min()
        summary.attrs["ms_per_term"] = 1000 * seconds / max(1, len(pred))
    return summary, pd.DataFrame(cov_rows), pd.DataFrame(errors)


def show_evaluation(summary, cov, errors):
    if not len(summary):
        st.info("No gold codes in this file, so there is nothing to score.")
        return
    a, b, c = st.columns(3)
    a.metric("Headline (lower of the two systems)", f"{summary.attrs['headline']:.3f}")
    b.metric("Terms", int(summary.n.max()))
    c.metric("Latency (batch)", f"{summary.attrs['ms_per_term']:.0f} ms / term")
    st.dataframe(summary.style.format({k: "{:.3f}" for k in summary.columns if k not in ("system", "n")},
                                      na_rep="–"), hide_index=True, width="stretch")
    st.markdown("**Accuracy at coverage**: answer only the most confident share of terms and send the rest to review.")
    st.dataframe(cov.pivot(index="coverage", columns="system", values="accuracy").loc[["100%", "90%", "80%", "70%", "50%"]]
                 .style.format("{:.3f}"), width="stretch")
    if len(errors):
        st.markdown(f"**Errors ({len(errors)})**: gold in the top 5 means a ranking error; otherwise the gold was not "
                    "retrieved.")
        e = errors.groupby("system").agg(errors=("gold", "size"), gold_in_top5=("gold in top 5", "mean"),
                                        same_category=("same 3-char category", "mean"),
                                        gold_seen=("gold seen in training", "mean"))
        st.dataframe(e.style.format({"gold_in_top5": "{:.0%}", "same_category": "{:.0%}", "gold_seen": "{:.0%}"}),
                     width="stretch")
        st.dataframe(errors, hide_index=True, width="stretch")


def code_card(row: pd.Series, s: str, review_below: float):
    c = COLS[s]
    conf, margin = float(row[f"{c}_confidence"] or 0), float(row[f"{c}_margin"] or 0)
    st.subheader(NAMES[s])
    st.markdown(f"### `{row[f'{c}_code']}`\n{row[f'{c}_display']}")
    x, y = st.columns(2)
    x.metric("confidence", f"{conf:.2f}")
    y.metric("margin over runner-up", f"{margin:.2f}")
    if conf < review_below:
        st.warning("Low confidence: route to a coder for review.")
    alts = []
    for part in str(row[f"{c}_top5"]).split(" | "):
        if part:
            code, *rest = part.split(":")
            alts.append({"code": code, "description": ":".join(rest[:-1]), "score": float(rest[-1])})
    st.dataframe(pd.DataFrame(alts), hide_index=True, width="stretch")


# --------------------------------------------------------------------------- page
st.title("TermMap: clinical term → ICD-10-CM + SNOMED CT")
with st.sidebar:
    st.header("Engine")
    engine_path = st.text_input("engine file", str(config.CACHE_DIR / "engine.joblib"))
    have_sel = DEFAULT_SELECTOR.exists()
    use_sel = st.checkbox("LLM selector (local Qwen2.5-7B + LoRA, GPU)", value=have_sel, disabled=not have_sel)
    adapter = st.text_input("selector adapter", str(DEFAULT_SELECTOR))
    review_below = st.slider("route to review below confidence", 0.0, 1.0, 0.5, 0.05)
    st.caption("Offline: no network calls at inference. The shipped configuration = engine + selector + exported stacker.")

tab1, tab2, tab3 = st.tabs(["Map a term", "Batch + evaluate", "Blind test"])

with tab1:
    demo_path = ROOT / "docs" / "demo" / "hard_cases.csv"
    presets = pd.read_csv(demo_path, dtype=str) if demo_path.exists() else pd.DataFrame(columns=["chunk_id", "clinical_entity", "chunk_text"])
    labels = ["(type your own)"] + [f"{r.chunk_id}: {r.clinical_entity}" for r in presets.itertuples()]
    def _apply_preset():
        i = labels.index(st.session_state.preset)
        if i:
            st.session_state.entity = presets.clinical_entity.iloc[i - 1]
            st.session_state.chunk = presets.chunk_text.iloc[i - 1]

    st.selectbox("hard-case presets: abbreviation collision, OCR, specificity, context, negation, laterality", labels,
                 key="preset", on_change=_apply_preset)
    entity = st.text_input("clinical entity", key="entity")
    chunk = st.text_area("chart text (chunk)", key="chunk", height=110)
    if st.button("Map", type="primary", disabled=not entity.strip()):
        eng = engine_for(engine_path, use_sel, adapter)
        items = pd.DataFrame({"chunk_id": ["live"], "clinical_entity": [entity], "entity": [entity], "chunk_text": [chunk]})
        pred, secs = run(eng, items)
        st.caption(f"mapped in {1000 * secs:.0f} ms")
        a, b = st.columns(2)
        with a:
            code_card(pred.iloc[0], ICD, review_below)
        with b:
            code_card(pred.iloc[0], SNOMED, review_below)

with tab2:
    st.markdown("Upload `chunk_id, clinical_entity[, chunk_text]` (problem schema) or the training schema. "
                "Add gold columns (`gold_icd10cm`, `gold_snomed`) to get the evaluation.")
    up = st.file_uploader("CSV", type="csv")
    if up is not None:
        raw = pd.read_csv(up, dtype=str, keep_default_na=False)
        items, gold = split_gold(raw)
        st.write(f"{len(items)} terms" + ("" if gold is None else ", gold codes found (hidden from the engine)"))
        if st.button("Run", type="primary"):
            eng = engine_for(engine_path, use_sel, adapter)
            with st.spinner("mapping…"):
                pred, secs = run(eng, items[[c for c in items.columns if c in ("chunk_id", "entity", "clinical_entity", "chunk_text")]])
            st.success(f"mapped {len(items)} terms in {secs:.1f} s ({1000 * secs / max(1, len(items)):.0f} ms/term)")
            sub = submission_frame(items, pred)
            st.dataframe(sub, hide_index=True, width="stretch")
            c1, c2 = st.columns(2)
            c1.download_button("predictions.csv (submission schema)", sub.to_csv(index=False), "predictions.csv")
            c2.download_button("details.csv (confidence, top-5)", pred.drop(columns=["chunk_text"], errors="ignore")
                               .to_csv(index=False), "details.csv")
            if gold is not None:
                show_evaluation(*evaluate(pred, gold, train_code_counts(), secs, review_below))
                st.caption("Note: if these terms are in the training file, the engine has seen them (exact-term lookup); "
                           "use the Blind test tab for an honest estimate.")

with tab3:
    st.markdown("Hide the answers of one cross-validation fold (grouped by chunk, as in `scripts/pipeline.py`), map it "
                "with an engine trained **without** that fold, and score against the hidden gold.")
    fold = st.number_input("fold", 0, 4, 0)
    blind_engine = config.CACHE_DIR / "blind" / f"f{fold}" / "engine.joblib"
    sel_ok = (config.CACHE_DIR / "stages" / "v2" / f"selector_f{fold}").exists()
    if not blind_engine.exists():
        st.warning(f"No engine trained without fold {fold}. Build it once (~20 min on the A6000):\n\n"
                   f"`python -m scripts.blind_test --fold {fold}`")
    else:
        no_ctx = st.checkbox("hide chunk_text too (entity only)")
        limit = st.slider("terms to map (subset for a quick demo)", 20, 600, 100, 20)
        st.caption("Fold 0 caveat: the exported stacker was fitted on fold 0's features, so its blind score is optimistic; "
                   "the clean fold-0 numbers are the pipeline's 0.716 / 0.685.")
        if st.button("Run blind test", type="primary"):
            from scripts.pipeline import cv_folds
            train = data.load_train()
            folds = cv_folds(train)
            items, gold = split_gold(train[folds == fold])
            items, gold = items.iloc[:limit].reset_index(drop=True), gold.iloc[:limit].reset_index(drop=True)
            if no_ctx:
                items["chunk_text"] = ""
            tr = train[(folds != fold) & (train.subconcept == "DiagnosisCode")]
            counts = {s: tr[tr.coding_system == s].code.value_counts() for s in (ICD, SNOMED)}
            eng = engine_for(str(blind_engine), use_sel and sel_ok, str(config.CACHE_DIR / "stages" / "v2" / f"selector_f{fold}"))
            with st.spinner(f"mapping {len(items)} hidden-answer terms…"):
                pred, secs = run(eng, items)
            show_evaluation(*evaluate(pred, gold, counts, secs, review_below))
