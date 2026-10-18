"""Side experiment, NOT part of the shipped pipeline: TypeSafe's Jev decision model choosing among ranker candidates.

    JEV_API_KEY=... python -m scripts.llm_sidecar --run cpu_v2 --fold 0 --n 100

Jev is a decision model (System One API, https://docs.typesafe.ai/api): it takes a `state` plus typed questions and
returns, for a Choice question, a probability for every option (up to 255) instead of generated text. Here each test
term is one request: state = the term, its ±150-character context, the coding system and (with --examples) similar
training-side terms with their gold codes; one Choice question whose options are the ranker's top-k candidates
(code -> official wording). The answer can only be one of the candidates.

Reads only the cached fold lists of a pipeline run (data/cache/stages/<run>/f<fold>_*); it never trains, never writes
stage caches and never changes engine files. Responses are cached in data/cache/llm_jev/ so reruns cost nothing.
Reports the ranker's top-1, Jev's pick, the gold-in-top-k ceiling, Jev on low-confidence terms only, and blends of
ranker and Jev log-probabilities.

DATA CAUTION: this sends chart text to a third-party API. The problem statement forbids sending the dataset to APIs
that retain data (TypeSafe's terms: no training on user data; zero data retention only for enterprise accounts).
Keep --n small and use it as a side experiment only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from termmap import config, data
from termmap.engine import CodeSystem, Memory
from termmap.normalize import context_window
from scripts.pipeline import cv_folds

CACHE = config.CACHE_DIR / "llm_jev"
INSTRUCTIONS = ("Which code would a professional medical coder assign to the clinical `term`, read in its `context`? "
                "Follow the conventions shown in `similar_coded_terms` (how specific to be, which concept they prefer, "
                "how negated findings are coded). Do not invent specificity the text does not state.")


def ask(state: dict, options: dict[str, str], model: str, key: str, base: str) -> dict:
    body = {"model": model, "state": state,
            "questions": {"code": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": options}}}
    h = hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()
    path = CACHE / f"{h}.json"
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(5):
        r = requests.post(f"{base.rstrip('/')}/v1/systemone", json=body, timeout=120,
                          headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        if r.status_code in (429, 500, 502, 503) and attempt < 4:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        break
    out = r.json()
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="cpu_v2")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n", type=int, default=100, help="test terms per system")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--examples", type=int, default=5, help="similar training terms in the state (0 = none)")
    ap.add_argument("--threshold", type=float, default=0.8, help="ask-only-when-unsure: ranker prob below this")
    ap.add_argument("--model", default=os.environ.get("JEV_MODEL", "jev-latest"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    key = os.environ.get("JEV_API_KEY") or os.environ.get("TYPESAFE_API_KEY")
    base = os.environ.get("JEV_BASE_URL", "https://api.typesafe.ai")
    if not key:
        raise SystemExit("JEV_API_KEY is not set in this session")

    cache = config.CACHE_DIR / "stages" / args.run
    train = data.load_train()
    tr_rows = train[cv_folds(train) != args.fold]
    report = {"run": args.run, "fold": args.fold, "model": args.model, "n": args.n, "k": args.k,
              "examples": args.examples}
    for system in config.SYSTEMS:
        cs = CodeSystem.load(system)
        test = pd.read_parquet(cache / f"f{args.fold}_{system}_test.parquet")
        items = pd.read_parquet(cache / f"f{args.fold}_{system}_test_items.parquet")
        ids = list(range(min(args.n, len(items))))
        top = (test[test.item.isin(ids)].sort_values(["item", "score", "code"], ascending=[True, False, True])
               .groupby("item").head(args.k))
        exs = (Memory(tr_rows, system).examples([items.entity[i] for i in ids], args.examples)
               if args.examples else [[] for _ in ids])
        jobs = []
        for i, ex in zip(ids, exs):
            g = top[top.item == i]
            options = {c: cs.pref.get(c, c) + (f" ({cs.tag[c]})" if cs.tag.get(c) else "") for c in g.code}
            state = {"coding_system": "ICD-10-CM" if system == config.ICD else "SNOMED CT",
                     "term": items.entity[i], "context": context_window(items.entity[i], items.chunk_text[i], 150)}
            if ex:
                state["similar_coded_terms"] = [{"term": e, "code": c, "meaning": cs.pref.get(c, "")} for e, c in ex]
            jobs.append((state, options))
        t0 = time.time()
        with ThreadPoolExecutor(args.workers) as pool:
            answers = list(pool.map(lambda j: ask(j[0], j[1], args.model, key, base), jobs))
        secs = time.time() - t0

        hit = {k: [] for k in ("ranker", "jev", "ceiling", "jev_if_unsure", "blend_0.5", "blend_1.0")}
        conf = []
        for i, ans in zip(ids, answers):
            g = top[top.item == i].reset_index(drop=True)
            gold = items.code[i]
            a = ans.get("answers", {}).get("code", {})
            probs = a.get("probabilities", {})
            pick = a.get("choice") or (g.code.iloc[0] if len(g) else "")
            rk = g.code.iloc[0] if len(g) else ""
            conf.append(a.get("confidence", np.nan))
            hit["ranker"].append(rk == gold)
            hit["jev"].append(pick == gold)
            hit["ceiling"].append(gold in set(g.code))
            unsure = float(g.prob.iloc[0]) < args.threshold if len(g) else False
            hit["jev_if_unsure"].append((pick if unsure else rk) == gold)
            for w in (0.5, 1.0):
                s = np.log(g.prob.clip(lower=1e-9)) + w * np.log(np.array([probs.get(c, 1e-9) for c in g.code]) + 1e-9)
                hit[f"blend_{w}"].append(g.code.iloc[int(np.argmax(s))] == gold if len(g) else False)
        r = {k: round(float(np.mean(v)), 4) for k, v in hit.items()}
        r.update(terms=len(ids), seconds=round(secs, 1), mean_confidence=round(float(np.nanmean(conf)), 3),
                 jev_only_right=int(np.sum(np.array(hit["jev"]) & ~np.array(hit["ranker"]))),
                 ranker_only_right=int(np.sum(np.array(hit["ranker"]) & ~np.array(hit["jev"]))),
                 model_answered=answers[0].get("model") if answers else None)
        report[system] = r
        print(f"[{system}] n={len(ids)} ranker={r['ranker']:.3f} jev={r['jev']:.3f} jev_if_unsure={r['jev_if_unsure']:.3f} "
              f"blend0.5={r['blend_0.5']:.3f} blend1.0={r['blend_1.0']:.3f} gold@{args.k}={r['ceiling']:.3f} "
              f"jev-only-right={r['jev_only_right']} ranker-only-right={r['ranker_only_right']} ({secs:.0f}s, "
              f"model {r['model_answered']})", flush=True)
    out = config.ROOT / "reports" / "stages" / f"jev_sidecar_{args.run}_f{args.fold}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
