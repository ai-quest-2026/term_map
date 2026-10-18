"""Side experiment, NOT part of the shipped pipeline: a hosted LLM (JEV) choosing among cached ranker candidates.

    JEV_API_KEY=... JEV_BASE_URL=https://.../v1 JEV_MODEL=<model> \\
        python -m scripts.llm_sidecar --run cpu_v2 --fold 0 --n 100

Reads only the cached fold lists of a pipeline run (data/cache/stages/<run>/f<fold>_*); it never trains, never
writes stage caches and never changes engine files. Calls go through termmap/llm.py's OpenAI-compatible client
(temperature 0, seed), with responses cached separately in data/cache/llm_jev/ so a rerun costs nothing.
For each of the first --n test terms per system: the top-k candidates, a ±150-character context and, with
--examples, similar training-side terms with their gold codes (few-shot conventions). Reports the ranker's
top-1, the LLM's pick, the gold-in-top-k ceiling, and two routing policies (LLM on low-confidence terms only;
LLM pick only when the ranker also finds it plausible).

DATA CAUTION: this sends chart text to a third-party API. The problem statement forbids sending the dataset to
APIs that retain data; use only a provider/tier that does not retain prompts, and keep --n small.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from termmap import config, data, llm
from termmap.engine import CodeSystem, Memory
from termmap.normalize import context_window
from scripts.pipeline import cv_folds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="cpu_v2")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n", type=int, default=100, help="test terms per system")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--examples", type=int, default=5, help="few-shot training examples per term (0 = none)")
    ap.add_argument("--threshold", type=float, default=0.8, help="routing: ask the LLM only below this ranker prob")
    ap.add_argument("--gate", type=float, default=0.3, help="routing: accept a pick only if prob >= gate x top")
    args = ap.parse_args()

    key, base, model = (os.environ.get(v, "") for v in ("JEV_API_KEY", "JEV_BASE_URL", "JEV_MODEL"))
    if not (key and base and model):
        raise SystemExit("set JEV_API_KEY, JEV_BASE_URL (OpenAI-compatible, ending in /v1) and JEV_MODEL")
    os.environ.update(LLM_PROVIDER="openai_compat", OPENAI_COMPAT_API_KEY=key, OPENAI_COMPAT_BASE_URL=base,
                      OPENAI_COMPAT_MODEL=model, TERMMAP_LLM_BATCH=str(args.batch))
    llm._CACHE = config.CACHE_DIR / "llm_jev"

    cache = config.CACHE_DIR / "stages" / args.run
    train = data.load_train()
    tr_rows = train[cv_folds(train) != args.fold]
    report = {"run": args.run, "fold": args.fold, "model": model, "n": args.n, "k": args.k}
    for system in config.SYSTEMS:
        cs = CodeSystem.load(system)
        test = pd.read_parquet(cache / f"f{args.fold}_{system}_test.parquet")
        items = pd.read_parquet(cache / f"f{args.fold}_{system}_test_items.parquet")
        ids = list(range(min(args.n, len(items))))
        top = (test[test.item.isin(ids)].sort_values(["item", "score", "code"], ascending=[True, False, True])
               .groupby("item").head(args.k))
        tasks = []
        for i in ids:
            g = top[top.item == i]
            cands = [(c, cs.pref.get(c, "") + (f" ({cs.tag[c]})" if cs.tag.get(c) else "")
                      + (f" [{c[:3]}: {cs.pref[c[:3]]}]" if system == config.ICD and c[:3] in cs.pref and c != c[:3] else ""))
                     for c in g.code]
            tasks.append({"entity": items.entity[i], "system": system,
                          "context": context_window(items.entity[i], items.chunk_text[i], 150), "candidates": cands})
        if args.examples:
            mem = Memory(tr_rows, system)
            for t, ex in zip(tasks, mem.examples([t["entity"] for t in tasks], args.examples)):
                t["examples"] = [(e, c, cs.pref.get(c, "")) for e, c in ex]
        picks = llm.adjudicate(tasks)

        hit = {k: [] for k in ("ranker", "llm", "ceiling", "routed", "gated")}
        answered = 0
        for i, pick in zip(ids, picks):
            g = top[top.item == i].reset_index(drop=True)
            gold, prob = items.code[i], dict(zip(g.code, g.prob))
            rk = g.code.iloc[0] if len(g) else ""
            answered += pick is not None
            p = pick or rk
            low = prob.get(rk, 0) < args.threshold
            gated = p if (pick and prob.get(pick, 0) >= args.gate * prob.get(rk, 0)) else rk
            hit["ranker"].append(rk == gold)
            hit["llm"].append(p == gold)
            hit["ceiling"].append(gold in set(g.code))
            hit["routed"].append((p if low else rk) == gold)
            hit["gated"].append((gated if low else rk) == gold)
        r = {k: round(float(np.mean(v)), 4) for k, v in hit.items()}
        r.update(terms=len(ids), answered=answered, llm_only_right=int(np.sum(np.array(hit["llm"]) & ~np.array(hit["ranker"]))),
                 ranker_only_right=int(np.sum(np.array(hit["ranker"]) & ~np.array(hit["llm"]))))
        report[system] = r
        print(f"[{system}] n={len(ids)} answered={answered} ranker={r['ranker']:.3f} llm={r['llm']:.3f} "
              f"routed(<{args.threshold})={r['routed']:.3f} gated={r['gated']:.3f} gold@{args.k}={r['ceiling']:.3f} "
              f"llm-only-right={r['llm_only_right']} ranker-only-right={r['ranker_only_right']}", flush=True)
    out = config.ROOT / "reports" / "stages" / f"llm_sidecar_{args.run}_f{args.fold}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
