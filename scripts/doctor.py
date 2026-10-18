"""Pre-flight check before any long run: prints PASS / WARN / FAIL per item and exits non-zero on a FAIL.

    python -m scripts.doctor
"""
from __future__ import annotations

import importlib
import os
import shutil
import sys

from termmap import config

results: list[tuple[str, str, str]] = []


def check(name: str, status: str, detail: str = "") -> None:
    results.append((status, name, detail))
    print(f"[{status:4s}] {name}{': ' + detail if detail else ''}", flush=True)


def main():
    check("python", "PASS", sys.version.split()[0])
    for f in (config.TRAIN_CSV, config.ICD_XLSX, config.SNOMED_XLSX):
        check(f"input {f.name}", "PASS" if f.exists() else "FAIL")
    for mod in ("pandas", "numpy", "sklearn", "lightgbm", "rapidfuzz", "scipy", "joblib", "pyarrow"):
        try:
            m = importlib.import_module(mod)
            check(f"module {mod}", "PASS", getattr(m, "__version__", ""))
        except ImportError as e:
            check(f"module {mod}", "FAIL", str(e))
    try:
        import torch
        import transformers
        cuda = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if cuda else "no GPU"
        free = f", {torch.cuda.mem_get_info()[0] / 2**30:.1f} GB free" if cuda else ""
        check("torch / transformers", "PASS", f"torch {torch.__version__}, transformers {transformers.__version__}")
        check("CUDA", "PASS" if cuda else "WARN", dev + free + ("" if cuda else " (SapBERT/cross-encoder will be slow)"))
    except ImportError:
        check("torch / transformers", "WARN", "not installed: SapBERT and cross-encoder disabled")
    check("CUBLAS_WORKSPACE_CONFIG", "PASS" if os.environ.get("CUBLAS_WORKSPACE_CONFIG") else "WARN",
          os.environ.get("CUBLAS_WORKSPACE_CONFIG", "unset"))
    wv = config.CACHE_DIR / "wordvec.npy"
    check("word vectors cache", "PASS" if wv.exists() else "WARN", "built on first run" if not wv.exists() else "")
    for s in config.SYSTEMS:
        p = config.CACHE_DIR / f"codesystem_{s}_{config.CACHE_VERSION}.joblib"
        check(f"codesystem cache {s} ({config.CACHE_VERSION})", "PASS" if p.exists() else "WARN",
              "" if p.exists() else "will be built (~6 min, plus SapBERT re-encoding of changed texts)")
    al = sorted(config.EXTERNAL_DIR.glob("*_aliases*.parquet"))
    check("open-source aliases", "PASS" if al else "WARN", f"{len(al)} files" if al else "run: python -m scripts.ingest all")
    llm = os.environ.get("LLM_PROVIDER", "none")
    check("LLM_PROVIDER", "PASS", f"{llm} (only used with --llm)")
    for s in config.SYSTEMS:
        p = config.CACHE_DIR / f"stacker_{s}.joblib"
        check(f"stacker {s}", "PASS" if p.exists() else "WARN",
              "" if p.exists() else "none: ranker+CE z-blend (export with scripts/pipeline.py --export-stacker)")
    du = shutil.disk_usage(config.ROOT)
    check("disk free", "PASS" if du.free > 20 * 2**30 else "WARN", f"{du.free / 2**30:.0f} GB")
    fails = [r for r in results if r[0] == "FAIL"]
    print(f"\n{len(fails)} FAIL, {sum(r[0] == 'WARN' for r in results)} WARN")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
