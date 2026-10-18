#!/usr/bin/env bash
# Bundle the trained artifacts the submission command needs (they are gitignored: too large for git).
#   bash scripts/pack_artifacts.sh [termmap_artifacts.tgz]
# Unpack at the repo root of a fresh clone with `tar xzf termmap_artifacts.tgz`; then the one-command
# predict in the README runs without retraining. It holds model weights, indexes and the engine's training-example
# memory (derived from hackathon_train.csv), so share it only inside the challenge (internal drive), like the data.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-termmap_artifacts.tgz}"
C=data/cache
FILES=("$C/engine.joblib" $C/engine_ce_* $C/stacker_*.joblib $C/stages/v2/selector_f0 $C/codesystem_*_v2.joblib)
for f in "${FILES[@]}"; do [ -e "$f" ] || { echo "missing $f" >&2; exit 1; }; done
# The cached dictionary indexes (SapBERT embeddings etc.) are rebuilt by setup.sh. Add them for an offline machine:
EXTRA=$(ls -d $C/sapbert* $C/*index* 2>/dev/null || true)
tar czf "$OUT" "${FILES[@]}" $EXTRA
ls -lh "$OUT"
