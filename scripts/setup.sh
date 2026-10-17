#!/usr/bin/env bash
# One-time setup on a fresh machine (Linux/macOS/WSL).
#
#   bash scripts/setup.sh                 # deps + scispacy word vectors + dictionary caches/indexes
#   bash scripts/setup.sh --ingest        # + open sources (CDC ICD-10-CM FY2026, DOID, Mondo, HPO)
#   bash scripts/setup.sh --with-sapbert  # + torch/transformers and SapBERT index (~30-40 min on CPU, minutes on GPU)
#
# SCISPACY_TGZ=/path/en_core_sci_lg-0.5.4.tar.gz  use an already-downloaded copy of the vectors.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
INGEST=0; SAPBERT=0
for a in "$@"; do
  case "$a" in
    --ingest) INGEST=1 ;;
    --with-sapbert) SAPBERT=1 ;;
    *) echo "unknown option $a"; exit 1 ;;
  esac
done
step() { echo; echo "==> [$(date +%H:%M:%S)] $*"; }

step "installing python dependencies"
$PY -m pip install -q -r requirements.txt
if [ "$SAPBERT" = 1 ]; then $PY -m pip install -q torch transformers; fi

mkdir -p data/external/scispacy data/cache
URL=https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
TGZ=${SCISPACY_TGZ:-data/external/scispacy/en_core_sci_lg-0.5.4.tar.gz}
tgz_ok() { [ -f "$1" ] && $PY -c "import sys, tarfile; tarfile.open(sys.argv[1]).getmembers()" "$1" 2>/dev/null; }

if [ ! -f data/cache/wordvec.npy ]; then
  if ! tgz_ok "$TGZ"; then
    [ -f "$TGZ" ] && { echo "existing $TGZ is incomplete or corrupt; resuming/re-downloading"; mv -f "$TGZ" "$TGZ.part"; }
    step "downloading scispacy en_core_sci_lg vectors (531 MB, resumable)"
    if command -v curl >/dev/null; then
      curl -L -C - --retry 5 --retry-delay 3 --progress-bar -o "$TGZ.part" "$URL"
    else
      wget -c -O "$TGZ.part" "$URL"
    fi
    tgz_ok "$TGZ.part" || { echo "download is corrupt; delete $TGZ.part and re-run"; exit 1; }
    mv "$TGZ.part" "$TGZ"
  fi
  step "extracting word vectors"
  $PY - "$TGZ" <<'EOF'
import sys, tarfile
with tarfile.open(sys.argv[1]) as t:
    members = [m for m in t.getmembers() if "/vocab/" in m.name or m.name.endswith("meta.json")]
    t.extractall("data/external/scispacy", members=members)
print("scispacy vectors extracted")
EOF
fi

if [ "$INGEST" = 1 ]; then
  step "ingesting open sources (CDC ICD-10-CM FY2026, DOID, Mondo, HPO)"
  $PY -m scripts.ingest all
fi
if [ "$SAPBERT" = 0 ]; then export TERMMAP_SAPBERT=0; fi

step "building dictionary caches and retrieval indexes (first run ~6 min$( [ "$SAPBERT" = 1 ] && echo ', plus SapBERT encoding' ))"
if [ "$SAPBERT" = 1 ] && [ ! -f data/cache/sapbert_ft_syn/config.json ]; then
  # base SapBERT index first (needed to mine hard negatives), then the leak-free synonyms-only fine-tune
  TERMMAP_SAPBERT_MODEL=base $PY -c "from termmap.engine import CodeSystem; from termmap import config; [CodeSystem.load(s) for s in config.SYSTEMS]"
  step "fine-tuning SapBERT on open synonym pairs (no labelled terms)"
  TERMMAP_SAPBERT_MODEL=base $PY -m scripts.finetune_sapbert --epochs 2 --no-train-terms --aliases 40000
fi
$PY -c "from termmap.engine import CodeSystem; from termmap import config; [CodeSystem.load(s) for s in config.SYSTEMS]"
step "setup done. Next: python -m scripts.evaluate   or   python -m scripts.predict --input <csv> --output preds.csv"
