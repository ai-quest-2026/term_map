#!/usr/bin/env bash
# One-time setup on a fresh machine (Linux/macOS; on Windows use Git Bash or WSL).
#
#   bash scripts/setup.sh                 # deps + scispacy word vectors + dictionary caches/indexes
#   bash scripts/setup.sh --ingest        # + open sources (CDC ICD-10-CM FY2026, DOID, Mondo, HPO)
#   bash scripts/setup.sh --with-sapbert  # + torch/transformers and SapBERT index (~30-40 min on CPU, minutes on GPU)
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

$PY -m pip install -q -r requirements.txt
[ "$SAPBERT" = 1 ] && $PY -m pip install -q torch transformers

mkdir -p data/external/scispacy data/cache
if [ ! -f data/cache/wordvec.npy ]; then
  $PY - <<'EOF'
import tarfile, urllib.request
from pathlib import Path
url = "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz"
dest = Path("data/external/scispacy/en_core_sci_lg-0.5.4.tar.gz")
if not dest.exists():
    print("downloading scispacy en_core_sci_lg (~530 MB)...")
    urllib.request.urlretrieve(url, dest)
with tarfile.open(dest) as t:
    members = [m for m in t.getmembers() if "/vocab/" in m.name or m.name.endswith("meta.json")]
    t.extractall("data/external/scispacy", members=members)
print("scispacy vectors extracted")
EOF
fi

[ "$INGEST" = 1 ] && $PY -m scripts.ingest all
[ "$SAPBERT" = 0 ] && export TERMMAP_SAPBERT=0

# Builds parquet dictionaries, word-vector cache and retrieval indexes (~6 min first time).
$PY -c "from termmap.engine import CodeSystem; from termmap import config; [CodeSystem.load(s) for s in config.SYSTEMS]"
echo "setup done. Next: python -m scripts.evaluate   or   python -m scripts.predict --input <csv> --output preds.csv"
