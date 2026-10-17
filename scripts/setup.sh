#!/usr/bin/env bash
# One-time setup on a fresh machine: python deps + scispacy word vectors + dictionary caches.
set -euo pipefail
cd "$(dirname "$0")/.."
pip install -q -r requirements.txt
mkdir -p data/external/scispacy data/cache
TGZ=data/external/scispacy/en_core_sci_lg-0.5.4.tar.gz
if [ ! -f data/cache/wordvec.npy ]; then
  [ -f "$TGZ" ] || curl -sSfL -o "$TGZ" https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
  tar xzf "$TGZ" -C data/external/scispacy --wildcards '*/vocab/*' '*/meta.json'
fi
# Builds parquet dictionaries, word-vector cache and retrieval indexes (~6 min first time).
python -c "from termmap.engine import CodeSystem; from termmap import config; [CodeSystem.load(s) for s in config.SYSTEMS]"
echo "setup done"
