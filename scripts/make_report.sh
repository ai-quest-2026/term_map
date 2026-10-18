#!/usr/bin/env bash
# Build the submission PDF from docs/writeup.md:  bash scripts/make_report.sh [out.pdf]
#   1. docs/architecture.mmd -> docs/architecture.png  (mermaid-cli via npx; skipped if node/npx is missing)
#   2. docs/writeup.md -> HTML (pandoc, docs/report.css) -> PDF (headless Chromium / Chrome)
# Needs pandoc and a Chromium/Chrome binary (CHROME=... to override; Playwright's copy is found automatically).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-docs/TermMap_writeup.pdf}"
CHROME="${CHROME:-$(command -v chromium || command -v chromium-browser || command -v google-chrome || \
  ls -d "${PLAYWRIGHT_BROWSERS_PATH:-$HOME/.cache/ms-playwright}"/chromium-*/chrome-linux/chrome 2>/dev/null | head -1 || true)}"
[ -x "$CHROME" ] || { echo "no Chromium/Chrome found; set CHROME=/path/to/chrome" >&2; exit 1; }
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

if command -v npx >/dev/null; then
  printf '{"executablePath":"%s","args":["--no-sandbox"]}' "$CHROME" > "$TMP/puppeteer.json"
  PUPPETEER_SKIP_DOWNLOAD=1 npx -y @mermaid-js/mermaid-cli@11.12.0 -p "$TMP/puppeteer.json" \
    -i docs/architecture.mmd -o docs/architecture.png -s 3 -b white -w 1400 >/dev/null
  echo "rendered docs/architecture.png"
fi

pandoc docs/writeup.md -f gfm -s --embed-resources --resource-path=docs --css docs/report.css \
  --metadata title="TermMap write-up" -o "$TMP/writeup.html"
"$CHROME" --headless --no-sandbox --disable-gpu --no-pdf-header-footer --print-to-pdf="$OUT" "$TMP/writeup.html" 2>/dev/null
python - "$OUT" <<'EOF' 2>/dev/null || echo "wrote $OUT"
import sys
from pypdf import PdfReader
print(f"wrote {sys.argv[1]} ({len(PdfReader(sys.argv[1]).pages)} pages)")
EOF
