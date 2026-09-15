#!/bin/bash
# Render paper/hone.md to paper/hone.pdf with figures inline.
# Needs pandoc and Google Chrome (headless print-to-PDF).
set -euo pipefail
cd "$(dirname "$0")/../paper"
CHROME=${CHROME:-"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"}
# For print, fold each "![Figure N](img)" + following "*Figure N. caption*"
# paragraph into one captioned figure so a page break cannot separate them.
# (The markdown keeps the separate caption so it shows on GitHub too.)
python3 - <<'PY' > hone.print.md
import re
s = open("hone.md").read()
s = re.sub(r"!\[Figure (\d+)\]\(([^)]+)\)\n\n\*(Figure \1\.[^\n]*?)\*\n",
           lambda m: f"![{m.group(3)}]({m.group(2)})\n", s)
# Title "Claim: Subtitle" -> claim as the title, subtitle on its own line.
s = re.sub(r"\A# ([^\n:]+): ([^\n]+)\n",
           lambda m: f"# {m.group(1)}\n\n<p class=\"subtitle\">{m.group(2)}</p>\n", s)
print(s, end="")
PY
# -smart keeps "--" as typed (no en/em dash substitution)
pandoc hone.print.md -f markdown-smart --standalone --css print.css --embed-resources \
  --metadata pagetitle="Speculative Decoding Pays Only at Small Active-Parameter Ratios on Apple Silicon" -o hone.html
rm -f hone.print.md
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf=hone.pdf "file://$PWD/hone.html" 2>/dev/null
rm -f hone.html
echo "wrote paper/hone.pdf"
