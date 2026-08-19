#!/usr/bin/env bash
# Assemble a static site: Zig autodoc + markdown guides.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

zig build docs -Dhip=off

SITE="$ROOT/zig-out/site"
rm -rf "$SITE"
mkdir -p "$SITE/api" "$SITE/guides"

if [[ -d "$ROOT/zig-out/docs/api" ]]; then
    cp -R "$ROOT/zig-out/docs/api/." "$SITE/api/"
fi

cp "$ROOT/README.md" "$SITE/guides/README.md"
cp "$ROOT/docs/"*.md "$SITE/guides/" 2>/dev/null || true
if [[ -d "$ROOT/docs/tutorials" ]]; then
    mkdir -p "$SITE/guides/tutorials" "$SITE/guides/stages"
    cp "$ROOT/docs/tutorials/"*.md "$SITE/guides/tutorials/" 2>/dev/null || true
    cp "$ROOT/docs/stages/"*.md "$SITE/guides/stages/" 2>/dev/null || true
fi

/usr/bin/python3 - "$SITE" <<'PY'
from pathlib import Path
import html
import sys

site = Path(sys.argv[1])
guides = sorted(p.relative_to(site / "guides") for p in (site / "guides").rglob("*.md"))
api_index = (site / "api" / "index.html").exists()

items = "\n".join(
    f'        <li><a href="guides/{html.escape(str(p))}">{html.escape(str(p))}</a></li>'
    for p in guides
)
api_link = (
    '      <p><a href="api/index.html">Zig autodoc API reference</a></p>'
    if api_index
    else "      <p>API autodoc was not generated.</p>"
)

(site / "index.html").write_text(
    f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>zynfer documentation</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 52rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.45; }}
    a {{ color: #0b57d0; }}
  </style>
</head>
<body>
  <h1>zynfer documentation</h1>
  <p>Generated from the repository. Markdown files are served as source;
     the API tree is Zig autodoc HTML.</p>
  <h2>API</h2>
{api_link}
  <h2>Guides</h2>
  <ul>
{items}
  </ul>
</body>
</html>
""",
    encoding="utf-8",
)
print(f"wrote {site / 'index.html'}")
PY

touch "$SITE/.nojekyll"
echo "site ready: $SITE"
