#!/usr/bin/env bash
# Run installed test binaries under kcov. Linux CI only.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if ! command -v kcov >/dev/null 2>&1; then
    echo "kcov is required. On Ubuntu: sudo apt-get install -y kcov" >&2
    exit 1
fi

zig build -Dhip=off -Dinstall-tests
OUT="$ROOT/zig-out/coverage"
rm -rf "$OUT"
mkdir -p "$OUT"

for bin in test-unit test-numerical test-smoke; do
    path="$ROOT/zig-out/bin/$bin"
    if [[ ! -x "$path" ]]; then
        echo "missing $path (did -Dinstall-tests install it?)" >&2
        exit 1
    fi
    kcov --include-pattern="$ROOT/src/" \
        --exclude-pattern=".zig-cache,/usr/" \
        "$OUT" \
        "$path"
done

# kcov writes cobertura.xml and an index. Summarize line coverage if present.
/usr/bin/python3 - <<'PY'
import json, os, pathlib, sys, xml.etree.ElementTree as ET
root = pathlib.Path("zig-out/coverage")
xmls = list(root.rglob("cobertura.xml")) + list(root.rglob("coverage.xml"))
pct = None
for path in xmls:
    try:
        tree = ET.parse(path)
        cov = tree.getroot()
        rate = cov.attrib.get("line-rate")
        if rate is not None:
            pct = float(rate) * 100.0
            break
    except Exception:
        pass
idx = root / "index.json"
if pct is None and idx.exists():
    data = json.loads(idx.read_text())
    pct = float(data.get("percent_covered", data.get("percent", 0)))
if pct is None:
    print("coverage percent: unknown (see zig-out/coverage)")
    sys.exit(0)
print(f"coverage percent: {pct:.1f}")
if "GITHUB_STEP_SUMMARY" in os.environ:
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh:
        fh.write(f"## Coverage\n\nLine coverage (src/): **{pct:.1f}%**\n")
PY
