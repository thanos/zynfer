#!/usr/bin/python3
"""Compare ops-bench JSON against a committed sanity/regression baseline.

GitHub-hosted runners are noisy. This gate catches catastrophic regressions
(hangs, Debug-vs-Release mixups, missing ops), not 5% jitter.

The ops-bench CLI prints a human table then a line `json` followed by one
JSON object. This script accepts the full stdout or a bare object.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def extract_payload(text: str) -> dict:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "json" and i + 1 < len(lines):
            return json.loads(lines[i + 1])
    start = text.rfind("{")
    if start < 0:
        raise SystemExit("no JSON object found in ops-bench output")
    return json.loads(text[start:])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, help="ops-bench stdout file")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("bench/baselines/ci-linux-cpu.json"),
    )
    args = parser.parse_args()

    payload = extract_payload(args.output.read_text())
    baseline = json.loads(args.baseline.read_text())

    if payload.get("backend") != baseline.get("expect_backend", "cpu"):
        print(
            f"backend mismatch: got {payload.get('backend')!r} "
            f"want {baseline.get('expect_backend')!r}",
            file=sys.stderr,
        )
        return 1

    ops = {row["name"]: row for row in payload.get("ops", [])}
    failures = []
    for name, spec in baseline["ops"].items():
        row = ops.get(name)
        if row is None:
            failures.append(f"{name}: missing from run")
            continue
        cpu_ns = row.get("cpu_ns")
        if cpu_ns is None or cpu_ns <= 0:
            failures.append(f"{name}: cpu_ns missing or non-positive ({cpu_ns})")
            continue
        max_ns = spec["cpu_ns_max"]
        if cpu_ns > max_ns:
            failures.append(f"{name}: cpu_ns={cpu_ns} exceeds ceiling {max_ns}")
        if baseline.get("require_apple_null", True):
            if row.get("apple_metal_ns") not in (None,):
                failures.append(f"{name}: expected apple_metal_ns=null on this runner")

    if failures:
        print("bench regression check failed:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1

    print("bench regression check passed")
    for name, row in ops.items():
        print(f"  {name}: cpu_ns={row.get('cpu_ns')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
