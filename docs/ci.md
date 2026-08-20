# CI / CD

GitHub Actions is the required gate. Local parity:

```bash
zig build ci -Dhip=off
```

That runs format check, unit/numerical/smoke tests, CLI integration tests,
and Zig autodoc. Coverage and benches are extra jobs because they need
kcov or a ReleaseSafe timed run.

## Workflows

| Workflow | File | What it enforces |
| --- | --- | --- |
| CI | `.github/workflows/ci.yml` | `zig fmt --check`, `zig build test`, `zig build integration` |
| Coverage | `.github/workflows/coverage.yml` | kcov over `src/` on Ubuntu; HTML artifact; optional Codecov |
| Bench | `.github/workflows/bench.yml` | ReleaseSafe CPU `ops-bench` vs `bench/baselines/ci-linux-cpu.json` |
| Docs | `.github/workflows/docs.yml` | autodoc + markdown site; GitHub Pages deploy from `main` |

HIP is always off in CI (`-Dhip=off`). GitHub-hosted runners do not have
ROCm. Apple Metal GPU tests skip on `macos-latest` via
`ZYNFER_SKIP_APPLE_GPU=1` because those images often lack the Metal
shader toolchain. Compile still links Metal on macOS. Full Metal
differential tests run on a developer Mac: `zig build test`.

## Regression tests

`zig build test` is the numerical/invariant regression suite: CPU oracle
ops, tensor construction, backend selection (unknown names fail), and
Metal-vs-CPU when the GPU path is enabled.

## Integration tests

`zig build integration` installs `zynfer` and checks CLI contracts:

- `help` / `env` / `caps --backend cpu` / `backends` exit 0
- `--backend cuda` and unknown commands exit 2
- `ops-bench --backend cpu` prints a JSON object with `cpu_ns`
- `block-bench --backend cpu` prints a JSON object with `cpu_prefill_ns`

## Coverage

Ubuntu 24.04 (`ubuntu-latest`) does not ship `kcov` in apt. CI builds
[kcov v43](https://github.com/SimonKagstrom/kcov) from source via
`scripts/ci/install-kcov.sh` and caches `.kcov-prefix`.

```bash
# Linux
bash scripts/ci/install-kcov.sh
bash scripts/ci/coverage.sh
```

Reports include `src/` only. There is no coverage quota yet; the job
publishes HTML so gaps are visible. Codecov upload is best-effort
(`continue-on-error`) until `CODECOV_TOKEN` is set.

## Regression benchmarking

```bash
zig build -Dhip=off -Doptimize=ReleaseSafe
ZYNFER_BACKEND=cpu ./zig-out/bin/zynfer ops-bench | tee /tmp/ops-bench.txt
python3 scripts/ci/compare-bench.py /tmp/ops-bench.txt
```

The baseline is a **sanity ceiling** for ubuntu-latest, not a 5%
performance gate. Runner noise is large. Tighten
`bench/baselines/ci-linux-cpu.json` from CI artifacts once they are
stable. Do not compare Debug CPU times to ReleaseSafe GPU times.

TTFT / tok/s are N/A until a model generates tokens.

## Documentation generation

```bash
bash scripts/build-docs-site.sh
# output: zig-out/site/
```

`zig build docs` emits Zig autodoc to `zig-out/docs/api`. The site
script copies markdown guides next to that tree. On `main`, the Docs
workflow deploys `zig-out/site` to GitHub Pages. Enable Pages in the
repository settings (source: GitHub Actions).

## Dependabot

`.github/dependabot.yml` opens weekly PRs for GitHub Actions updates.
