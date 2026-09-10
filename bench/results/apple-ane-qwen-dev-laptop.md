# Apple Stage M7 — ANE / Core ML at Qwen scale (dev laptop)

Stage 7 rejected Core ML/ANE as a tiny-block experiment. Stage M7 reopens
the question now that a real Qwen Metal engine exists (M0–M6) and **closes
it** with a final Qwen-scale **REJECT**. No zombie half-support.

```text
date:              2026-08-24
host:              MacBook-Pro (Apple Silicon)
OS:                Darwin 25.5.0 (arm64)
Zig:               0.16.0
commands:
  zig build test -Dhip=off
  zig build stageM7 -Dhip=off
  ./zig-out/bin/zynfer stageM7
  ./zig-out/bin/zynfer coreml-smoke tools/fixtures/coreml_toy.mlpackage
  ZYNFER_FORCE_COREML=1 ./zig-out/bin/zynfer caps   # expect exit 2
```

## Probe results

| Signal | Value |
| --- | --- |
| Core ML framework linked | yes |
| `MLModelConfiguration` All / CPUAndNeuralEngine | ok |
| `MLState` class available | yes (Darwin ≥ 24) |
| Qwen `.mlmodel` / `.mlpackage` | **no** |
| Toy fixture `tools/fixtures/coreml_toy.mlpackage` | **yes** (load smoke only) |
| ANE execution verified (Instruments) | **no** |
| Core ML inference path retained | **no** |

## Optional polish — toy load smoke + xctrace

Does **not** clear retain. Confirms the bridge can compile/load/predict and
that the Instruments recipe is runnable.

### Toy model

```bash
# Dev-time only (needs coremltools):
ASDF_PYTHON_VERSION=3.12.9 python3 tools/fixtures/make_coreml_toy.py
# Checked-in fixture is already present (~12K .mlpackage).

./zig-out/bin/zynfer coreml-smoke tools/fixtures/coreml_toy.mlpackage
```

Observed (2026-08-24): `status=0`, `load_ok=yes`, `predict_ok=yes`,
`compute_units=CPUAndNeuralEngine`,
`y≈[1.0205, 1.1670, 1.3125, 1.4590]` (ones input). Bridge uses
`MLModel.compileModel(at:)` then load. **ANE still unverified.**

### xctrace recipe (run once)

```bash
xctrace record --template 'Core ML' --time-limit 15s \
  --output /tmp/zynfer-coreml-toy.trace --target-stdout - \
  --launch -- ./zig-out/bin/zynfer coreml-smoke tools/fixtures/coreml_toy.mlpackage

xctrace export --input /tmp/zynfer-coreml-toy.trace --toc
# Optional deeper look (GUI Instruments is authoritative for ANE lanes):
# open /tmp/zynfer-coreml-toy.trace
```

Observed (2026-08-24): recording completed (~1.4s, target exit 0). TOC lists
`coreml-os-signpost` and `ane-hw-intervals-internal` tables; xpath export of
those schemas returned **empty row sets** for this tiny matmul. That is **not**
ANE confirmation — open the `.trace` in Instruments to inspect Neural Engine
lanes. Probe keeps `ane_execution_verified=false`. No Metal A/B (no Qwen Core
ML path).

## Candidates

| Candidate | Decision | Why |
| --- | --- | --- |
| (a) Fixed-length prefill subgraph | SKIP | No Qwen converter/model — building one *is* most of a fair A/B. Handoff/repack vs resident Metal weights taxes TTFT (unified memory ≠ free Metal↔Core ML). |
| (b) Stateful decode (`MLState`) | SKIP | API present; no exported Qwen KV graph; mutable KV / dynamic shapes; no Instruments ANE proof. Presence ≠ engine. |
| (c) Nothing | **CHOSEN** | All four retain criteria unmet; starting (a)/(b) was a multi-month bet against Metal |

SKIP means we did **not** implement a Qwen Core ML path and lose a race. It
means we declined to start that project. See tutorial 21 (“Why we did not
try (a) and (b)” and “Unified memory ≠ free handoff”).

## Retain criteria checklist

| Criterion | Met? |
| --- | --- |
| Instruments confirms ANE (not CPU/GPU fallback) | **no** — toy xctrace run ≠ Qwen ANE proof |
| e2e TTFT or tok/s > M6 Metal @ equal quality | **N/A** — no Core ML Qwen path to A/B |
| I/O copies do not erase the win | Would hurt vs resident Metal (layout/ownership handoff, not PCIe) |
| Graph amortizes compilation | No stable Qwen Core ML graph |

## Instruments evidence

**Toy recipe exercised; Qwen ANE evidence not collected.** Retain stays closed.

### How to collect if reopened later

1. Dev-time: `coremltools` → fixed-shape Qwen subgraph `.mlpackage`.
2. Load with `MLComputeUnitsCPUAndNeuralEngine` (see `zynfer_coreml_smoke`).
3. `xctrace record --template 'Core ML' … --launch -- …` (recipe above).
4. Confirm work on **ANE** in Instruments GUI; screenshot / note intervals.
5. A/B vs `qwen-bench` Metal (`bf16` / `int8`).
6. Retain only if all four criteria above pass.

The probe will never set `ane_execution_verified` from config enums alone.

## E2E A/B vs M6 Metal

**N/A.** No Core ML Qwen inference path. Metal remains the measured engine
(`qwen-bench`, `mem-report`, Stage M6 ITL ledger).

## Decision

| Item | Result |
| --- | --- |
| Core ML / ANE at Qwen scale | **REJECT (final)** |
| Retained Apple path | Metal M0–M6 + Accelerate Stage 5 |
| `ZYNFER_FORCE_COREML` | exit **2** (loud) |

**Not in scope for M7:** becoming a Core ML–specialized product (“NInfer
for Core ML”). That would be a mission change. M8’s “NInfer philosophy”
milestone is a **registered Qwen3-4B** on Metal, not ANE.

## Next

Stage M8 — Apple capstone (`docs/stages/M8-apple-capstone.md`): quantized
Qwen3-4B + final matrix → **Apple-complete**.

## Files

- `src/backends/apple/coreml.zig` + `coreml_bridge.[hm]` (`probe` + `smoke`)
- CLI: `zynfer stageM7`, `zynfer coreml-smoke [PATH]`
- Fixture: `tools/fixtures/coreml_toy.mlpackage`, `make_coreml_toy.py`
- Docs: `docs/stages/M7-ane-coreml-qwen.md`, tutorial 21
  (plain English, skip rationale, unified-memory handoff)
