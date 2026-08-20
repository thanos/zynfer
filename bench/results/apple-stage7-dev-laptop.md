# Apple Stage 7 — SME / SME2 and Core ML / ANE experiments

Stage 7 attempts optional CPU SME and Core ML/ANE paths, then **retains
only** what is measured and maintainable. On this development laptop both
experimental inference paths are **rejected** with explicit probes and
loud force failures. Metal Stage 6 and Accelerate (Stage 5) remain.

```text
date:              2026-08-20
host:              MacBook-Pro-2.local
OS:                Darwin 25.5.0 (arm64)
chip:              Apple M1 Max
Zig:               0.16.0
command:           zig build && ./zig-out/bin/zynfer stage7
```

## Probe results

| Signal | Value |
| --- | --- |
| `hw.optional.arm.FEAT_SME` | 0 |
| `hw.optional.arm.FEAT_SME2` | 0 |
| Zig target `Feature.sme` | off (host default) |
| Core ML framework linked | yes |
| `MLModelConfiguration` All / CPUAndNeuralEngine | ok |
| ANE execution verified (Instruments) | **no** |
| SME inference path retained | **no** |
| Core ML inference path retained | **no** |

```text
./zig-out/bin/zynfer stage7
ZYNFER_FORCE_SME=1 ./zig-out/bin/zynfer stage7     # exits 2
ZYNFER_FORCE_COREML=1 ./zig-out/bin/zynfer caps     # exits 2
```

## Case studies

### SME / SME2

- **Hypothesis:** M4+ SME could beat Accelerate/Metal on some CPU shapes.
- **Evidence:** This host is M1 Max (`FEAT_SME=0`). Zig 0.16 exposes
  `Feature.sme*` for the target, but there is no maintainable public
  Zig/Clang SME kernel path in-tree. Prompt forbids brittle assembly.
- **Change:** `cpu.sme.probe()` + `matmul`/`matvec` → `Unsupported`.
- **Decision:** **REJECT** kernels. Keep detection so M4+ hosts report
  hardware honestly without claiming execution.

### Core ML / ANE

- **Hypothesis:** Offload a stable subgraph to ANE via Core ML.
- **Evidence:** Framework links; compute-unit enums accept
  `CPUAndNeuralEngine`. No `.mlmodel`, no Qwen weights, no Instruments
  confirmation of ANE placement. Tiny-block attention/KV is a poor ANE
  candidate (prompt). End-to-end opportunity waits on Stages 10–12.
- **Change:** `apple.coreml` bridge probe; ops → `Unsupported`.
- **Decision:** **REJECT** inference path. Do not report ANE wins.

### Accelerate / AMX attribution

- **Evidence:** Stage 5 size-gated `vDSP_mmul` is measured and tested.
- **Decision:** **RETAIN** Accelerate. Do **not** claim direct AMX;
  Apple may use the matrix coprocessor inside Accelerate privately.

## Path matrix rows (Stage 7)

| Path | Status |
| --- | --- |
| CPU optimized / SME/SME2 | detected when present; kernels **disabled** |
| Core ML/ANE hybrid | framework probe only; path **disabled** |
| CPU Accelerate vDSP | **retained** (Stage 5) |
| Metal fused tiny-block | **retained** (Stage 6) |

## Files

- `src/backends/cpu/sme.zig`
- `src/backends/apple/coreml.zig` + `coreml_bridge.[hm]`
- CLI: `zynfer stage7`, caps Stage 7 section, `ZYNFER_FORCE_{SME,COREML}`
