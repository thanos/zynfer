# Benchmarks

Zynfer records measurements so later "faster" claims have a baseline.
Do not invent TTFT or tok/s numbers before a model generates tokens.

## Required metadata

Every recorded result should include:

```text
date
host
OS / kernel
device (Apple GPU name or AMD marketing name)
backend / execution path
Zig version
command
warmup
iterations
what was measured
```

Do not compare engines casually. Same model, precision, prompt, generated
length, device, concurrency, and metric definition are required.

## Commands

```bash
zig build bench                 # HIP enumeration timing (AMD host)
zig build ops-bench             # CPU vs Apple op microbenchmarks
zig build run -- ops-bench --backend cpu
```

`ops-bench` prints a human table plus a `json` line. Apple times include
per-op shared-buffer fill and `waitUntilCompleted`. Init/shader compile
is reported separately as `metal_device_create_plus_shader_compile_ns`.

On a machine without HIP, `zig build bench` still runs and reports that
GPU timing is unavailable.

## Where results live

Committed templates:

- `bench/results/stage0-dev-laptop.md`
- `bench/results/apple-ops-dev-laptop.md`

Local captures that should not be committed:

- `bench/results/local/`

## Inference metrics (not yet measurable)

Prefill and decode are reported separately once the engine exists.

| Metric | Status |
| --- | --- |
| Time to first token | N/A — no tokenizer or forward pass |
| Prefill tok/s | N/A |
| Decode tok/s | N/A |
| ITL p50/p95/p99 | N/A |
| Energy/token | N/A |
| Peak model/KV/scratch memory | N/A — no loaded weights |

Interactive (later): time to first token, inter-token latency,
single-request decode tok/s.

Throughput (later): aggregate committed tok/s, concurrency scaling.

## Profiling

See `docs/apple-backend.md` for Instruments / Metal System Trace steps
on the Mac path. AMD ROCProfiler instructions belong with the HIP kernel
stages, not this baseline.

## Final path matrix

| Path | Device/API | Model/quant | Workload | TTFT | Prefill tok/s | Decode tok/s | ITL p50/p95/p99 | Peak memory | Effective bandwidth | Energy/token | Status/notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CPU reference | Zig scalar f32 | none | op microbench | N/A | N/A | N/A | N/A | N/A | N/A | N/A | correctness oracle |
| CPU optimized | Accelerate BNNS/vecLib | — | — | N/A | N/A | N/A | N/A | N/A | N/A | N/A | not built |
| CPU optimized | SME/SME2 | — | — | N/A | N/A | N/A | N/A | N/A | N/A | N/A | not implemented |
| Metal baseline | naive MSL f32 | none | op microbench | N/A | N/A | N/A | N/A | N/A | N/A | N/A | implemented; see apple-ops result |
| Metal optimized | simdgroup_matrix | — | — | N/A | N/A | N/A | N/A | N/A | N/A | N/A | capability discovered; kernel not shipped |
| Metal fused | fused MSL | — | — | N/A | N/A | N/A | N/A | N/A | N/A | N/A | not built |
| Core ML/ANE hybrid | — | — | — | N/A | N/A | N/A | N/A | N/A | N/A | N/A | not built |
| HIP | RDNA 4 | — | device enum | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Stage 0 probe only |
