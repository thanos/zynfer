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
zig build block-bench           # tiny-block prefill + decode timings
zig build run -- ops-bench --backend cpu
zig build run -- block-bench --backend cpu
```

`ops-bench` and `block-bench` print a human table plus a `json` line.
Apple times include per-op shared-buffer fill and `waitUntilCompleted`.
Init/shader compile is reported separately as
`metal_device_create_plus_shader_compile_ns`.

`block-bench` times a synthetic tiny transformer block (not Qwen).
Prefill (8 tokens) and decode (8 steps) are reported separately.

When Metal is much slower than CPU on this fixture, that is expected for
the Stage 4 baseline. See
[Why Metal is slower on the tiny block](apple-backend.md#why-metal-is-slower-on-the-tiny-block).

On a machine without HIP, `zig build bench` still runs and reports that
GPU timing is unavailable.

## Where results live

Committed templates:

- `bench/results/stage0-dev-laptop.md`
- `bench/results/apple-ops-dev-laptop.md`
- `bench/results/apple-block-dev-laptop.md`
- `bench/results/apple-stage5-dev-laptop.md`
- `bench/results/apple-stage6-dev-laptop.md`
- `bench/results/apple-stage7-dev-laptop.md`
- `bench/results/apple-stage8-dev-laptop.md`

Local captures that should not be committed:

- `bench/results/local/`

## Inference metrics (tiny-block fixture only)

Vocabulary TTFT still N/A — no tokenizer (curriculum Stage 12; was an
Apple Stage 6 gate item deferred for lack of a loaded model). The tiny
block reports wall time for a synthetic residual stream. Remaining
post–Stage 6 work is mapped in `docs/apple-backend.md` (Apple-7/8,
Stages 10–12 / 16)—do not treat it as dropped Stage 6 scope.

| Metric | Status |
| --- | --- |
| Time to first token | N/A — no tokenizer or Qwen forward pass |
| Prefill tok/s | N/A as an LLM metric; `block-bench` reports `cpu_prefill_ns` / `apple_prefill_ns` |
| Decode tok/s | N/A as an LLM metric; `block-bench` reports `*_decode_ns_per_token` |
| ITL p50/p95/p99 | N/A |
| Energy/token | N/A |
| Peak model/KV/scratch memory | N/A — no loaded Qwen weights |

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
| CPU reference | Zig scalar f32 | tiny-block f32 | prefill 8 / decode 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | `block-bench`; ns not tok/s |
| CPU optimized | Accelerate vDSP | f32 | matmul ≥64³ / matvec ≥256² | N/A | N/A | N/A | N/A | N/A | N/A | N/A | size-gated; stage5 result |
| Metal baseline | naive MSL f32 | none | op microbench | N/A | N/A | N/A | N/A | N/A | N/A | N/A | implemented; see apple-ops result |
| Metal baseline | naive MSL f32 | tiny-block f32 | prefill 8 / decode 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | path=`baseline_per_op` (`ZYNFER_APPLE_BLOCK=baseline`) |
| Metal optimized | simdgroup / x4 | f32 | matmul 64³ / 256³ | N/A | N/A | N/A | N/A | N/A | N/A | N/A | auto ≥64³; x4 force-only |
| Metal int8 GEMV/GEMM | matvec/matmul_q8_f32 | int8×f32 scale | 256² / 128³ prepacked | N/A | N/A | N/A | N/A | N/A | N/A | N/A | explicit API; not auto over f32 |
| Metal fused | Stage 6 `batched_resident_kv_fused` | tiny-block f32 | prefill 8 / decode 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | ~8× vs `baseline_per_op`; stage6 result |
| Core ML/ANE hybrid | Core ML probe | — | — | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Stage 7: framework ok; path **rejected** |
| CPU optimized | SME/SME2 | — | — | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Stage 7: FEAT_SME probed; kernels **rejected** |
| HIP | RDNA 4 | — | device enum | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Stage 0 probe only |
