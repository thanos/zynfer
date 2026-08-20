# Apple M-series backend

This is the development-laptop path: Zig host plus a thin Objective-C
Metal bridge. It is not the AMD production target.

## What runs

- CPU f32 oracle in `src/backends/cpu/ops.zig`
- Metal f32 kernels: add, mul, silu_mul, rmsnorm, softmax, matmul,
  matvec, rope (Qwen3-style split-half), causal GQA attention (`kv_len` ≤ 256)
- Stage 5: capability-gated `matmul_f32_simdgroup` / `matmul_f32_simdgroup_x4`
  (Apple7+, size-gated), `matvec_q8_f32` / `matmul_q8_f32` (per-row or
  per-tensor scale), Accelerate `vDSP_mmul` for large CPU matmul and matvec
- Stage 6: tiny-block **one command buffer / one wait**, persistent
  shared buffers, Metal-resident KV (`kv_append_f32` + GPU permutes),
  fused `add_rmsnorm_f32`, persistent int8 weights (`Q8DeviceWeights`).
  A/B with `ZYNFER_APPLE_BLOCK=baseline` (per-op waits).
- Stage 7: SME/SME2 **hardware probe** + Core ML **framework probe**;
  both inference paths **rejected** (see `zynfer stage7` /
  `bench/results/apple-stage7-dev-laptop.md`). Accelerate retained;
  do not claim AMX.
- Stage 8: attention `kv_len` ≤ **256**, opt-in signposts
  (`ZYNFER_SIGNPOSTS=1`: prefill/decode/weights_upload + encode/batch),
  `peak_rss_bytes` in block-bench + benchmark matrix, dual-`Gpu`
  concurrency (one owner thread per `Gpu`); ICB/fp16/extra tiny-block
  fusions **rejected** (`zynfer stage8`).
- Tiny transformer block: RMSNorm → QKV → RoPE → KV append →
  attention → O + residual → SwiGLU residual
- Separate `prefill` and `decode` entry points; decode does not allocate
  host tensors
- Capability dump: `zig build run -- caps` / `zynfer stage7` / `zynfer stage8`
- Op microbenchmarks: `zig build ops-bench`
- Block timings: `zig build block-bench`

`simdgroup_matrix` is **used** when hardware reports Apple7+ and the
matmul shape meets the measured size gate (`M·N·K ≥ 64³`, dims ≥ 8).
`matmul_f32_simdgroup_x4` is implemented and forceable
(`ZYNFER_MATMUL_PATH=simdgroup_x4`) but **not** auto-selected: at 256³
on M1 Max it was slower than the 1-SG kernel under the per-op wait
baseline. Smaller shapes keep the naive `matmul_f32` fallback.
Force with `ZYNFER_MATMUL_PATH=naive|simdgroup|simdgroup_x4`.

Measured selection evidence: `bench/results/apple-stage5-dev-laptop.md`.

## Commands

```bash
zig build test
zig build run -- caps
zig build run -- caps --backend cpu
zig build run -- backends
zig build ops-bench
zig build block-bench
ZYNFER_BACKEND=cpu zig build ops-bench
ZYNFER_BACKEND=cpu zig build block-bench
```

Invalid names fail:

```bash
zig build run -- caps --backend cuda   # exit 2
```

Runtime shader compile needs Apple's Metal Toolchain on current Xcode:

```bash
xcodebuild -downloadComponent MetalToolchain
xcrun -sdk macosx metal -c src/backends/apple/kernels.metal -o /tmp/zynfer_kernels.air
```

## Profiling

Labels are set on command buffers and encoders (kernel name). The
**ops** path submits one command buffer per op and waits. The **tiny-block
Stage 6** path batches ~20 encodes into one CB and waits once.

### Recipe: capture Stage 6 vs baseline on `block-bench`

Run on a Mac with the Metal Toolchain installed:

```bash
zig build -Doptimize=ReleaseSafe
./zig-out/bin/zynfer block-bench --backend apple
ZYNFER_APPLE_BLOCK=baseline ./zig-out/bin/zynfer block-bench --backend apple
```

1. Open **Instruments** → **Metal System Trace** (or **Game Performance**
   with Metal).
2. Choose the `zynfer` process (or launch from Instruments).
3. Record one `block-bench` run (~seconds). Stop.
4. In the timeline, select GPU / command-buffer lanes. Expect:
   - **Stage 6:** one command buffer per forward, many short dispatches,
     then a **single** `waitUntilCompleted`;
   - **baseline:** a gap after every dispatch in
     `zynfer_mtl_encode_and_wait`.
5. With `ZYNFER_SIGNPOSTS=1`, os_signpost intervals also label
   `prefill` / `decode` / `weights_upload` and the encode/batch waits
   under subsystem `com.zynfer.metal` / category `stage8`.

| Signal | Meaning |
| --- | --- |
| One wait gap per forward (Stage 6) | CB batching working |
| Serial gaps between every kernel (baseline) | Per-op `waitUntilCompleted` |
| Short GPU kernels vs long wall time (baseline) | Launch/sync bound |
| Nested `prefill`/`decode` around batch wait | Session-level signposts |

Measured A/B: `bench/results/apple-stage6-dev-laptop.md` (~8× decode).

### Case study (baseline, not an optimization)

- **Hypothesis:** per-op `Gpu.init` (shader compile) would dominate
  `ops-bench` if the device were created inside the timed loop.
- **Change:** compile the library once per `ops-bench` process; time
  init separately as `metal_device_create_plus_shader_compile_ns`.
- **Correctness:** `zig build test` still compares every Metal op to CPU.
- **Decision:** keep one-shot compile for ops microbenches.

### Case study (Stage 6: one wait + resident KV)

- **Hypothesis:** Metal `block-bench` was slower than CPU mainly because
  each tiny-block op paid encode + alloc + `waitUntilCompleted`.
- **Change:** persistent weights/scratch/KV buffers; GPU permute +
  `kv_append_f32`; `batchBegin`/`batchCommit` → one wait per forward.
- **Evidence:** ReleaseSafe A/B in
  `bench/results/apple-stage6-dev-laptop.md` (~8.3× decode ns/token).
- **Decision:** Stage 6 is the default tiny-block path; keep
  `ZYNFER_APPLE_BLOCK=baseline` for regression A/B.

## Prefill vs decode (tiny block)

`src/model/tiny_block.zig` owns the schedule and the **host** KV layout
(CPU path / baseline A/B). The default Apple path is
`src/backends/apple/block.zig`: persistent shared weights/scratch,
**Metal-resident KV** (`kv_append_f32` + GPU permutes), and **one**
command buffer / one `waitUntilCompleted` per prefill or decode
(`path=batched_resident_kv_fused`). Set `ZYNFER_APPLE_BLOCK=baseline`
to force the Stage 4/5 per-op schedule (host permute/append + upload +
wait per kernel) for differential A/B only.

Fixture limits:

| Limit | Value | Notes |
| --- | ---: | --- |
| `max_seq` (CPU + host KV) | 32 | Prefill/decode tested through full length |
| Metal `attention_f32` `kv_len` | ≤ 256 | Thread-local scores (Stage 8); larger → `Unsupported` |
| Host alloc in decode/prefill after init | 0 | Scratch preallocated; multi-reset stress tested |

See [Why Metal was slower…](#why-metal-was-slower-on-the-tiny-block-stage-4-baseline)
for the Stage 4 baseline diagnosis; Stage 6 closed the wait/alloc story
for this fixture.

## Why Metal was slower on the tiny block (Stage 4 baseline)

Historical diagnosis of the **per-op** path (`ZYNFER_APPLE_BLOCK=baseline`
/ Stage 4 schedule). The **default** Stage 6 path no longer waits per
kernel and no longer re-uploads KV each step—see
[What changed the picture](#what-changed-the-picture-stage-5-6).
Metal remains slower than the CPU oracle on this microscopic fixture
because FLOPs are tiny; Stage 6 removed wait dominance (~8× vs baseline),
not CPU absolute wins.

The Stage 4 path is a **correctness baseline**, not a tuned decode
pipeline. The GPU does almost no useful math per launch; most of the
time goes to setup, copies, and waiting.

This is expected. It is not a bug and does not mean the M-series GPU
is bad at inference. It means that schedule paid full driver and
sync cost on every op while the tensors were too small for compute to
matter.

### 1. Every op pays full launch + wait (baseline only)

Each per-op Metal call goes through `zynfer_mtl_encode_and_wait` in
`src/backends/apple/bridge.m`, which ends with a blocking wait:

```objc
[cb commit];
[cb waitUntilCompleted];
```

One decode token on the baseline schedule runs the full block in
`src/model/tiny_block.zig`:

- RMSNorm (attention)
- matmul ×3 (Q, K, V)
- RoPE ×2 (Q, K)
- causal GQA attention
- matmul (output projection)
- add (residual)
- RMSNorm (MLP)
- matmul ×2 (gate, up)
- SiLU × up
- matmul (down)
- add (final residual)

That is roughly **14–15 separate Metal submissions per forward**, each
with its own command buffer and `waitUntilCompleted`. Stage 6 collapses
this to **one** wait.

At roughly 0.3–0.7 ms per op (consistent with `ops-bench` on small
kernels), launch + wait alone account for about **4–6 ms per decode
token** on the baseline path.

### 2. Fresh GPU buffers on every op (baseline / ops path)

Each call in the per-op `src/backends/apple/ops.zig` helpers allocates
new shared buffers, copies host data in, runs the kernel, copies out,
then frees. The Stage 6 Session keeps persistent device weights/scratch/KV.

### 3. The work is too small for the GPU

The fixture is deliberately tiny:

- hidden = 8, head_dim = 4, inter = 16
- decode = **1 token** at a time

A single-token matmul is a handful of FMAs. The CPU oracle runs that in
microseconds. Even one-CB Metal still pays encode + sync overhead that
dominates microscopic FLOPs.

### 4. Attention re-uploads the whole KV cache (baseline only)

On the baseline path, attention reads the **entire** growing K/V cache
from host tensors and uploads them again every step. Stage 6 keeps KV
Metal-resident via `kv_append_f32`.

### 5. Some work never hits the GPU (baseline only)

On the baseline schedule, layout permute and KV `append` stay on the
CPU between kernels. Stage 6 moves permutes and append onto the GPU
inside the single command buffer.

### What the numbers mean (Stage 4 baseline)

| Observation | Interpretation |
| --- | --- |
| Metal ~500× slower than CPU on decode (Stage 4) | ~14 synchronous micro-kernel launches on microscopic tensors |
| `ops-bench` Metal often ~2× slower even on 4096-element adds | Launch + wait dominate before compute does |
| ReleaseSafe helps CPU much more than Metal | Bottleneck is encode/sync/buffer churn, not scalar loops |
| Prefill gap similar to decode | Same per-op pattern; prefill just runs more tokens in one forward |

**Bottom line (historical):** the Stage 4 gap measured **14+ synchronous
micro-kernel launches on tiny tensors**, not “M1 Max matmul vs scalar
CPU.” Stage 4 proved correctness and schedule; Stage 6 is the retained
tiny-block throughput path.

### What changed the picture (Stage 5–6)

Delivered on the tiny-block fixture:

- **Persistent device buffers** — weights, scratch, and KV (Stage 6)
- **One command buffer per forward** — one wait, not ~15 (Stage 6)
- **Metal-resident KV** — `kv_append_f32`; attention reads cache in place
- **Optional `simdgroup_matrix`** — size-gated (Stage 5)

Still needed for a real decode story:

- **Realistic shapes** — enough FLOPs that Metal beats scalar CPU
- **Qwen-scale weights / logits** — checkpoint loader and golden tests

Stage 6 A/B: `bench/results/apple-stage6-dev-laptop.md`.

## Fallback matrix

| Path | Status |
| --- | --- |
| CPU scalar f32 | implemented; oracle |
| CPU Accelerate vDSP matmul | size-gated on macOS (`M·N·K ≥ 64³`); differential vs scalar |
| CPU Accelerate vDSP matvec | size-gated on macOS (`M·K ≥ 256²`); differential vs scalar |
| Metal naive f32 | implemented; differentially tested |
| Metal `simdgroup_matrix` matmul | Apple7+; auto when M·N·K≥64³; `_x4` force-only (slower at 256³) |
| Metal int8 GEMV/GEMM (`matvec_q8_f32` / `matmul_q8_f32`) | explicit API; fair (prepacked) benches; not auto over f32 |
| Metal fused / batched tiny-block | Stage 6 one CB/wait + resident KV; ~8× vs per-op baseline |
| Metal attention long context | Stage 8: `kv_len` ≤ 256 |
| SME / SME2 | hardware probed (`FEAT_SME`); kernels **rejected** (Stage 7) |
| Core ML / ANE | framework probed; inference path **rejected** (no verified subgraph) |
| fp16 / bf16 Metal | **rejected** Stage 8 (`Unsupported`); kernels remain f32 |
| HIP transformer ops | not implemented (probe only) |

`zig build run -- caps` and `zynfer stage7` print the Stage 7 ledger.
Forced `--backend apple` on a non-macOS build exits 2.
`ZYNFER_FORCE_SME=1` / `ZYNFER_FORCE_COREML=1` exit 2 (paths not retained).

## Deferred work (mapped to future stages)

Stage 6 is **closed for the tiny-block fixture**. Nothing below is
silent debt—each item has an owner stage. Keep this table in sync when
Apple Stage 7/8 or curriculum Stages 10–12 / 16 land.

### Done in Stage 6 (do not reopen)

| Item | Evidence |
| --- | --- |
| One CB / one wait + persistent weights/scratch/KV | `apple.block`; ~8× vs baseline |
| Metal-resident KV + GPU permutes | `kv_append_f32`, permute kernels |
| `add_rmsnorm_f32` + existing `silu_mul` | tiny-block fused path |
| Persistent int8 for ops | `Q8DeviceWeights` / `*Q8Persistent` |

### Closed — Apple Stage 7

| Item | Notes |
| --- | --- |
| **SME / SME2** | **done (rejected)** — `cpu.sme` detects FEAT_SME/SME2; kernels not retained (`bench/results/apple-stage7-dev-laptop.md`) |
| **Core ML / ANE** | **done (rejected)** — framework probe only; no verified ANE subgraph |

### Closed — Apple Stage 8

| Item | Notes |
| --- | --- |
| **Fused vs `baseline` numerical A/B test** | **done** — Stage 6 |
| **Attention `kv_len` cap** | **done** — raised to **256**; tested at 96 |
| **Signposts** | **done** — `ZYNFER_SIGNPOSTS=1` (`prefill`/`decode`/`weights_upload` + encode/batch) |
| **Peak RSS** | **done** — `peak_rss_bytes` in block-bench JSON; Peak memory in `docs/benchmarks.md` |
| **Stress / cancel paths** | **done** — repeated Session + batch abort + dual-`Gpu` concurrency |
| **fp16 / bf16 Metal** | **rejected** — `Unsupported` stubs |
| **Reusable execution encoding** (ICB) | **rejected** — see stage8 results |
| **Further MSL fusions** | **rejected** for tiny-block; → Stage 16 |
| **Int8 weights in tiny-block Session** | **rejected** for now; ops path keeps `Q8DeviceWeights` |
| **Benchmark matrix TTFT/tok/s fill-in** | After Stages 10–12 produce tokens |
| **Energy/token** | N/A — not measured |

Stage 8 ledger: `bench/results/apple-stage8-dev-laptop.md`.

### Still open — curriculum (not Apple-6)

| Item | Lands in |
| --- | --- |
| **Qwen loader / artifact** | Stage 10 |
| **Full Qwen forward + golden logits** | Stage 11 |
| **Tokenizer / sampling** (real TTFT) | Stage 12 |
| **Qwen-scale / HIP fusion ledger** | Stage 16 |

Stage 5 matrix paths: `bench/results/apple-stage5-dev-laptop.md`.

Stage 6 block A/B: `bench/results/apple-stage6-dev-laptop.md`.

Historical wait-bound writeup (pre-Stage 6):
[Why Metal was slower…](#why-metal-was-slower-on-the-tiny-block-stage-4-baseline).

### Still true

- Tiny block is a synthetic residual stream, not Qwen3-0.6B.
- Metal attention hard-caps `kv_len` at **256**; fixture `max_seq` is 32.
- Energy/token is not claimed; `peak_rss_bytes` is reported when available.
- Stage 6 Metal is still slower than scalar CPU on this tiny shape.

## Measured on this development laptop (2026-08-19)

Apple M1 Max, Darwin 25.5.0, Zig 0.16.0, `zig build ops-bench`,
warmup 2, iters 8. Times include per-op buffer fill and wait.

| Op | CPU ns/iter | Metal ns/iter |
| --- | ---: | ---: |
| add_f32_4096 | 278182 | 550218 |
| silu_mul_f32_4096 | 308057 | 724234 |
| matvec_f32_256x256 | 949156 | 1227098 |
| matmul_f32_32x64x64 | 863552 | 674625 |

Shader compile + device create: 112987125 ns (~113 ms). Full record:
`bench/results/apple-ops-dev-laptop.md`.

Tiny-block `zig build block-bench` on the same machine (Debug, warmup 1,
iters 4, 8 prefill tokens, 8 decode steps):

| Path | Prefill ns | Decode ns/token |
| --- | ---: | ---: |
| CPU | 43416 | 10079 |
| Metal | 6012323 | 5855778 |

Full record: `bench/results/apple-block-dev-laptop.md`. For why Metal
was orders of magnitude slower on the Stage 4 baseline path, see
[Why Metal was slower on the tiny block (Stage 4 baseline)](#why-metal-was-slower-on-the-tiny-block-stage-4-baseline).
Stage 6 (~8× vs that baseline) is the retained tiny-block schedule.
