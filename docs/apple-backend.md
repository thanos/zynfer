# Apple M-series backend

This is the development-laptop path: Zig host plus a thin Objective-C
Metal bridge. It is not the AMD production target.

## What runs

- CPU f32 oracle in `src/backends/cpu/ops.zig`
- Metal f32 kernels: add, mul, silu_mul, rmsnorm, softmax, matmul,
  matvec, rope (Qwen3-style split-half), causal GQA attention (`kv_len` ≤ 64)
- Tiny transformer block: RMSNorm → QKV → RoPE → host KV append →
  attention → O + residual → SwiGLU residual
- Separate `prefill` and `decode` entry points; decode does not allocate
  host tensors
- Capability dump: `zig build run -- caps`
- Op microbenchmarks: `zig build ops-bench`
- Block timings: `zig build block-bench`

`simdgroup_matrix` hardware may be present on Apple7+ GPUs. This build
does **not** launch a simdgroup matrix kernel. The capability flag is
discovery only.

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
baseline submits one command buffer per op and waits.

1. **Time Profiler** (Instruments): CPU cost of encoding, buffer
   alloc/fill, and `waitUntilCompleted`. Hypothesis for this baseline:
   launch + wait dominate small ops.
2. **Metal System Trace**: queue occupancy and the gap after each wait.
   Expect serial kernels, not overlap.
3. **Xcode GPU capture** / Metal debugger: kernel duration vs memory
   traffic for `matmul_f32` / `matvec_f32`.

Signposts are not wired yet. Cheap compile-time instrumentation can be
added around `zynfer_mtl_encode_and_wait` without changing the op API.

### Case study (baseline, not an optimization)

- **Hypothesis:** per-op `Gpu.init` (shader compile) would dominate
  `ops-bench` if the device were created inside the timed loop.
- **Change:** compile the library once per `ops-bench` process; time
  init separately as `metal_device_create_plus_shader_compile_ns`.
- **Correctness:** `zig build test` still compares every Metal op to CPU.
- **Decision:** keep one-shot compile; still wait after every kernel.
  Removing the wait is deferred until a fused/overlapped decode path
  exists to measure overlap against.

## Prefill vs decode (tiny block)

`src/model/tiny_block.zig` owns the schedule and the host KV cache.
`src/backends/apple/block.zig` runs the same ops on a persistent Metal
device. Layout permute and cache append stay on the host. Each Metal
kernel still waits. Attention uploads the full host K/V tensors every
call; that is not a Metal-resident cache.

See [Why Metal is slower on the tiny block](#why-metal-is-slower-on-the-tiny-block)
for a breakdown of the measured gap vs the CPU oracle.

## Why Metal is slower on the tiny block

Metal is much slower than the CPU oracle on `block-bench` because the
Stage 4 path is a **correctness baseline**, not a tuned decode pipeline.
The GPU does almost no useful math per launch; most of the time goes to
setup, copies, and waiting.

This is expected. It is not a bug and does not mean the M-series GPU
is bad at inference. It means the current schedule pays full driver and
sync cost on every op while the tensors are too small for compute to
matter.

### 1. Every op pays full launch + wait

Each Metal op goes through `zynfer_mtl_encode_and_wait` in
`src/backends/apple/bridge.m`, which ends with a blocking wait:

```objc
[cb commit];
[cb waitUntilCompleted];
```

One decode token runs the full block schedule in
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

That is roughly **14 separate Metal submissions per forward**, each with
its own command buffer and `waitUntilCompleted`. There is no batching
and no overlap between kernels.

At roughly 0.3–0.7 ms per op (consistent with `ops-bench` on small
kernels), launch + wait alone account for about **4–6 ms per decode
token**, which matches the ~5.9 ms measured in Debug on this laptop.

### 2. Fresh GPU buffers on every op

Each call in `src/backends/apple/ops.zig` allocates new shared buffers,
copies host data in, runs the kernel, copies out, then frees:

```zig
fn upload(gpu: *Gpu, t: Tensor) !Buffer {
    var buf = try gpu.allocShared(...);
    @memcpy(buf.f32s()[0..src.len], src);
    return buf;
}
```

Decode avoids **host** tensor allocation (scratch is preallocated), but
Metal still **creates and destroys buffers per kernel**. That
alloc/memcpy/wait cost dominates when each matmul is tiny (for example
`1×8×8`).

### 3. The work is too small for the GPU

The fixture is deliberately tiny:

- hidden = 8, head_dim = 4, inter = 16
- decode = **1 token** at a time

A single-token matmul is a handful of FMAs. The CPU oracle runs that in
microseconds. The Metal path still pays driver encode, dispatch, and
sync — often hundreds of microseconds to milliseconds regardless of
FLOPs.

ReleaseSafe makes this clear: CPU decode dropped to ~2 µs/token while
Metal stayed ~4 ms/token. Optimizing scalar CPU math barely moved Metal
because Metal was never bound on compute.

### 4. Attention re-uploads the whole KV cache

On decode, attention reads the **entire** growing K/V cache from host
tensors and uploads them again every step. After `n` tokens that is
`O(n)` host→shared-buffer traffic **per decode step**, with no
Metal-resident cache. That is intentional for Stage 4 correctness, not
performance.

### 5. Some work never hits the GPU

Layout permute (`tokensHeadsToHeadsTokens`) and KV `append` stay on the
CPU. Metal only runs the op wrappers; the schedule still bounces through
host memory between kernels.

### What the numbers mean

| Observation | Interpretation |
| --- | --- |
| Metal ~500× slower than CPU on decode | ~14 synchronous micro-kernel launches on microscopic tensors |
| `ops-bench` Metal often ~2× slower even on 4096-element adds | Launch + wait dominate before compute does |
| ReleaseSafe helps CPU much more than Metal | Bottleneck is encode/sync/buffer churn, not scalar loops |
| Prefill gap similar to decode | Same per-op pattern; prefill just runs more tokens in one forward |

**Bottom line:** the gap measures **14+ synchronous micro-kernel launches
on tiny tensors**, not “M1 Max matmul vs scalar CPU.” Stage 4 proves
correctness and schedule; it does not claim GPU throughput.

### What would change the picture (Stage 5–6)

These are deferred until profiled against a real decode loop:

- **Persistent device buffers** — weights, scratch, and KV; no per-op alloc
- **One command buffer per decode step** (or fused kernels) — one wait, not fourteen
- **Metal-resident KV** — append on GPU; attention reads cache in place
- **Realistic shapes** — enough FLOPs to amortize launch cost
- **Optional `simdgroup_matrix`** where measurements show a win

Do not treat a large Metal speedup from any one of these in isolation as
end-to-end decode improvement until `block-bench` (or a Qwen fixture)
confirms it.

## Fallback matrix

| Path | Status |
| --- | --- |
| CPU scalar f32 | implemented; oracle |
| Metal naive f32 | implemented; differentially tested |
| Metal `simdgroup_matrix` | hardware may report yes; kernel not shipped |
| Accelerate / BNNS / AMX | not wired |
| SME / SME2 | not implemented |
| Core ML / ANE | not implemented |
| fp16 / bf16 Metal | dtype exists; kernels are f32 only |
| HIP transformer ops | not implemented (probe only) |

`zig build run -- caps` prints the disabled-path list. Forced
`--backend apple` on a non-macOS build exits 2.

## Deferred work (evidence)

- **Qwen3-0.6B loader / tokenizer / sampling** — no checkpoint compiler
  or tokenizer in the tree; the tiny block is a synthetic fixture.
- **Metal-resident KV cache / remove per-op wait and GPU buffer alloc** —
  decode no longer allocates host tensors; each Metal op still creates
  shared buffers and `waitUntilCompleted`. Needs a profiled decode loop
  with persistent device buffers before claiming overlap.
- **Fused kernels** — launch overhead is visible on tiny shapes; fusion
  waits until a representative Qwen block is scheduled.
- **Quantized GEMV/GEMM** — no quantized weights loaded.
- **Accelerate / SME / ANE experiments** — optional; only retain if a
  named workload improves without becoming a hard dependency.

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
is orders of magnitude slower on this fixture, see
[Why Metal is slower on the tiny block](apple-backend.md#why-metal-is-slower-on-the-tiny-block)
in `docs/apple-backend.md`.
