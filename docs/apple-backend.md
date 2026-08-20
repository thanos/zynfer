# Apple M-series backend

This is the development-laptop path: Zig host plus a thin Objective-C
Metal bridge. It is not the AMD production target.

## What runs

- CPU f32 oracle in `src/backends/cpu/ops.zig`
- Metal f32 kernels: add, mul, silu_mul, rmsnorm, softmax, matmul,
  matvec, rope (Qwen3-style split-half)
- Tiny SwiGLU residual fixture: RMSNorm → gate/up GEMM → SiLU×Up → down
  GEMM → residual add
- Capability dump: `zig build run -- caps`
- Op microbenchmarks: `zig build ops-bench`

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
ZYNFER_BACKEND=cpu zig build ops-bench
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
  Removing the wait is deferred until a real prefill/decode schedule
  exists to measure overlap against.

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
  or tokenizer in the tree; the SwiGLU residual is the fixture.
- **Prefill vs decode + production KV cache** — no sequence model yet;
  a cache layout without a forward pass would be speculative.
- **Remove per-token alloc and per-op wait** — needs a real decode loop
  to profile; current tests allocate host tensors per op on purpose.
- **Fused kernels** — launch overhead is visible on tiny shapes; fusion
  waits until a representative block is scheduled.
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
