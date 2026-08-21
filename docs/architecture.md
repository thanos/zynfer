# Architecture

Zynfer is a specialized LLM inference engine. The host runtime is Zig.
Execution backends are adapters: a deterministic CPU oracle, an Apple
Metal path on macOS, and an AMD HIP probe that will later own RDNA 4
kernels. Model math must not import Metal or HIP types.

```text
Model architecture (fixture: tiny block; Stage 10: `.zynfer` + Qwen3-0.6B meta;
forward not loaded yet)
        |
        v
Backend-neutral tensors, KV cache, block schedule
        |
        +-----------+-----------+
        |           |           |
     CPU f32     Apple      AMD HIP
     oracle      Metal      probe only
        |           |           |
   scalar loops  MSL f32    hip_probe.c
                 kernels
```

`BackendKind` (`cpu`, `apple`, `amd_hip`) is the execution API family.
`DeviceArchitecture` (`generic_cpu`, `apple_m`, `amd_rdna`) is the
hardware family used for capability reports and later kernel selection.
They are not the same enum.

## Ownership

| Layer | Owner | Language | Notes |
| --- | --- | --- | --- |
| CLI | `src/main.zig` | Zig | `env`, `gpu`, `caps`, `inspect`, `artifact-compile`, benches, stage ledgers |
| Env report | `src/env_report.zig` | Zig | Host/toolchain; HIP optional; Apple compile flag |
| Backend identity | `src/runtime/backend.zig` | Zig | Kind vs architecture; fallback reasons |
| Tensors | `src/runtime/tensor.zig` | Zig | Rank ≤ 4, contiguous, overflow-safe byte sizes |
| KV cache | `src/runtime/kv_cache.zig` | Zig | Host `[n_kv, max_seq, head_dim]`; append/used |
| Tiny block | `src/model/tiny_block.zig` | Zig | Prefill/decode schedule; no Metal/HIP imports |
| Artifact | `src/model/artifact.zig` | Zig | `.zynfer` v1 build/validate/load (Stage 10) |
| Qwen3 dims | `src/model/qwen3.zig` | Zig | HF architecture constants; forward is Stage 11 |
| CPU oracle | `src/backends/cpu/ops.zig` | Zig | Scalar f32; correctness reference |
| Apple bridge | `src/backends/apple/bridge.m` | ObjC/ARC | Device, shared buffers, compile, wait |
| Apple kernels | `src/backends/apple/kernels.metal` | MSL | Naive f32 + simdgroup(+x4) + q8 + Stage 6 permute/kv_append |
| Apple dispatch | `src/backends/apple/gpu.zig`, `ops.zig`, `block.zig` | Zig | Policy, batch CB, resident KV schedule, CPU differential tests |
| CPU Accelerate | `src/backends/cpu/accelerate.zig` | Zig+vDSP | Size-gated matmul/matvec; oracle remains scalar |
| HIP probe | `src/hip.zig` + `src/hip_probe.c` | Zig + C | Enumeration only until the AMD op path exists |

The smallest end-to-end fixture is one tiny transformer block with
separate prefill/decode entry points. The CPU path uses an explicit host
KV cache; the default Apple Stage 6 path keeps KV Metal-resident. Apple
Stages 0–8 are closed for that fixture. Deferred leftovers are
mapped to curriculum Stages 10–12 / 16 (see `docs/apple-backend.md`)—
Qwen weights, tokenizer, sampling, further fusions at model scale, and
vocabulary TTFT are not silent drops. SME/Core ML (Stage 7) and
ICB/fp16/extra tiny-block fusions (Stage 8) were probed/rejected with
evidence.

## Apple memory and synchronization

Shared `MTLBuffer` storage (`MTLResourceStorageModeShared`) is
CPU-writable on unified memory. Filling `contents` is not a PCIe copy.

The **tiny-block Stage 6 path** encodes ~20 dispatches into one command
buffer and waits once. The **per-op `apple.ops` path** (ops-bench /
baseline) still calls `waitUntilCompleted` after every kernel. Do not
describe either as free overlap with the CPU.

## HIP boundary

Zig does not include HIP headers directly. `src/hip_probe.c` is compiled
only when a ROCm prefix is found. If HIP is absent (typical on a Mac)
the binary still builds and device APIs return `error.HipUnavailable`.

## What is required to generate a token?

Nothing in this tree generates vocabulary tokens. The tiny-block fixture
runs a synthetic residual stream through one attention+MLP block with a
host KV cache. A Qwen forward pass, tokenizer, and sampler are deferred.

```text
x[t,H] → RMSNorm → QKV → RoPE → KV append → causal GQA → O + residual → SwiGLU residual
```

## Non-goals that already constrain the architecture

- No Python in the runtime.
- No generic tensor framework.
- No CUDA-shaped graph runtime.
- No silent backend fallback: `--backend cuda` exits 2.
- Do not claim AMX, SME/SME2, or ANE/Core ML **execution**; Stage 7
  probes SME hardware and Core ML availability, then keeps those
  inference paths disabled. `simdgroup_matrix` and Accelerate are used
  only when capability- and size-gated selection chooses them.
- No multi-GPU / model-sharding path across devices. Concurrent Apple
  sessions use one `Gpu` (and buffers) per owner thread.
