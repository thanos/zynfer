# Architecture

Zynfer is a specialized LLM inference engine. The host runtime is Zig.
Execution backends are adapters: a deterministic CPU oracle, an Apple
Metal path on macOS, and an AMD HIP probe that will later own RDNA 4
kernels. Model math must not import Metal or HIP types.

```text
Model architecture (not loaded yet: Qwen3-0.6B)
        |
        v
Backend-neutral tensors, dtypes, LLM ops
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
| CLI | `src/main.zig` | Zig | `env`, `gpu`, `caps`, `backends`, `ops-bench`, `bench` |
| Env report | `src/env_report.zig` | Zig | Host/toolchain; HIP optional; Apple compile flag |
| Backend identity | `src/runtime/backend.zig` | Zig | Kind vs architecture; fallback reasons |
| Tensors | `src/runtime/tensor.zig` | Zig | Rank ≤ 4, contiguous, overflow-safe byte sizes |
| CPU oracle | `src/backends/cpu/ops.zig` | Zig | Scalar f32; correctness reference |
| Apple bridge | `src/backends/apple/bridge.m` | ObjC/ARC | Device, shared buffers, compile, wait |
| Apple kernels | `src/backends/apple/kernels.metal` | MSL | Naive f32; no `simdgroup_matrix` |
| Apple dispatch | `src/backends/apple/gpu.zig`, `ops.zig` | Zig | Policy, launches, CPU differential tests |
| HIP probe | `src/hip.zig` + `src/hip_probe.c` | Zig + C | Enumeration only until the AMD op path exists |

Model / tokenizer / KV-cache / engine modules do not exist yet. The
smallest end-to-end fixture is a SwiGLU residual fragment compared
CPU vs Metal.

## Apple memory and synchronization

Shared `MTLBuffer` storage (`MTLResourceStorageModeShared`) is
CPU-writable on unified memory. Filling `contents` is not a PCIe copy.
It is also not free coherence: the baseline path calls
`waitUntilCompleted` after every kernel. Do not describe this as
zero-copy overlap.

## HIP boundary

Zig does not include HIP headers directly. `src/hip_probe.c` is compiled
only when a ROCm prefix is found. If HIP is absent (typical on a Mac)
the binary still builds and device APIs return `error.HipUnavailable`.

## What is required to generate a token?

Nothing in this tree generates tokens. The functions that will sit on
that path are not present. What exists today:

```text
host tensor → CPU op or Metal kernel → host tensor
```

A Qwen forward pass, tokenizer, sampler, and KV cache are deferred.

## Non-goals that already constrain the architecture

- No Python in the runtime.
- No generic tensor framework.
- No CUDA-shaped graph runtime.
- No silent backend fallback: `--backend cuda` exits 2.
- Do not claim AMX, SME/SME2, ANE/Core ML, or `simdgroup_matrix`
  execution; those capabilities may be discovered and are disabled with
  an explicit reason.
- No multi-GPU path.
