# Architecture

Zynfer is a specialized LLM inference engine. The host runtime is Zig.
GPU work is launched through the ROCm HIP runtime onto a single AMD RDNA 4
GPU.

The intended long-term boundary is:

```text
                    zynfer
                        |
            +-----------+-----------+
            |                       |
        Zig runtime             GPU kernels
            |                       |
     HIP C runtime API       HIP/AMDGCN initially
            |                       |
            +-----------+-----------+
                        |
                    ROCm/HIP
                        |
                 AMD RDNA 4 GPU
```

Stage 0 implements only the leftmost host path far enough to answer:

> Is the machine we are sitting on actually the machine we think it is?

No transformer math, no tensors, and no custom GPU kernels live here yet.

## Ownership

| Layer | Owner | Language | Notes |
| --- | --- | --- | --- |
| CLI / env report | `src/main.zig` | Zig | No GPU work required |
| HIP bindings | `src/hip.zig` + `src/hip_probe.c` | Zig + C | Thin; no `hipDeviceProp_t` in Zig |
| Device handle | `src/device.zig` | Zig | Index + properties only |
| GPU kernels | `kernels/` | HIP later | Empty until Stage 2 |
| Model / engine | `src/model.zig` etc. | Zig later | Not created yet |

## HIP boundary

Zig does not include HIP headers directly.

`src/hip_probe.c` is compiled only when a ROCm prefix is found. It copies
the fields we care about into `ZynferGpuInfo`, a struct whose layout we
control. That keeps ROCm version churn on the C side of a one-page
adapter.

If HIP is absent (typical on a development laptop) the Zig binary still
builds. Device APIs return `error.HipUnavailable`.

## What is required to generate a token?

Nothing in Stage 0 generates tokens. The files that will eventually sit
on that path are listed in the repository layout in `README.md`. After
each later stage this document should be able to name the exact functions
that implement:

```text
prompt → tokens → embeddings → blocks → logits → sample → next token
```

## Non-goals that already constrain the architecture

- No Python in the runtime.
- No CUDA, Vulkan, or Metal backend.
- No generic tensor framework.
- No multi-GPU path.
- Native Zig GPU codegen is interesting later and must not block HIP.
