# Zynfer

> **From-scratch LLM inference in Zig, specialized for AMD RDNA 4 GPUs.**

Zynfer is an experimental LLM inference engine written primarily in **Zig** and built specifically to explore high-performance inference on modern **AMD GPUs**.

The initial hardware target is the **AMD Radeon AI PRO R9700**, an RDNA 4 GPU exposed by ROCm as `gfx1201`. The initial bootstrap model is **Qwen3-0.6B**.

Zynfer deliberately starts narrow.

It is not intended to become a universal machine-learning framework or an abstraction over every GPU vendor. Instead, the project asks a more focused question:

> **How fast can a small, understandable inference runtime become when the model, runtime, memory system, kernels, and GPU architecture are designed for each other?**

The project is also educational. Every major subsystem is developed in stages, beginning with Zig/HIP interoperability and a single GPU kernel and eventually progressing through transformer inference, KV caching, quantization, kernel fusion, graph execution, and architecture-specific RDNA 4 optimization.

---

## Status

**CPU oracle + Apple Metal tiny-block prefill/decode; Stage 5 matrix paths.**

The Zig repository, HIP device enumeration (when ROCm is present),
environment report, CPU f32 reference ops, naive Metal kernels, a tiny
transformer-block fixture with an explicit KV cache, capability-gated
`simdgroup_matrix` matmul, int8 GEMV, and Accelerate vDSP matmul exist.
Qwen3-0.6B is not loaded. Tokenizer and sampling do not exist.

On a Mac, `zig build test` differential-checks Metal against CPU.
`--backend` / `ZYNFER_BACKEND` select `cpu`, `apple`, or `amd-hip`.
Unknown names fail; they do not fall back.

See `docs/apple-backend.md` and `docs/architecture.md`.

The development sequence begins with:

```text
Zig
 ↓
HIP runtime
 ↓
AMD GPU kernel
 ↓
tensor primitives
 ↓
transformer primitives
 ↓
one transformer block
 ↓
Qwen3-0.6B
 ↓
KV-cached autoregressive inference
 ↓
profiling
 ↓
RDNA 4 optimization
```

Do not expect arbitrary model compatibility or production stability at this stage.

---

## Philosophy

Zynfer takes inspiration from the specialization philosophy demonstrated by NInfer: explicitly selected checkpoints, explicitly selected hardware, and an inference runtime designed around those constraints rather than around general framework compatibility.

The AMD implementation is not intended to be a mechanical CUDA-to-HIP port.

AMD GPUs have their own execution model, memory hierarchy, matrix capabilities, compiler behavior, and performance characteristics. Zynfer treats those differences as part of the project.

The core principles are:

### Specialize deliberately

The first target is one GPU architecture:

```text
AMD RDNA 4
gfx1201
```

and initially one physical GPU:

```text
AMD Radeon AI PRO R9700
```

Supporting fewer configurations gives us permission to make decisions based on the actual hardware.

### Understand the entire inference path

Zynfer should make it possible to trace:

```text
prompt
  ↓
tokenizer
  ↓
token IDs
  ↓
embeddings
  ↓
transformer blocks
  ↓
attention + KV cache
  ↓
MLP
  ↓
logits
  ↓
sampling
  ↓
next token
```

For every step, the project should eventually be able to answer:

- What mathematics are we computing?
- Where does the data live?
- Which GPU kernels execute?
- How much memory moves?
- How long does it take?
- What limits its performance?
- Why is the result numerically correct?

### Correctness before optimization

Every important GPU operation begins with a simple reference implementation.

The workflow is:

```text
reference
   ↓
test
   ↓
benchmark
   ↓
profile
   ↓
optimize
   ↓
test again
   ↓
benchmark again
```

An optimization that cannot demonstrate preserved correctness and measurable improvement does not belong in the optimized path.

### Measure rather than guess

Performance work is driven by evidence.

Relevant measurements include:

- time to first token;
- inter-token latency;
- decode tokens/sec;
- prefill tokens/sec;
- memory bandwidth;
- kernel execution time;
- launch overhead;
- VRAM consumption;
- register pressure;
- LDS usage;
- occupancy;
- speculative-token acceptance rates.

The project should never describe something as faster merely because it looks clever.

---

## Why Zig?

Zig provides an unusually good environment for this experiment.

The runtime needs:

- explicit memory ownership;
- predictable allocation;
- straightforward C interoperability;
- compile-time specialization;
- low runtime overhead;
- control over binary layouts;
- a small systems-level abstraction layer.

Zynfer uses Zig for the host-side inference runtime rather than placing a Python framework between the application and the GPU.

Conceptually:

```text
┌───────────────────────────────┐
│             Zynfer            │
│                               │
│  model loader                 │
│  memory planner               │
│  tokenizer                    │
│  inference engine             │
│  KV cache                     │
│  sampling                     │
│  scheduler                    │
│  benchmark infrastructure     │
│                               │
│             Zig               │
└───────────────┬───────────────┘
                │
                │ C ABI
                ▼
┌───────────────────────────────┐
│          ROCm / HIP           │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│       AMD GPU kernels         │
│                               │
│  GEMM / GEMV                  │
│  RMSNorm                      │
│  RoPE                         │
│  attention                    │
│  activation / gating          │
│  quantization                 │
│  fused operations             │
└───────────────┬───────────────┘
                │
                ▼
          AMD RDNA 4 GPU
```

GPU kernels may initially be compiled through the ROCm/Clang/HIP toolchain and loaded by the Zig runtime.

Writing every GPU kernel directly in Zig is **not a prerequisite** for the project.

That boundary can be investigated later without preventing the inference engine from becoming useful first.

---

## Initial Hardware Target

The initial target is:

**AMD Radeon AI PRO R9700**

Relevant characteristics include:

```text
Architecture        RDNA 4
LLVM target         gfx1201
Compute Units       64
VRAM                32 GiB GDDR6
Memory bandwidth    640 GB/s
```

The R9700 also exposes hardware acceleration for lower-precision matrix computation, making FP8, INT8, and INT4 interesting future targets.

Zynfer does not assume that theoretical peak throughput translates directly into LLM performance.

In particular, autoregressive decode frequently has very different performance characteristics from prompt prefill.

Understanding that distinction is one of the project's primary goals.

---

## Linux and ROCm

Linux is the reference platform.

The intended development environment is approximately:

```text
Linux
  │
  ├── Zig
  ├── ROCm
  │    ├── HIP runtime
  │    ├── compiler toolchain
  │    ├── hipBLASLt
  │    └── profiling tools
  │
  └── AMD Radeon AI PRO R9700
```

AMD currently lists the R9700 as a supported ROCm GPU with LLVM target `gfx1201`.

Exact supported Linux distributions, ROCm releases, drivers, and compiler versions will be pinned as Zynfer's development environment stabilizes.

---

## Initial Model

The bootstrap model is:

**Qwen3-0.6B**

The small model is intentional.

The purpose of the first model is not to demonstrate the largest model that fits into 32 GiB of VRAM. It is to exercise essentially the complete transformer inference stack while keeping debugging cycles short.

It gives us real implementations of:

- embeddings;
- RMSNorm;
- rotary positional embeddings;
- grouped-query attention;
- causal attention;
- KV caching;
- gated MLPs;
- residual connections;
- output projection;
- tokenization;
- sampling;
- prompt prefill;
- autoregressive decoding.

Once the implementation is correct and profiled, development will move to larger explicitly selected checkpoints.

A likely progression is:

```text
Qwen3-0.6B
    │
    │ correctness
    ▼
Qwen3-4B
    │
    │ optimization
    ▼
larger selected Qwen model
    │
    │ quantization + specialization
    ▼
32 GiB-class optimized workload
```

Model support will remain explicit rather than automatically accepting arbitrary checkpoints.

---

## Prefill and Decode

Zynfer treats prompt processing and token generation as different workloads.

### Prefill

Given a prompt containing many tokens, the model processes those tokens in parallel.

This tends to create relatively large matrix operations:

```text
many tokens
     ×
model dimensions
```

and can make effective use of highly parallel matrix hardware.

### Decode

Once prefill is complete, autoregressive generation repeatedly processes approximately one new token per sequence:

```text
token
  ↓
28-ish transformer layers
  ↓
logits
  ↓
next token
  ↓
repeat
```

The matrix shapes, memory behavior, and launch overhead are now very different.

Decode can become dominated by:

- reading model weights;
- reading/writing KV cache;
- small matrix operations;
- kernel launches;
- synchronization;
- memory bandwidth.

Therefore Zynfer benchmarks **prefill and decode independently**.

An optimization that improves prefill is not automatically assumed to improve decode.

---

## KV Cache

Without caching, autoregressive generation would repeatedly recompute keys and values for the entire existing context.

Zynfer will maintain a persistent KV cache containing the attention state from previous tokens.

Conceptually:

```text
layer
  │
  ├── token 0 ── K,V
  ├── token 1 ── K,V
  ├── token 2 ── K,V
  │
  └── token N ── K,V
```

Each decode step computes only the new token's K/V state and attends against the cached prefix.

KV-cache layout is a performance-sensitive design decision and will eventually be optimized around measured AMD memory-access behavior rather than inherited blindly from another GPU architecture.

Future work may include:

- quantized KV storage;
- block-based allocation;
- prefix reuse;
- cache eviction;
- long-context memory planning.

---

## GPU Kernels

The project will progressively implement and study the operations required by transformer inference.

Early kernels include:

```text
vector operations
reductions
softmax
RMSNorm
RoPE
SiLU
gating
```

followed by:

```text
GEMV
GEMM
QKV projection
attention
MLP
sampling
```

and eventually fused operations.

Every important performance kernel should document:

```text
input shape
output shape
dtype
workgroup dimensions
wave assumptions
LDS usage
register considerations
memory access pattern
synchronization
target architecture
reason for specialization
```

Magic launch parameters without an explanation are considered technical debt.

---

## Matrix Operations

Matrix multiplication is central to transformer inference.

Zynfer will initially establish several baselines:

```text
CPU reference
      ↓
naive GPU implementation
      ↓
vendor-optimized baseline
      ↓
shape-specific experiments
```

ROCm libraries such as hipBLASLt can provide both a correctness reference and a strong performance baseline.

The objective is **not** to rewrite a vendor GEMM library merely for ideological purity.

Custom kernels are justified when Zynfer's known inference shapes or fusion opportunities allow something meaningfully better for the targeted workload.

---

## Attention

The conceptual operation is:

```text
Attention(Q, K, V) =
    softmax(QKᵀ / √d) V
```

but an efficient implementation should avoid unnecessarily materializing large intermediate attention matrices.

Zynfer will first implement an intentionally understandable attention path and verify its intermediate tensors.

Only then will it investigate:

- tiled attention;
- streaming softmax;
- LDS reuse;
- reduced global-memory traffic;
- fused Q/K preparation;
- fused RoPE;
- architecture-specific layouts.

Grouped-query attention will be implemented according to the selected model rather than treated as an optional generic feature.

---

## Quantization

Quantization comes **after** the floating-point implementation is correct.

Potential paths include:

- FP8;
- INT8;
- low-bit packed weights;
- groupwise quantization;
- quantized KV cache.

The chosen formats will be based on the capabilities and measured behavior of RDNA 4 rather than on formats designed specifically around NVIDIA hardware.

For each quantization strategy Zynfer should measure:

```text
artifact size
VRAM usage
prefill throughput
decode throughput
quality regression
conversion/dequantization overhead
```

A smaller model file is not sufficient evidence that a quantization scheme is useful.

---

## Kernel Fusion

Once individual operations are understood and benchmarked, adjacent operations may be fused to reduce:

- kernel launches;
- intermediate writes;
- intermediate reads;
- synchronization;
- memory traffic.

Possible experiments include:

```text
RMSNorm + projection preparation

Q/K preparation + RoPE

SiLU + gate multiplication

residual + normalization

sampling pipeline operations
```

Every fusion experiment keeps the unfused path available as a correctness and performance baseline.

---

## HIP Graphs

Autoregressive decode repeatedly executes nearly the same GPU operation graph.

Zynfer will investigate HIP graph execution as a way to reduce repeated CPU-side launch overhead.

Conceptually:

```text
normal decode

CPU → kernel
CPU → kernel
CPU → kernel
CPU → kernel
...
```

versus:

```text
captured decode graph
        │
CPU ────┴──→ GPU execution graph
```

Graph execution will only become the default if benchmarking demonstrates a useful improvement.

---

## Memory Philosophy

Steady-state decode should perform as little dynamic memory management as possible.

The desired eventual state is:

```text
startup
  │
  ├── load model
  ├── allocate weights
  ├── allocate KV cache
  ├── allocate scratch memory
  ├── initialize kernels
  └── prepare execution resources
          │
          ▼
       decode loop
          │
          ├── no general allocation
          ├── no model transfer over PCIe
          ├── minimal synchronization
          └── predictable memory access
```

Memory ownership should always be explicit.

---

## Artifact Format

Zynfer will eventually use a small native artifact format rather than making the hot runtime understand every upstream checkpoint format.

The development pipeline will conceptually be:

```text
Hugging Face checkpoint
        │
        ▼
Zynfer artifact compiler
        │
        ▼
     .zynfer
        │
        ▼
Zynfer runtime
```

A native artifact can encode:

- model identity;
- architecture parameters;
- tensor directory;
- dtype information;
- aligned tensor payloads;
- quantization metadata;
- version;
- integrity information.

Development tools may use Python to inspect or convert upstream model files.

Python is not intended to be part of the inference execution path.

---

## Benchmarking

Zynfer treats benchmarks as part of the implementation.

Every meaningful result should record enough metadata to reproduce it:

```text
GPU
GPU architecture
ROCm version
driver
Zig version
model
artifact version/hash
precision
quantization
prompt length
generated length
batch size
concurrency
KV format
graph mode
sampling configuration
```

### Primary interactive metrics

```text
time to first token
inter-token latency
single-request decode tokens/sec
```

### Throughput metrics

```text
aggregate tokens/sec
tokens/sec/request
concurrency scaling
```

### Prefill

```text
prompt tokens/sec
prefill latency
```

### Memory

```text
model VRAM
KV-cache VRAM
scratch VRAM
peak VRAM
```

### Kernel metrics

Where profiling support allows:

```text
kernel duration
effective memory bandwidth
occupancy
register pressure
LDS usage
launch count
```

---

## Profiling

Optimization should begin with a profile of one generated token.

Eventually Zynfer should be able to produce something conceptually similar to:

```text
Operation                     Time
----------------------------------
RMSNorm                       ...
QKV projection                ...
RoPE                          ...
attention                     ...
output projection             ...
MLP                           ...
sampling                      ...
runtime / launch overhead     ...
----------------------------------
complete decode token         ...
```

That profile determines what gets optimized next.

Not intuition.

---

## Project Roadmap

### Phase 0 — Environment

- Linux development environment
- Zig toolchain
- ROCm/HIP
- detect `gfx1201`
- enumerate GPU from Zig

### Phase 1 — GPU fundamentals

- GPU allocation
- host/device copies
- streams
- first custom kernel
- timing infrastructure

### Phase 2 — Tensor primitives

- tensor representation
- memory arena
- reductions
- softmax
- RMSNorm
- RoPE
- SiLU

### Phase 3 — Linear algebra

- naive GEMM
- vendor GEMM baseline
- model-specific matrix shapes
- prefill/decode benchmarking

### Phase 4 — Transformer

- Q/K/V projection
- grouped-query attention
- MLP
- residual path
- complete transformer block

### Phase 5 — Model

- checkpoint inspection
- artifact compiler
- Qwen3-0.6B loader
- complete forward pass
- tokenizer
- sampling

### Phase 6 — Real inference

- autoregressive generation
- KV cache
- separate prefill/decode paths
- deterministic verification

### Phase 7 — Performance

- full-token profiling
- memory planning
- kernel fusion
- HIP graphs
- architecture-specific kernels

### Phase 8 — Low precision

- FP8 experiments
- INT8 experiments
- low-bit weight formats
- quantized KV cache

### Phase 9 — Serving

- batching
- scheduling
- prefix reuse
- HTTP API
- streaming generation

### Phase 10 — Advanced inference

- speculative decoding
- multi-token prediction where supported
- larger selected checkpoints
- increasingly aggressive `gfx1201` specialization

---

## Educational Goal

Zynfer is intended to be readable as an executable course in LLM inference.

The accompanying tutorials will cover topics such as:

```text
How a GPU executes
Zig ↔ C interoperability
HIP
GPU memory
AMD waves and workgroups
tensor representation
parallel reductions
softmax
RMSNorm
matrix multiplication
prefill vs decode
RoPE
attention
grouped-query attention
transformer blocks
model checkpoints
tokenization
sampling
KV caches
profiling
kernel fusion
HIP graphs
quantization
RDNA 4 optimization
batching
prefix caching
speculative decoding
LLM serving
```

The objective is that someone who works through the repository should understand not merely **how to run an LLM**, but what the machine actually does to generate each token.

---

## Non-goals

Version 1 deliberately does **not** target:

- NVIDIA GPUs;
- CUDA;
- Windows;
- macOS GPU execution;
- Vulkan;
- WebGPU;
- arbitrary Hugging Face models;
- training;
- fine-tuning;
- multimodal inference;
- distributed inference;
- multi-node execution;
- Kubernetes;
- a general tensor framework.

Some may become future experiments.

They are not allowed to complicate the initial architecture.

---

## Why Not Just Use llama.cpp or vLLM?

If your objective is simply to run an LLM, you probably should.

Zynfer exists for a different reason.

It is an experiment in understanding and controlling the complete inference stack:

```text
model
 +
numerics
 +
memory layout
 +
runtime
 +
GPU kernels
 +
hardware architecture
```

General inference frameworks necessarily optimize across many combinations of models, hardware, operating modes, and workloads.

Zynfer intentionally gives up that generality.

The hypothesis is that specialization makes both **deeper understanding** and **different optimization choices** possible.

---

## Inspiration

Zynfer is inspired in part by **NInfer**, which demonstrates an intentionally narrow approach to high-performance inference: selected checkpoints, a selected GPU target, native artifacts, specialized kernels, optimized KV-cache behavior, graph execution, and speculative decoding.

NInfer currently targets C++/CUDA and an NVIDIA RTX 5090.

Zynfer explores a related philosophy from a different direction:

```text
NInfer                         Zynfer

C++                            Zig
CUDA                           ROCm / HIP
NVIDIA                         AMD
RTX 5090                       Radeon AI PRO R9700
Blackwell                      RDNA 4
sm_120a                        gfx1201
selected checkpoints           selected checkpoints
```

Zynfer is an independent project and is not intended to be a source-level port of NInfer.

---

## Development Environment

The intended workflow supports a remote GPU host:

```text
development laptop
        │
        │ SSH
        ▼
Linux GPU host
        │
        ├── Zig
        ├── ROCm
        ├── profiler
        ├── Zynfer
        └── model artifacts
                │
                ▼
          R9700 / gfx1201
```

The development laptop does not need an AMD GPU.

---

## Building

Pinned Zig version: **0.16.0** (see `.tool-versions`).

```bash
# host-only diagnostic (development laptop, no AMD GPU required)
./scripts/check-env.sh
zig build
zig build test
zig build integration
zig build ci            # fmt + test + integration + autodoc
zig build run

# Linux GPU host with ROCm: HIP is linked automatically when found
zig build -Dhip=on -Dhip-path=/opt/rocm
zig build test
zig build run -- gpu
```

`-Dhip=auto` (the default) links HIP when `HIP_PATH`, `ROCM_PATH`, or
`/opt/rocm` contains `include/hip/hip_runtime_api.h`. `-Dhip=off` forces
a host-only binary.

GPU kernel compilation is not part of the AMD Stage 0 probe. On macOS
the Apple backend compiles embedded MSL at runtime (`newLibraryWithSource`).
That requires the Metal Toolchain:

```bash
xcodebuild -downloadComponent MetalToolchain
```

---

## Running

```bash
zig build run              # environment report + capabilities + HIP probe
zig build run -- env
zig build run -- gpu
zig build run -- caps
zig build run -- backends
zig build ops-bench        # CPU vs Apple Metal op timings
zig build block-bench      # tiny-block prefill/decode timings
zig build bench            # time HIP property queries
zig build integration      # CLI contracts against the installed binary
zig build docs             # Zig autodoc → zig-out/docs/api
./scripts/check-env.sh     # shell diagnostic, does not require a build
```

CI workflows (format, tests, coverage, benches, docs) are described in
`docs/ci.md`.

`--backend cpu|apple|amd-hip` or `ZYNFER_BACKEND` forces a backend.
Unknown names exit 2.

The eventual inference CLI is expected to resemble:

```bash
zynfer run models/qwen3-0.6b.zynfer \
  --prompt "Explain why the sky is blue." \
  --max-new 128
```

That interface is not implemented yet.

---

## Repository Layout

```text
zynfer/
├── .github/workflows/     # CI, coverage, bench, docs
├── build.zig
├── build.zig.zon
├── README.md
├── scripts/
│   ├── check-env.sh
│   ├── build-docs-site.sh
│   └── ci/
├── docs/
│   ├── architecture.md
│   ├── apple-backend.md
│   ├── ci.md
│   ├── roadmap.md
│   ├── benchmarks.md
│   ├── numerics.md
│   ├── hardware-r9700.md
│   └── tutorials/
│       ├── 00-development-environment.md
│       └── 01-how-a-gpu-executes.md
├── src/
│   ├── main.zig
│   ├── root.zig
│   ├── hip.zig
│   ├── hip_probe.c
│   ├── device.zig
│   ├── env_report.zig
│   ├── runtime/           # dtype, tensor, backend, compare
│   └── backends/
│       ├── cpu/ops.zig
│       └── apple/         # bridge.m, kernels.metal, gpu.zig, ops.zig
├── kernels/                 # empty until Stage 2
├── tools/
├── tests/
│   ├── unit/
│   ├── numerical/
│   └── integration/
└── bench/results/
```

Later stages add `model.zig`, `tokenizer.zig`, `kv_cache.zig`, and
`engine.zig`. They are omitted until they have a job.

This will evolve as implementation reveals the correct boundaries.

---

## Contributing

Zynfer is currently an exploratory project.

Contributions should favor:

- measurable improvements;
- clear explanations;
- small changes;
- deterministic tests;
- reproducible benchmarks;
- explicit hardware assumptions.

Performance changes should include before/after measurements whenever practical.

Architecture-specific optimizations should document why they work.

A complicated optimization with no demonstrated benefit is not an optimization.

---

## License

License to be selected before the first public release.

---

## Current Target

```text
Language       Zig
Platform       Linux (AMD target) and macOS (Apple backend)
GPU vendor     AMD (target) / Apple (dev laptop)
GPU            Radeon AI PRO R9700 / Apple M-series
Architecture   RDNA 4 / Apple GPU family 7+
LLVM target    gfx1201 (AMD)
GPU runtime    ROCm/HIP (AMD) and Metal (Apple)
Bootstrap LLM  Qwen3-0.6B (not loaded yet)
Execution      Single GPU
Status         CPU oracle + Metal baseline ops; no token generation
```

---

## The Goal

Zynfer is ultimately an attempt to answer a simple question in unreasonable detail:

> **What actually has to happen between receiving a token and producing the next one — and how fast can we make that process when we understand every layer involved?**

The goal is not merely to make an LLM run on an AMD GPU.

The goal is to understand the machine well enough that performance becomes explainable.