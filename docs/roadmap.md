# Roadmap

Zynfer is built as a staged curriculum. Each stage produces code, tests,
a tutorial, a benchmark command, and a short report. Do not skip ahead
when correctness is unresolved.

| Stage | Title | Status |
| --- | --- | --- |
| 0 | Reproducible development environment | **done on the Mac and HIP-absent hosts** |
| Apple-1 | Backend-neutral types + CPU oracle | **done** |
| Apple-2 | Metal device / shared buffers / trivial kernel | **done** |
| Apple-3 | Metal LLM ops vs CPU + SwiGLU fixture | **done** |
| Apple-4 | Prefill/decode + KV cache | **done for tiny-block fixture** (closeout polish: long-context + no-growth tests, Instruments recipe, deferred list) |
| Apple-5 | simdgroup_matrix / quantized GEMV / Accelerate | **done** (size-gated; see `bench/results/apple-stage5-dev-laptop.md`) |
| Apple-6 | Fusion / fewer waits / Metal-resident KV | **done** (one CB/wait + resident KV; see `bench/results/apple-stage6-dev-laptop.md`) |
| Apple-7 | SME / Core ML experiments | deferred (carries Stage 6 SME/ANE leftover) |
| Apple-8 | Hardening + Stage 6 leftovers | not started (ICB/replay, further fusions, kv_len, fp16, signposts/metrics; fused vs baseline A/B **done**) |
| 1 | Zig meets HIP (alloc, copy, streams) | not started |
| 2 | First AMD kernel (vector add) | not started |
| 3 | Tensor representation and memory planning | partial (host tensors exist; no GPU planner) |
| 4 | Reductions, softmax, RMSNorm | CPU + Metal f32; not HIP |
| 5 | GEMV and GEMM | CPU + Metal f32; not HIP |
| 6 | SiLU and SwiGLU | CPU + Metal f32; not HIP |
| 7 | RoPE | CPU + Metal f32; not HIP |
| 8 | Attention from scratch | CPU + Metal f32 (`kv_len` ≤ 64 on Metal) |
| 9 | One complete transformer block | **done as tiny fixture**; not a Qwen block |
| 10 | Checkpoint inspection and artifact compiler | not started (absorbs Qwen loader deferred from Apple-6) |
| 11 | Full Qwen3-0.6B forward pass | not started (absorbs golden logits deferred from Apple-6) |
| 12 | Tokenizer and sampling | not started (absorbs TTFT/tok/s deferred from Apple-6) |
| 13 | KV cache | **host layout + Metal-resident for tiny block** |
| 14 | Prefill vs decode | **done for tiny block** |
| 15 | Profiling the whole token | not started |
| 16 | Kernel fusion | Apple-6 CB batching + `silu_mul`/`add_rmsnorm` done for tiny-block; Qwen-scale / further fusions / ICB → Apple-8 + this stage |
| 17 | HIP graphs | not started |
| 18 | Quantization | not started |
| 19 | AMD-specific kernel tuning | not started |
| 20 | Static decode memory plan | not started |
| 21 | Batching and scheduling | not started |
| 22 | Prefix reuse / cache management | not started |
| 23 | Speculative / multi-token prediction | not started |
| 24 | Server | not started |
| 25 | Move to a serious model | not started |

## Milestones

- **A. Hello, GPU** — Zig controls the R9700 through HIP. Stage 0 starts this; Stage 1 finishes memory copies.
- **B. We own the math** — transformer primitives on our kernels.
- **C. One block** — one Qwen block matches the oracle.
- **D. It is an LLM** — Qwen3-0.6B greedy tokens match the reference.
- **E. It is an inference engine** — KV-cached generation.
- **F. We understand the bottleneck** — one decode token is accounted for.
- **G. AMD-native** — important kernels tuned for `gfx1201`.
- **H. Specialized** — quantization, fusion, graphs help and are measured.
- **I. Useful** — concurrent requests.
- **J. NInfer philosophy** — a larger registered checkpoint on one R9700.

## Current rule

The AMD curriculum still starts at Stage 1 (HIP alloc/copy) on the
R9700. The development laptop additionally has a CPU oracle, naive
Apple Metal ops, and a tiny transformer-block prefill/decode fixture.
Apple Stages 0–6 are closed for the tiny fixture: measured matrix paths
(Stage 5) and one-CB/wait + Metal-resident KV (Stage 6). Do not treat
Qwen loading or further speculative fusions as unfinished Stage 6 work—
they are mapped to Apple-7/8 and curriculum Stages 10–12 / 16 (see
`docs/apple-backend.md` deferred table).

Do not start a full Qwen forward pass until a checkpoint loader exists.

See `docs/apple-backend.md` for the deferred→stage map and the
Instruments recipe for wait dominance.
