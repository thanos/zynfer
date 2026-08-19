# Roadmap

Zynfer is built as a staged curriculum. Each stage produces code, tests,
a tutorial, a benchmark command, and a short report. Do not skip ahead
when correctness is unresolved.

| Stage | Title | Status |
| --- | --- | --- |
| 0 | Reproducible development environment | **in progress / current** |
| 1 | Zig meets HIP (alloc, copy, streams) | not started |
| 2 | First AMD kernel (vector add) | not started |
| 3 | Tensor representation and memory planning | not started |
| 4 | Reductions, softmax, RMSNorm | not started |
| 5 | GEMV and GEMM | not started |
| 6 | SiLU and SwiGLU | not started |
| 7 | RoPE | not started |
| 8 | Attention from scratch | not started |
| 9 | One complete transformer block | not started |
| 10 | Checkpoint inspection and artifact compiler | not started |
| 11 | Full Qwen3-0.6B forward pass | not started |
| 12 | Tokenizer and sampling | not started |
| 13 | KV cache | not started |
| 14 | Prefill vs decode | not started |
| 15 | Profiling the whole token | not started |
| 16 | Kernel fusion | not started |
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

Finish Stage 0, then stop. Stage 1 is GPU allocation and copies, not
kernels and not transformers.
