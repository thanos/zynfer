# Numerics

The CPU backend is the floating-point oracle. Accelerated ops are
accepted only when they match it within an explicit tolerance.

## Current defaults

| Quantity | Choice | Notes |
| --- | --- | --- |
| CPU oracle | f32 | Scalar loops in `src/backends/cpu/ops.zig` |
| Metal baseline | f32 | Same formulas; reduction in threadgroup memory |
| Weights / activations in a real model | not loaded | Qwen3-0.6B dtypes are not pinned yet |
| RMSNorm / softmax accumulation | f32 | Stability before speed |
| RoPE | f32 split-half | Matches the CPU Qwen3-style pairing |

fp16 and bf16 exist as `DType` tags. No CPU or Metal kernel uses them
yet. `caps` reports those paths as disabled.

## Tolerance policy

Every numerical test states atol/rtol. Exact equality is not required
for GPU results. Tolerances are not loosened to hide a mismatch.

Current Metal vs CPU checks use roughly `1e-5` for elementwise ops and
`1e-4`–`2e-4` for reductions, matmul, RoPE, and the SwiGLU residual.

On mismatch, `src/runtime/compare.zig` prints max abs, max rel, RMS,
failing index, and expected/actual.

## Questions still open

- FP32 vs BF16 vs FP16 for each tensor class once a checkpoint is read
- Whether softmax/RMSNorm should accumulate in higher precision on GPU
- What quantization changes, once a floating-point baseline model exists
