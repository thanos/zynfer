# Numerics

The CPU backend is the floating-point oracle. Accelerated ops are
accepted only when they match it within an explicit tolerance.

## Current defaults

| Quantity | Choice | Notes |
| --- | --- | --- |
| CPU oracle | f32 | Scalar loops in `src/backends/cpu/ops.zig` |
| Metal baseline (M3 default) | f32 weights + f32 KV | Batched schedule |
| Metal half path (M4) | **bf16** weights + **bf16** KV | `ZYNFER_QWEN_METAL=bf16`; f32 activations and accumulators |
| Metal int8 path (M5) | **int8** projections + f32 scales; f32 KV | `ZYNFER_QWEN_METAL=int8`; fused dequant in GEMM/GEMV |
| Checkpoint / `.zynfer` | BF16 payloads (tag `2`); optional i8 tag `3` | Converter / `quantize_zynfer_int8.py` |
| RMSNorm / softmax accumulation | f32 | Stability before speed |
| RoPE | f32 split-half | Matches the CPU Qwen3-style pairing |

`caps` reports Apple `fp16`/`bf16` as available. CPU still does not run
half-precision kernels (oracle stays f32).

## Tolerance policy

Every numerical test states atol/rtol. Exact equality is not required
for GPU results. Tolerances are not loosened to hide a mismatch.

Current Metal vs CPU checks use roughly `1e-5` for elementwise ops and
`1e-4`–`3e-4` for reductions, matmul (including simdgroup), RoPE,
attention, SwiGLU, int8 GEMV vs its dequant oracle, and the tiny
transformer block. Quantized vs full-precision f32 uses a looser bound
on purpose (packing error).

**M4 bf16 path vs CPU logits:** **5e-3** atol (BF16 ~3–4 decimal digits).
**M5 int8 path vs CPU logits:** **5e-2** atol (per-row packing error).
Greedy tokens on the mini fixture match CPU for bf16; full-model greedy is gated
behind `ZYNFER_FULL_MODEL_TESTS=1` (slow CPU oracle).

On mismatch, `src/runtime/compare.zig` prints max abs, max rel, RMS,
failing index, and expected/actual.

## Questions still open

- Whether softmax/RMSNorm should accumulate in higher precision on GPU
  beyond current f32 reductions
- 4-bit weights only after int8 shows a measured decode win (M5 ledger)