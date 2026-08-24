# Stage M4 — bf16/fp16 Metal weights + KV

**Status: done.** Halves GPU-resident weight and KV bytes before quantization.

Part of **Phase M** ([`fable-5-prompt.md`](../../baoulo/prompts/fable-5-prompt.md)).
Reopens the Stage 8 **reject** on fp16/bf16 Metal.

## Goal

```bash
zig build stageM4 -Dhip=off
ZYNFER_QWEN_METAL=bf16 ./zig-out/bin/zynfer qwen-bench --mini --max-tokens 4
ZYNFER_QWEN_METAL=bf16 ./zig-out/bin/zynfer qwen-bench models/qwen3-0.6b.zynfer \
  --prompt "Explain gravity simply." --max-tokens 2
```

Default Apple path remains M3 f32 (`batched_resident_kv_fused`).
Half path: `ZYNFER_QWEN_METAL=bf16|half|fp16` → `batched_resident_kv_bf16`.

## Numerics

| Tensor class | Storage | Compute |
| --- | --- | --- |
| Weights (all projections + norms) | **bf16** on GPU | widen to f32 in kernel |
| KV cache | **bf16** on GPU | attention dot/softmax in f32 |
| Activations | **f32** | RoPE, SiLU, residual adds |
| Softmax / RMSNorm reductions | f32 | same as M3 |

Tolerance vs CPU f32 oracle: **5e-3 atol** (BF16 ~3–4 decimal digits).

## Gate

1. Mini batched bf16 logits within 5e-3 of CPU; f32 batched still 3e-3.
2. Greedy tokens match CPU on mini; full-model greedy behind
   `ZYNFER_FULL_MODEL_TESTS=1` (PASS recorded in ledger).
3. `bytes_per_tok_est` halves when half path active.
4. Measured A/B + roofline in `bench/results/stageM4-dev-laptop.md`
   (prefill ~1.5×; decode flat on naive kernels — characterized).
5. Native artifact f16/bf16 → GPU upload (`copyArtifactToBf16`).
6. Tutorial 18.

## Commands

```bash
zig build test -Dhip=off
zig build stageM4 -Dhip=off
zig build integration -Dhip=off
```

## Files

| Path | Role |
| --- | --- |
| `src/backends/apple/kernels.metal` | `matmul_bf16_f32`, `matvec_bf16_f32`, KV + attention bf16 |
| `src/backends/apple/ops.zig` | encode*Bf16 + fp16 matmul/matvec API |
| `src/backends/apple/qwen_schedule.zig` | `MetalStack.half_mode`, path `batched_resident_kv_bf16` |
| `src/runtime/bf16.zig` / `float16.zig` | encode/decode; GPU upload helpers |
| `src/model/qwen_weights.zig` | `copyArtifactToBf16` native half upload |
| `docs/tutorials/18-half-precision-inference.md` | Walkthrough |
| `bench/results/stageM4-dev-laptop.md` | Before/after bytes + bench |
