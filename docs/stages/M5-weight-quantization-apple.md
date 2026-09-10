# Stage M5 — Weight quantization on Apple

**Status: done (int8).** Per-row symmetric int8 projections on the Qwen Metal
path; 4-bit deferred until int8 decode is measurably faster *and* quality
allows.

Part of **Phase M** ([`fable-5-prompt.md`](../../baoulo/prompts/fable-5-prompt.md)).
Old curriculum Stage 18 (Apple half). Reopens Stage 8 Session-int8 reject.

## Goal

```bash
zig build stageM5 -Dhip=off
ZYNFER_QWEN_METAL=int8 ./zig-out/bin/zynfer qwen-bench --mini --max-tokens 4
ZYNFER_QWEN_METAL=int8 ./zig-out/bin/zynfer qwen-bench models/qwen3-0.6b.zynfer \
  --prompt "Explain gravity simply." --max-tokens 2
# vs M4:
ZYNFER_QWEN_METAL=bf16 ./zig-out/bin/zynfer qwen-bench models/qwen3-0.6b.zynfer \
  --prompt "Explain gravity simply." --max-tokens 2
```

Default Apple path remains M3 f32. Half: `bf16`. Int8: `int8|q8`.

## Scheme (measured choice)

| Property | Choice |
| --- | --- |
| Group | **Full output row** (per-channel / per-row) |
| Scale | `max_abs(row) / 127`, symmetric, **no zero-point** |
| Storage | i8 `[out, in]` (HF layout) + f32 `scale[out]` |
| Pack site | Prefer on-disk i8 + `.qscale` → Metal; else pack host f32 `[in,out]` |
| Dequant | **Fused** in `matmul_aq8_f32` / `matvec_q8_f32` |
| Not quantized | Norms (f32); embed table (bf16 gather); activations (f32) |
| KV | **bf16** on the int8 Metal path (post-M8; was f32 in M5 v1) |

Why per-row (not group-32): already implemented and differentially tested at
ops level (Stage 5); Qwen decode is bandwidth-bound on weight traffic; row
scales are tiny vs weight bytes. Sub-row groups deferred if quality needs them.

## Gate

1. Mini int8 logits within **5e-2** of CPU (packing-justified).
2. `B/tok_est` via `estimateDecodeBytesPerTokenQ8` ≪ bf16 estimate.
3. Full-model A/B vs M4 bf16 in ledger (decode and/or prefill).
4. Quality: token-drift / logit proxy documented separately from speed.
5. Converter + CPU decoder (`quantize_zynfer_int8.py`, `qwen_quant`) —
   smoke: worst abs err ≈ 4.9e-3 on qwen3-0.6b.
6. Tutorial 19.
7. Full-model greedy under `ZYNFER_FULL_MODEL_TESTS=1`.

## Commands

```bash
zig build test -Dhip=off
zig build stageM5 -Dhip=off
zig build integration -Dhip=off
python3 tools/checkpoint/quantize_zynfer_int8.py \
  --in models/qwen3-0.6b.zynfer --out models/qwen3-0.6b-int8.zynfer
```

## Files

| Path | Role |
| --- | --- |
| `src/model/qwen_quant.zig` | Scheme + pack/dequant CPU decoder |
| `src/backends/apple/kernels.metal` | `matmul_aq8_f32` |
| `src/backends/apple/qwen_schedule.zig` | `LayerQ8Weights`, artifact i8 upload, bf16 KV, path `batched_resident_kv_q8` |
| `src/model/qwen_weights.zig` | `loadForAppleQ8` (norms-only host when artifact is i8) |
| `tools/checkpoint/quantize_zynfer_int8.py` | Dev-time i8 artifact writer |
| `docs/tutorials/19-quantization-on-apple-silicon.md` | Walkthrough |
| `bench/results/stageM5-dev-laptop.md` | A/B + quality |
