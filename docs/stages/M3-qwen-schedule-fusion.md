# Stage M3 — Qwen-scale schedule + fusion

**Status: done.** Stage 6’s one-CB / resident-KV discipline, re-earned on
Qwen3 shapes, with a measured fusion retain/reject ledger.

Part of **Phase M** ([`fable-5-prompt.md`](../../baoulo/prompts/fable-5-prompt.md)).
Old curriculum Stage 16 (Apple half).

## Goal

```bash
zig build stageM3 -Dhip=off
./zig-out/bin/zynfer qwen-bench --mini --max-tokens 4
ZYNFER_QWEN_METAL=baseline ./zig-out/bin/zynfer qwen-bench --mini --max-tokens 4
./zig-out/bin/zynfer qwen-bench models/qwen3-0.6b.zynfer \
  --prompt "Explain gravity simply." --max-tokens 2
```

Default Apple path: `batched_resident_kv_fused` (~2 waits/forward).
A/B: `ZYNFER_QWEN_METAL=baseline` → M0 per-op (~17 waits/layer).

## Why this stage

M2 showed launch/upload tax dominating Metal families and CPU LM head as
the largest wall slice. M3 collapses waits and keeps weights/KV resident;
it does **not** invent fusions without a ledger.

## Retained

| Change | Why |
| --- | --- |
| One CB for all layers + one CB for final norm / LM head | Waits 476 → **2** per decode |
| Resident per-layer weights | Kill per-op weight re-upload |
| Metal-resident KV (`kv_append_f32` + GPU permute) | Kill full-KV host upload |
| `silu_mul` | Already fused; free in encode path |
| `add_rmsnorm_f32` (post-attn residual + norm) | Removes one launch/layer |
| Metal LM-head `matvec_f32` | M2 top cost; moves vocab GEMV onto GPU |

## Rejected (with reason)

| Candidate | Decision |
| --- | --- |
| Q/K-prep + RoPE fused kernel | **REJECT** — batching already covers; no measured need for new MSL |
| Attention tiling / online-softmax | **REJECT for M3** — teach later; parity risk at Qwen scale without dedicated kernel work |
| Dequant-into-GEMV | **DEFER → M5** — placeholder only until int8 session weights |
| ICB / encode-once replay | **REJECT** — decode mutates KV every step (same as Stage 8) |

## Gate

1. Measured decode tok/s and TTFT improve vs M0 baseline (itemized).
2. Mini Metal batched logits ≈ CPU (and ≈ baseline Metal).
3. Fusion ledger in `bench/results/stageM3-dev-laptop.md`.
4. Tutorial 17.

## Commands

```bash
zig build test -Dhip=off
zig build stageM3 -Dhip=off
```

## Files

| Path | Role |
| --- | --- |
| `src/backends/apple/qwen_schedule.zig` | Batched stack + resident KV/weights |
| `src/model/qwen_forward.zig` | Routes Apple → `MetalStack` unless baseline |
| `docs/tutorials/17-fusion-at-model-scale.md` | Walkthrough |
| `bench/results/stageM3-dev-laptop.md` | A/B numbers + ledger |
