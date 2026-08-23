# Stage M2 — Profile one decode token

**Status: done.** One Metal (or CPU) decode token is accounted for by
family, with measured bandwidth and a transparent roofline.

Part of **Phase M** ([`fable-5-prompt.md`](../../baoulo/prompts/fable-5-prompt.md)).
Old curriculum Stage 15.

## Goal

```bash
zig build qwen-profile -Dhip=off
./zig-out/bin/zynfer qwen-profile --mini
# full model (local):
./zig-out/bin/zynfer qwen-profile models/qwen3-0.6b.zynfer \
  --prompt "Explain gravity simply."
# Instruments labels:
ZYNFER_SIGNPOSTS=1 ./zig-out/bin/zynfer qwen-profile --mini
```

Answer: **where does one decode token go?** — with a table, top-3, and
JSON.

## Why this stage

M1 showed prefill vs decode as separate regimes. M2 names the costs
*inside* one decode token before any fusion (M3). Optimizing the wrong
family wastes the next stage.

## Metrics

| Item | Source |
| --- | --- |
| Per-family wall ms | timed sections in `qwen_block.forward` + embed / LM head / sample |
| Families | RMSNorm, QKV, RoPE, attention, O-proj, MLP, host layout, embed, LM head, sampling |
| Metal launches | `Gpu.total_encodes` / `total_waits` for the profiled token |
| Empty encode+wait | microbench → estimated launch overhead (embedded in Metal families on M0 path) |
| Bandwidth | Metal STREAM triad (`stream_triad_f32`), measured GB/s |
| Roofline | `ideal_tok_s ≈ BW / B/tok_est`; fraction = measured / ideal |
| Signposts | `ZYNFER_SIGNPOSTS=1` → `qwen.*` family intervals + existing encode/batch |

## In scope

| Item | Notes |
| --- | --- |
| `zynfer qwen-profile` | Human table + top3 + json |
| `ProfilingAdapter` | Metal ops + buckets + optional signposts |
| STREAM bandwidth | `apple.ops.measureSustainableBandwidth` |

## Explicitly not M2

| Item | Owner |
| --- | --- |
| One-CB / fusion ledger | M3 |
| Moving LM head to Metal | M0 open / later |
| Closing remaining M0 gate items | Still open |

## Gate

1. One command → per-family table + top3 on Apple (Mac).
2. Mini fixture works without weights.
3. Documented answer names the top three costs.
4. Tutorial + ledger.

## Commands

```bash
zig build test -Dhip=off
zig build stageM2 -Dhip=off
zig build qwen-profile -Dhip=off
```

## Files

| Path | Role |
| --- | --- |
| `src/model/decode_profile.zig` | Families, top3, roofline helpers |
| `src/model/qwen_block.zig` | Timed / signed family sections |
| `src/backends/apple/qwen_adapter.zig` | `ProfilingAdapter` |
| `src/backends/apple/ops.zig` | STREAM + empty-launch microbenches |
| `src/main.zig` | `qwen-profile`, `stageM2` |
| `docs/tutorials/16-profiling-a-token-on-apple.md` | Walkthrough |
| `bench/results/stageM2-dev-laptop.md` | Local numbers |
