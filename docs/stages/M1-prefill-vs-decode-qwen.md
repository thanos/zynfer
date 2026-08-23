# Stage M1 — Prefill vs decode on Qwen

**Status: done.** Every Qwen generate path reports prefill and decode as
separate regimes; `qwen-bench` emits a CPU + Metal split table and JSON.

Part of **Phase M** ([`fable-5-prompt.md`](../../baoulo/prompts/fable-5-prompt.md)).
Old curriculum Stage 14.

## Goal

```bash
zig build qwen-bench -Dhip=off
./zig-out/bin/zynfer qwen-bench --mini --max-tokens 4
# full model (local):
./zig-out/bin/zynfer qwen-bench models/qwen3-0.6b.zynfer \
  --prompt "Explain gravity simply." --max-tokens 8
```

One command prints prefill and decode metrics on **cpu** and **apple**
(when Metal is buildable).

## Why this stage

Prefill ≈ large GEMMs (compute-rich). Decode ≈ small matvecs + KV reads
(bandwidth / launch-bound). Optimizing one can hurt the other; M1 makes
the split mandatory so later stages (M2–M3) do not blur the regimes.

## Metrics

| Regime | Fields |
| --- | --- |
| Prefill | `prefill_ms`, `prefill_tok_s`, GEMM shapes for prompt length `t` |
| Decode | `decode_tok_s`, `decode_ms_per_tok`, `B/tok_est`, measured `enc/tok` / `wait/tok` |
| Shared | `ttft_ms`, prompt/generated token counts |

`enc/tok` and `wait/tok` are **measured** Metal launch counters from
`Gpu.total_encodes` / `total_waits` during each `decodeToken` (M0 per-op
path: encode ≈ wait). CPU reports 0.

`B/tok_est` ≈ f32 weight reads + KV read at end-of-run `kv_len`.

## In scope

| Item | Notes |
| --- | --- |
| `zynfer qwen-bench` | Human table + `json` line |
| `run` / `chat` footer | Always prints `prefill_tok_s` and `decode_ms_per_tok` |
| Arch helpers | `describePrefillGemms`, `estimateDecodeBytesPerToken`, … |

## Explicitly not M1

| Item | Owner |
| --- | --- |
| Per-op decode profile / roofline | M2 |
| Batched one-CB Metal schedule | M3 |
| Closing remaining M0 gate items | Still open — see M0 stage doc |

## Gate

1. One command → split report on both backends (Mac).
2. Mini fixture works in CI without weights.
3. Tutorial + ledger + bench note.

## Commands

```bash
zig build test -Dhip=off
zig build stageM1 -Dhip=off
zig build qwen-bench -Dhip=off
```

## Files

| Path | Role |
| --- | --- |
| `src/main.zig` | `qwen-bench`, `stageM1`, split footer |
| `src/model/qwen3.zig` | GEMM / bytes / launch estimates |
| `docs/tutorials/15-prefill-vs-decode-on-qwen.md` | Walkthrough |
| `bench/results/stageM1-dev-laptop.md` | Local numbers |
