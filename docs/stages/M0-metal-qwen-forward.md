# Stage M0 — Metal Qwen forward and generate

**Status: done (baseline; residual checklist closed by M1–M8).** First Qwen3
forward and KV-cached generate on Metal, f32, correctness before speed.
Curriculum Phase M is Apple-complete at M8; this doc keeps the M0 history.

Part of **Phase M** ([`fable-5-prompt.md`](../../baoulo/prompts/fable-5-prompt.md)).

## Goal

```bash
./zig-out/bin/zynfer forward-golden zig-out/stage11-mini.zynfer \
  --tokens 2,3 --backend apple
./zig-out/bin/zynfer run models/qwen3-0.6b.zynfer \
  --prompt "Hello" --backend apple --max-tokens 8 --no-stream
```

Metal greedy logits/tokens match CPU within documented f32 tolerances.

## What changed from the tiny-block path

| Tiny block (Apple 0–8) | Qwen M0 |
| --- | --- |
| `hidden=8`, `max_seq=32` | `hidden=1024`, real vocab |
| No QK-norm | Qwen3 QK-norm before RoPE |
| `apple/block.zig` batched schedule | Per-op `qwen_adapter` (M3 adds batching) |
| `kv_len ≤ 256` only | **≤ 2048** (`attention_f32_buf` device scores) |

Embed, final RMSNorm, and LM head stay on **CPU** in M0 (blocks on Metal).

## Done (baseline)

| Item | Location |
| --- | --- |
| Backend routing | `Session.initWithBackend`, `forwardBlock` |
| Metal block adapter | `src/backends/apple/qwen_adapter.zig` |
| Attention cap lift | `attention_f32` + `attention_f32_buf`, cap 2048 |
| CLI | `--backend apple` on `run`, `chat`, `forward-golden` |
| CI test | mini artifact Metal vs CPU logits |
| Docs | this file, tutorial 14, `stageM0` ledger |

## Open checklist — closed by later stages

Do **not** reopen M0 for these; they were owned by M1–M8.

| # | Item | Closed by |
| --- | --- | --- |
| 1 | Full-model Metal greedy vs CPU | M4/M5 full-model tests + M8 chat |
| 2 | Honest Metal TTFT / decode tok/s | M1 `qwen-bench` + capstone ledger |
| 3 | Per-layer differential ladder | M2 profile + M3 schedule |
| 4 | LM-head path on Metal | M3+ (Q8 / bf16 matvec on stack) |
| 5 | `kv_len` ceiling parity | M3 attention cap 2048 + tests |
| 6 | Persistent weights / resident KV | **M3** (`qwen_schedule`) |

## Gate (exit criterion) — satisfied for curriculum

1. Mini Metal vs CPU logits (CI).
2. Full-model paths measured in later M stages / M8 ledger.
3. `zynfer stageM0` + tutorial 14 remain as the baseline entry point.

## Not M0

| Item | Owner |
| --- | --- |
| One-CB / 28-layer batched schedule | M3 |
| Metal embed / LM head end-to-end | M3+ |
| fp16 / quant | M4–M5 |
| Context > 2048 on Metal | M3 streaming / tiling |
| ANE | M7 |
| Prefill/decode split report | **M1 (done)** |

## Commands

```bash
zig build test -Dhip=off
zig build stageM0 -Dhip=off

./zig-out/bin/zynfer forward-golden zig-out/stage11-mini.zynfer \
  --tokens 2,3 --backend cpu
./zig-out/bin/zynfer forward-golden zig-out/stage11-mini.zynfer \
  --tokens 2,3 --backend apple
```

## Files

| Path | Role |
| --- | --- |
| `src/backends/apple/qwen_adapter.zig` | Per-op Metal adapter for `qwen_block` |
| `src/model/qwen_forward.zig` | `initWithBackend`, `forwardBlock` |
| `src/backends/apple/kernels.metal` | `attention_f32_buf` |
| `docs/tutorials/14-qwen-on-metal.md` | Walkthrough |
| `bench/results/stageM0-dev-laptop.md` | Local numbers + open checklist |
