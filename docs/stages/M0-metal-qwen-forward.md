# Stage M0 — Metal Qwen forward and generate

**Status: in progress (baseline landed; gate not closed).** First Qwen3
forward and KV-cached generate on Metal, f32, correctness before speed.

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

## Open — required to close the M0 gate

Do **not** mark M0 done until these are checked off. Source of truth:
[`fable-5-prompt.md`](../../baoulo/prompts/fable-5-prompt.md) Stage M0.

| # | Item | Status | Notes |
| --- | --- | --- | --- |
| 1 | Full-model Metal greedy tokens match CPU golden on fixture prompts | **OPEN** | Mini logits only today; need short `chat`/`run --backend apple` vs cpu parity |
| 2 | First honest Metal TTFT / decode tok/s recorded | **OPEN** | Use `qwen-bench` (M1) on full model; fill `bench/results/stageM0-dev-laptop.md` |
| 3 | Per-layer differential ladder (embed → block0 → … → logits → tokens) | **OPEN** | Stage 11 dump-hook style for Metal |
| 4 | LM-head GEMV path selection (naive vs simdgroup vs Accelerate) | **OPEN** | Vocab 151936; measure, do not assume; still CPU in M0 |
| 5 | Ceiling differential tests at new `kv_len` boundary | **OPEN** | Parity at 257 / 512 / 1024 (device scores path) |
| 6 | Stage 6 reuse where free (persistent weights / resident KV) | **DONE (M3)** | `qwen_schedule` batched_resident_kv_fused |

## Gate (exit criterion)

1. Greedy tokens / logits match CPU golden on fixture prompts (**full model**, not only mini).
2. Per-layer tolerances documented.
3. First honest Metal TTFT / decode tok/s reported (expected: slow vs later M3).
4. `zynfer stageM0` ledger + tutorial + bench note.

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
