# Stage M6 — Static decode plan on Apple

**Status: done.** After warm-up, the Apple Qwen hot path is resident weights +
KV to `max_seq` + fixed GPU scratch + Session-owned sample scratch, with
**asserted** zero host heap growth per decode token. ICB/replay stays
**REJECT**.

Part of **Phase M** ([`fable-5-prompt.md`](../../baoulo/prompts/fable-5-prompt.md)).
Old curriculum Stage 20 (Apple half).

## Goal

```bash
zig build stageM6 -Dhip=off
./zig-out/bin/zynfer mem-report --mini
./zig-out/bin/zynfer mem-report models/qwen3-0.6b.zynfer --max-tokens 128
# ITL variance (p50/p95/p99):
./zig-out/bin/zynfer run models/qwen3-0.6b.zynfer \
  --prompt "Explain gravity simply." --max-tokens 16 --temperature 0 --no-stream
```

## What is static

| Piece | Policy |
| --- | --- |
| Weights | GPU-resident at `MetalStack.init` (f32 / bf16 / int8) |
| KV | Preallocated to session `max_seq`; only `used` advances |
| Scratch | Fixed MTLBuffers; no per-token GPU alloc |
| Pipelines | Library + PSO cache once per `Gpu` |
| Sample | `Session.sample_probs` / `sample_idx`; reuse `Session.logits` |
| `out_ids` | `ensureTotalCapacity` then `appendAssumeCapacity` |
| Host twin | Batched Metal: KV/scratch **mirror** (counter only, no tensors) |

## Gate

1. `FailingAllocator` flat across warm decode / `generateCached` (unit tests).
   Mini long run fills `max_seq`; optional `ZYNFER_FULL_MODEL_TESTS=1` for 32
   tokens on qwen3-0.6b.
2. `mem-report` prints weights / KV / scratch / peak RSS (+ JSON); run under
   `ZYNFER_QWEN_METAL=bf16|int8` for path-specific metal bytes.
3. ITL p50/p95/p99 in `run`, `qwen-bench` (default max_new=16 full model), ledger.
4. ICB/replay **REJECT** finalized (Stage 8 + M3 evidence).
5. Tutorial 20.

## ICB (final)

**REJECT.** Per-decode `q_len` / KV length change invalidates a frozen ICB.
Encode count remains high; waits are already 2/forward. No ICB code path.

## Commands

```bash
zig build test -Dhip=off
zig build stageM6 -Dhip=off
zig build integration -Dhip=off
```

## Files

| Path | Role |
| --- | --- |
| `src/model/qwen_forward.zig` | Sample scratch, generate capacity, `memoryReport`, M6 tests |
| `src/model/qwen_block.zig` | `initWithHostCompute` (optional host scratch/KV) |
| `src/runtime/kv_cache.zig` | `initMirror` for Metal-owned KV |
| `src/backends/apple/qwen_schedule.zig` | `residentBytes` / KV byte helpers |
| `src/main.zig` | `stageM6`, `mem-report` |
| `docs/tutorials/20-static-decode-plan.md` | Walkthrough |
| `bench/results/stageM6-dev-laptop.md` | Ledger |
