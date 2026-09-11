# Stage S2 — Prefix reuse / cache management

**Status: done (dense contiguous).** Exact-prefix registry + same-Session
`truncateTo` + `prefillContinue` so repeated shared prompts skip redoing the
prefix prefill. Paged/block KV is **deferred** until contiguous reuse is clear.

Part of **Phase S** ([`fable-5-prompt.md`](../../baoulo/prompts/fable-5-prompt.md)).
Old curriculum Stage 22.

## Goal

```bash
zig build stageS2 -Dhip=off
./zig-out/bin/zynfer stageS2
./zig-out/bin/zynfer prefix-bench --mini --prefix-len 8 --suffix-len 2 --trials 4
ZYNFER_QWEN_METAL=int8 ./zig-out/bin/zynfer prefix-bench \
  models/qwen3-0.6b-int8.zynfer --prefix-len 64 --suffix-len 8 --trials 4 --backend apple
```

## What landed

| Piece | Policy |
| --- | --- |
| Prefix identity | Exact token-id sequences in `PrefixCache` |
| Lookup | Longest registered exact prefix of a prompt |
| Eviction | LRU at `max_entries` |
| KV reuse | `Session.truncateTo(n)` + `prefillContinue(suffix)` |
| Layout | Dense contiguous `used` counter (capacity beyond `n` ignored) |
| Metrics | Cold vs warm `prefill_ns`, `savings_ratio`, logits match |

## Explicit non-goals

- Paged / block-allocator KV
- Packed `n_seq` Metal continuous batching
- Speculative / MTP (S3)
- HTTP (S4)

## Gate

1. Warm suffix logits match cold full prefill (mini unit + `prefix-bench`)
2. `prefix-bench` shows warm trial `prefill_ns` ≪ cold; `savings_ratio > 0` + JSON
3. Tutorial covers identity, truncate, fragmentation vs paged
4. Non-goals documented above

## Commands

```bash
zig build test -Dhip=off
zig build stageS2 -Dhip=off
zig build prefix-bench -Dhip=off
zig build integration -Dhip=off
```

## Files

| Path | Role |
| --- | --- |
| `src/runtime/prefix_cache.zig` | `PrefixCache`, `runColdWarmPrefill` |
| `src/model/qwen_forward.zig` | `kvLen`, `truncateTo`, `prefillContinue` |
| `src/runtime/kv_cache.zig` | `truncateTo` |
| `src/backends/apple/qwen_schedule.zig` | Metal `truncateTo` |
| `src/main.zig` | `stageS2`, `prefix-bench` |
| `docs/tutorials/24-prefix-reuse.md` | Walkthrough |
| `bench/results/stageS2-dev-laptop.md` | Ledger |
