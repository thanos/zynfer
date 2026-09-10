# Stage S1 — Batching and scheduling

**Status: done (request-level).** FIFO admit into `max_inflight` independent
`Session` slots; round-robin one decode step across active requests. Serial
generate is the A/B baseline. Packed continuous batching in one Metal forward
and HTTP serving are **not** this stage.

Part of **Phase S** ([`fable-5-prompt.md`](../../baoulo/prompts/fable-5-prompt.md)).
Old curriculum Stage 21.

## Goal

```bash
zig build stageS1 -Dhip=off
./zig-out/bin/zynfer stageS1
./zig-out/bin/zynfer batch-bench --mini --batch-size 2 --max-tokens 4
./zig-out/bin/zynfer batch-bench --mini --batch-size 4 --max-inflight 2 --max-tokens 4
```

## What landed

| Piece | Policy |
| --- | --- |
| Admit | FIFO; at most one new job per scheduler tick into a free slot |
| Decode | Round-robin: one sample (+ optional `decodeToken`) per tick |
| Cache | Each request owns its `Session` / KV (no sharing) |
| Baseline | `runSerial` = one Session at a time |
| Metrics | concurrency, aggregate tok/s, per-req tok/s, TTFT, mean ITL, queue wait |
| Parity | greedy (`temperature=0`) token ids match serial vs scheduled |

## Explicit non-goals

- Packed `n_seq` Metal forward / continuous batch in one command buffer
- Prefix / paged KV reuse (S2)
- Speculative / MTP (S3)
- HTTP server (S4)
- Confusing this with **Metal CB packing** (Stage M3)

## Gate

1. ≥2 scheduled requests; greedy token parity vs serial (mini unit + `batch-bench`)
2. `batch-bench` reports TTFT / aggregate tok/s + JSON; serial A/B
3. Tutorial distinguishes request scheduling vs Metal CB batching
4. Non-goals documented above

## Commands

```bash
zig build test -Dhip=off
zig build stageS1 -Dhip=off
zig build batch-bench -Dhip=off
zig build integration -Dhip=off
```

## Files

| Path | Role |
| --- | --- |
| `src/runtime/scheduler.zig` | `Job`, `runSerial`, `runScheduled`, unit tests |
| `src/main.zig` | `stageS1`, `batch-bench` |
| `docs/tutorials/23-batching-and-scheduling.md` | Walkthrough |
| `bench/results/stageS1-dev-laptop.md` | Ledger |
