# Stage S1 — batching and scheduling (dev laptop)

Request-level FIFO admit + round-robin decode over independent Sessions.
Not Metal CB packing (M3); not packed `n_seq` forward; not HTTP.

```text
date:              2026-09-10
host:              MacBook-Pro (Apple Silicon)
OS:                Darwin 25.5.0 (arm64)
Zig:               0.16.0
commands:
  zig build test -Dhip=off
  zig build stageS1 -Dhip=off
  zig build batch-bench -Dhip=off
  ./zig-out/bin/zynfer batch-bench --mini --batch-size 2 --max-inflight 2 --max-tokens 4 --backend cpu
  ./zig-out/bin/zynfer batch-bench --mini --batch-size 4 --max-inflight 2 --max-tokens 4 --backend cpu
  ZYNFER_QWEN_METAL=int8 ./zig-out/bin/zynfer batch-bench \
    models/qwen3-0.6b-int8.zynfer --batch-size 2 --max-inflight 2 \
    --max-tokens 8 --backend apple --seed 1
```

## Correctness

| Check | Result |
| --- | --- |
| Unit: greedy token parity serial vs scheduled (mini, inflight=2) | PASS |
| `batch-bench --mini` token_parity | PASS |
| Apple 0.6B int8 `batch-bench` token_parity | PASS |
| Empty batch / zero inflight reject | PASS |

## Mini CPU A/B (representative run)

`batch_size=2`, `max_inflight=2`, `max_new=4`, backend=cpu

| Mode | wall_ms | agg_tok/s | notes |
| --- | ---: | ---: | --- |
| serial | ~9.1 | ~877 | one Session at a time |
| scheduled | ~10.0 | ~797 | 2 slots; RR decode |

Per-request (scheduled): req0 TTFT ≈ serial TTFT; req1 waits in queue then
shares the RR wheel — mean ITL rises (fairness tax). Aggregate wall is **not**
expected to beat serial on a single CPU thread with no async overlap; the
stage proves **concurrency + metrics + parity**, not a throughput win on mini.

`batch_size=4`, `max_inflight=2`: token_parity PASS; later requests show
higher `queue_ms` / TTFT as slots stay full.

## Apple Qwen3-0.6B int8 A/B

`models/qwen3-0.6b-int8.zynfer`, `ZYNFER_QWEN_METAL=int8`, backend=apple,
`batch_size=2`, `max_inflight=2`, `max_new=8`, `seed=1`

| Mode | wall_ms | agg_tok/s | gen_tok | token_parity |
| --- | ---: | ---: | ---: | --- |
| serial | 1470.6 | 10.88 | 16 | — |
| scheduled | **1289.8** | **12.41** | 16 | PASS |

Per-request (scheduled):

| id | ttft_ms | e2e_ms | mean_itl_ms | tok/s | queue_ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 251.8 | 1229.3 | 139.6 | 6.51 | 0.2 |
| 1 | 569.0 | 1258.4 | 98.5 | 6.36 | 299.4 |

**Reading:** on a real Metal path, scheduled aggregate wall beats serial
(~12% less wall / higher agg tok/s) even though each Session still loads its
own resident weights and RR stretches per-request ITL. That is the Stage 21
lesson: concurrency can improve **system** throughput while trading
**per-request** latency. Cold Session init (weight upload) still dominates
early wall; shared weight residency across slots is future work, not S1.

## Scheduling policy (documented)

1. Each outer step: admit **at most one** FIFO job into a free slot (prefill).
2. Then round-robin **one** sample/decode on an alive slot.
3. Each request owns its Session/KV (no sharing).

## Non-goals (unchanged)

- Packed continuous batch Metal forward
- Prefix / paged KV (S2)
- Speculative (S3)
- HTTP (S4)

## See also

- `docs/stages/S1-batching-and-scheduling.md`
- `docs/tutorials/23-batching-and-scheduling.md`
