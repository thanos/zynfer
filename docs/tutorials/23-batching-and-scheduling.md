# Tutorial — Batching and scheduling (Stage S1)

Moving from a single-prompt demo toward a real inference server starts with
**request scheduling**, not with packing more sequences into one GPU kernel.

## In plain English

You have two users. Each needs their own conversation state (KV cache). A
simple engine runs them **one after another**. A scheduler keeps several
sessions **in flight** and takes turns generating the next token — so while
one request is waiting on the GPU/CPU for its decode, another can be admitted
or stepped.

That is **not** the same as Stage M3’s Metal **command-buffer packing**, which
batches *ops inside one forward* for one sequence. S1 batches *requests* at
the session level.

## Latency vs throughput

| Mode | What you optimize | Typical effect |
| --- | --- | --- |
| Serial | Per-request wall time | Best TTFT for the first job; poor machine utilization with a queue |
| Scheduled (`max_inflight>1`) | Aggregate tokens/sec across the batch | Higher system throughput; each request’s ITL may stretch (fairness trade) |

S1 reports both: **aggregate tok/s** (batch wall) and **per-request tok/s /
TTFT / mean ITL**.

## What zynfer does in S1

1. **FIFO admit (one per tick)** — a free slot takes the next queued job and
   runs **prefill** (at most one admit before the next decode step).
2. **Round-robin decode** — among alive slots, take one sample (and decode
   unless finished). Fairness is “one token each,” not equal wall time.
3. **Independent Sessions** — no shared KV, no prefix cache (that is S2).

```text
queue ──► [slot0 Session] ──┐
         [slot1 Session] ──┼── round-robin decode steps
         [slotN Session] ──┘
```

## Continuous batching (concept vs this code)

Industry “continuous batching” often means: as soon as a sequence finishes,
its slot is filled by a waiting request **without draining the whole batch**,
and sometimes tokens from many sequences share one packed forward.

S1 does the **admit-when-slot-free** half (continuous in the queue sense).
It does **not** pack multiple sequences into one Metal matmul. Padding waste
and shared-kernel continuous batching stay future work.

## Try it

```bash
zig build stageS1 -Dhip=off
./zig-out/bin/zynfer batch-bench --mini --batch-size 2 --max-inflight 2 --max-tokens 4
```

Expect `token_parity: PASS` (greedy serial vs scheduled) and a JSON line with
concurrency metrics.

## What is not here

- Prefix reuse / paged KV — Stage S2 (tutorial 24; dense path done; paged deferred)
- Speculative decoding — Stage S3
- HTTP — Stage S4
- Metal CB fusion — Stage M3 (already done)

## See also

- [`docs/stages/S1-batching-and-scheduling.md`](../stages/S1-batching-and-scheduling.md)
- [`bench/results/stageS1-dev-laptop.md`](../../bench/results/stageS1-dev-laptop.md)
- Stage M3 tutorial for the other meaning of “batching”
