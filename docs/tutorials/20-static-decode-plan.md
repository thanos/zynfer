# Tutorial — Static decode plan on Apple Silicon (Stage M6)

A fast decode path that still mallocs every token is not finished. Stage M6
makes the Apple Qwen hot path **boring**: allocate everything at session init
(or first warm-up), then only mutate buffers and counters.

## What “static” means here

1. **Weights** live in Metal buffers for the whole session.
2. **KV** is sized to the session context bound (`max_seq`) once.
3. **Scratch** (activations, scores, logits buffer on GPU) is fixed.
4. **Pipelines** compile once and hit the PSO cache.
5. **Sampling** uses Session-owned `probs`/`idx` — not fresh heap slices.
6. **Output ids** grow into a pre-reserved `ArrayList` capacity.

Batched Metal also drops the old host per-layer KV/scratch twin: the host only
mirrors `used` so the Session API stays consistent.

## Prove it

```bash
# Unit tests use FailingAllocator around decode / generateCached
zig build test -Dhip=off

# Memory breakdown
./zig-out/bin/zynfer mem-report --mini
./zig-out/bin/zynfer mem-report models/qwen3-0.6b.zynfer --max-tokens 128
```

Look for `host_kv_cap = 0` on Apple batched (mirror), non-zero `metal_*`, and
a `peak_rss` line.

## Latency variance

`zynfer run` records inter-token latency. Use enough tokens that p95 is meaningful:

```bash
./zig-out/bin/zynfer run models/qwen3-0.6b.zynfer \
  --prompt "Explain gravity simply." \
  --max-tokens 32 --temperature 0 --no-stream
```

Read `itl_ms p50/p95/p99`. `qwen-bench` also prints ITL per backend (default
`max_new=16` on full model when `--max-tokens` is omitted) and includes
`itl_p50_ns` / `itl_p95_ns` / `itl_p99_ns` in JSON.

## Why not ICB

Indirect command buffers / encode-once replay looked attractive when encode
dominated. At Qwen scale the KV length (and attention `q_len`) still change
every decode step, so a frozen ICB does not apply. Stage 8 and M3 already
rejected ICB with that evidence; M6 **finalizes REJECT** — no zombie path.

**M7** — ANE / Core ML at Qwen scale: **REJECT (final)** —
`docs/tutorials/21-the-neural-engine-question.md`.

## Next

**M8** — Apple capstone: quantized Qwen3-4B + final benchmark matrix
([`docs/stages/M8-apple-capstone.md`](../stages/M8-apple-capstone.md)).