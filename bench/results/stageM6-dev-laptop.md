# Stage M6 — static decode plan (dev laptop)

```text
date:              2026-08-24
host:              MacBook-Pro (Apple Silicon)
Zig:               0.16.0
commands:
  zig build test -Dhip=off
  ZYNFER_FULL_MODEL_TESTS=1 zig build test -Dhip=off   # optional long alloc
  zig build stageM6 -Dhip=off
  ./zig-out/bin/zynfer mem-report models/qwen3-0.6b.zynfer --max-tokens 128
  ZYNFER_QWEN_METAL=bf16 ./zig-out/bin/zynfer mem-report models/qwen3-0.6b.zynfer --max-tokens 128
  ZYNFER_QWEN_METAL=int8 ./zig-out/bin/zynfer mem-report models/qwen3-0.6b.zynfer --max-tokens 128
  ZYNFER_QWEN_METAL=bf16 ./zig-out/bin/zynfer run models/qwen3-0.6b.zynfer \
    --prompt "Explain gravity simply." --max-tokens 32 --temperature 0 --no-stream
  ZYNFER_QWEN_METAL=bf16 ./zig-out/bin/zynfer qwen-bench models/qwen3-0.6b.zynfer \
    --prompt "Explain gravity simply."
```

## Correctness (allocation)

| Check | Result |
| --- | --- |
| Metal `decodeToken` ×8 after warm-up | PASS — `FailingAllocator` flat |
| `generateCached` 12 tokens after capacity reserve | PASS — flat |
| Mini long run (30 decode @ max_seq=32) | PASS — flat |
| Full-model long generate (32 tok, `ZYNFER_FULL_MODEL_TESTS=1`) | PASS — flat after warm-up |
| Streaming `pending` pre-capacity (`max_tokens×8`) | done in `run`/`chat` |
| Host KV twin on batched Metal | **0 bytes** (`initMirror`) |
| `wait/tok` | **2** (unchanged from M3) |

## Memory report (`max_seq=128`)

| Path | host_weights | metal_weights | metal_kv | metal_scratch | accounted | peak_rss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| f32 Metal (default) | 2.38e9 | 2.38e9 | 2.94e7 | 1.63e7 | 4.82e9 | 6.29e9 |
| bf16 (`ZYNFER_QWEN_METAL=bf16`) | 2.38e9 | **1.19e9** | **1.47e7** | 1.63e7 | 3.61e9 | **5.10e9** |
| int8 (`ZYNFER_QWEN_METAL=int8`) | 2.38e9 | **1.22e9** | 2.94e7 | 1.63e7 | 3.65e9 | 5.13e9 |

Host weights stay f32 (CPU oracle). Metal resident halves on bf16 weights+KV;
int8 metal weights ≈ i8+scales (slightly above half f32 due to scale rows).

## ITL variance (bf16, greedy)

| max_new | decode_ms/tok | itl p50 | itl p95 | itl p99 | n |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 (ledger baseline) | 143 | 131.2 | 137.4 | 137.7 | 15 |
| 32 | 143 | **137.8** | **148.6** | **150.1** | 31 |
| 64 | 153 | **151.6** | **171.3** | **175.1** | 63 |

p95−p50 widens slightly at 64 tokens (~20 ms) — still modest; thermal/scheduling
not dominant on this run.

## qwen-bench (Stage M6)

- Default full-model `max_new` raised **8 → 16** when `--max-tokens` omitted.
- Apple + CPU rows print **ITL p50/p95/p99**; JSON includes `itl_*` fields.
- Example apple ITL @ max_new=16: p50=131 ms, p95=138 ms, p99=140 ms.

## ICB / replay

| Decision | **REJECT** (final) |
| --- | --- |
| Reason | KV / attention `q_len` change every decode; frozen ICB invalid |
| Evidence | Stage 8 ledger + M3 fusion ledger |
| Code | none |

## Retained

| Item | Reason |
| --- | --- |
| Session sample scratch + logits reuse | Zero alloc/token |
| Host KV/scratch mirror on batched Metal | RSS; Metal owns real buffers |
| `mem-report` CLI | Gate visibility |
| Long alloc tests + qwen-bench ITL | M6 polish |

## Deferred

| Item | Notes |
| --- | --- |
| Skip host f32 weight twin on Apple-only sessions | Still needed for load/oracle; large RSS |
| Runtime load of on-disk i8 without host f32 pack | From M5 polish |
| ICB | permanently rejected unless shapes freeze |
