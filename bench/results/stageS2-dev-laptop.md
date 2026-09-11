# Stage S2 — prefix reuse / cache management (dev laptop)

Dense contiguous KV: `truncateTo` + `prefillContinue`. Exact `PrefixCache`
identity / LRU. Not paged KV; not packed continuous batching; not HTTP.

```text
date:              2026-09-10
host:              MacBook-Pro (Apple Silicon)
OS:                Darwin 25.5.0 (arm64)
Zig:               0.16.0
commands:
  zig build test -Dhip=off
  zig build stageS2 -Dhip=off
  zig build prefix-bench -Dhip=off
  ./zig-out/bin/zynfer prefix-bench --mini --prefix-len 8 --suffix-len 2 --trials 4 --backend cpu
  ZYNFER_QWEN_METAL=int8 ./zig-out/bin/zynfer prefix-bench \
    models/qwen3-0.6b-int8.zynfer --prefix-len 64 --suffix-len 8 --trials 4 --backend apple
```

## Correctness

| Check | Result |
| --- | --- |
| Unit: cold vs warm logits (mini) | PASS |
| PrefixCache insert / longest lookup / LRU | PASS |
| `prefix-bench --mini` logits_match | PASS |
| Apple 0.6B int8 `prefix-bench` logits_match | PASS |

## Mini CPU A/B

`prefix_len=8`, `suffix_len=2`, `trials=4`, backend=cpu

| Mode | prefill_ms (sum) | tokens | savings_ratio |
| --- | ---: | ---: | ---: |
| cold | 0.699 | 40 | — |
| warm (suffix trials) | 0.169 | 8 | **0.758** |
| warm prefix (once) | 0.232 | 8 | (excluded from ratio) |

## Apple Qwen3-0.6B int8 A/B

`ZYNFER_QWEN_METAL=int8`, backend=apple, `prefix_len=64`, `suffix_len=8`, `trials=4`

| Mode | prefill_ms (sum) | tokens | savings_ratio |
| --- | ---: | ---: | ---: |
| cold | 1533.8 | 288 | — |
| warm (suffix trials) | 420.3 | 32 | **0.726** |
| warm prefix (once) | 364.4 | 64 | (excluded from ratio) |

**Reading:** skipping the shared 64-token prefix on each trial cuts trial
prefill wall by ~73% at matched logits. One-time warm prefix cost amortizes
across turns that share the same system/tool preamble. Dense truncate leaves
capacity beyond `used` allocated (documented “fragmentation” lesson vs paged KV).

## Policy

1. Register exact prefix identities in `PrefixCache` (LRU eviction).
2. Prime Session with `prefill(P)`; later `truncateTo(|P|)` + `prefillContinue(S)`.
3. No cross-Session buffer sharing in S2.

## Non-goals (unchanged)

- Paged / block-allocator KV
- Packed `n_seq` Metal continuous batching
- Speculative (S3)
- HTTP (S4)

## See also

- `docs/stages/S2-prefix-reuse.md`
- `docs/tutorials/24-prefix-reuse.md`
