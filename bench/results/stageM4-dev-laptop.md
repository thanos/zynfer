# Stage M4 — bf16 Metal weights + KV (dev laptop)

```text
date:              2026-08-24
host:              MacBook-Pro (Apple Silicon)
Zig:               0.16.0
commands:
  zig build test -Dhip=off
  ZYNFER_FULL_MODEL_TESTS=1 zig build test -Dhip=off   # optional full greedy
  zig build stageM4 -Dhip=off
  ./zig-out/bin/zynfer qwen-bench models/qwen3-0.6b.zynfer \
    --prompt "Explain gravity simply." --max-tokens 2
  ZYNFER_QWEN_METAL=bf16 ./zig-out/bin/zynfer qwen-bench \
    models/qwen3-0.6b.zynfer --prompt "Explain gravity simply." --max-tokens 2
  ./zig-out/bin/zynfer qwen-profile models/qwen3-0.6b.zynfer \
    --prompt "Explain gravity simply." --backend apple
  ZYNFER_QWEN_METAL=bf16 ./zig-out/bin/zynfer qwen-profile \
    models/qwen3-0.6b.zynfer --prompt "Explain gravity simply." --backend apple
```

## Exit criterion

- Mini bf16 logits within 5e-3 of CPU; greedy tokens match (unit test)
- Full-model greedy CPU vs bf16 Metal: **PASS** (`ZYNFER_FULL_MODEL_TESTS=1`)
- `B/tok_est` halves on half path
- Measured A/B + roofline recorded below
- Native artifact bf16→GPU copy (no f32 promote on upload)

## Bytes / token (estimate @ kv_len≈19)

| Path | `B/tok_est` | Notes |
| --- | ---: | --- |
| M3 f32 | **2 388 295 680** | `estimateDecodeBytesPerToken` |
| M4 bf16 | **1 194 147 840** | `estimateDecodeBytesPerTokenHalf` (÷2) |

## Correctness

| Check | Result |
| --- | --- |
| Mini bf16 vs CPU logits | PASS @ 5e-3 atol |
| Mini greedy tokens CPU vs bf16 | PASS |
| Full-model greedy (17 prompt tok, 2 new) | PASS |
| `wait/tok` bf16 batched | **2** |

## Full-model A/B (qwen3-0.6b, 17 prompt tok, max-tokens 2)

| Path | prefill_ms | prefill_t/s | decode_ms/tok | wait/tok | B/tok_est |
| --- | ---: | ---: | ---: | ---: | ---: |
| M3 f32 batched | 261 | 65.2 | **234** | 2 | 2.39e9 |
| M4 bf16 batched | **170** | **100** | 254 | 2 | **1.19e9** |

| Metric | f32 → bf16 | Delta |
| --- | --- | --- |
| Prefill latency | 261 → 170 ms | **~1.54×** faster |
| Decode ms/tok | 234 → 254 | ~**0.92×** (flat / slight regression) |
| Bytes/tok estimate | 2.39e9 → 1.19e9 | **2×** fewer |

### Roofline (`qwen-profile`, Apple)

| Path | STREAM GB/s | ideal tok/s | measured tok/s | fraction |
| --- | ---: | ---: | ---: | ---: |
| M3 f32 | 192 | 80.2 | 8.43 | **0.105** |
| M4 bf16 | 211 | 177 | 7.94 | **0.045** |

Ideal doubles with halved bytes, but **measured decode does not** on this
naive matmul path: still encode/compute bound (590 encodes/tok), not
memory-bandwidth bound enough for a 2× decode win. Prefill (larger GEMMs)
sees the bandwidth benefit. **Retain** bf16 for footprint + prefill;
expect decode tok/s to catch up after M5 (int8 / fused dequant GEMV) or
simdgroup half GEMM.

## Artifact / converter

| Item | Status |
| --- | --- |
| `.zynfer` dtype tags f16/bf16 | already in format |
| Converter copies raw BF16 + prints dtype summary | **done** |
| Runtime `float16` decode | **done** |
| GPU upload: `copyArtifactToBf16` (native bytes, transpose in u16) | **done** |
| CPU load still promotes to f32 for oracle | intentional |

## Retained

| Item | Reason |
| --- | --- |
| bf16 resident weights | Halves weight storage + read bytes; native artifact copy |
| bf16 KV cache | Halves KV bytes as context grows |
| f32 activations + accumulators | Softmax / norms / logits stability |
| M3 two-CB schedule | Waits stay ≈2/forward |

## Deferred

| Item | Stage |
| --- | --- |
| int8 session weights | M5 |
| simdgroup bf16 GEMM | after M5 if still bound |
| Skipping CPU f32 promote entirely on Apple-only sessions | M6 memory report |
