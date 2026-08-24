# Stage M5 — int8 Metal weights (dev laptop)

```text
date:              2026-08-24
host:              MacBook-Pro (Apple Silicon)
Zig:               0.16.0
commands:
  zig build test -Dhip=off
  ZYNFER_FULL_MODEL_TESTS=1 zig build test -Dhip=off
  zig build stageM5 -Dhip=off
  ASDF_PYTHON_VERSION=3.12.9 python3 tools/checkpoint/quantize_zynfer_int8.py \
    --in models/qwen3-0.6b.zynfer --out models/qwen3-0.6b-int8.zynfer
  ZYNFER_QWEN_METAL=int8 ./zig-out/bin/zynfer qwen-bench --mini --max-tokens 4
  ZYNFER_QWEN_METAL=bf16 ./zig-out/bin/zynfer qwen-bench \
    models/qwen3-0.6b.zynfer --prompt "Explain gravity simply." --max-tokens 2
  ZYNFER_QWEN_METAL=int8 ./zig-out/bin/zynfer qwen-bench \
    models/qwen3-0.6b.zynfer --prompt "Explain gravity simply." --max-tokens 2
  # quality proxy (greedy):
  ZYNFER_QWEN_METAL=bf16|int8 ./zig-out/bin/zynfer run models/qwen3-0.6b.zynfer \
    --prompt "Explain gravity simply." --max-tokens 2 --temperature 0 --no-stream
  ./zig-out/bin/zynfer run models/qwen3-0.6b.zynfer \
    --prompt "Explain gravity simply." --max-tokens 2 --temperature 0 --no-stream --backend cpu
```

## Scheme ledger

| Item | Value |
| --- | --- |
| Group | per-row (full `in` dim) |
| Scale | max_abs/127, symmetric, no ZP |
| Alignment | dense i8; scales f32 contiguous |
| Tail | none |
| Dequant | fused `matmul_aq8_f32` / `matvec_q8_f32` |

## Correctness

| Check | Result |
| --- | --- |
| `qwen_quant` pack/dequant round-trip | PASS @ 2e-2 |
| Mini int8 vs CPU logits | PASS @ **5e-2** atol |
| Full-model greedy (`ZYNFER_FULL_MODEL_TESTS=1`) | PASS (CPU == int8 Metal, 2 tok) |
| Converter smoke (`quantize_zynfer_int8.py`) | PASS — 197 projs, worst abs err **4.86e-3** (< 0.05) |
| `wait/tok` | **2** |

## Bytes / token (estimate @ kv≈19)

| Path | B/tok_est | Notes |
| --- | ---: | --- |
| M3 f32 | ~2.39e9 | weights+KV f32 |
| M4 bf16 | ~1.19e9 | weights+KV bf16 |
| M5 int8 | ~0.60e9 | i8 weights + f32 scales + **f32 KV** |

## Full-model A/B (qwen3-0.6b, 17 prompt tok, max-tokens 2)

| Path | prefill_ms | prefill_t/s | decode_ms/tok | wait/tok | B/tok_est |
| --- | ---: | ---: | ---: | ---: | ---: |
| M4 bf16 batched | 170 | 100 | 253 | 2 | 1.19e9 |
| M5 int8 batched | **126** | **134** | **119** | 2 | **0.60e9** |

| Metric | bf16 → int8 | Delta |
| --- | --- | --- |
| Prefill latency | 170 → 126 ms | **~1.34×** faster |
| Decode ms/tok | 253 → 119 | **~2.12×** faster |
| Bytes/tok estimate | 1.19e9 → 0.60e9 | **~2.0×** fewer |

**Decision:** **retain int8.** Decode is measurably faster than bf16 on this
hardware (weight traffic dominated); footprint win matches the estimator.
4-bit still deferred — int8 already clears the M5 gate.

## Quality proxy

Fixed prompt `"Explain gravity simply."`, greedy `max-tokens 2`,
`--temperature 0`:

| Backend | Generated text (2 tokens) |
| --- | --- |
| CPU | `Gravity is` |
| Apple bf16 | `Gravity is` |
| Apple int8 | `Gravity is` |

No token divergence on this fixture. Logit tolerance remains packing-justified
(5e-2); do not loosen further to hide packing error.

## 4-bit

**Deferred.** Int8 path is the M5 deliverable; 4-bit only after a measured
need beyond int8 quality/footprint.

## Retained

| Item | Reason |
| --- | --- |
| Per-row int8 projections + LM head | Bandwidth / footprint + decode win |
| Fused dequant kernels | No full f32 materialization |
| f32 KV + norms | Numerics |

## Deferred

| Item | Stage |
| --- | --- |
| 4-bit weights | later if justified |
| Runtime load of on-disk i8 artifact (skip host f32) | polish / M6 |
