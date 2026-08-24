# Stage M4 — bf16 Metal weights + KV (dev laptop)

Machine: Apple Silicon dev laptop (same host as M1–M3 ledgers).

## Reproduce

```bash
zig build stageM4 -Dhip=off

# f32 batched (M3 default)
./zig-out/bin/zynfer qwen-bench models/qwen3-0.6b.zynfer \
  --prompt "Explain gravity simply." --max-tokens 2

# bf16 batched (M4)
ZYNFER_QWEN_METAL=bf16 ./zig-out/bin/zynfer qwen-bench models/qwen3-0.6b.zynfer \
  --prompt "Explain gravity simply." --max-tokens 2
```

Mini CI:

```bash
ZYNFER_QWEN_METAL=bf16 ./zig-out/bin/zynfer qwen-bench --mini --max-tokens 4
```

## Bytes / token (estimate)

| Path | `B/tok_est` @ kv≈19 | Notes |
| --- | --- | --- |
| M3 f32 | ~2.39×10⁹ | `estimateDecodeBytesPerToken` |
| M4 bf16 | ~1.20×10⁹ | `estimateDecodeBytesPerTokenHalf` (weights+KV ÷2) |

Activations and f32 softmax scratch are **not** in this estimate (same as M1/M2).

## Correctness

| Check | Result |
| --- | --- |
| Mini bf16 vs CPU logits | PASS @ 5e-3 atol |
| Mini f32 batched vs CPU | PASS @ 3e-3 atol |
| `wait/tok` bf16 batched | **2** (same schedule as M3) |

## Performance (fill on hardware)

Run the reproduce commands above and paste measured prefill/decode rows.
Expected: decode speedup toward **1.3–1.8×** vs M3 f32 when bandwidth-bound
(not full 2× — activations and f32 attention scores remain).

## Retained

| Item | Reason |
| --- | --- |
| bf16 resident weights | Halves weight read bytes |
| bf16 KV cache | Halves KV read bytes as context grows |
| f32 activations + accumulators | Numerics (softmax, norms, logits) |
| M3 two-CB schedule | Waits stay ≈2/forward |

## Deferred

| Item | Stage |
| --- | --- |
| int8 weights in session | M5 |
| simdgroup bf16 GEMM | measure after M5 if still bound |
