# Stage M3 — Qwen schedule + fusion (dev laptop)

```text
date:              2026-08-23
host:              MacBook-Pro (Apple Silicon)
Zig:               0.16.0
commands:
  zig build test -Dhip=off
  zig build stageM3 -Dhip=off
  ./zig-out/bin/zynfer qwen-bench --mini --max-tokens 4
  ZYNFER_QWEN_METAL=baseline ./zig-out/bin/zynfer qwen-bench --mini --max-tokens 4
  ./zig-out/bin/zynfer qwen-bench models/qwen3-0.6b.zynfer \
    --prompt "Explain gravity simply." --max-tokens 2
  ZYNFER_QWEN_METAL=baseline ./zig-out/bin/zynfer qwen-bench \
    models/qwen3-0.6b.zynfer --prompt "Explain gravity simply." --max-tokens 2
```

## Exit criterion

- Default Apple path = `batched_resident_kv_fused` (~2 waits/forward)
- Mini logits: batched ≈ CPU ≈ baseline Metal
- Full-model decode/TTFT improved vs M0 baseline; ledger below
- Fusion retain/reject table filled

## Mini A/B (`qwen-bench --mini --max-tokens 4`)

| Path | prefill_ms | decode_ms/tok | enc/tok | wait/tok |
| --- | ---: | ---: | ---: | ---: |
| **batched** (default) | 10.7 | **1.61** | 23 | **2** |
| baseline (`ZYNFER_QWEN_METAL=baseline`) | 19.0 | 8.64 | 17 | 17 |

Decode ≈ **5.4×** vs baseline on the tiny fixture (launch-bound).

## Full model A/B (qwen3-0.6b, 17 prompt tok, max-tokens 2)

| Path | prefill_ms | prefill_t/s | decode_ms/tok | wait/tok |
| --- | ---: | ---: | ---: | ---: |
| **batched** | **177** | **95.9** | **231** | **2** |
| baseline (M0) | 1341–1383 | ~12.5 | 2561–2722 | 476 |

| Metric | Baseline → Batched | Speedup |
| --- | --- | ---: |
| Prefill latency | ~1360 ms → 177 ms | **~7.7×** |
| Decode ms/tok | ~2640 ms → 231 ms | **~11.4×** |
| Waits / decode tok | 476 → 2 | **238×** fewer waits |

`qwen-profile` after M3: wall ≈ **119 ms/tok**, STREAM ≈ 213 GB/s,
roofline fraction ≈ **0.095** (was ~0.01 on M0).

## Fusion ledger

| Candidate | Unfused / prior | Change | Numerics | Bench | Decision |
| --- | --- | --- | --- | --- | --- |
| One-CB stack + resident weights/KV | 476 waits/tok | encode all layers then 1 wait; KV on Metal | mini logits vs CPU OK (3e-3) | decode **~11×** | **RETAIN** |
| `silu_mul` | separate silu+mul | already fused encode | covered by block parity | launch −1/layer | **RETAIN** |
| `add_rmsnorm_f32` | add + rmsnorm | fused post-attn | covered by mini parity | launch −1/layer | **RETAIN** |
| Metal LM-head matvec | CPU tied/dot loop | `matvec_f32` on GPU | mini parity OK | part of ~11× | **RETAIN** |
| Q/K + RoPE fuse | separate encodes | — | — | no A/B need after wait collapse | **REJECT** |
| Attention tiling / online-softmax | `attention_f32(_buf)` | — | — | not implemented | **REJECT (M3)** |
| Dequant-into-GEMV | — | — | — | needs M5 weights | **DEFER → M5** |
| ICB / encode-once | — | — | — | KV mutates each decode | **REJECT** |

## Notes

- Encodes remain O(ops); the win is **waits** + no weight/KV re-upload.
- Second CB for final norm + LM head keeps the seam simple (still 2 waits).
- M0 full-model token-parity checklist can still be open for long chats;
  M3 mini logits parity is closed.
