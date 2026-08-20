# Apple Stage 8 — hardening + Stage 6 leftovers

Stage 8 closes Apple-backend carry-forwards that do not need a Qwen
loader. Items that still need a real model or larger shapes are handed
to curriculum Stages 10–12 / 16 with reasons (not silent drops).

```text
date:              2026-08-20
host:              MacBook-Pro-2.local
OS:                Darwin 25.5.0 (arm64)
chip:              Apple M1 Max
Zig:               0.16.0
commands:
  zig build test
  ./zig-out/bin/zynfer stage8
  ./zig-out/bin/zynfer block-bench --backend apple
```

## Decisions

| Item | Decision | Evidence / reason |
| --- | --- | --- |
| Fused vs baseline A/B | **RETAIN** (done Stage 6) | `Metal Stage 6 path matches baseline path and CPU` |
| Attention `kv_len` cap | **RAISE 64→256** | Thread-local `scores[256]`; CPU differential at kv_len=96; >256 still `Unsupported` |
| Signposts | **RETAIN** (opt-in) | `ZYNFER_SIGNPOSTS=1` → `os_signpost` on encode_and_wait / batch begin+commit |
| Peak RSS | **RETAIN** | `peak_rss_bytes` in block-bench JSON via `getrusage` |
| Energy/token | **N/A** | Not measured; JSON field null |
| Stress / error paths | **RETAIN** | Repeated Session init; fill `max_seq`; overflow; batch abort |
| fp16/bf16 Metal | **REJECT** | `matmulF16`/`matvecF16` → `Unsupported`; stay on f32 |
| ICB / encode-once | **REJECT** | Decode changes KV/`q_len` each step; Stage 6 already removed wait dominance |
| Extra MSL fusions | **REJECT** (tiny-block) | `add_rmsnorm` did not beat unfused Stage 6 ns; revisit Stage 16 |
| Int8 Session weights | **REJECT** (for now) | `Q8DeviceWeights` kept for ops; Session stays f32 until Qwen shapes |
| TTFT / tok/s | **N/A** | Stages 10–12 |

## Commands

```bash
zig build test
./zig-out/bin/zynfer stage8
./zig-out/bin/zynfer caps --backend apple
ZYNFER_SIGNPOSTS=1 ./zig-out/bin/zynfer block-bench --backend apple
```

## Notes

- Fixture `max_seq` remains 32; Metal attention can now run longer contexts
  up to 256 when a larger fixture/spec is used.
- Concurrent multi-session GPU ownership is still single-threaded per
  `Gpu` (bridge is not internally synchronized).
- Apple Stages 0–8 are closed for the tiny-block curriculum path.
