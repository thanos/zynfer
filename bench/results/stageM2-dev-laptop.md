# Stage M2 — Profile one decode token (dev laptop)

```text
date:              2026-08-23
host:              MacBook-Pro (Apple Silicon)
Zig:               0.16.0
commands:
  zig build test -Dhip=off
  zig build stageM2 -Dhip=off
  zig build qwen-profile -Dhip=off
  ./zig-out/bin/zynfer qwen-profile --mini
  ./zig-out/bin/zynfer qwen-profile models/qwen3-0.6b.zynfer \
    --prompt "Explain gravity simply."
```

## Exit criterion

- `zynfer stageM2` ledger
- `zynfer qwen-profile --mini` → family table + `top3` + `json`
- Documented answer to “where does one decode token go?” (top three)

## Mini fixture (`qwen-profile --mini`)

```text
backend=apple  layers=1  metal_encodes=waits=17
top3: MLP (~36%), RMSNorm (~22%), QKV (~14%)
empty_encode_wait ~0.27 ms → est launch overhead ~4.6 ms of ~6.2 ms wall
STREAM triad (~4 MiB elems): ~49 GB/s (small working set; not the machine peak)
```

Mini is launch-bound; absolute GB/s is not the story.

## Full model (Qwen3-0.6B, prompt 17 tok, one profiled decode)

```text
backend=apple  layers=28  kv_len=19  metal_encodes=waits=476
wall decode token ≈ 1062 ms  → measured_tok_s ≈ 0.94

operation                     ms      %
RMSNorm                     32.9    3.1
QKV projection             122.6   11.6
RoPE                        12.6    1.2
attention                   38.1    3.6
output projection           63.6    6.0
MLP                        233.1   22.0
host layout / KV append      0.2    0.0
embedding                    0.0    0.0
LM head + final norm       556.7   52.5
sampling                     0.4    0.0

top3:
  1. LM head + final norm (52.5%)   — CPU tied matvec today
  2. MLP (22.0%)                    — Metal, per-op encode+wait
  3. QKV projection (11.6%)

empty_encode_wait_ns ≈ 256 µs → est_launch_overhead ≈ 122 ms
  (embedded in Metal families; ~476 launches)
STREAM triad (64 MiB elems): ≈ 228.5 GB/s (measured, sustainable)
bytes_per_tok_est ≈ 2.39e9
roofline ideal_tok_s ≈ 95.7   fraction ≈ 0.0098 (~1% of BW bound)
```

## Answer (gate)

**Where does one Metal decode token go on this machine?**

1. **CPU LM head** (tied embed matvec) — largest single slice  
2. **MLP** Metal matmuls / silu under M0 per-op launches  
3. **QKV** Metal projections, same launch tax  

Launch overhead (~122 ms est.) is real but already inside the Metal
rows; M3 should collapse waits. LM-head Metal path remains an open M0
item.

## Notes

- Family wall times include upload + encode + wait on the M0 path.
- Do not treat mini STREAM GB/s as the machine number; use the full-model
  STREAM line (~228 GB/s here).
- M0 gate checklist remains open — see `docs/stages/M0-metal-qwen-forward.md`.
