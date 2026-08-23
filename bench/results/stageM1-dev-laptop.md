# Stage M1 — Prefill vs decode on Qwen (dev laptop)

```text
date:              2026-08-23
host:              MacBook-Pro (Apple Silicon)
Zig:               0.16.0
commands:
  zig build test -Dhip=off
  zig build stageM1 -Dhip=off
  zig build qwen-bench -Dhip=off
  ./zig-out/bin/zynfer qwen-bench --mini --max-tokens 4
  ./zig-out/bin/zynfer qwen-bench models/qwen3-0.6b.zynfer \
    --prompt "Explain gravity simply." --max-tokens 2
```

## Exit criterion (CI)

- `zynfer stageM1` ledger
- `zynfer qwen-bench --mini` → table with `cpu` + `apple` + `json`
- Unit helpers: decode bytes estimate grows with `kv_len`
- Measured Metal `enc/tok` / `wait/tok` on Apple rows (not estimated)

## Mini fixture (`qwen-bench --mini --max-tokens 4`)

```text
backend    prefill_ms  prefill_t/s      ttft_ms   decode_t/s  decode_ms/tok    B/tok_est    enc/tok   wait/tok
cpu             0.054    37037.037        0.091    35767.511          0.028        10112        0.0        0.0
apple          13.413      149.104       13.481      157.285          6.358        10112       17.0       17.0
```

Metal is launch-bound on the tiny fixture (expected for M0 per-op path).
Measured: **17 encodes = 17 waits** per decode token (1 layer). Prefill also 17/17.

## Full model (`qwen3-0.6b.zynfer`, prompt 17 tok, `--max-tokens 2`)

```text
backend    prefill_ms  prefill_t/s      ttft_ms   decode_t/s  decode_ms/tok    B/tok_est    enc/tok   wait/tok
cpu         36241.291        0.469    36241.755        0.189       5304.464   2388295680        0.0        0.0
apple        1211.258       14.035     1211.723        0.465       2149.220   2388295680      476.0      476.0
```

Apple notes from this run:

- prefill: encodes=476 waits=476
- decode: encodes=952 waits=952 over 2 steps → **476 / tok**
- 476 = **17 × 28 layers** (old M1 estimate was `18 × layers` = 504)

Metal decode remains slow until M3 (batched one-CB schedule). Absolute speed is not the M1 gate — the split columns and measured launch counts are.

## Notes

- `run` / `chat` footers print `prefill_tok_s` and `decode_ms_per_tok`.
- `enc/tok` / `wait/tok` come from `Gpu.total_encodes` / `total_waits` around each
  `decodeToken` (and prefill snapshot in the bench notes).
- M0 gate checklist remains open — see `docs/stages/M0-metal-qwen-forward.md`.
