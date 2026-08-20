# Apple Stage 6 — fusion / fewer waits / Metal-resident KV

Measured on the development laptop. Stage 6 tiny-block **default** path
label is `batched_resident_kv_fused`: persistent shared buffers,
Metal-resident KV, one command buffer / one `waitUntilCompleted`, and
`add_rmsnorm_f32` (~19 encodes). The Stage 4/5 per-op path remains
available via `ZYNFER_APPLE_BLOCK=baseline` → `baseline_per_op`.

`block-bench` JSON fields: `apple_block_path`, `apple_block_waits`,
`apple_block_encodes`. Correctness: Stage 6 vs baseline vs CPU is covered
by `zig build test` (`Metal Stage 6 path matches baseline path and CPU`).

```text
date:              2026-08-20
host:              MacBook-Pro-2.local
OS:                Darwin 25.5.0 (arm64)
chip:              Apple M1 Max
Zig:               0.16.0
command:           zig build -Doptimize=ReleaseSafe block-bench
warmup:            1
iterations:        4
prefill_tokens:    8
decode_steps:      8
```

## Selection decisions (profile-proven)

| Path label | What changed | vs baseline | Decision |
| --- | --- | --- | --- |
| `batched_resident_kv_fused` | one CB/wait; resident KV; GPU permute + `kv_append_f32`; `add_rmsnorm` | decode ~7–8× | **default** |
| `baseline_per_op` | 15 upload+wait kernels (Stage 4/5) | — | A/B only (`ZYNFER_APPLE_BLOCK=baseline`) |
| Further MSL fusions / ICB / TTFT | — | — | Apple-8 + Stages 10–12 / 16 (`docs/apple-backend.md`) |

## Results (ReleaseSafe)

| Mode | path | waits | encodes | prefill ns | decode ns/token |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU oracle | — | — | — | 8041 | 1945 |
| Stage 6 (historical, pre-`add_rmsnorm`) | `batched_resident_kv` | 1 | 20 | 386843 | 437394 |
| Stage 6 default | `batched_resident_kv_fused` | 1 | 19 | 544197 | 524888 |
| Stage 4/5 baseline | `baseline_per_op` | 15 | 15 | 3297843 | 3637569 |

Absolute ns varies run-to-run; A/B vs baseline remains ~7–8×. The
historical 20-encode row is kept for the pre-`add_rmsnorm` capture; the
live default label is always `batched_resident_kv_fused`.

Default (current code path name; ns from fused re-measure):

```json
{"backend":"apple","fixture":"tiny-block","hidden":8,"prefill_tokens":8,"decode_steps":8,"warmup":1,"iters":4,"metal_init_ns":136557959,"cpu_prefill_ns":8041,"cpu_decode_ns_per_token":1945,"apple_prefill_ns":544197,"apple_decode_ns_per_token":524888,"apple_block_path":"batched_resident_kv_fused","apple_block_waits":1,"apple_block_encodes":19}
```

Baseline A/B:

```json
{"backend":"apple","fixture":"tiny-block","hidden":8,"prefill_tokens":8,"decode_steps":8,"warmup":1,"iters":4,"metal_init_ns":52495375,"cpu_prefill_ns":7770,"cpu_decode_ns_per_token":1902,"apple_prefill_ns":3297843,"apple_decode_ns_per_token":3637569,"apple_block_path":"baseline_per_op","apple_block_waits":15,"apple_block_encodes":15}
```

## Notes

- Metal is still slower than the scalar CPU oracle on this **tiny** shape;
  Stage 6 removes wait/alloc dominance, not FLOP dominance.
- `ops-bench` / `apple.ops` still use one wait per kernel (microbench
  baseline). The Stage 6 schedule lives in `apple.block`.
- Host layout permutes and KV append moved onto GPU so the whole block
  fits in one CB without mid-schedule CPU sync.
- Further kernel fusion is optional; Instruments should now show one
  wait gap per forward, not fifteen.
