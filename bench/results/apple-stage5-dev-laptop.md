# Apple Stage 5 — matrix / quantized / Accelerate paths

Measured on the development laptop. Times include per-op shared-buffer
fill and `waitUntilCompleted` on Metal (same baseline as Stage 3–4).
Accelerate is CPU-only.

```text
date:              2026-08-20
host:              MacBook-Pro-2.local
OS:                Darwin 25.5.0 (arm64)
chip:              Apple M1 Max
Zig:               0.16.0
command:           zig build -Doptimize=ReleaseSafe ops-bench
warmup:            2
iterations:        8
```

## Selection decisions (profile-proven)

| Path | Selected region | vs fallback | Decision |
| --- | --- | --- | --- |
| `matmul_f32_simdgroup` | Apple7+ and M,N,K≥8 and M·N·K≥64³ | 64³ ~1.05×; 256³ ~1.46× vs naive Metal | **auto-select** |
| `matmul_f32` (naive) | elsewhere | — | default fallback |
| Accelerate `vDSP_mmul` | macOS and M·N·K≥64³ | ~23× vs scalar CPU at 64³ | **auto-select** via `cpu.accelerate.matmulAuto` |
| `matvec_q8_f32` | explicit int8 weights API | not faster than f32 when both re-upload each call | **retained API**, not auto-replacing f32 matvec |

Override Metal matmul: `ZYNFER_MATMUL_PATH=naive|simdgroup`.

## Results (ReleaseSafe)

| Op | CPU ns/iter | Apple Metal ns/iter |
| --- | ---: | ---: |
| add_f32_4096 | 6083 | 403843 |
| silu_mul_f32_4096 | 12432 | 391536 |
| matvec_f32_256x256 | 128989 | 781906 |
| matmul_f32_32x64x64 | 84671 | 359546 |
| matmul_f32_naive_64x64x64 | 167942 | 499187 |
| matmul_f32_simdgroup_64x64x64 | 168015 | 474515 |
| matmul_f32_naive_256x256x256 | 10586921 | 935984 |
| matmul_f32_simdgroup_256x256x256 | 9930145 | 639046 |
| matmul_f32_auto_64x64x64 | 167322 | 472520 |
| matvec_q8_f32_256x256 | 173062 | 785234 |
| matmul_accelerate_64x64x64 | 7203 | N/A |

```json
{"backend":"apple","zig":"0.16.0","warmup":2,"iters":8,"metal_init_ns":55632541,"ops":[{"name":"add_f32_4096","cpu_ns":6083,"apple_metal_ns":403843},{"name":"silu_mul_f32_4096","cpu_ns":12432,"apple_metal_ns":391536},{"name":"matvec_f32_256x256","cpu_ns":128989,"apple_metal_ns":781906},{"name":"matmul_f32_32x64x64","cpu_ns":84671,"apple_metal_ns":359546},{"name":"matmul_f32_naive_64x64x64","cpu_ns":167942,"apple_metal_ns":499187},{"name":"matmul_f32_simdgroup_64x64x64","cpu_ns":168015,"apple_metal_ns":474515},{"name":"matmul_f32_naive_256x256x256","cpu_ns":10586921,"apple_metal_ns":935984},{"name":"matmul_f32_simdgroup_256x256x256","cpu_ns":9930145,"apple_metal_ns":639046},{"name":"matmul_f32_auto_64x64x64","cpu_ns":167322,"apple_metal_ns":472520},{"name":"matvec_q8_f32_256x256","cpu_ns":173062,"apple_metal_ns":785234},{"name":"matmul_accelerate_64x64x64","cpu_ns":7203,"apple_metal_ns":null}]}
```

## Notes

- Metal times are still launch/sync-bound for tiny shapes; simdgroup
  wins grow with problem size (clear at 256³).
- int8 GEMV is for stored quantized weights. With host pack+upload every
  call it does not beat f32 matvec; do not claim a decode win until
  persistent int8 weight buffers exist (later stage).
- Accelerate is an optimized **CPU** path checked against the scalar
  oracle; it is not used by the Metal block schedule.
