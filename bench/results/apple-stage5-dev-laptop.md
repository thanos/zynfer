# Apple Stage 5 — matrix / quantized / Accelerate paths

Measured on the development laptop. Times include per-op shared-buffer
fill and `waitUntilCompleted` on Metal (same baseline as Stage 3–4),
except fair q8 rows which pack once outside the timed loop.
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
| `matmul_f32_simdgroup` | Apple7+ and M,N,K≥8 and M·N·K≥64³ | 64³ ~1.04×; 256³ ~1.20× vs naive Metal | **auto-select** |
| `matmul_f32_simdgroup_x4` | force `ZYNFER_MATMUL_PATH=simdgroup_x4` | 256³ ~0.94× vs 1-SG (slower) | **force-only** (retained kernel) |
| `matmul_f32` (naive) | elsewhere | — | default fallback |
| Accelerate `vDSP_mmul` | macOS and M·N·K≥64³ | ~21× vs scalar CPU at 64³ | **auto-select** via `cpu.accelerate.matmulAuto` |
| Accelerate `vDSP` matvec | macOS and M·K≥256² | ~1.12× vs scalar at 256² | **auto-select** via `cpu.accelerate.matvecAuto` |
| `matvec_q8_f32` / `matmul_q8_f32` | explicit int8 weights API | prepacked Metal q8 matvec slightly under f32; q8 GEMM slower than f32 at 128³ under upload-per-call | **retained API**, not auto-replacing f32 |

Override Metal matmul: `ZYNFER_MATMUL_PATH=naive|simdgroup|simdgroup_x4`.

## Results (ReleaseSafe)

| Op | Path | CPU ns/iter | Apple Metal ns/iter |
| --- | --- | ---: | ---: |
| add_f32_4096 | add_f32 | 6088 | 494349 |
| silu_mul_f32_4096 | silu_mul_f32 | 12203 | 267838 |
| matvec_f32_256x256 | matvec_f32 | 91296 | 711864 |
| matmul_f32_32x64x64 | matmul_auto | 75541 | 340734 |
| matmul_f32_naive_64x64x64 | matmul_f32 | 156088 | 388427 |
| matmul_f32_simdgroup_64x64x64 | matmul_f32_simdgroup | 148244 | 372640 |
| matmul_f32_naive_256x256x256 | matmul_f32 | 9808906 | 790880 |
| matmul_f32_simdgroup_256x256x256 | matmul_f32_simdgroup | 10449140 | 660187 |
| matmul_f32_simdgroup_x4_256x256x256 | matmul_f32_simdgroup_x4 | 9854640 | 698338 |
| matmul_f32_auto_256x256x256 | matmul_auto | 11283109 | 688276 |
| matmul_accelerate_64x64x64 | accelerate_vDSP_mmul | 7375 | N/A |
| matvec_accelerate_256x256 | accelerate_vDSP_matvec | 81776 | N/A |
| matvec_q8_f32_256x256_prepacked | matvec_q8_f32_per_row | 55921 | 586333 |
| matmul_q8_f32_128x128x128_prepacked | matmul_q8_f32_per_row | 845994 | 561796 |
| matvec_f32_256x256_ref | matvec_f32 | 91234 | 694046 |
| matmul_f32_128x128x128_ref | matmul_auto | 1316541 | 456843 |

```json
{"backend":"apple","zig":"0.16.0","warmup":2,"iters":8,"metal_init_ns":124752333,"ops":[{"name":"add_f32_4096","path":"add_f32","cpu_ns":6088,"apple_metal_ns":494349},{"name":"silu_mul_f32_4096","path":"silu_mul_f32","cpu_ns":12203,"apple_metal_ns":267838},{"name":"matvec_f32_256x256","path":"matvec_f32","cpu_ns":91296,"apple_metal_ns":711864},{"name":"matmul_f32_32x64x64","path":"matmul_auto","cpu_ns":75541,"apple_metal_ns":340734},{"name":"matmul_f32_naive_64x64x64","path":"matmul_f32","cpu_ns":156088,"apple_metal_ns":388427},{"name":"matmul_f32_simdgroup_64x64x64","path":"matmul_f32_simdgroup","cpu_ns":148244,"apple_metal_ns":372640},{"name":"matmul_f32_naive_256x256x256","path":"matmul_f32","cpu_ns":9808906,"apple_metal_ns":790880},{"name":"matmul_f32_simdgroup_256x256x256","path":"matmul_f32_simdgroup","cpu_ns":10449140,"apple_metal_ns":660187},{"name":"matmul_f32_simdgroup_x4_256x256x256","path":"matmul_f32_simdgroup_x4","cpu_ns":9854640,"apple_metal_ns":698338},{"name":"matmul_f32_auto_256x256x256","path":"matmul_auto","cpu_ns":11283109,"apple_metal_ns":688276},{"name":"matmul_accelerate_64x64x64","path":"accelerate_vDSP_mmul","cpu_ns":7375,"apple_metal_ns":null},{"name":"matvec_accelerate_256x256","path":"accelerate_vDSP_matvec","cpu_ns":81776,"apple_metal_ns":null},{"name":"matvec_q8_f32_256x256_prepacked","path":"matvec_q8_f32_per_row","cpu_ns":55921,"apple_metal_ns":586333},{"name":"matmul_q8_f32_128x128x128_prepacked","path":"matmul_q8_f32_per_row","cpu_ns":845994,"apple_metal_ns":561796},{"name":"matvec_f32_256x256_ref","path":"matvec_f32","cpu_ns":91234,"apple_metal_ns":694046},{"name":"matmul_f32_128x128x128_ref","path":"matmul_auto","cpu_ns":1316541,"apple_metal_ns":456843}]}
```

## Notes

- Metal times are still launch/sync-bound for tiny shapes; simdgroup
  wins grow with problem size (clear at 256³). The 4-SG tile did not
  beat 1-SG at 256³ under the per-op wait baseline.
- Fair int8 GEMV (pack excluded) beats scalar CPU f32 and is slightly
  under Metal f32 while still uploading weights each call. Do not claim
  a decode win until persistent int8 weight buffers exist.
- Fair int8 GEMM at 128³ is slower than Metal f32 under the same upload
  model; keep as an explicit API for stored quantized weights.
- Accelerate matmul/matvec are optimized **CPU** paths checked against
  the scalar oracle; they are not used by the Metal block schedule.
