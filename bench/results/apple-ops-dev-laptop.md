# Apple Metal op microbenchmarks — development laptop

Not an inference result. These timings include host tensor allocation
inside each iteration and, on the Apple path, shared-buffer fill plus
`waitUntilCompleted` after every kernel. Small elementwise ops are
launch-bound; do not read them as a GPU vs CPU throughput ranking.

```text
date:              2026-08-19
host:              MacBook-Pro-2.local
OS:                Darwin 25.5.0 (arm64)
chip:              Apple M1 Max
memory:            64 GiB
Metal device:      Apple M1 Max (GPU family Apple7)
Zig:               0.16.0
command:           zig build test && zig build ops-bench
tests:             CPU oracle + Metal differential tests passed
warmup:            2
iterations:        8
shader compile:    metal_device_create_plus_shader_compile_ns=112987125
```

| Op | CPU ns/iter | Apple Metal ns/iter |
| --- | ---: | ---: |
| add_f32_4096 | 278182 | 550218 |
| silu_mul_f32_4096 | 308057 | 724234 |
| matvec_f32_256x256 | 949156 | 1227098 |
| matmul_f32_32x64x64 | 863552 | 674625 |

Machine-readable line from the same run:

```json
{"backend":"apple","zig":"0.16.0","warmup":2,"iters":8,"metal_init_ns":112987125,"ops":[{"name":"add_f32_4096","cpu_ns":278182,"apple_metal_ns":550218},{"name":"silu_mul_f32_4096","cpu_ns":308057,"apple_metal_ns":724234},{"name":"matvec_f32_256x256","cpu_ns":949156,"apple_metal_ns":1227098},{"name":"matmul_f32_32x64x64","cpu_ns":863552,"apple_metal_ns":674625}]}
```

HIP enumeration was not measured on this host (`zig build bench` reports
HIP is not linked).
