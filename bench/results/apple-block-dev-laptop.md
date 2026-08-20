# Tiny-block prefill/decode — development laptop

Not Qwen3 and not a production decode path. The fixture is hidden=8,
n_q=2, n_kv=1, head_dim=4, inter=16, max_seq=32. Apple times include
per-op shared-buffer fill and `waitUntilCompleted`. Layout permute and
KV append stay on the host. Decode does not allocate host tensors;
Metal still allocates a shared buffer per kernel.

```text
date:              2026-08-19
host:              MacBook-Pro-2.local
OS:                Darwin 25.5.0 (arm64)
chip:              Apple M1 Max
memory:            64 GiB
Metal device:      Apple M1 Max
Zig:               0.16.0
tests:             zig build test (CPU + Metal differential) passed
prefill_tokens:    8
decode_steps:      8
warmup:            1
iterations:        4
```

## Debug (`zig build block-bench`)

shader compile: `metal_device_create_plus_shader_compile_ns=113589750`

| Path | Prefill ns (8 tok) | Decode ns/token |
| --- | ---: | ---: |
| CPU | 43416 | 10079 |
| Apple Metal | 6012323 | 5855778 |

```json
{"backend":"apple","fixture":"tiny-block","hidden":8,"prefill_tokens":8,"decode_steps":8,"warmup":1,"iters":4,"metal_init_ns":113589750,"cpu_prefill_ns":43416,"cpu_decode_ns_per_token":10079,"apple_prefill_ns":6012323,"apple_decode_ns_per_token":5855778}
```

## ReleaseSafe (`zig build -Doptimize=ReleaseSafe block-bench`)

shader compile: `metal_device_create_plus_shader_compile_ns=44651584`

| Path | Prefill ns (8 tok) | Decode ns/token |
| --- | ---: | ---: |
| CPU | 8229 | 2002 |
| Apple Metal | 3988260 | 3972990 |

```json
{"backend":"apple","fixture":"tiny-block","hidden":8,"prefill_tokens":8,"decode_steps":8,"warmup":1,"iters":4,"metal_init_ns":44651584,"cpu_prefill_ns":8229,"cpu_decode_ns_per_token":2002,"apple_prefill_ns":3988260,"apple_decode_ns_per_token":3972990}
```

Metal is slower than the CPU oracle on this shape because launch + wait
dominate. That is the Stage 4 baseline, not a regression to hide.

Explanation: [docs/apple-backend.md#why-metal-is-slower-on-the-tiny-block](../docs/apple-backend.md#why-metal-is-slower-on-the-tiny-block)
