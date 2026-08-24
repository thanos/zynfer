# Tutorial — Quantization on Apple Silicon (Stage M5)

Decode reads almost every weight every token. After M4 halved those bytes
with bf16, **int8** cuts weight traffic again (~4× vs f32, ~2× vs bf16 on
the projection matrices), if dequant stays fused into GEMV/GEMM.

## Scheme (what we ship)

Not a copy of llama.cpp GGUF. Chosen from what Metal already executes well:

1. **Per-row (per-output-channel) symmetric int8**
2. `scale[r] = max_abs(row_r) / 127`
3. `q = clamp(round(w / scale), -127, 127)` — no zero-point
4. Weights stored as HF `[out, in]` for the pack; schedule uses
   `matmul_aq8_f32`: `C[t,out] = A[t,in] @ dequant(W[out,in])ᵀ`

Norms, embeddings (gather), and the KV cache stay higher precision.
Activations stay f32.

## How to run

```bash
# Pack happens at Session init from the usual bf16/f32 .zynfer
ZYNFER_QWEN_METAL=int8 ./zig-out/bin/zynfer qwen-bench \
  models/qwen3-0.6b.zynfer --prompt "Explain gravity simply." --max-tokens 2
```

Optional on-disk int8 artifact (dev converter; validates dequant error;
requires NumPy):

```bash
python3 tools/checkpoint/quantize_zynfer_int8.py \
  --in models/qwen3-0.6b.zynfer \
  --out models/qwen3-0.6b-int8.zynfer
```

Smoke on qwen3-0.6b: 197 projections, worst CPU dequant abs err ≈ **4.9e-3**
(budget 0.05). Runtime still packs from host f32 for the Metal path in M5 v1;
the converter proves the scheme and dtype tag `3=i8` + `.qscale` siblings.

Full-model greedy parity (CPU vs Metal int8, 2 tokens) runs under
`ZYNFER_FULL_MODEL_TESTS=1 zig build test -Dhip=off`.

## Quality vs speed

Always report separately:

| Lens | What to look at |
| --- | --- |
| Speed | `qwen-bench` Apple row vs `ZYNFER_QWEN_METAL=bf16` |
| Bytes | `B/tok_est` from `estimateDecodeBytesPerTokenQ8` |
| Quality | Logit atol (mini 5e-2); greedy token drift on a fixed prompt |

On this laptop (see `bench/results/stageM5-dev-laptop.md`), int8 beat bf16
on both prefill (~1.3×) and decode (~2.1×) with matching greedy text on the
fixture prompt. Encode count is unchanged (590/tok); the win is cheaper
weight traffic in the fused dequant kernels.

## Why not 4-bit yet

M5 gate: go further **only if** int8 is measurably faster than fp16/bf16
with acceptable quality. Int8 already clears that bar here — 4-bit is
deferred until there is a measured need beyond int8 footprint/quality.

## Next

**M6** — static decode plan: resident weights, fixed scratch, zero
per-token heap allocation.
