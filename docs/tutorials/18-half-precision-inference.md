# Tutorial — Half-precision inference on Apple Metal (Stage M4)

Decode on Qwen3-0.6B is **memory-bandwidth bound**: each token reads
almost all weights plus the growing KV cache. Halving those bytes is the
largest win available **before** int8 quantization (M5).

## What stays f32 (and why)

- **Activations** (hidden states, Q/K/V linears before cache write): RoPE,
  SiLU, and residual adds need headroom; keeping activations f32 avoids
  error compounding across 28 layers.
- **Softmax and RMSNorm reductions**: reductions in f16/bf16 drift quickly;
  kernels widen to f32 for sums, max, and `rsqrt`.
- **CPU oracle**: correctness tests still compare against f32 CPU forward.

## What goes half on GPU

- **All resident weights** (projections, layer norms, embed, LM head): stored
  as **bf16** in Metal buffers — matches Qwen3 `.zynfer` checkpoints
  (`dtype=2`).
- **KV cache**: appended as bf16 after Q/K/V linears (still computed in f32
  before narrow).

Upload path: CPU loads BF16→f32 for the oracle; `bf16.encodeFromF32`
narrows once when copying to GPU (same bit pattern as checkpoint bf16).

## Selecting the path

```bash
# M3 default (f32 weights + f32 KV)
./zig-out/bin/zynfer qwen-bench models/qwen3-0.6b.zynfer --max-tokens 4

# M4 half path
ZYNFER_QWEN_METAL=bf16 ./zig-out/bin/zynfer qwen-bench models/qwen3-0.6b.zynfer \
  --max-tokens 4
```

`qwen-bench` reports `B/tok_est` using `estimateDecodeBytesPerTokenHalf`
(≈ half the f32 estimate) when the half path is active.

## Tolerance budget

BF16 mantissa is 7 bits (~2–3 decimal digits). Expect logits to differ
slightly from f32 CPU; zynfer uses **5e-3 absolute tolerance** on the mini
fixture. Greedy tokens usually still match; when they diverge, inspect
top-logit gaps before loosening bounds.

## Roofline

If decode were purely bandwidth-limited, ideal speedup would be **2×**.
On the current naive Metal schedule, **prefill** improves (~1.5× measured)
while **decode** stays roughly flat — still encode/compute bound
(~590 encodes/tok). Ideal tok/s doubles; measured fraction drops until
kernels catch the bandwidth. See `bench/results/stageM4-dev-laptop.md`.

```bash
./zig-out/bin/zynfer qwen-bench models/qwen3-0.6b.zynfer --max-tokens 2
ZYNFER_QWEN_METAL=bf16 ./zig-out/bin/zynfer qwen-bench \
  models/qwen3-0.6b.zynfer --max-tokens 2
```

## Next

**M5** quantizes weights to int8 (and maybe 4-bit) with dequant fused into
GEMV — another step down in bytes/token with a quality ledger.
