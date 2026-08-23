# Tutorial — Why the KV cache exists (Stage 13)

Stage 12 already generated text with a cache. This stage slows down and
explains the trick.

Full reference: [`docs/stages/13-kv-cache.md`](../stages/13-kv-cache.md).

## 1. Three different “memories”

| Name | Lifetime | Grows with sequence? |
| --- | --- | --- |
| Weights | Whole process | No |
| Activations | One forward call | Only for that call’s shapes |
| KV cache | Prefill + every decode step | Yes — one slot per token |

If you confuse KV with weights, you will mis-size VRAM. If you confuse it
with activations, you will wonder why decode still allocates after the first
token.

## 2. Inefficient decoder (baseline)

Without a cache, each new token rebuilds the full prefix and re-runs the
forward. Zynfer keeps this path for teaching:

```bash
./zig-out/bin/zynfer chat "Explain gravity simply." \
  --max-tokens 8 --no-kv-cache --no-stream
```

Expect much slower decode as the prefix lengthens.

## 3. Cached decoder (default)

Default `run` / `chat` prefills once, then appends one K/V position per
decode step.

```bash
./zig-out/bin/zynfer chat "Explain gravity simply." --max-tokens 8 --no-stream
```

Footer includes `kv_cache=on` and `kv_bytes_used` / `kv_bytes_cap`.

## 4. Side-by-side bench

```bash
# Always works (mini fixture)
./zig-out/bin/zynfer kv-bench --mini --max-tokens 4

# Full Qwen locally
./zig-out/bin/zynfer kv-bench models/qwen3-0.6b.zynfer --max-tokens 8
```

Look for:

- `token_parity: PASS` — same greedy ids
- `decode_speedup` — uncached wall / cached wall (should be ≫ 1×)

## 5. Why this physical layout

```bash
./zig-out/bin/zynfer kv-bench --layout
```

Two candidates:

| Layout | Append one token | Decode attention scan |
| --- | --- | --- |
| `[n_kv, max_seq, head_dim]` (retained) | Slightly slower | Contiguous rows — **~3× faster** |
| `[max_seq, n_kv, head_dim]` | Contiguous write | Strided gather |

Attention dominates, so heads-outer wins. See
[`bench/results/stage13-dev-laptop.md`](../../bench/results/stage13-dev-laptop.md).

## 6. Memory formula

```text
bytes ≈ layers × n_kv × seq × head_dim × 2 × 4
```

Qwen3-0.6B at `seq=4096` is on the order of **~900 MiB** of f32 KV alone.
That is why long context is expensive even before weights.

## 7. What Stage 13 does *not* claim

Metal-resident KV for the tiny block already exists (Apple Stage 6). This
stage is the **Qwen CPU curriculum** for cache correctness and education.
Paged KV, quantized KV, and serving reuse come later.
