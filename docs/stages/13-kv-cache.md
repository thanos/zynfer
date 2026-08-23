# Stage 13 — KV cache

**Status: done (CPU Qwen).** Persistent K/V across decode steps, with an
intentional full-recompute baseline for comparison.

Stage 12 already streamed with a cache. Stage 13 makes the cache the lesson:
why it exists, how much memory it costs, and how much slower life is without it.

## Goal

```bash
zig build stage13 -Dhip=off
zig build kv-bench -Dhip=off
./zig-out/bin/zynfer kv-bench --mini --max-tokens 4
# local full model:
./zig-out/bin/zynfer kv-bench models/qwen3-0.6b.zynfer --max-tokens 8
```

Cached greedy tokens match uncached; decode with cache is dramatically faster.

## In plain English

- **Without a KV cache** — every new token re-runs attention over the whole
  prompt + everything generated so far. Cost grows like O(n²).
- **With a KV cache** — after prefill, each decode step only computes K/V for
  the new token and appends them. Attention reuses prior K/V.
- **Three memories**
  - **Weights** — fixed model parameters (the `.zynfer` file).
  - **Activations** — scratch for the current forward (hidden states, Q, …).
  - **KV cache** — growing history of keys and values per layer.

## Layout

Logical tree (per layer):

```text
layer
  └── sequence position
       └── KV head
            └── head dimension
```

Physical host layout in Zynfer: dense `[n_kv, max_seq, head_dim]` for K and
for V (`src/runtime/kv_cache.zig`). Attention reads `used` positions with
stride `max_seq`.

### Layout bake-off (measured)

Compared to `[max_seq, n_kv, head_dim]` with Qwen3-0.6B-shaped dims
(`n_q=16`, `n_kv=8`, `head_dim=128`, `kv_len=1024`):

| Path | Winner | Why |
| --- | --- | --- |
| Decode attention K/V scan | **heads-outer (retained)** | Contiguous `head_dim` rows; ~3× faster gather |
| Append one token | seq-outer | One contiguous write across heads (ns-scale) |

Attention dominates decode wall time, so **`[n_kv, max_seq, head_dim]` is retained**.
Re-run locally: `./zig-out/bin/zynfer kv-bench --layout` / `zig build kv-layout`.

### Memory formula (f32)

```text
bytes ≈ num_layers × n_kv_heads × seq_len × head_dim × 2 × sizeof(f32)
```

For Qwen3-0.6B (`28 × 8 × seq × 128 × 2 × 4`):

| seq | KV bytes (approx) |
| --- | --- |
| 1 | ~224 KiB |
| 4096 | ~896 MiB |
| 40960 (max) | ~8.75 GiB |

## In scope

| Item | Notes |
| --- | --- |
| Host `KvCache` | append / reset / bytesUsed / bytesCapacity |
| Cached generate | default `Session.generate` |
| Uncached generate | `use_kv_cache=false` / `--no-kv-cache` |
| Parity | unit test + `kv-bench` |
| Bench | `zynfer kv-bench` decode_tok_s with vs without |
| Layout bake-off | `zynfer kv-bench --layout` |

## Explicitly not Stage 13

| Item | Owner |
| --- | --- |
| Metal-resident Qwen KV | Later (tiny-block already has Apple Stage 6) |
| Prefill vs decode metric split | Stage 14 |
| Paged / quantized KV | Stages 18 / 22 |

## Prerequisites

- Stages 11–12
- Zig 0.16

## Commands

```bash
zig build test -Dhip=off
zig build stage13 -Dhip=off
zig build kv-bench -Dhip=off
zig build kv-layout -Dhip=off

# Mini fixture (CI / no weights)
./zig-out/bin/zynfer kv-bench --mini --max-tokens 4

# Layout bake-off (no weights)
./zig-out/bin/zynfer kv-bench --layout

# Full model (local)
./zig-out/bin/zynfer kv-bench models/qwen3-0.6b.zynfer \
  --prompt "Explain gravity simply." --max-tokens 8

# Force uncached path on normal generate
./zig-out/bin/zynfer chat "Explain gravity simply." \
  --max-tokens 8 --no-kv-cache --no-stream
```

## Exit criterion

1. Inefficient decoder exists (`use_kv_cache=false`).
2. Cached path is the default generate path.
3. `kv-bench` reports with vs without cache.
4. Cached greedy tokens match uncached.
5. Docs teach weights / activations / KV + the memory formula.

## Files

| Path | Role |
| --- | --- |
| `src/runtime/kv_cache.zig` | Layout + memory helpers |
| `src/model/qwen_forward.zig` | `generateCached` / `generateUncached` |
| `src/main.zig` | `stage13`, `kv-bench`, `--no-kv-cache` |
| `docs/tutorials/13-kv-cache.md` | Walkthrough |
| `bench/results/stage13-dev-laptop.md` | Local numbers |
