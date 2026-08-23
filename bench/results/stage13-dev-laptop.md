# Stage 13 — KV cache (dev laptop)

```text
date:              2026-08-22
host:              MacBook-Pro (Apple Silicon)
Zig:               0.16.0
commands:
  zig build test -Dhip=off
  zig build stage13 -Dhip=off
  zig build kv-bench -Dhip=off
  zig build kv-layout -Dhip=off
  ./zig-out/bin/zynfer kv-bench --mini --max-tokens 4
  ./zig-out/bin/zynfer kv-bench --layout
  ./zig-out/bin/zynfer kv-bench models/qwen3-0.6b.zynfer \
    --prompt "Explain gravity simply." --max-tokens 8
```

## Exit criterion (CI)

- Unit test: mini greedy cached ids == uncached ids
- `KvCache` bytesUsed / capacity + Qwen3-0.6B formula test
- Layout index + bake-off microbench smoke test
- `zynfer stage13` ledger
- `zynfer kv-bench --mini` → `token_parity: PASS`
- `zynfer kv-bench --layout` → `RETAIN [n_kv, max_seq, head_dim]`

## Mini fixture (`kv-bench --mini --max-tokens 4`)

```text
token_parity: PASS
cached:   decode_tok_s≈34749
uncached: decode_tok_s≈11778
decode_speedup (uncached/cached wall): 2.95×
```

(Mini is tiny; speedup is illustrative. Full model shows the real cost.)

## Layout bake-off (`kv-bench --layout`)

Qwen3-0.6B-shaped: `n_q=16`, `n_kv=8`, `head_dim=128`, `max_seq=2048`,
`kv_len=1024`, warmup=3, iters=25.

```text
RETAIN  [n_kv, max_seq, head_dim]
  attn_ns≈13091130  append_ns≈306
reject  [max_seq, n_kv, head_dim]
  attn_ns≈44989218  append_ns≈141

attn speedup (seq-outer / heads-outer): 3.44×
append cost ratio (heads-outer / seq-outer): 2.17×
```

Decision: retain heads-outer. Decode attention is the hot path (~ms);
append is ns-scale even when 2× slower.

## Full model (local)

Greedy, prompt `"Explain gravity simply."`, `--max-tokens 8` (2026-08-22):

```text
token_parity: PASS
cached_ids:   38409,374,264,5344,429,60091,1449,1633
uncached_ids: 38409,374,264,5344,429,60091,1449,1633

cached:   prefill_ms≈41083  decode_tok_s≈0.284  kv_bytes_used=5734400
uncached: prefill_ms≈38177  decode_tok_s≈0.018
decode_speedup (uncached/cached wall): 16.16×
```

Earlier `--max-tokens 4` run: parity PASS, speedup **16.53×**.

CPU decode remains a correctness/education path, not a speed target. The
point is the **ratio**: uncached recomputes the growing prefix every step.

## Notes

- Default generate uses the cache; `--no-kv-cache` forces the educational path.
- Host layout: `[n_kv, max_seq, head_dim]` f32 K and V per layer.
- Formula: `layers × n_kv × seq × head_dim × 2 × 4` bytes.
