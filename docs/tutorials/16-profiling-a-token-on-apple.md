# Tutorial — Profiling a decode token on Apple (Stage M2)

Stage M1 split prefill from decode. Stage **M2** asks where **one**
Metal decode token spends its wall time — before fusion.

Full reference: [`docs/stages/M2-profile-one-decode-token.md`](../stages/M2-profile-one-decode-token.md).

## 1. One command

```bash
# CI / no weights
./zig-out/bin/zynfer qwen-profile --mini

# Full Qwen locally
./zig-out/bin/zynfer qwen-profile models/qwen3-0.6b.zynfer \
  --prompt "Explain gravity simply."
```

Optional Instruments labels:

```bash
ZYNFER_SIGNPOSTS=1 ./zig-out/bin/zynfer qwen-profile --mini
```

Look for `qwen.rmsnorm`, `qwen.qkv`, `qwen.mlp`, … plus existing
`encode_and_wait` intervals.

## 2. Reading the table

Each row is **wall time** for that family (Metal ops include per-op
upload + encode + wait on the M0 path).

- **sum(families)** should nearly match **wall decode token**
- **empty_encode_wait_ns × metal_encodes** estimates launch overhead
  *already inside* the Metal family rows — it is not additive again
- **top3** is the gate answer for “where does the token go?”

## 3. Roofline

```text
ideal_tok_s ≈ measured_STREAM_GB/s / bytes_per_tok_est
fraction    = measured_tok_s / ideal_tok_s
```

Decode that is a few percent of roofline is expected while the path is
still per-op encode+wait and (today) CPU LM head.

## 4. What the full-model profile said (dev laptop)

On Qwen3-0.6B Metal, one decode token’s top three were:

1. **LM head + final norm** (CPU tied matvec — still host-side)
2. **MLP** (Metal matmuls + silu, launch-heavy)
3. **QKV projection**

That ranking — not intuition — drives M3 (schedule collapse) and the
open M0 LM-head path work.

## 5. Next

**M3** collapses waits across 28 layers and runs the fusion ledger only
for bottlenecks M2 named.
