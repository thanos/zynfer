# Tutorial — Fusion at model scale (Stage M3)

Stage 6 taught one command buffer / one wait on a **tiny** block. Stage
**M3** re-earns that on Qwen3-0.6B and only keeps fusions that measure
well.

Full reference: [`docs/stages/M3-qwen-schedule-fusion.md`](../stages/M3-qwen-schedule-fusion.md).

## 1. Two paths

```bash
# Default: batched_resident_kv_fused (~2 waits/forward)
./zig-out/bin/zynfer qwen-bench --mini --max-tokens 4

# M0 per-op A/B (~17 waits/layer)
ZYNFER_QWEN_METAL=baseline ./zig-out/bin/zynfer qwen-bench --mini --max-tokens 4
```

Compare `wait/tok` and `decode_ms/tok`. Encodes stay high (many kernels);
**waits** are what collapsed.

## 2. What the schedule does

1. Upload all layer weights once at session init (resident MTLBuffers).
2. Keep K/V caches on Metal; append with `kv_append_f32`.
3. `batchBegin` → encode **all** layers → `batchCommit` (1 wait).
4. Second tiny batch: final RMSNorm + LM-head matvec (1 wait).
5. Host only sees embeddings in and logits out.

## 3. Fusion ledger (not a shopping list)

| Kept | Why |
| --- | --- |
| `silu_mul` | Already fused MLP activation |
| `add_rmsnorm_f32` | Post-attn residual + norm, one launch |
| Metal LM head | Was ~half of M2 decode wall on CPU |

| Rejected / deferred | Why |
| --- | --- |
| Q/K+RoPE fuse | No evidence after wait collapse |
| Attention tiling | Needs dedicated kernel + parity work |
| Dequant GEMV | M5 |
| ICB replay | KV changes every decode |

## 4. What to look at next

M3 moves the bottleneck toward **bytes/token** (weights + KV). **M4**
halves those with fp16; **M5** quantizes weights.
