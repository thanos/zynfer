# Tutorial — Prefill vs decode on Qwen (Stage M1)

Stage M0 got Qwen blocks onto Metal. Stage **M1** stops treating
“tokens per second” as one number.

Full reference: [`docs/stages/M1-prefill-vs-decode-qwen.md`](../stages/M1-prefill-vs-decode-qwen.md).

## 1. Two workloads

| Phase | What happens | Bottleneck feel |
| --- | --- | --- |
| **Prefill** | Whole prompt through all layers once | Large GEMMs (`t × hidden`) |
| **Decode** | One new token; KV grows | Small matvecs + KV bandwidth + launches |

An optimization that speeds prefill can slow decode (and vice versa).
Always look at the split.

## 2. One command

```bash
# CI / no weights
./zig-out/bin/zynfer qwen-bench --mini --max-tokens 4

# Full Qwen locally
./zig-out/bin/zynfer qwen-bench models/qwen3-0.6b.zynfer \
  --prompt "Explain gravity simply." --max-tokens 8
```

Expect a table with rows for `cpu` and `apple`, then a `json` line.

## 3. Reading the columns

- **prefill_t/s** — prompt tokens / prefill wall time  
- **decode_t/s** — (generated − 1) / decode wall (intervals after first token)  
- **B/tok_est** — approximate f32 bytes read per decode token  
- **enc/tok / wait/tok** — measured Metal kernel encodes and GPU waits per
  `decodeToken` (M0 per-op: usually equal)

## 4. Everyday `run` / `chat`

Footers now always include `prefill_tok_s` and `decode_ms_per_tok` so
routine generates keep the M1 split.

## 5. Next

**M2** profiles *where* one Metal decode token goes (and the bandwidth
roofline). Do not start fusion (M3) until that table exists.
