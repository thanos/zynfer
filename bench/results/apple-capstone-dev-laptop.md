# Apple Stage M8 — Capstone ledger (dev laptop)

Declares **Backend 1 (Apple) complete**: registered Qwen3-4B, quantized
artifact path, final matrix, external comparison notes.

```text
date:              2026-08-25
host:              MacBook-Pro (Apple Silicon)
OS:                Darwin 25.5.0 (arm64)
Zig:               0.16.0
commands:
  zig build test -Dhip=off
  zig build stageM8 -Dhip=off
  ./zig-out/bin/zynfer stageM8
  python3 tools/setup_qwen.py --model 4b --quantize --skip-golden
  ZYNFER_QWEN_METAL=int8 ./zig-out/bin/zynfer chat models/qwen3-4b-int8.zynfer "…"
```

## Registry

| ID | HF repo | h / layers / heads | artifact | int8 |
| --- | --- | --- | --- | --- |
| `qwen3-0.6b` | Qwen/Qwen3-0.6B | 1024 / 28 / 16+8 | `models/qwen3-0.6b.zynfer` | `…-int8.zynfer` |
| `qwen3-4b` | Qwen/Qwen3-4B | 2560 / 36 / 32+8 | `models/qwen3-4b.zynfer` | `…-int8.zynfer` |

Dims are validated on load when `vocab_size == 151936` (`registry.validateArch`).
CI mini fixtures (tiny vocab) skip that check. Artifact load size limit raised
to 32 GiB for 4B bf16.

## KV budget (Qwen3-4B, **bf16** K/V on int8 Metal path)

| max_seq | KV (MiB) | notes |
| --- | --- | --- |
| 256 | ~36 | comfortable |
| 512 | ~72 | comfortable (`mem-report` metal_kv_cap) |
| 1024 | ~144 | fine on ≥32 GB unified |
| 2048 | ~288 | Metal `kv_len` cap; Unsupported above |

Int8 weight working set for 4B is ~4.1 GiB Metal-resident (i8 projs + scales +
bf16 embed table; norms f32). Prefer `models/qwen3-4b-int8.zynfer` on the laptop.

## Final Apple matrix — Qwen3-0.6B calibration (same machine)

Prompt `"Hi"` → 13 tokens; `max_new=8` (CPU row `max_new=4` in one run).
Warm process; first listed apple row after load. **Metal schedule** =
`ZYNFER_QWEN_METAL`.

| Path | Prefill TTFT (ms) | Decode tok/s (ITL) | Notes |
| --- | --- | --- | --- |
| CPU reference | ~31800 | ~0.35 | f32 oracle |
| Accelerate | *(same CPU row)* | *(same)* | large matmul via vDSP inside CPU path |
| Metal f32 (`baseline`) | ~1397 | ~0.63 | per-op waits (476 waits/tok) — not the shipped schedule |
| Metal bf16 | ~183 | ~7.0 | 2 waits/tok (M3/M6) |
| Metal int8 | ~110 | ~14.7 | shipped quantized path |
| Core ML / ANE | — | — | **REJECT** (Stage M7) |

Derived from `qwen-bench` JSON (`prefill_ns` / `ttft_ns` / `decode_ns`,
`generated_tokens−1` intervals):

- bf16 apple: `decode_ns=1.007e9` / 7 → ~7.0 tok/s  
- int8 apple: `decode_ns=4.75e8` / 7 → ~14.7 tok/s  

### Long prompt (0.6B / 4B int8 Metal)

`printf 'word %.0s' {1..200}` → **213** prompt tokens; `--max-tokens 8`; fresh process.

| Model | Prefill TTFT (ms) | Prefill tok/s | Decode tok/s |
| --- | --- | --- | --- |
| 0.6B int8 | ~1407 | ~151 | ~4.2 |
| 4B int8 | ~15352 | ~13.9 | ~1.9 |

Long decode is slower than short-prompt decode (attention scales with `kv_len`).

### Cold vs warm (4B int8, prompt `"Hi"`, max_new=8)

| | Prefill TTFT (ms) | Decode tok/s |
| --- | --- | --- |
| Cold (fresh process, includes load) | ~910 (forward only; load ~20 s wall) | ~3.6 |
| Warm haiku chat (after prior load in other runs) | ~1080 | ~3.9 |

Load is no longer dominated by host f32 dequant/pack; wall ~20 s is mostly
mmap + Metal upload of i8 + bf16 embed.

## Qwen3-4B matrix (int8 artifact, lab machine)

Artifact: `models/qwen3-4b-int8.zynfer` (4.1 GiB)  
SHA-256: `c7ef40e3312623f1958f5e4473313eb1bc06fbca1ec89721d4f15d2b26d0b541`  
`ZYNFER_QWEN_METAL=int8`, prompt `"Hi"` (13 tok), `max_new=8`.

| Path | Prefill TTFT (ms) | Decode tok/s | waits/tok |
| --- | --- | --- | --- |
| CPU | ~252700 | ~0.039 | — |
| Metal int8 | ~910 | ~3.6 | 2 |

4B bf16/f32 Metal rows: not filled (7.5 GiB bf16 artifact + host twin exceeds
comfortable laptop headroom vs int8 path). Use 0.6B bf16 row above for half path.

Chat smoke (`--max-tokens 24`): prefill ~1.1 s (19 tok), decode ~3.9 tok/s.

Haiku example:

```bash
ZYNFER_QWEN_METAL=int8 ./zig-out/bin/zynfer chat models/qwen3-4b-int8.zynfer \
  "give me a haiku on snow"
```

```text
White silence falls,
crystal whispers on the ground—
winter's breath is still.

---
prompt_tokens=19 generated_tokens=20 kv_cache=on backend=apple
prefill_ms≈1080  decode_tok_s≈3.88  itl_ms p50≈261
```

### Memory (`mem-report`, max_seq=512, int8 Metal)

| Item | Bytes |
| --- | --- |
| host_weights (RMS norms only) | ~0.75 MiB |
| metal_weights (i8 projs + scales + bf16 embed) | ~4.1 GiB |
| metal_kv_cap (bf16 K/V) | ~72 MiB |
| peak_rss | ~8.3 GiB |

Note: `loadForAppleQ8` skips host embed + proj f32 twins. Embed gathers from
Metal-resident bf16 table; tied lm_head uses bf16 matvec on that table (no
vocab-wide i8 pack at init). Projections upload artifact i8→Metal directly.

## External comparison (same Mac)

| Engine | Status on this host | Notes |
| --- | --- | --- |
| llama.cpp Metal | `llama-cli` present (`/opt/homebrew/bin`) | Compare with same prompt/length; convert 4B to GGUF separately — not automated here |
| MLX | `mlx_lm` **not** found | Install to fill; document gap until then |

Hypothesis template for gaps: bytes/token, batching, kernel fusion, quant
scheme mismatch (GGUF vs per-row int8). No matched llama.cpp/MLX numbers yet
(no Qwen3-4B GGUF on disk; `mlx_lm` absent).

## Why zynfer feels much slower than Ollama

**Plain English:** Ollama is a product tuned to feel fast on Apple Silicon.
Zynfer Phase M is a measured Metal engine with a real 4B path — not a
finished competitor. Comparing them without that context looks like a bug;
it is mostly **different goals + unpaid optimization debt**.

**Observed (this host):** Qwen3-4B int8 Metal ≈ **3.6–3.9 tok/s** decode and
≈ **0.9–1.1 s** TTFT for a short prompt (forward). Ollama (typically
llama.cpp Metal + a denser GGUF) is often many× faster for interactive chat.

| Factor | Zynfer (M8 4B int8) | Ollama (typical) |
| --- | --- | --- |
| Mission | Curriculum engine; retain/reject by ledger | Product UX / tok/s |
| Backend | Our Metal schedule (M0–M6) | llama.cpp Metal (years of kernels) |
| Quant | Per-row int8 (M5), documented | Common GGUF Q4_K / similar (fewer bytes/token) |
| Weight residency | Disk i8 → Metal i8 (~4.1 GiB) + bf16 embed; host norms only | Quantized weights stay closer to GPU; denser GGUF |
| Cold start | ~20 s load (mmap + Metal upload); no host f32 twin | Much lighter load path |
| Peak RSS | ~8.3 GiB (`mem-report`) | Usually far lower for 4B-class GGUF |
| Serving polish | Single-request `chat` / `qwen-bench` | Persistent server, tuned sampling/batching |

**Hypotheses for the gap (to re-check when running a matched A/B):**

1. **Bytes/token** — Ollama’s default quant is often lower bit-width than
   our int8; that alone moves the roofline.
2. **Kernel maturity** — attention / GEMV fusion and memory layouts in
   llama.cpp vs our educational op set.
3. **Fairness** — same model size, quant family, prompt length, max tokens,
   and warm vs cold. Do not compare a warm Ollama server to a cold
   `zynfer chat` process.

**Deferred (not this pass):** denser than int8 weights (4-bit / GGUF-like);
further Metal kernel polish. Phase S / follow-ons.

**What this does *not* mean:** Metal is “wrong,” or M8 failed. Capstone
numbers are an honest baseline.

**Matched comparison recipe (when filling numbers):**

```bash
# Zynfer (warm process preferred for decode tok/s)
ZYNFER_QWEN_METAL=int8 ./zig-out/bin/zynfer qwen-bench models/qwen3-4b-int8.zynfer \
  --prompt "Hi" --max-tokens 32

# Ollama / llama.cpp — same prompt length & new-token count; note model tag/quant
# ollama run qwen3:4b   # or llama-cli -m <gguf> -p "Hi" -n 32
```

Record Ollama/llama.cpp TTFT and tok/s here when run; until then the gap
is explained qualitatively above.

## Decision

| Item | Result |
| --- | --- |
| Apple Backend 1 | **complete** (Metal M0–M6 + Accelerate; Core ML rejected M7) |
| Capstone model | Registered **Qwen3-4B** (int8 preferred) |
| Gate | `zynfer stageM8` + chat on registered quantized artifact |

## Files

- `src/model/registry.zig`, `qwen3.zig` (`qwen3_4b`)
- `tools/setup_qwen.py --model 4b --quantize`
- `tools/checkpoint/safetensors_to_zynfer.py --model-id`
- CLI: `zynfer stageM8`
- Docs: `docs/stages/M8-apple-capstone.md`, tutorial 22
