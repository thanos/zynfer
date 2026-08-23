# Tutorial — Tokenizer, sampling, and `zynfer run` (Stage 12)

Stage 11 proved the math. Stage 12 turns a **sentence** into model input and
prints generated text (streaming).

Full reference: [`docs/stages/12-tokenizer-sampling.md`](../stages/12-tokenizer-sampling.md).

## 0. One-shot setup (local only)

```bash
zig build -Dhip=off
./zig-out/bin/zynfer setup
```

This runs `tools/setup_qwen.py`:

1. `pip install` huggingface_hub, safetensors, numpy (+ torch/transformers for golden)
2. Download `Qwen/Qwen3-0.6B` → `models/Qwen3-0.6B`
3. Convert → `models/qwen3-0.6b.zynfer`
4. Optional golden → `ref_logits.f32`

Flags: `--skip-golden`, `--skip-pip`, `--skip-download`.

**Never run setup in CI** (large download).

## 1. Encode check

```bash
zig build test -Dhip=off
```

## 2. Chat (recommended)

```bash
./zig-out/bin/zynfer chat "Explain gravity simply." --max-tokens 32
```

Streams tokens as they generate. Defaults:

- artifact: `models/qwen3-0.6b.zynfer`
- tokenizer: next to the artifact (`models/Qwen3-0.6B`) if present

Metrics footer includes `ttft_ms`, `decode_tok_s`, and `itl_ms p50/p95/p99`.

```bash
./zig-out/bin/zynfer chat "What is the value of pi?" --max-tokens 16
./zig-out/bin/zynfer chat --no-stream "Say hi." --max-tokens 8
```

## 3. Explicit `run`

```bash
./zig-out/bin/zynfer run models/qwen3-0.6b.zynfer \
  --prompt "Explain gravity simply." \
  --max-tokens 32
```

## What's next

| Item | Stage |
| --- | --- |
| KV curriculum depth | **13–14** |
| Profiling one token | **15** |
| Metal Qwen path | After CPU text path is trusted |
