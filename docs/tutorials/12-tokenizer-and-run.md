# Tutorial — Tokenizer, sampling, and `zynfer run` (Stage 12)

Stage 11 proved the math. Stage 12 turns a **sentence** into model input and
prints generated text.

Full reference: [`docs/stages/12-tokenizer-sampling.md`](../stages/12-tokenizer-sampling.md).

## Prerequisites

```bash
# Artifact from Stage 10/11
ls models/qwen3-0.6b.zynfer
ls models/Qwen3-0.6B/vocab.json models/Qwen3-0.6B/merges.txt
zig build -Dhip=off
```

## 1. Encode check (optional)

Tokenizer unit tests cover `"Explain gravity simply."` →
`840, 20772, 23249, 4936, 13` when the HF dir exists.

```bash
zig build test -Dhip=off
```

## 2. Run generation

```bash
./zig-out/bin/zynfer run models/qwen3-0.6b.zynfer \
  --prompt "Explain gravity simply." \
  --tokenizer models/Qwen3-0.6B \
  --max-tokens 32
```

You should see:

1. Generated assistant text
2. A `---` separator
3. Metrics: `prompt_tokens`, `generated_tokens`, `ttft_ms`, `prefill_ms`,
   and `decode_tok_s` when more than one new token was produced

Default sampling is **greedy** (`--temperature 0`). CPU Qwen is slow — start
with `--max-tokens 16` while debugging.

## 3. What the command does

1. Loads BPE from `--tokenizer`
2. Wraps the prompt in the Qwen3 non-thinking chat template (`--raw` skips this)
3. Encodes to token IDs
4. Prefills the prompt (fills KV cache)
5. Samples the next token, decodes one step, repeats
6. Decodes generated IDs to UTF-8 and prints them

## 4. Sampling knobs

```bash
./zig-out/bin/zynfer run models/qwen3-0.6b.zynfer \
  --prompt "Explain gravity simply." \
  --temperature 0.6 --top-k 20 --top-p 0.95 --seed 42 \
  --max-tokens 32
```

Same `--seed` → same tokens for the same prompt and settings.

## What's next

| Item | Stage |
| --- | --- |
| Deeper KV / prefill-decode curriculum focus | **13–14** (host KV already used here) |
| Profiling one token | **15** |
| Metal Qwen path | After CPU text path is trusted |
