# Stage M8 — Apple capstone: quantized Qwen3-4B + release

**Status: done (Apple-complete).** Declares **Backend 1 (Apple) complete**
after registering Qwen3-4B, wiring the quantized path, and publishing the
final matrix ledger.

Part of **Phase M** ([`fable-5-prompt.md`](../../baoulo/prompts/fable-5-prompt.md)).
Old stage 25 (Apple slice). Follows M7’s final Core ML/ANE **REJECT**.

## In plain English

M0–M7 built and settled the Apple engine (Metal + Accelerate; ANE out).
**M8 is graduation:** register **Qwen3-4B**, prefer an int8 `.zynfer`, fill
the scoreboard (and optional llama.cpp / MLX notes), and call Backend 1 done.

You do **not** get AMD (Phase R) or an HTTP server / batching (Phase S)
from M8 alone.

## Goal

```bash
zig build stageM8 -Dhip=off
./zig-out/bin/zynfer stageM8
python3 tools/setup_qwen.py --model 4b --quantize --skip-golden
ZYNFER_QWEN_METAL=int8 ./zig-out/bin/zynfer chat models/qwen3-4b-int8.zynfer \
  "give me a haiku on snow"
```

Example (warm process, lab laptop):

```text
White silence falls,
crystal whispers on the ground—
winter's breath is still.

---
prompt_tokens=19 generated_tokens=20 kv_cache=on backend=apple
prefill_ms≈1088  decode_tok_s≈3.86  itl_ms p50≈262
```

## Scope (delivered)

- Explicit **registry** (`src/model/registry.zig`): ID, HF repo, paths, dims,
  artifact version; dim check on full-vocab loads.
- Qwen3-4B Arch (h=2560, 36 layers, GQA 32/8); converter `--model-id` /
  dim inference; setup `--model 4b --quantize`.
- Artifact mmap limit raised to 32 GiB.
- KV budget table in `stageM8` + ledger.
- Final Apple matrix in `bench/results/apple-capstone-dev-laptop.md`
  (0.6B measured; 4B filled after local quantize).
- External comparison: llama.cpp present / MLX absent on lab host — noted.
- Tutorial 22 retrospective.

## Gate — “Apple-complete”

A user on an M-series Mac can `zynfer run` / `zynfer chat` a registered
quantized model at competitive, reproducible, documented speed, and every
claim in the matrix has a ledger behind it.

## What M8 is not

- Not reopening Core ML/ANE.
- Not “NInfer for Core ML.” Milestone **J** = registered larger **Metal**
  checkpoint.
- Not Phase R or Phase S.

## Related

- Ledger: [`bench/results/apple-capstone-dev-laptop.md`](../../bench/results/apple-capstone-dev-laptop.md)
- Tutorial: [`docs/tutorials/22-what-makes-apple-inference-fast.md`](../tutorials/22-what-makes-apple-inference-fast.md)
- Prior: [`M7-ane-coreml-qwen.md`](M7-ane-coreml-qwen.md)
