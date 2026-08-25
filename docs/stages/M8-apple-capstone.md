# Stage M8 — Apple capstone: quantized Qwen3-4B + release

**Status: not started.** Declares **Backend 1 (Apple) complete** after
proving the Metal engine on a model people actually want to run.

Part of **Phase M** ([`fable-5-prompt.md`](../../baoulo/prompts/fable-5-prompt.md)).
Old stage 25 (Apple slice). Follows M7’s final Core ML/ANE **REJECT**.

## In plain English

M0–M7 built and settled the Apple engine (Metal + Accelerate; ANE out).
**M8 is graduation:** ship **Qwen3-4B**, fill the scoreboard, compare to
MLX / llama.cpp-Metal on the same Mac, and call the Apple backend done.

You do **not** get AMD (Phase R) or an HTTP server / batching (Phase S)
from M8 alone — those come after **Apple-complete**.

## Goal

Prove the engine on a registered quantized Qwen3-4B artifact and publish
the final Apple benchmark matrix with ledgers behind every claim.

## Scope (from fable-5)

- Register Qwen3-4B (ID, architecture, dimensions, artifact version,
  checksum). Validate that nothing silently assumed 0.6B dimensions.
- Quantized artifact sized for the dev machine’s unified memory; KV
  budget documented per context length.
- **Final Apple benchmark matrix:** CPU reference / Accelerate / Metal
  f32 / Metal fp16 / Metal quantized / Core ML–ANE hybrid **or its
  documented rejection (M7)** — each × short + long prefill and batch-1
  decode, cold and warm.
- External comparison now permitted: llama.cpp-Metal and MLX on the same
  machine, same model/quant/prompt/length/sampling, same measurement
  definitions. Report honestly, including losses; each gap gets a
  hypothesis in the ledger.
- Tutorial `docs/tutorials/22-what-makes-apple-inference-fast.md` —
  retrospective: which optimizations mattered, in what order, and what
  the roofline says about what remains.

## Gate — “Apple-complete”

A user on an M-series Mac can `zynfer run` / `zynfer chat` a registered
quantized model at competitive, reproducible, documented speed, and every
claim in the matrix has a ledger behind it.

## What M8 is not

- Not reopening Core ML/ANE (unless new evidence meets M7’s four criteria).
- Not “NInfer for Core ML.” Roadmap milestone **J (NInfer philosophy)**
  here means a **registered larger checkpoint** on the retained Metal
  path — specialize the *model registry*, not switch accelerators.
- Not Phase R (AMD) or Phase S (serving).

## Commands (expected)

```bash
# Exact commands land with the stage implementation + ledger.
zig build stageM8 -Dhip=off   # when wired
./zig-out/bin/zynfer stageM8
./zig-out/bin/zynfer chat   # registered quantized Qwen3-4B
```

## Related

- Prior stage: [`M7-ane-coreml-qwen.md`](M7-ane-coreml-qwen.md)
- Roadmap: [`docs/roadmap.md`](../roadmap.md)
- Spec: fable-5 § Stage M8
