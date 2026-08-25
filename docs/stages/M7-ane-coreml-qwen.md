# Stage M7 — ANE / Core ML at Qwen scale

**Status: done (REJECT).** Reopens the Stage 7 Core ML reject with a real
compiled-subgraph context (M0–M6 Qwen Metal path exists) and **closes** the
ANE question for this project: no retained Core ML inference path.

Part of **Phase M** ([`fable-5-prompt.md`](../../baoulo/prompts/fable-5-prompt.md)).
Reopens Stage 7 reject.

## In plain English

We asked whether the Neural Engine could make Qwen faster. The answer is
**no for this project** — not because we raced Core ML and lost, but because
starting that race meant months of converter + hybrid runtime work with a
weak chance of beating **resident Metal**, and without Instruments we could
not honestly claim ANE anyway.

**Deliverable:** a final reject decision + probe/smoke tooling + ledger.
**Not delivered:** a Core ML Qwen inference path.

Full teaching writeup: [`docs/tutorials/21-the-neural-engine-question.md`](../tutorials/21-the-neural-engine-question.md).

## Goal

```bash
zig build stageM7 -Dhip=off
./zig-out/bin/zynfer stageM7
./zig-out/bin/zynfer coreml-smoke
ZYNFER_FORCE_COREML=1 ./zig-out/bin/zynfer caps   # exit 2
```

## Honest framing

- ANE is **only** reachable through Core ML; there is no public ANE ISA.
- Zynfer cannot hand-write ANE kernels; it can only offload compiled graphs.
- `coremltools` would be **dev-time only** (like the checkpoint converter) —
  never in the runtime path.
- Modern APIs of interest: **MLState** (macOS 15+ / Darwin 24+), fixed-shape
  prefill graphs, Core ML weight compression.
- Dynamic shapes and mutable KV remain the known hazards.
- **Unified memory ≠ free Metal↔Core ML handoff.** Same DRAM still means
  separate compiled weights / `MLMultiArray` packaging / possible ANE
  staging. Prefill is where that tax hurts most (see tutorial).

## Candidates (a → b → c)

| Candidate | Outcome |
| --- | --- |
| (a) Fixed-length prefill subgraph | **SKIP** — no Qwen `.mlmodel`/converter; handoff tax vs resident Metal; fair race is most of the stage |
| (b) Stateful decode (MLState) | **SKIP** — class present on Darwin 24+, but no exported Qwen KV graph; mutable KV / dynamic shapes; no Instruments ANE proof |
| (c) Nothing | **CHOSEN** |

SKIP ≠ “we implemented it and it was slow.” It means we did not start the
large project required before a single honest A/B number.

## Retain criteria (all required — none met)

1. Instruments confirms ANE placement (not CPU/GPU fallback inside Core ML)
2. End-to-end TTFT or tok/s beats **M6 Metal** at equal quality
3. I/O / layout handoffs do not erase the win (still true on unified memory)
4. Graph stable enough to amortize compilation

## Gate

1. `bench/results/apple-ane-qwen-dev-laptop.md` with decision + repro commands
2. `ZYNFER_FORCE_COREML` exits loudly (exit 2)
3. Tutorial 21
4. Probe reports MLState availability without claiming ANE execution

## Optional polish (done; does not clear retain)

- Toy `.mlpackage` + `zynfer coreml-smoke` (compile → load → one predict)
- `xctrace` Core ML template recorded once on that toy (see ledger)

## Decision

**REJECT (final).** Retained Apple engine = Metal (M0–M6) + Accelerate (Stage 5).
No zombie half-support.

Becoming “NInfer for Core ML” (few exported graphs, Core ML as the main
product path) would be a **mission change**, not an M7 reopen. Milestone J
at M8 is a registered **larger Metal checkpoint**, not ANE specialization.

## Next

[`M8 — Apple capstone`](M8-apple-capstone.md): quantized Qwen3-4B + final
benchmark matrix → **Apple-complete**.

## Files

| Path | Role |
| --- | --- |
| `src/backends/apple/coreml.zig` + bridge | Probe (+ MLState / Darwin); `smoke`; ops Unsupported |
| `tools/fixtures/coreml_toy.mlpackage` | Tiny fixed-shape matmul for load smoke |
| `tools/fixtures/make_coreml_toy.py` | Dev-time rebuild (`coremltools`) |
| `src/main.zig` | `stageM7`, `coreml-smoke` |
| `docs/tutorials/21-the-neural-engine-question.md` | Walkthrough (plain English + rationale) |
| `bench/results/apple-ane-qwen-dev-laptop.md` | Ledger |
