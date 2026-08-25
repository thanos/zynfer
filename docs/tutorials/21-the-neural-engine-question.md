# Tutorial — The Neural Engine question (Stage M7)

Apple Silicon includes a Neural Engine (ANE). It is tempting to treat it as
another accelerator backend beside Metal. That is the wrong mental model for
zynfer.

## In plain English

We asked: can the Neural Engine make Qwen faster in zynfer?

**Answer: no — not for this project.** We did not ship an ANE backend. We
shipped a **decision**: keep Apple inference on **Metal (+ Accelerate)**,
reject Core ML as an inference path, and write that down so nobody
half-builds it later.

Why that sounds surprising: people assume you can “just use” the Neural
Engine like Metal. You cannot. There is no public way to write Neural
Engine kernels. The only supported door is Core ML — export a model, hand
it to Apple’s runtime, and *hope* it runs on the Neural Engine (it might
run on CPU or GPU instead). Claiming “ANE wins” from a config enum alone
is incorrect.

What M7 delivered instead of a new engine:

- A stronger Core ML **probe** that never claims ANE execution
- `zynfer stageM7` + a reject ledger with reproduction commands
- A loud fail if someone forces Core ML (`ZYNFER_FORCE_COREML=1` → exit 2)
- Optional polish: toy `.mlpackage` load smoke + one `xctrace` recipe
  (proves plumbing works; does **not** clear retain)

## ANE is not a Metal peer

There is **no public ANE instruction set**. You cannot write ANE kernels the
way you write Metal Shading Language. The only supported path is:

```text
your graph → Core ML model (.mlmodel / .mlpackage) → Core ML runtime → (maybe) ANE
```

Core ML may place work on CPU, GPU, or ANE. Placement is opaque without
Instruments.

## What Stage 7 already knew

Stage 7 linked Core ML, probed `MLModelConfiguration` compute units, and
**rejected** an inference path: no model, no subgraph, no Instruments proof.
It deferred the question to “when a stable compiled subgraph exists.”

## What Stage M7 reopens — and closes

M0–M6 built a real Qwen Metal engine (resident weights, fused schedule,
int8, static decode). That is exactly the condition Stage 7 named for a
revisit. M7 therefore asks:

1. Can a **fixed-shape prefill** subgraph beat Metal TTFT?
2. Can a **stateful decode** graph (`MLState`, macOS 15+) carry KV and win
   tok/s?
3. Or is the honest answer still **nothing**?

We chose **(3)**. See [Why we did not try (a) and (b)](#why-we-did-not-try-a-and-b)
below — “skip” means we did not start a multi-month converter + hybrid
runtime project with a weak chance of beating resident Metal, not “we ran
the race and lost.”

### MLState

`MLState` makes autoregressive / stateful Core ML graphs representable.
The class is present on Darwin 24+ (`zynfer stageM7` prints
`MLState class available`). Presence ≠ a Qwen decode engine. Mutable KV
length and attention shapes remain hostile to frozen graphs.

### coremltools

Allowed **only** as a development-time converter (like
`safetensors_to_zynfer.py`). Never as a runtime dependency.

## Why we did not try (a) and (b)

### (a) Fixed-shape prefill on Core ML

To try this properly you would need: export a real Qwen piece as Core ML,
wire load/predict beside Metal, copy or repack activations, prove ANE in
Instruments, and show faster TTFT than Metal.

We did not start that because the **starting position already looked bad**:

- There was **no Qwen Core ML graph** — the converter/model work *is* most
  of the stage, not a quick A/B.
- Metal already keeps weights **resident**. Splicing Core ML for a chunk
  fights that win (see [Unified memory ≠ free handoff](#unified-memory--free-handoff)).
- Prefill is where tensor volume is largest, so any handoff tax shows up
  hardest on time-to-first-token — the metric a Core ML prefill offload
  would try to win.
- A successful load still would not mean “ANE won.”

So (a) is **SKIP**: high cost to get a fair race; race handicapped against
resident Metal.

### (b) Stateful decode with `MLState`

Same story, harder:

- The API **exists** on Darwin 24+ (we checked).
- Presence ≠ a decoder. You still need an exported graph that carries
  Qwen KV / attention state.
- Decode length and cache shapes **change every token**. Frozen Core ML
  graphs hate that; Metal’s path is built for it.
- Again: no Instruments ANE proof and no Metal A/B until that graph exists.

So (b) is **SKIP**: API available; the Qwen engine on top of it is not, and
building it is the whole adventure.

### Retain bar

M7 required **all** of: real ANE proof, beat Metal e2e, copies do not erase
the win, graph stable enough. We had **none** of those for Qwen. Getting
them meant months of work *before* a single honest number — so the
deliverable is a final reject ledger, not a zombie half-path.

## Unified memory ≠ free handoff

Apple Silicon has **one DRAM pool** (CPU, GPU, and Neural Engine share
physical memory). That removes classic PCIe “copy to a discrete GPU”
costs. It does **not** mean Metal ↔ Core ML is free.

“Shipping data across” in the M7 ledger means **across frameworks /
layouts / ownership**, not across a PCIe bus:

1. **Weights often live twice (or in two layouts).** Metal already has
   Qwen weights in `MTLBuffer`s shaped for zynfer kernels. Core ML wants
   a compiled `.mlmodelc` with its own packing. You usually cannot point
   Core ML at the same Metal weight buffer.
2. **Activations still get packaged.** Core ML I/O (`MLMultiArray`,
   feature providers) often requires a required shape/dtype/layout.
   Metal → Core ML → Metal may repack even when both buffers sit in the
   same DRAM.
3. **ANE is pickier than “same RAM.”** Getting work onto the Neural Engine
   can involve framework-managed staging. Instruments is required.

```text
Same chip, same DRAM
        │
        ├─ Metal path:  weights already in GPU-friendly buffers you own
        │
        └─ Core ML path: separate compiled model + its I/O contract
```

That is why criterion (3) still bites on unified memory: a Core ML prefill
splice would fight the residency win M6 already paid for.

## Retain bar (checklist)

| Criterion | Status in zynfer |
| --- | --- |
| Instruments shows ANE (not CPU/GPU fallback) | **not for Qwen** — toy xctrace run only; see ledger |
| e2e TTFT or tok/s beats M6 Metal | **N/A** — no Core ML Qwen path |
| I/O copies do not erase the win | Would tax resident-Metal path (even with unified memory) |
| Graph amortizes compilation | No stable Qwen Core ML graph |

## Toy smoke + xctrace (optional polish)

A tiny fixed-shape matmul lives at `tools/fixtures/coreml_toy.mlpackage`
(rebuild with `make_coreml_toy.py` + `coremltools`). It proves the ObjC
bridge can **compile → load → predict** under `CPUAndNeuralEngine`:

```bash
./zig-out/bin/zynfer coreml-smoke tools/fixtures/coreml_toy.mlpackage
```

Record once with Instruments / `xctrace` (does **not** clear the M7 reject):

```bash
xctrace record --template 'Core ML' --time-limit 15s \
  --output /tmp/zynfer-coreml-toy.trace --target-stdout - \
  --launch -- ./zig-out/bin/zynfer coreml-smoke tools/fixtures/coreml_toy.mlpackage
```

Open the `.trace` in Instruments to inspect Core ML / Neural Engine lanes.
`ane_execution_verified` stays false unless you confirm ANE placement there
**and** you have a Qwen-scale Metal A/B win — neither applies to the toy.

## What “NInfer for Core ML” would be (and why zynfer is not that)

Projects like **NInfer** specialize: few Qwen checkpoints, one stack
(CUDA), custom artifact, max single-GPU tok/s, CLI + HTTP APIs.

A Core ML analogue would be: few exported Qwen graphs, Core ML as the
**main** path, custom `.mlpackage`, product serving — closer to rewriting
the mission than to Stage M7. That would drop multi-backend purity and
fight public-API limits (no ANE ISA; private `_ANE*` paths are a different
risk profile).

Zynfer’s choice: **Metal is Backend 1.** Core ML stays a documented no.
Roadmap milestone **J (“NInfer philosophy”)** at M8 means a **registered
larger checkpoint** (Qwen3-4B) on the retained Metal engine — not “become
NInfer on the Neural Engine.”

## How to collect Instruments evidence (if you reopen later)

1. Export a fixed-shape **Qwen** Core ML subgraph (dev-time `coremltools`).
2. Load with `MLComputeUnitsCPUAndNeuralEngine` (or All).
3. Instruments / `xctrace` with the Core ML template (recipe above).
4. Confirm work on **ANE**, not CPU/GPU fallback.
5. A/B TTFT and tok/s against `ZYNFER_QWEN_METAL=bf16` (or int8) Metal.
6. Only then consider retain — and still require no handoff tax.

Without that proof, keep `ane_execution_verified = false` and do not claim
ANE wins.

## Decision

**REJECT (final for this project).** The Apple story is Metal + Accelerate.
`ZYNFER_FORCE_COREML=1` exits with code 2 — no silent half-path.

```bash
./zig-out/bin/zynfer stageM7
./zig-out/bin/zynfer coreml-smoke
ZYNFER_FORCE_COREML=1 ./zig-out/bin/zynfer caps   # exit 2
```

Ledger: [`bench/results/apple-ane-qwen-dev-laptop.md`](../../bench/results/apple-ane-qwen-dev-laptop.md).

## Next — what Stage M8 gets you

**M8** is the Apple **capstone**, not another accelerator experiment.

After M8 (“Apple-complete” Backend 1):

- Registered **Qwen3-4B** (dims / artifact / checksum — prove nothing assumed 0.6B)
- Quantized artifact sized for the Mac; KV budget documented
- Final Apple **benchmark matrix** (CPU / Accelerate / Metal f32 / fp16 /
  quantized / Core ML–ANE or its documented reject) × prefill × decode ×
  cold/warm
- External comparison permitted: llama.cpp-Metal and MLX on the same
  machine, same definitions — report wins and losses
- Tutorial 22 retrospective: what made Apple inference fast

You do **not** get Phase R (AMD) or Phase S (HTTP / batching / speculative)
from M8 alone — those follow after Apple-complete.

See [`docs/stages/M8-apple-capstone.md`](../stages/M8-apple-capstone.md).
