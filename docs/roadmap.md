# Roadmap

Zynfer is built as a staged curriculum. Each stage produces code, tests,
a tutorial, a benchmark command, and a short report.

**Planning authority:** [`baoulo/prompts/fable-5-prompt.md`](../baoulo/prompts/fable-5-prompt.md)
supersedes the stage ordering in the master development prompt. Golden
Rules, Definition of Done, benchmark philosophy, and teaching obligations
from the master prompt remain binding.

## Phase overview

```text
Phase M — Apple Qwen backend, complete and fast   (done — Apple-complete)
Phase R — AMD gfx1201 campaign                    (when hardware lands)
Phase S — Serving & scale, backend-neutral        (next; overlaps R)
```

| Phase | Goal | Precondition |
| --- | --- | --- |
| **M** | Maximally fast Apple M-series Qwen inference (Metal primary) | **done** (M8) |
| **R** | Maximally fast RDNA 4 HIP inference | Physical `gfx1201` hardware |
| **S** | Batching, prefix cache, speculative, HTTP server | M8 Apple-complete (**met**) |

## Milestones

- **A. Hello, GPU** — HIP alloc/copy (Phase R0)
- **B. We own the math** — transformer primitives on our kernels
- **C. One block** — one Qwen block matches the oracle
- **D. It is an LLM** — Qwen3-0.6B greedy tokens match reference (**done**)
- **E. It is an inference engine** — KV-cached generation (**done**, Stage 13)
- **F. We understand the bottleneck** — one decode token accounted for (M2)
- **G. AMD-native** — kernels tuned for `gfx1201` (Phase R)
- **H. Specialized** — quant, fusion, graphs measured (M4–M6, R5–R7)
- **I. Useful** — concurrent requests (Phase S)
- **J. NInfer philosophy** — registered larger checkpoint on the **retained**
  backend (M8 Apple Metal 4B; R10 AMD) — not “NInfer for Core ML / ANE”

---

## Closed foundation (do not reopen without trigger)

| Area | Status |
| --- | --- |
| Apple Stages 0–8 (tiny-block) | **Closed** — one CB/wait, resident KV, simdgroup, int8 ops |
| Stage 10 | **Done** — `.zynfer` v1 |
| Stage 11 | **Done** — Qwen CPU forward + golden |
| Stage 12 | **Done** — tokenizer + sampling |
| Stage 13 | **Done** — KV cache vs uncached |

Tiny-block rejects with **reopen triggers** (see fable-5 §1): fp16, Session
int8, ICB, extra fusions, Core ML/ANE — reopened at Qwen scale in M3–M7.

---

## Phase M — Apple M-series (strict order)

| Stage | Title | Old # | Status |
| --- | --- | --- | --- |
| **M0** | Metal Qwen forward + generate (f32) | — | **done** — baseline + residual checklist closed by M1–M8; [`docs/stages/M0-metal-qwen-forward.md`](stages/M0-metal-qwen-forward.md) |
| **M1** | Prefill vs decode on Qwen | 14 | **done** — [`docs/stages/M1-prefill-vs-decode-qwen.md`](stages/M1-prefill-vs-decode-qwen.md) |
| **M2** | Profile one decode token | 15 | **done** — [`docs/stages/M2-profile-one-decode-token.md`](stages/M2-profile-one-decode-token.md) |
| **M3** | Qwen-scale scheduling + fusion | 16 (Apple) | **done** — [`docs/stages/M3-qwen-schedule-fusion.md`](stages/M3-qwen-schedule-fusion.md) |
| **M4** | fp16/bf16 Metal path | 8 reject reopen | **done** |
| **M5** | Weight quantization (Apple) | 18 (Apple) | **done** (int8) |
| **M6** | Static decode plan (Apple) | 20 (Apple) | **done** |
| **M7** | ANE / Core ML gated experiment | 7 reject reopen | **done (REJECT)** — [`docs/stages/M7-ane-coreml-qwen.md`](stages/M7-ane-coreml-qwen.md) |
| **M8** | Capstone: quantized Qwen3-4B + matrix | 25 (Apple) | **done (Apple-complete)** — [`docs/stages/M8-apple-capstone.md`](stages/M8-apple-capstone.md) |

**M0 gate (closed for curriculum):** Metal Qwen baseline shipped; remaining
checklist items were satisfied or superseded by M1–M8 (bench/profile,
batched schedule, half/int8, 4B capstone). See stage M0 doc.

**M1 gate:** `zynfer qwen-bench` split report on CPU + Apple.

**M2 gate:** `zynfer qwen-profile` names top-3 costs for one decode token
(+ measured STREAM bandwidth / roofline).

**M3 gate:** batched Qwen path improves decode/TTFT vs M0 baseline;
fusion ledger in `bench/results/stageM3-dev-laptop.md`.

**M7 gate (closed):** final Qwen-scale Core ML/ANE reject ledger
(`bench/results/apple-ane-qwen-dev-laptop.md`); `ZYNFER_FORCE_COREML`
exits 2; tutorial 21. Candidates (a)/(b) **skipped** (no Qwen Core ML
graph; handoff tax vs resident Metal; unified memory ≠ free splice) —
not “raced and lost.” Toy `coreml-smoke` + one `xctrace` do not clear
retain.

**M8 gate (Apple-complete — closed):** registered quantized Qwen3-4B;
final Apple benchmark matrix with ledgers
(`bench/results/apple-capstone-dev-laptop.md`); `zynfer stageM8`;
tutorial 22. Optional llama.cpp-Metal / MLX comparison on the same
machine. Does **not** include Phase R (AMD) or Phase S (serving).
Milestone **J** = larger **registered Metal** checkpoint, not Core ML.

---

## Phase R — AMD RDNA 4 (when hardware available)

| Stage | Content | Old # |
| --- | --- | --- |
| R0 | HIP env + alloc/copy/streams | 0–1 |
| R1 | First kernel + module loader | 2 |
| R2 | Port op set vs CPU oracle | 3–8 |
| R3 | Qwen block → full forward + golden | 9, 11 |
| R4 | Prefill/decode split + ROCm profiling | 14–15 |
| R5 | Fusion ledger on RDNA 4 | 16 |
| R6 | HIP graphs (measurement-gated) | 17 |
| R7 | Quantization for `gfx1201` | 18 |
| R8 | `gfx1201` tuning notebook | 19 |
| R9 | Static decode plan (discrete VRAM) | 20 |
| R10 | R9700 validation + large checkpoints | 25 |

Until hardware lands: keep backend seam clean, compile-only stubs, **no
fabricated AMD numbers**.

---

## Phase S — Serving (after M8, overlaps R)

| Stage | Title | Old # |
| --- | --- | --- |
| S1 | Batching and scheduling | 21 |
| S2 | Prefix reuse / cache management | 22 |
| S3 | Speculative / MTP | 23 |
| S4 | HTTP server | 24 |

---

## Legacy stage index (traceability)

| Old | Maps to |
| --- | --- |
| 0–9 | Foundation + tiny block (partial HIP) |
| 10–13 | **Done** (CPU Qwen path) |
| 14–16 | M1–M3 |
| 17–19 | R6–R8 |
| 18, 20 | M5–M6 (Apple); R7, R9 (AMD) |
| 21–24 | S1–S4 |
| 25 | M8 (Apple 4B) + R10 (large AMD) |

See `docs/apple-backend.md` for Apple tiny-block deferred map and
Instruments recipes.
