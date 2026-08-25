# Tutorial — What makes Apple inference fast (Stage M8)

A retrospective of Phase M on Apple Silicon: which changes actually moved
tok/s and TTFT, in what order, and what the roofline still says is left.

## In plain English

We did not make Apple inference fast by chasing the Neural Engine. We made
it fast by **keeping work on Metal**, **cutting bytes and launches**, and
**stopping allocations on the hot path** — then proving it on a real-size
model (Qwen3-4B).

## Order of impact (what mattered)

1. **Resident weights + one wait structure (M0–M3 / M6)**  
   Encode once per forward (batched schedule), keep weights on GPU, fuse
   the loudest pairs (`add_rmsnorm`, SiLU×mul). Waiting per tiny op killed
   tok/s long before FLOPs did.

2. **Half precision then int8 (M4–M5)**  
   bf16 cut traffic; int8 weights with fused dequant into GEMV/GEMM cut it
   again. Quality stayed within a documented proxy. 4-bit stays deferred
   until int8 stops winning.

3. **Static decode plan (M6)**  
   Preallocate KV to max seq, fixed scratch, session sample buffers, host
   mirror skip on batched Metal. Zero heap on decode — asserted, not hoped.

4. **What did *not* make the cut**  
   - ICB / encode-once replay — REJECT (KV/`q_len` change each token)  
   - Core ML / ANE — REJECT at Qwen scale (M7): no public ANE ISA; handoff
     tax vs resident Metal; retain bar unmet  
   - SME kernels — REJECT (Stage 7)

## Roofline reminder

Decode on Apple Silicon for these models is largely **memory-bound**:
bytes moved per token (weights + KV) dominate. That is why int8 + short
KV budgets beat clever kernels that still stream f32 weights.

Estimate helpers live on `Arch` (`estimateDecodeBytesPerTokenQ8`) and
`zynfer mem-report` / `stageM8` KV tables.

## Capstone model (M8)

Registered **Qwen3-4B** (hidden 2560, 36 layers, GQA 32/8) with an int8
`.zynfer` sized for unified memory. Chat defaults prefer the registered
int8 artifact when present:

```bash
python3 tools/setup_qwen.py --model 4b --quantize --skip-golden
ZYNFER_QWEN_METAL=int8 ./zig-out/bin/zynfer chat models/qwen3-4b-int8.zynfer "…"
./zig-out/bin/zynfer stageM8
```

## External comparisons

llama.cpp-Metal and MLX are allowed on the **same Mac**, same prompt /
length / sampling / definitions. Losses get hypotheses in
`bench/results/apple-capstone-dev-laptop.md` — not excuses without numbers.

### Why zynfer can feel much slower than Ollama

Ollama is a product (usually llama.cpp Metal + aggressive GGUF quants).
Zynfer’s M8 4B int8 path still expands weights to a **host f32 twin**
before Metal residency, uses a curriculum int8 scheme (not Q4_K), and
lacks years of kernel polish. On this laptop that showed up as
~3.6 tok/s decode vs Ollama often many× faster for chat — see the
ledger section **“Why zynfer feels much slower than Ollama.”** That gap
is expected baseline debt, not a silent failure of the Apple-complete gate.

## What “Apple-complete” means

Backend 1 is done when a user can run a **registered quantized** model at
competitive, reproducible, documented speed with a matrix ledger. Serving
(Phase S) and AMD (Phase R) are separate.

## Related

- Stage: [`docs/stages/M8-apple-capstone.md`](../stages/M8-apple-capstone.md)
- Ledger: [`bench/results/apple-capstone-dev-laptop.md`](../../bench/results/apple-capstone-dev-laptop.md)
- ANE close: tutorial 21
