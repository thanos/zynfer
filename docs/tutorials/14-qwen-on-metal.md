# Tutorial — Qwen on Metal (Stage M0)

Stage 13 proved KV-cached **CPU** generation. Stage **M0** runs the
**transformer blocks** on Metal while keeping the CPU oracle for embed,
final norm, LM head, and golden checks.

Reference: [`docs/stages/M0-metal-qwen-forward.md`](../stages/M0-metal-qwen-forward.md).

## 1. Why a new stage?

Apple Stages 0–8 optimized a **tiny synthetic block** (`hidden=8`). Qwen3-0.6B
uses `hidden=1024`, GQA 16/8, QK-norm, and vocab 151936. The LM-head GEMV
alone is the largest matvec the Metal stack has seen.

M0 is **parity first**: same tokens as CPU, then M1–M3 make it fast.

## 2. Backend selection

```bash
# CPU oracle (default)
./zig-out/bin/zynfer forward-golden zig-out/stage11-mini.zynfer --tokens 2,3

# Metal blocks (embed/norm/head still CPU in M0)
./zig-out/bin/zynfer forward-golden zig-out/stage11-mini.zynfer \
  --tokens 2,3 --backend apple

ZYNFER_BACKEND=apple ./zig-out/bin/zynfer chat "Hello" --max-tokens 4 --no-stream
```

Invalid backend names **fail loudly** — no silent fallback.

## 3. What runs where (M0)

```text
CPU:  token embed → [Metal: 28× Qwen block] → CPU: final norm → CPU: LM head
```

Inside each block on Metal (per-op path):

RMSNorm → QKV matmuls → **QK-norm** → RoPE → KV append → attention →
O-proj → residual → MLP → residual.

## 4. Attention context cap

Metal attention supports `kv_len ≤ 2048`:

- `kv_len ≤ 256`: thread-local scores (fast)
- `257 … 2048`: device scores buffer (`attention_f32_buf`)

Full 40960 context needs M3 tiling — do not assume Metal chat works at
max context yet.

## 5. Verify parity

```bash
zig build test -Dhip=off          # includes mini Metal vs CPU test
./zig-out/bin/zynfer stageM0
```

## 6. What comes next

| Stage | Focus | Status |
| --- | --- | --- |
| M1 | Prefill vs decode metrics on Qwen | **done** — `qwen-bench` |
| M2 | Profile one decode token + roofline | **done** — `qwen-profile` |
| M3 | 28-layer one-CB schedule + fusion ledger | **done** — `qwen_schedule` |
| M4 | fp16/bf16 Metal path | next |

M0 gate items still open (full-model parity, Metal TTFT ledger, …) —
see [`docs/stages/M0-metal-qwen-forward.md`](../stages/M0-metal-qwen-forward.md).
