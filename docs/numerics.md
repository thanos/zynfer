# Numerics

Stage 0 does not compute model numerics. This file exists so later
stages have a single place to pin dtypes, accumulation rules, and
tolerances.

## Planned defaults

| Quantity | Initial choice | Notes |
| --- | --- | --- |
| Weights / activations | BF16 or FP16 as appropriate | Confirm against the Qwen3-0.6B checkpoint |
| Reductions / softmax acc | FP32 | Stability before speed |
| RMSNorm acc | FP32 | Same reason |
| Reference oracle | CPU Zig + PyTorch fixtures | Oracle is never a runtime dependency |

## Tolerance policy

Every numerical test must state an explicit tolerance. "Looks close" is
not a criterion.

Until the first kernel lands, there is nothing to tolerate.

## Questions this file will answer

- FP32 vs BF16 vs FP16 for each tensor class
- Why some ops accumulate wider
- What "numerical stability" means for softmax and RMSNorm
- What quantization changes, once we have a floating-point baseline
