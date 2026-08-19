# Benchmarks

Zynfer records measurements from the first stage so later "faster"
claims have a baseline.

## Required metadata

Every recorded result should include:

```text
date
host
OS / kernel
GPU name
GPU architecture (LLVM target)
ROCm version
HIP runtime version
Zig version
command
warmup
iterations
what was measured
```

Do not compare engines casually. Same model, precision, prompt, generated
length, GPU, concurrency, and metric definition are required before a
comparison is meaningful.

## Stage 0 command

```bash
zig build bench
# equivalent:
zig build run -- bench
```

This times `hipGetDeviceCount` plus `hipGetDeviceProperties` for device 0.
It is a runtime-query microbenchmark, not a kernel benchmark.

On a machine without HIP the command still runs and reports that GPU
timing is unavailable.

## Where results live

Committed templates:

- `bench/results/stage0-dev-laptop.md`

Local captures that should not be committed:

- `bench/results/local/`

## Later metrics

Prefill and decode are reported separately once the engine exists.

Interactive: time to first token, inter-token latency, single-request
decode tok/s.

Throughput: aggregate committed tok/s, concurrency scaling.

Memory: model VRAM, KV VRAM, scratch VRAM, peak VRAM.

Kernel: duration, effective bandwidth, occupancy, register/LDS pressure
where the profiler provides them.
