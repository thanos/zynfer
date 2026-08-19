# Stage 0 result — development laptop (no AMD GPU)

Recorded on the macOS development host used to bootstrap the Zig
repository. This is not a GPU performance result. It exists so later
stages have a place to put reproducible numbers.

```text
date:              2026-08-18
host:              MacBook-Pro-2.local
OS:                Darwin 25.5.0
machine:           arm64 (compile target aarch64-macos)
GPU:               none (HIP not linked)
Zig:               0.16.0
clang:             Apple clang 17.0.0
ROCm:              not installed
command:           zig build test && zig build run -- bench
tests:             11 passed, 1 skipped (HIP device enumeration)
measured result:   HIP is not linked; enumeration latency was not measured
```

On the Linux GPU host, replace this file's GPU section with the output of:

```bash
./scripts/check-env.sh
zig build -Dhip=on
zig build test
zig build run -- bench
```

The GPU host report must identify the Radeon AI PRO R9700 and LLVM target
`gfx1201` before Stage 0 is fully closed.
