# Tutorial 00 — Development environment

## 1. What we are building

A reproducible way to ask a machine:

- which OS and kernel it is running;
- which Zig compiler will build zynfer;
- whether ROCm/HIP is installed;
- whether an AMD GPU is visible;
- what that GPU calls itself, including its LLVM target.

The Stage 0 program is `zynfer`. On a development laptop it prints a
honest "HIP not linked" report. On the Linux GPU host it should identify
the Radeon AI PRO R9700 as `gfx1201`.

## 2. Why it exists

Inference performance work is meaningless if we cannot name the machine.
Kernel timings change with driver, ROCm, clocks, and which GPU we
accidentally selected. Environment drift is a common way to invent a
speedup that was actually a different host.

Stage 0 also forces the repository to exist as a real Zig project before
anyone writes a transformer block.

## 3. The underlying concept

A GPU inference stack is several layers that people casually call "ROCm":

```text
application (zynfer, Zig)
        │
        ▼
HIP runtime API          user-space, process-local
        │
        ▼
ROCm user-space stack    hip runtime, thunk, queues
        │
        ▼
amdgpu kernel driver     Linux kernel module, firmware
        │
        ▼
GPU hardware             RDNA 4, gfx1201
```

These are not synonyms.

- **ROCm** is AMD's user-space compute platform: runtimes, compilers,
  libraries, profilers.
- **HIP** is the portable GPU programming API zynfer calls. The C
  runtime API is enough to enumerate devices. Kernel language HIP comes
  later.
- **AMDGPU** is LLVM's backend name for AMD GPUs. `gfx1201` is an AMDGPU
  processor ID.
- **amdgpu** (lowercase, kernel) is the Linux driver that binds the
  device and exposes compute to user space.

Host code runs on the CPU. Device code runs on the GPU. Stage 0 is
entirely host code. It only *asks questions* of the runtime.

Reproducibility matters because later we will claim "this kernel is
faster." That sentence is false unless the before and after numbers
share a recorded environment.

## 4. How the hardware sees it

From the GPU's point of view, Stage 0 does almost nothing. The HIP
runtime opens a connection to the driver, queries PCI and firmware
properties, and returns a C struct. No compute units execute a kernel.
No VRAM is allocated for model weights.

That is still a real hardware interaction: user space talking to the
driver about a discrete GPU on PCIe. If this fails, nothing else in
zynfer can work.

## 5. How the math maps to code

There is no model math yet. The only "formulas" are administrative:

```text
HIP packed version = major * 10_000_000 + minor * 100_000 + patch

VRAM GiB = totalGlobalMem / 1024³
```

Device 0 is the default until we have a reason to select another.

## 6. How the Zig implementation works

`zig build` produces `zig-out/bin/zynfer`.

- `src/env_report.zig` collects OS, Zig, compiler paths, ROCm files,
  amdgpu sysfs, and a best-effort `rocminfo` parse.
- `src/hip.zig` is a small Zig API over optional HIP.
- `src/hip_probe.c` is compiled **only** when a ROCm prefix exists. Zig
  never includes `hip_runtime_api.h`, so `hipDeviceProp_t` layout changes
  cannot silently break the host.
- `src/device.zig` stores a device index and the copied properties.
- `src/main.zig` prints `env`, `gpu`, or `bench`.

HIP detection (in `build.zig`):

1. `-Dhip-path=`
2. `HIP_PATH`
3. `ROCM_PATH`
4. `/opt/rocm`
5. `/usr`

`-Dhip=auto` (default) links HIP when that probe succeeds. `-Dhip=off`
builds a host-only binary. `-Dhip=on` fails the build if HIP is missing,
which is what we want on the GPU host.

The project pins Zig **0.16.0** via `.tool-versions`.

## 7. How the GPU implementation works

There is no GPU kernel. The only device-side-adjacent code is HIP's
`hipGetDeviceCount`, `hipGetDeviceProperties`, and version queries,
called from C and wrapped in Zig.

If those symbols cannot be linked, the binary is still useful: it tells
you the host is not the GPU machine.

## 8. How correctness is checked

```bash
zig build test
```

Tests cover:

- packed HIP version decoding;
- C-string slicing of `gfx1201`;
- target GPU name/arch matching;
- `rocminfo` marketing-name parsing;
- `error.HipUnavailable` on a host-only build;
- on Linux with HIP linked: at least one device, non-empty name, arch,
  and non-zero VRAM.

The human check on the GPU host is:

```bash
./scripts/check-env.sh
zig build run -- gpu
```

The report must show a marketing name consistent with the R9700 and an
ISA/arch of `gfx1201`.

## 9. How performance is measured

```bash
zig build bench
```

This warms up, then times repeated device-count + property queries. The
number is CPU/runtime overhead, not teraflops. We record it so Stage 1
allocation latency has a neighboring measurement.

On a laptop without HIP the benchmark command still exits 0 and says so.

## 10. What surprised us

- A current macOS development machine can compile the entire Stage 0
  host path. That is useful, and it is also a trap: success on the laptop
  is not success on the GPU.
- HIP device properties are ABI-unstable enough that copying them into
  our own struct is cheaper than pretending Zig can `#include` HIP
  forever.
- `rocminfo` and HIP may disagree slightly on naming. The LLVM target
  string is the field we trust.

## 11. Exercises

1. Run `./scripts/check-env.sh` on your laptop and on the GPU host. Diff
   the two reports.
2. Build with `zig build -Dhip=off` even on the GPU host and confirm
   `zynfer gpu` refuses HIP rather than crashing.
3. Read `hipGetDeviceProperties` in the HIP runtime API docs and list
   five fields we do **not** copy yet. Which of them will matter for
   occupancy later?
4. Find `/opt/rocm/.info/version` (or equivalent) and pin that string in
   `docs/hardware-r9700.md`.

## 12. Further experiments

- Compare `rocminfo` ISA name with `hipDeviceProp_t.gcnArchName`. If they
  ever diverge, treat that as an incident, not a curiosity.
- Record the HIP query latency from `zig build bench` under idle vs load.
- Once Stage 1 exists, compare this query latency with `hipMalloc`
  latency. Allocation should be much more expensive; if it is not, the
  benchmark is wrong.

## Stage 0 report

```text
Correctness:
  Host-only Zig binary builds and tests on macOS. HIP path is compiled
  only when ROCm headers are present. GPU identity is not yet verified
  on the R9700 because this session ran on a development laptop.

Performance:
  No GPU kernel. Bench command exists; HIP query latency is unmeasured
  until the Linux host is used.

Known limitations:
  Developer laptop has no ROCm and no AMD GPU. macOS is not an execution
  target. rocminfo parsing is heuristic. Wave size and CU count are
  planning numbers until HIP reports them.

Technical debt:
  C adapter must be compiled with Zig's clang against HIP headers; if a
  future ROCm release makes hip_runtime_api.h C++-only we may need hipcc.
  Fingerprint / ROCm versions are not pinned to a GPU-host measurement.

What we learned:
  The first product of an inference engine is an environment report.
  Optional HIP linking is what makes laptop development possible without
  lying about GPU presence.

Next bottleneck:
  Actually seeing the R9700 via HIP, then Stage 1 allocation and copies.
```

## References

- [AMD ROCm documentation](https://rocm.docs.amd.com/)
- [HIP runtime API](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [Zig language documentation](https://ziglang.org/documentation/)
- [LLVM AMDGPU usage](https://llvm.org/docs/AMDGPUUsage.html)
