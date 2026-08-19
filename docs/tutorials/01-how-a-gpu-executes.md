# Tutorial 01 — How a GPU executes

## 1. What we are building

A mental model of AMD GPU execution that Stage 2 can attach a real
kernel to. Stage 0 does not launch work. This tutorial exists so the
first kernel is not the first time we hear the words *wave*, *workgroup*,
or *LDS*.

## 2. Why it exists

People import CUDA vocabulary onto AMD hardware and then tune the wrong
thing. Zynfer targets RDNA 4 / `gfx1201`. The programming API is HIP,
which *looks* like CUDA, but the machine is not an NVIDIA SM.

If we cannot explain how a kernel is launched, we cannot explain launch
overhead, occupancy, or why decode is often not a compute problem.

## 3. The underlying concept

A **kernel** is a function compiled for the GPU and launched over a
**grid** of **work items**.

HIP/OpenCL vocabulary (what zynfer will use):

| HIP / AMD | Rough CUDA analogue | Meaning |
| --- | --- | --- |
| kernel | kernel | GPU function |
| grid | grid | the whole launch |
| workgroup / block | thread block | items that can share LDS and sync |
| work item / thread | thread | one instance of the kernel |
| wave / wavefront | warp | lockstep group of lanes |
| lane | lane | one work item inside a wave |
| LDS | shared memory | on-chip memory per workgroup |
| global memory | global memory | VRAM (and its caches) |
| VGPR / SGPR | registers | per-lane and per-wave registers |

A **wave** on RDNA is typically 32 lanes. Confirm on the R9700 with
`warpSize` from Stage 0's GPU report; do not hard-code 64 because older
GCN wavefronts were 64.

The compiler and occupancy calculator care about:

- how many VGPRs each wave needs;
- how much LDS the workgroup requested;
- how many waves the compute unit can hold.

**Occupancy** is "how many waves actually fit," not "how busy the ALU
looks." High occupancy can hide memory latency. It is not automatically
higher performance; register-heavy kernels often run fewer waves on
purpose.

**Coalescing:** neighboring lanes should touch neighboring addresses in
global memory. Scattered 64-byte loads from a wave waste bandwidth.

**Arithmetic intensity:** FLOPs per byte moved. Softmax and RMSNorm on
long vectors are usually bandwidth-bound. A large prefill GEMM can be
compute-bound. Decode GEMV-like shapes often sit in an unhappy middle:
not enough reuse, too many launches.

## 4. How the hardware sees it

Very roughly, for one kernel launch:

```text
CPU (zynfer)
  │  HIP runtime: build command packet
  ▼
GPU command processor
  │  dispatch workgroups onto compute units
  ▼
compute unit
  │  execute waves, 32 lanes typical
  ▼
each lane
     runs the same kernel text
     on different data
     with its own VGPRs
```

LDS lives on the compute unit and is shared by the workgroup. Registers
are the closest storage; spilling them to memory is a performance event.
Global memory is VRAM across the memory controllers, cached.

The GPU does not "run the transformer." It runs whatever kernel we
dispatch, thousands of work items at a time, until we dispatch the next
one. Launching is not free: the CPU must talk to the runtime, the
runtime must enqueue work, and we often `hipDeviceSynchronize` because it
is convenient. Synchronization is a correctness tool and a performance
cost.

## 5. How the math maps to code

Vector add is the Stage 2 kernel, not Stage 0, but it is the right
picture:

```text
C[i] = A[i] + B[i]
```

If we launch `N` work items with one item per `i`, the mapping is
trivial. The interesting questions are already visible:

- Is `i` computed from `blockIdx`/`threadIdx` (HIP) correctly for all `N`?
- Do consecutive lanes read consecutive `float`s?
- Did we pay more to launch the kernel than to add the numbers?

For tiny `N`, a benchmark is measuring launch overhead. For huge `N`, it
is measuring memory bandwidth. Saying "the kernel is fast" without
saying which of those you measured is how performance folklore starts.

## 6. How the Zig implementation works

Today Zig only:

1. optionally links the HIP runtime;
2. asks how many devices exist;
3. prints properties including wave size, compute unit count, LDS per
   block, and VRAM.

That is the execution context. The launch API (`hipModuleLoad`,
`hipModuleLaunchKernel`, or HIP's `<<<>>>` compiled with hipcc) is Stage
2.

The important Stage 0 implication: **the host binary and the device ISA
are different programs.** Zig 0.16.0 compiles the host. AMD Clang/HIP
will compile kernels to a code object for `gfx1201`. Zig will load that
object. Native Zig GPU codegen is explicitly not a blocker.

## 7. How the GPU implementation works

There is still no custom kernel. The GPU implementation of "how a GPU
executes" in this stage is: the device firmware and command processor
already know how to dispatch work; we have not given them any.

When we do, a HIP kernel will look conceptually like:

```c
__global__ void saxpy(const float *x, float *y, float a, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
```

Workgroup size (`blockDim`) is a tuning knob we will have to justify.
"256 because CUDA tutorials use 256" is not a justification on RDNA 4.

## 8. How correctness is checked

Stage 0 cannot check kernel results. It can check that we are talking to
the right machine:

- LLVM target string starts with `gfx1201` for the intended GPU;
- VRAM is in the 32 GiB class;
- wave size is whatever HIP reports, recorded in
  `docs/hardware-r9700.md`.

Stage 2 will compare every GPU vector-add output to a CPU loop.

## 9. How performance is measured

Not yet with a kernel profiler. Use Stage 0's `zig build bench` as a
lower bound on "how long it takes to poke HIP at all." Kernel launch
overhead in Stage 2 should be larger than a property query and much
smaller than a cold `hipMalloc`. If that ordering is violated, the
timer is wrong or we are synchronizing accidentally.

Later, ROCprofiler becomes the authority for GPU time vs wall time.

## 10. What surprised us

HIP's CUDA-like surface makes it easy to believe occupancy, wave size,
and shared-memory rules transferred unchanged. They did not. RDNA
execution, cache, and matrix instructions are first-class subjects, not
footnotes to an NVIDIA mental model.

Also: "the GPU is idle" and "our process has a HIP context" can both be
true. Enumerating devices does not mean we are using the machine.

## 11. Exercises

1. From the GPU report, write down wave size, CU count, LDS/block, and
   VRAM. Put them in `docs/hardware-r9700.md` with `Verified: yes`.
2. Estimate how many 32-wide waves fit in one workgroup of 256 threads.
   Then estimate how many such workgroups you would need to cover a
   1,048,576-element vector.
3. Explain in one paragraph why a 16-element vector add cannot show
   memory-bandwidth limits.
4. Contrast "occupancy" with "ALU utilization." Give a case where
   raising occupancy could hurt.

## 12. Further experiments

- Read AMD's HIP programming guide on grid/workgroup mapping and compare
  the terminology table above to the current ROCm docs. Update this
  tutorial if names shifted.
- Once Stage 2 lands, vary workgroup size and plot time vs size. Find
  the region that is launch-bound.
- After the first real kernel, dump compiler stats (VGPR, SGPR, LDS) and
  start the hardware notebook's occupancy section with evidence rather
  than guesses.

## References

- [HIP programming guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [LLVM AMDGPU usage](https://llvm.org/docs/AMDGPUUsage.html)
- [ROCprofiler](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/)
