# Hardware notebook: AMD Radeon AI PRO R9700

Living notes about the first target GPU. Values here are planning
assumptions until the Linux host's `zig build run -- gpu` report
replaces them.

## Identity

| Field | Planning value | Verified |
| --- | --- | --- |
| Product | AMD Radeon AI PRO R9700 | no |
| Architecture | RDNA 4 | no |
| LLVM / AMDGPU target | `gfx1201` | no |
| Compute units | 64 | no |
| VRAM | 32 GiB GDDR6 | no |
| Memory bandwidth | 640 GB/s | no |
| Wave size | 32 (RDNA family typical; confirm) | no |

The LLVM target name is the one that matters to the compiler. Marketing
names can vary; `gfx1201` should not.

## Software stack to pin

Record the exact versions from the GPU host:

```text
Linux distro + kernel
amdgpu driver / module version
ROCm release
HIP runtime
hipcc / AMD Clang
Zig
```

AMD documents R9700 as a ROCm-supported GPU with `gfx1201`. Re-verify
against current ROCm release notes before depending on a version-specific
API. See [ROCm documentation](https://rocm.docs.amd.com/).

## What Stage 0 needs from the hardware

Visibility. If HIP can enumerate the device, print its name, LLVM
target, and VRAM, the rest of the engine has a place to run.

## What later stages will measure here

- wave size actually reported by `hipDeviceProp_t.warpSize`
- LDS per workgroup
- register file pressure on decode kernels
- whether important decode ops are bandwidth-bound or launch-bound
- matrix instruction usefulness on Qwen3-0.6B shapes
- FP8 / INT8 / packed low-bit support that is real on this stack, not
  copied from NVIDIA naming

## Magic numbers

None yet. When a kernel later uses a tile such as 64×128 or a workgroup
of a particular size, the reason belongs in this notebook and in the
kernel header comment.
