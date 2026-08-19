/* HIP runtime probe used by the Zig host.
 *
 * This file is compiled only when ROCm/HIP headers and libamdhip64 are
 * available. It exists so Zig never depends on hipDeviceProp_t layout.
 *
 * Input shape:  one HIP device index
 * Output shape: ZynferGpuInfo
 * DType:        n/a
 * Workgroup:    n/a (host runtime API)
 * Wave assumptions: n/a
 * LDS usage:    n/a
 * Expected access pattern: host-side property query
 * Synchronization: none
 * Known target: gfx1201 / Radeon AI PRO R9700, any ROCm HIP device
 * Reason for specialization: isolate HIP ABI churn from the Zig runtime
 */

#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif

#include "hip_probe.h"

#include <hip/hip_runtime_api.h>
#include <string.h>

static void copy_cstr(char *dst, size_t dst_len, const char *src) {
    if (dst_len == 0) {
        return;
    }
    memset(dst, 0, dst_len);
    if (src == NULL) {
        return;
    }
    strncpy(dst, src, dst_len - 1);
}

int zynfer_hip_runtime_version(int *out) {
    if (out == NULL) {
        return (int)hipErrorInvalidValue;
    }
    return (int)hipRuntimeGetVersion(out);
}

int zynfer_hip_driver_version(int *out) {
    if (out == NULL) {
        return (int)hipErrorInvalidValue;
    }
    return (int)hipDriverGetVersion(out);
}

int zynfer_hip_device_count(int *out) {
    if (out == NULL) {
        return (int)hipErrorInvalidValue;
    }
    return (int)hipGetDeviceCount(out);
}

int zynfer_hip_describe_device(int device, ZynferGpuInfo *out) {
    hipDeviceProp_t prop;
    hipError_t err;

    if (out == NULL) {
        return (int)hipErrorInvalidValue;
    }

    memset(&prop, 0, sizeof(prop));
    err = hipGetDeviceProperties(&prop, device);
    if (err != hipSuccess) {
        return (int)err;
    }

    memset(out, 0, sizeof(*out));
    out->index = device;
    copy_cstr(out->name, sizeof(out->name), prop.name);
    copy_cstr(out->gcn_arch, sizeof(out->gcn_arch), prop.gcnArchName);
    out->total_mem_bytes = (uint64_t)prop.totalGlobalMem;
    out->multiprocessor_count = prop.multiProcessorCount;
    out->warp_size = prop.warpSize;
    out->max_threads_per_block = prop.maxThreadsPerBlock;
    out->clock_rate_khz = prop.clockRate;
    out->memory_clock_rate_khz = prop.memoryClockRate;
    out->memory_bus_width_bits = prop.memoryBusWidth;
    out->l2_cache_bytes = prop.l2CacheSize;
    out->shared_mem_per_block = (int32_t)prop.sharedMemPerBlock;
    out->regs_per_block = prop.regsPerBlock;
    out->pci_domain = prop.pciDomainID;
    out->pci_bus = prop.pciBusID;
    out->pci_device = prop.pciDeviceID;
    out->integrated = prop.integrated ? 1 : 0;
    out->can_map_host_memory = prop.canMapHostMemory ? 1 : 0;
    out->concurrent_kernels = prop.concurrentKernels ? 1 : 0;
    return (int)hipSuccess;
}

const char *zynfer_hip_error_string(int err) {
    return hipGetErrorString((hipError_t)err);
}
