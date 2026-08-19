#ifndef ZYNFER_HIP_PROBE_H
#define ZYNFER_HIP_PROBE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Stable ABI between the HIP C runtime and the Zig host.
 *
 * HIP's hipDeviceProp_t layout changes across ROCm releases. This struct is
 * the only device-property layout the Zig runtime is allowed to depend on.
 */
typedef struct ZynferGpuInfo {
    int32_t index;
    char name[256];
    char gcn_arch[256];
    uint64_t total_mem_bytes;
    int32_t multiprocessor_count;
    int32_t warp_size;
    int32_t max_threads_per_block;
    int32_t clock_rate_khz;
    int32_t memory_clock_rate_khz;
    int32_t memory_bus_width_bits;
    int32_t l2_cache_bytes;
    int32_t shared_mem_per_block;
    int32_t regs_per_block;
    int32_t pci_domain;
    int32_t pci_bus;
    int32_t pci_device;
    int32_t integrated;
    int32_t can_map_host_memory;
    int32_t concurrent_kernels;
} ZynferGpuInfo;

int zynfer_hip_runtime_version(int *out);
int zynfer_hip_driver_version(int *out);
int zynfer_hip_device_count(int *out);
int zynfer_hip_describe_device(int device, ZynferGpuInfo *out);
const char *zynfer_hip_error_string(int err);

#ifdef __cplusplus
}
#endif

#endif /* ZYNFER_HIP_PROBE_H */
