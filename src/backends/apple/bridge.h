#ifndef ZYNFER_APPLE_BRIDGE_H
#define ZYNFER_APPLE_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ZynferMtlDevice ZynferMtlDevice;
typedef struct ZynferMtlBuffer ZynferMtlBuffer;

enum {
    ZYNFER_MTL_OK = 0,
    ZYNFER_MTL_NO_DEVICE = 1,
    ZYNFER_MTL_COMPILE = 2,
    ZYNFER_MTL_PIPELINE = 3,
    ZYNFER_MTL_OOM = 4,
    ZYNFER_MTL_ENCODE = 5,
    ZYNFER_MTL_UNSUPPORTED = 6,
    ZYNFER_MTL_INVALID = 7,
};

typedef struct ZynferMtlCaps {
    char device_name[256];
    uint64_t recommended_working_set_bytes;
    uint64_t max_buffer_bytes;
    uint32_t max_threads_per_threadgroup;
    int unified_memory;
    int gpu_family_apple7;
    int gpu_family_apple8;
    int gpu_family_apple9;
    int simdgroup_matrix_available;
} ZynferMtlCaps;

/* Ownership: create functions return a +1 object. destroy releases it.
 * Buffers remain valid until destroy. Command buffers wait on the CPU in
 * encode_and_wait; do not free a buffer while a wait is in flight.
 * Threading: the device is not internally synchronized; one owner thread.
 */
int zynfer_mtl_device_create(ZynferMtlDevice **out);
void zynfer_mtl_device_destroy(ZynferMtlDevice *dev);
int zynfer_mtl_caps(const ZynferMtlDevice *dev, ZynferMtlCaps *out);
int zynfer_mtl_compile_library(ZynferMtlDevice *dev, const char *source);
const char *zynfer_mtl_last_error(const ZynferMtlDevice *dev);

int zynfer_mtl_buffer_create(ZynferMtlDevice *dev, size_t bytes, ZynferMtlBuffer **out);
void zynfer_mtl_buffer_destroy(ZynferMtlBuffer *buf);
void *zynfer_mtl_buffer_contents(ZynferMtlBuffer *buf);
size_t zynfer_mtl_buffer_length(const ZynferMtlBuffer *buf);

int zynfer_mtl_encode_and_wait(
    ZynferMtlDevice *dev,
    const char *kernel,
    uint32_t grid_x,
    uint32_t grid_y,
    uint32_t grid_z,
    uint32_t tg_x,
    uint32_t tg_y,
    uint32_t tg_z,
    ZynferMtlBuffer **bufs,
    uint32_t nbufs,
    const void *params,
    uint32_t params_len,
    uint32_t threadgroup_mem_bytes,
    int dispatch_threadgroups);

/* Stage 6: encode many dispatches into one command buffer, one wait.
 * begin → encode* → commit_and_wait. abort cancels without waiting.
 * Nested begin, or encode outside a batch, returns ZYNFER_MTL_INVALID.
 */
int zynfer_mtl_batch_begin(ZynferMtlDevice *dev);
int zynfer_mtl_batch_encode(
    ZynferMtlDevice *dev,
    const char *kernel,
    uint32_t grid_x,
    uint32_t grid_y,
    uint32_t grid_z,
    uint32_t tg_x,
    uint32_t tg_y,
    uint32_t tg_z,
    ZynferMtlBuffer **bufs,
    uint32_t nbufs,
    const void *params,
    uint32_t params_len,
    uint32_t threadgroup_mem_bytes,
    int dispatch_threadgroups);
int zynfer_mtl_batch_commit_and_wait(ZynferMtlDevice *dev);
void zynfer_mtl_batch_abort(ZynferMtlDevice *dev);

#ifdef __cplusplus
}
#endif

#endif
