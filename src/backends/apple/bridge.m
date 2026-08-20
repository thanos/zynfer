#import "bridge.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdlib.h>
#include <string.h>

struct ZynferMtlDevice {
    __strong id<MTLDevice> device;
    __strong id<MTLCommandQueue> queue;
    __strong id<MTLLibrary> library;
    __strong NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipelines;
    __strong NSString *last_error;
    char last_error_utf8[2048];
};

struct ZynferMtlBuffer {
    __strong id<MTLBuffer> buffer;
};

static void set_error(ZynferMtlDevice *dev, NSString *msg) {
    if (dev == NULL) {
        return;
    }
    NSString *copy = msg ? [msg copy] : @"unknown Metal error";
    dev->last_error = copy;
    const char *utf8 = copy.UTF8String ?: "unknown Metal error";
    strncpy(dev->last_error_utf8, utf8, sizeof(dev->last_error_utf8) - 1);
    dev->last_error_utf8[sizeof(dev->last_error_utf8) - 1] = 0;
}

int zynfer_mtl_device_create(ZynferMtlDevice **out) {
    if (out == NULL) {
        return ZYNFER_MTL_INVALID;
    }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        return ZYNFER_MTL_NO_DEVICE;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
        return ZYNFER_MTL_ENCODE;
    }
    ZynferMtlDevice *dev = (ZynferMtlDevice *)calloc(1, sizeof(ZynferMtlDevice));
    if (dev == NULL) {
        return ZYNFER_MTL_OOM;
    }
    dev->device = device;
    dev->queue = queue;
    dev->pipelines = [NSMutableDictionary dictionary];
    dev->last_error = nil;
    dev->last_error_utf8[0] = 0;
    *out = dev;
    return ZYNFER_MTL_OK;
}

void zynfer_mtl_device_destroy(ZynferMtlDevice *dev) {
    if (dev == NULL) {
        return;
    }
    dev->pipelines = nil;
    dev->library = nil;
    dev->queue = nil;
    dev->device = nil;
    dev->last_error = nil;
    free(dev);
}

int zynfer_mtl_caps(const ZynferMtlDevice *dev, ZynferMtlCaps *out) {
    if (dev == NULL || out == NULL) {
        return ZYNFER_MTL_INVALID;
    }
    memset(out, 0, sizeof(*out));
    NSString *name = dev->device.name ?: @"";
    strncpy(out->device_name, name.UTF8String, sizeof(out->device_name) - 1);
    out->recommended_working_set_bytes = dev->device.recommendedMaxWorkingSetSize;
    out->max_buffer_bytes = dev->device.maxBufferLength;
    MTLSize max_tg = dev->device.maxThreadsPerThreadgroup;
    out->max_threads_per_threadgroup = (uint32_t)max_tg.width;
    out->unified_memory = dev->device.hasUnifiedMemory ? 1 : 0;
    if (@available(macOS 10.15, *)) {
        out->gpu_family_apple7 = [dev->device supportsFamily:MTLGPUFamilyApple7] ? 1 : 0;
        out->gpu_family_apple8 = [dev->device supportsFamily:MTLGPUFamilyApple8] ? 1 : 0;
    }
    if (@available(macOS 13.0, *)) {
        out->gpu_family_apple9 = [dev->device supportsFamily:MTLGPUFamilyApple9] ? 1 : 0;
    }
    /* SIMD-group matrix multiply is available on Apple7+ GPUs. This flag
     * means the hardware can run such kernels, not that zynfer currently does. */
    out->simdgroup_matrix_available = out->gpu_family_apple7;
    return ZYNFER_MTL_OK;
}

int zynfer_mtl_compile_library(ZynferMtlDevice *dev, const char *source) {
    if (dev == NULL || source == NULL) {
        return ZYNFER_MTL_INVALID;
    }
    NSError *error = nil;
    NSString *src = [NSString stringWithUTF8String:source];
    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> lib = [dev->device newLibraryWithSource:src options:opts error:&error];
    if (lib == nil) {
        set_error(dev, error.localizedDescription ?: @"library compile failed");
        return ZYNFER_MTL_COMPILE;
    }
    dev->library = lib;
    [dev->pipelines removeAllObjects];
    return ZYNFER_MTL_OK;
}

const char *zynfer_mtl_last_error(const ZynferMtlDevice *dev) {
    if (dev == NULL || dev->last_error_utf8[0] == 0) {
        return "";
    }
    return dev->last_error_utf8;
}

int zynfer_mtl_buffer_create(ZynferMtlDevice *dev, size_t bytes, ZynferMtlBuffer **out) {
    if (dev == NULL || out == NULL || bytes == 0) {
        return ZYNFER_MTL_INVALID;
    }
    id<MTLBuffer> buffer = [dev->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        set_error(dev, @"shared buffer allocation failed");
        return ZYNFER_MTL_OOM;
    }
    ZynferMtlBuffer *buf = (ZynferMtlBuffer *)calloc(1, sizeof(ZynferMtlBuffer));
    if (buf == NULL) {
        return ZYNFER_MTL_OOM;
    }
    buf->buffer = buffer;
    *out = buf;
    return ZYNFER_MTL_OK;
}

void zynfer_mtl_buffer_destroy(ZynferMtlBuffer *buf) {
    if (buf == NULL) {
        return;
    }
    buf->buffer = nil;
    free(buf);
}

void *zynfer_mtl_buffer_contents(ZynferMtlBuffer *buf) {
    if (buf == NULL || buf->buffer == nil) {
        return NULL;
    }
    return buf->buffer.contents;
}

size_t zynfer_mtl_buffer_length(const ZynferMtlBuffer *buf) {
    if (buf == NULL || buf->buffer == nil) {
        return 0;
    }
    return buf->buffer.length;
}

static id<MTLComputePipelineState> pipeline_for(ZynferMtlDevice *dev, const char *kernel) {
    NSString *name = [NSString stringWithUTF8String:kernel];
    id<MTLComputePipelineState> pso = dev->pipelines[name];
    if (pso != nil) {
        return pso;
    }
    if (dev->library == nil) {
        set_error(dev, @"no Metal library compiled");
        return nil;
    }
    id<MTLFunction> fn = [dev->library newFunctionWithName:name];
    if (fn == nil) {
        set_error(dev, [NSString stringWithFormat:@"missing kernel %s", kernel]);
        return nil;
    }
    NSError *error = nil;
    pso = [dev->device newComputePipelineStateWithFunction:fn error:&error];
    if (pso == nil) {
        set_error(dev, error.localizedDescription ?: @"pipeline compile failed");
        return nil;
    }
    dev->pipelines[name] = pso;
    return pso;
}

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
    uint32_t params_len)
{
    if (dev == NULL || kernel == NULL || grid_x == 0 || tg_x == 0) {
        return ZYNFER_MTL_INVALID;
    }
    id<MTLComputePipelineState> pso = pipeline_for(dev, kernel);
    if (pso == nil) {
        return ZYNFER_MTL_PIPELINE;
    }
    id<MTLCommandBuffer> cb = [dev->queue commandBuffer];
    if (cb == nil) {
        set_error(dev, @"command buffer create failed");
        return ZYNFER_MTL_ENCODE;
    }
    cb.label = [NSString stringWithUTF8String:kernel];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (enc == nil) {
        set_error(dev, @"encoder create failed");
        return ZYNFER_MTL_ENCODE;
    }
    enc.label = cb.label;
    [enc setComputePipelineState:pso];
    for (uint32_t i = 0; i < nbufs; i++) {
        if (bufs[i] == NULL || bufs[i]->buffer == nil) {
            [enc endEncoding];
            return ZYNFER_MTL_INVALID;
        }
        [enc setBuffer:bufs[i]->buffer offset:0 atIndex:i];
    }
    if (params != NULL && params_len > 0) {
        [enc setBytes:params length:params_len atIndex:nbufs];
    }
    MTLSize grid = MTLSizeMake(grid_x, grid_y == 0 ? 1 : grid_y, grid_z == 0 ? 1 : grid_z);
    MTLSize tg = MTLSizeMake(tg_x, tg_y == 0 ? 1 : tg_y, tg_z == 0 ? 1 : tg_z);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error != nil) {
        set_error(dev, cb.error.localizedDescription);
        return ZYNFER_MTL_ENCODE;
    }
    return ZYNFER_MTL_OK;
}
