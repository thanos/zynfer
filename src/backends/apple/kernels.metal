// Baseline Metal Shading Language kernels for zynfer's Apple backend.
//
// These are correctness-first f32 kernels. They do not use simdgroup_matrix.
// Tile sizes are documented next to each dispatch in the Zig caller.
//
// DType: f32
// Layout: dense row-major
// Known target: Apple GPU via MTLGPUFamilyApple7+ (M1 and later)
// Reason for specialization: none yet; this is the fallback path.

#include <metal_stdlib>
using namespace metal;

kernel void add_f32(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        c[gid] = a[gid] + b[gid];
    }
}

kernel void mul_f32(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        c[gid] = a[gid] * b[gid];
    }
}

kernel void silu_mul_f32(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        float x = gate[gid];
        float silu = x * (1.0f / (1.0f + exp(-x)));
        out[gid] = silu * up[gid];
    }
}

struct RmsNormParams {
    uint rows;
    uint cols;
    float eps;
};

kernel void rmsnorm_f32(
    device const float *x [[buffer(0)]],
    device const float *w [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant RmsNormParams &p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid2 [[thread_position_in_threadgroup]],
    uint2 tgs2 [[threads_per_threadgroup]])
{
    uint row = gid.y;
    uint lid = lid2.x;
    uint tgs = tgs2.x;
    if (row >= p.rows) {
        return;
    }
    threadgroup float scratch[256];
    device const float *row_x = x + row * p.cols;
    device float *row_y = y + row * p.cols;

    float acc = 0.0f;
    for (uint i = lid; i < p.cols; i += tgs) {
        float v = row_x[i];
        acc += v * v;
    }
    scratch[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgs / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(scratch[0] / float(p.cols) + p.eps);
    for (uint i = lid; i < p.cols; i += tgs) {
        row_y[i] = w[i] * row_x[i] * inv;
    }
}

struct SoftmaxParams {
    uint rows;
    uint cols;
};

kernel void softmax_f32(
    device const float *x [[buffer(0)]],
    device float *y [[buffer(1)]],
    constant SoftmaxParams &p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid2 [[thread_position_in_threadgroup]],
    uint2 tgs2 [[threads_per_threadgroup]])
{
    uint row = gid.y;
    uint lid = lid2.x;
    uint tgs = tgs2.x;
    if (row >= p.rows) {
        return;
    }
    threadgroup float scratch[256];
    device const float *row_x = x + row * p.cols;
    device float *row_y = y + row * p.cols;

    float max_v = -INFINITY;
    for (uint i = lid; i < p.cols; i += tgs) {
        max_v = max(max_v, row_x[i]);
    }
    scratch[lid] = max_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgs / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] = max(scratch[lid], scratch[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    max_v = scratch[0];

    float sum = 0.0f;
    for (uint i = lid; i < p.cols; i += tgs) {
        float e = exp(row_x[i] - max_v);
        row_y[i] = e;
        sum += e;
    }
    scratch[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgs / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0f / scratch[0];
    for (uint i = lid; i < p.cols; i += tgs) {
        row_y[i] *= inv;
    }
}

struct MatmulParams {
    uint m;
    uint n;
    uint k;
};

kernel void matmul_f32(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant MatmulParams &p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    if (row >= p.m || col >= p.n) {
        return;
    }
    float acc = 0.0f;
    for (uint t = 0; t < p.k; t++) {
        acc += a[row * p.k + t] * b[t * p.n + col];
    }
    c[row * p.n + col] = acc;
}

kernel void matvec_f32(
    device const float *a [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant MatmulParams &p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.m) {
        return;
    }
    float acc = 0.0f;
    for (uint t = 0; t < p.k; t++) {
        acc += a[gid * p.k + t] * x[t];
    }
    y[gid] = acc;
}

struct RopeParams {
    uint tokens;
    uint n_heads;
    uint head_dim;
    uint pos0;
    float theta;
};

kernel void rope_f32(
    device float *x [[buffer(0)]],
    constant RopeParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    uint half_d = p.head_dim / 2;
    uint total = p.tokens * p.n_heads * half_d;
    if (gid >= total) {
        return;
    }
    uint pair = gid;
    uint token = pair / (p.n_heads * half_d);
    uint rem = pair % (p.n_heads * half_d);
    uint head = rem / half_d;
    uint i = rem % half_d;
    uint base = (token * p.n_heads + head) * p.head_dim;
    float freq = pow(p.theta, -float(i) / float(half_d));
    float angle = float(p.pos0 + token) * freq;
    float c = cos(angle);
    float s = sin(angle);
    float a = x[base + i];
    float b = x[base + i + half_d];
    x[base + i] = a * c - b * s;
    x[base + i + half_d] = a * s + b * c;
}
