// Baseline Metal Shading Language kernels for zynfer's Apple backend.
//
// Correctness-first f32 kernels, plus capability-gated Stage 5 paths:
// Stage 5 also: `matmul_f32_simdgroup` / `_x4` (Apple7+), `matvec_q8_f32`,
// `matmul_q8_f32` (int8 weights, per-row or per-tensor scale).
//
// DType: f32 activations; optional int8 weights for q8 kernels
// Layout: dense row-major
// Known target: Apple GPU via MTLGPUFamilyApple7+ (M1 and later)
// Reason for specialization: simdgroup path uses hardware MMA when selected.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
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

// sum_out = x + residual; norm_out = rmsnorm(sum_out, w).
// Keeps the residual stream for later use (e.g. MLP residual add).
kernel void add_rmsnorm_f32(
    device const float *x [[buffer(0)]],
    device const float *residual [[buffer(1)]],
    device const float *w [[buffer(2)]],
    device float *sum_out [[buffer(3)]],
    device float *norm_out [[buffer(4)]],
    constant RmsNormParams &p [[buffer(5)]],
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
    device const float *row_r = residual + row * p.cols;
    device float *row_sum = sum_out + row * p.cols;
    device float *row_norm = norm_out + row * p.cols;

    float acc = 0.0f;
    for (uint i = lid; i < p.cols; i += tgs) {
        float v = row_x[i] + row_r[i];
        row_sum[i] = v;
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
        row_norm[i] = w[i] * row_sum[i] * inv;
    }
}

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

// Int8 weights with per-row f32 scale: y[m] = (W_q[m,k] * scale[m]) @ x[k]
kernel void matvec_q8_f32(
    device const char *wq [[buffer(0)]],
    device const float *scale [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device float *y [[buffer(3)]],
    constant MatmulParams &p [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.m) {
        return;
    }
    device const char *row = wq + gid * p.k;
    float acc = 0.0f;
    for (uint t = 0; t < p.k; t++) {
        acc += float(row[t]) * x[t];
    }
    y[gid] = acc * scale[gid];
}

struct MatmulQ8Params {
    uint m;
    uint n;
    uint k;
    uint scale_mode; // 0 = per-row (len m), 1 = per-tensor (len 1)
};

// C[m,n] = dequant(W_q[m,k]) @ B[k,n]. One thread per output element.
kernel void matmul_q8_f32(
    device const char *wq [[buffer(0)]],
    device const float *scale [[buffer(1)]],
    device const float *b [[buffer(2)]],
    device float *c [[buffer(3)]],
    constant MatmulQ8Params &p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    if (row >= p.m || col >= p.n) {
        return;
    }
    device const char *wrow = wq + row * p.k;
    float acc = 0.0f;
    for (uint t = 0; t < p.k; t++) {
        acc += float(wrow[t]) * b[t * p.n + col];
    }
    float s = (p.scale_mode == 1u) ? scale[0] : scale[row];
    c[row * p.n + col] = acc * s;
}

// simdgroup_matrix GEMM: one simdgroup (32 threads) per 8x8 C tile.
// Stages A/B into threadgroup memory with zero fill so ragged dims are OK.
// Requires Apple GPU family 7+. Caller must set threadgroup memory to 768 bytes
// and dispatchThreadgroups with threadsPerThreadgroup = (32,1,1).
kernel void matmul_f32_simdgroup(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant MatmulParams &p [[buffer(3)]],
    threadgroup float *tile [[threadgroup(0)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint TILE = 8u;
    const uint CELLS = 64u;
    const uint LANES = 32u;

    uint tile_row = tg_pos.y * TILE;
    uint tile_col = tg_pos.x * TILE;
    if (tile_row >= p.m || tile_col >= p.n) {
        return;
    }

    threadgroup float *a_stage = tile;
    threadgroup float *b_stage = tile + CELLS;
    threadgroup float *c_stage = tile + 2u * CELLS;

    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint kk = 0u; kk < p.k; kk += TILE) {
        for (uint e = lane; e < CELLS; e += LANES) {
            uint r = e / TILE;
            uint col = e % TILE;
            uint ar = tile_row + r;
            uint ak = kk + col;
            a_stage[e] = (ar < p.m && ak < p.k) ? a[ar * p.k + ak] : 0.0f;
            uint bk = kk + r;
            uint bc = tile_col + col;
            b_stage[e] = (bk < p.k && bc < p.n) ? b[bk * p.n + bc] : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 a_frag;
        simdgroup_float8x8 b_frag;
        simdgroup_load(a_frag, a_stage, TILE);
        simdgroup_load(b_frag, b_stage, TILE);
        simdgroup_multiply_accumulate(acc, a_frag, b_frag, acc);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc, c_stage, TILE);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = lane; e < CELLS; e += LANES) {
        uint r = e / TILE;
        uint col = e % TILE;
        uint row = tile_row + r;
        uint out_col = tile_col + col;
        if (row < p.m && out_col < p.n) {
            c[row * p.n + out_col] = c_stage[e];
        }
    }
}

// Four simdgroups per threadgroup: each owns an 8x8 tile; TG covers 8x32 of C.
// threadgroups: (ceil(N/32), ceil(M/8), 1), threadsPerThreadgroup (128,1,1),
// threadgroup memory 4 * 3 * 64 * 4 = 3072 bytes.
kernel void matmul_f32_simdgroup_x4(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant MatmulParams &p [[buffer(3)]],
    threadgroup float *tile [[threadgroup(0)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint TILE = 8u;
    const uint CELLS = 64u;
    const uint LANES = 32u;
    const uint SG = 4u;

    uint tile_row = tg_pos.y * TILE;
    uint tile_col = (tg_pos.x * SG + sg_id) * TILE;
    if (tile_row >= p.m || tile_col >= p.n || sg_id >= SG) {
        return;
    }

    threadgroup float *a_stage = tile + sg_id * (3u * CELLS);
    threadgroup float *b_stage = a_stage + CELLS;
    threadgroup float *c_stage = b_stage + CELLS;

    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint kk = 0u; kk < p.k; kk += TILE) {
        for (uint e = lane; e < CELLS; e += LANES) {
            uint r = e / TILE;
            uint col = e % TILE;
            uint ar = tile_row + r;
            uint ak = kk + col;
            a_stage[e] = (ar < p.m && ak < p.k) ? a[ar * p.k + ak] : 0.0f;
            uint bk = kk + r;
            uint bc = tile_col + col;
            b_stage[e] = (bk < p.k && bc < p.n) ? b[bk * p.n + bc] : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 a_frag;
        simdgroup_float8x8 b_frag;
        simdgroup_load(a_frag, a_stage, TILE);
        simdgroup_load(b_frag, b_stage, TILE);
        simdgroup_multiply_accumulate(acc, a_frag, b_frag, acc);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc, c_stage, TILE);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = lane; e < CELLS; e += LANES) {
        uint r = e / TILE;
        uint col = e % TILE;
        uint row = tile_row + r;
        uint out_col = tile_col + col;
        if (row < p.m && out_col < p.n) {
            c[row * p.n + out_col] = c_stage[e];
        }
    }
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

struct AttentionParams {
    uint n_q;
    uint n_kv;
    uint q_len;
    uint kv_len;
    uint kv_stride;
    uint head_dim;
};

// One thread per (q_token, q_head). Scores live in thread-local memory.
// kv_len is capped at 256 (Stage 8); Zig returns Unsupported above that.
kernel void attention_f32(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant AttentionParams &p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint tq = gid.x;
    uint h = gid.y;
    if (tq >= p.q_len || h >= p.n_q || p.kv_len > 256) {
        return;
    }
    uint group = p.n_q / p.n_kv;
    uint kv_h = h / group;
    uint d = p.head_dim;
    float scale = rsqrt(float(d));
    thread float scores[256];

    device const float *qrow = q + (h * p.q_len + tq) * d;
    float max_s = -INFINITY;
    for (uint tk = 0; tk < p.kv_len; tk++) {
        bool causal_ok = (p.kv_len - p.q_len + tq) >= tk;
        if (!causal_ok) {
            scores[tk] = -INFINITY;
            continue;
        }
        device const float *krow = k + (kv_h * p.kv_stride + tk) * d;
        float dot = 0.0f;
        for (uint i = 0; i < d; i++) {
            dot += qrow[i] * krow[i];
        }
        scores[tk] = dot * scale;
        max_s = max(max_s, scores[tk]);
    }

    float sum = 0.0f;
    for (uint tk = 0; tk < p.kv_len; tk++) {
        if (isinf(scores[tk])) {
            scores[tk] = 0.0f;
        } else {
            scores[tk] = exp(scores[tk] - max_s);
            sum += scores[tk];
        }
    }
    float inv = (sum == 0.0f) ? 0.0f : 1.0f / sum;
    device float *orow = out + (h * p.q_len + tq) * d;
    for (uint i = 0; i < d; i++) {
        orow[i] = 0.0f;
    }
    for (uint tk = 0; tk < p.kv_len; tk++) {
        float w = scores[tk] * inv;
        if (w == 0.0f) {
            continue;
        }
        device const float *vrow = v + (kv_h * p.kv_stride + tk) * d;
        for (uint i = 0; i < d; i++) {
            orow[i] += w * vrow[i];
        }
    }
}

// Stage 6 helpers for Metal-resident tiny-block schedule.

struct PermuteParams {
    uint tokens;
    uint n_heads;
    uint head_dim;
};

// in [tokens, n_heads, d] → out [n_heads, tokens, d]
kernel void permute_tokens_heads_f32(
    device const float *in [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant PermuteParams &p [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint n = p.tokens * p.n_heads * p.head_dim;
    if (gid >= n) {
        return;
    }
    uint d = p.head_dim;
    uint elem = gid;
    uint t = elem / (p.n_heads * d);
    uint rem = elem % (p.n_heads * d);
    uint h = rem / d;
    uint i = rem % d;
    out[(h * p.tokens + t) * d + i] = in[(t * p.n_heads + h) * d + i];
}

// in [n_heads, tokens, d] → out [tokens, n_heads, d]
kernel void permute_heads_tokens_f32(
    device const float *in [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant PermuteParams &p [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint n = p.tokens * p.n_heads * p.head_dim;
    if (gid >= n) {
        return;
    }
    uint d = p.head_dim;
    uint elem = gid;
    uint h = elem / (p.tokens * d);
    uint rem = elem % (p.tokens * d);
    uint t = rem / d;
    uint i = rem % d;
    out[(t * p.n_heads + h) * d + i] = in[(h * p.tokens + t) * d + i];
}

struct KvAppendParams {
    uint n_kv;
    uint t;
    uint head_dim;
    uint max_seq;
    uint used;
};

// k_new/v_new are [n_kv, t, d]; caches are [n_kv, max_seq, d].
kernel void kv_append_f32(
    device const float *k_new [[buffer(0)]],
    device const float *v_new [[buffer(1)]],
    device float *k_cache [[buffer(2)]],
    device float *v_cache [[buffer(3)]],
    constant KvAppendParams &p [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint n = p.n_kv * p.t * p.head_dim;
    if (gid >= n) {
        return;
    }
    uint d = p.head_dim;
    uint elem = gid;
    uint h = elem / (p.t * d);
    uint rem = elem % (p.t * d);
    uint i_t = rem / d;
    uint i = rem % d;
    uint src = ((h * p.t) + i_t) * d + i;
    uint dst = ((h * p.max_seq) + (p.used + i_t)) * d + i;
    k_cache[dst] = k_new[src];
    v_cache[dst] = v_new[src];
}
