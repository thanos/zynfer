//! Metal implementations of the LLM ops, differentially tested against CPU.
//!
//! Shared MTLBuffers are CPU-writable. Filling `contents` is not a PCIe copy;
//! `encode_and_wait` still waits for GPU cache visibility.

const std = @import("std");
const build_options = @import("build_options");
const Tensor = @import("../../runtime/tensor.zig").Tensor;
const compare = @import("../../runtime/compare.zig");
const cpu = @import("../cpu/ops.zig");
const gpu_mod = @import("gpu.zig");

pub const have_apple = gpu_mod.have_apple;
pub const Error = gpu_mod.Error || cpu.OpsError;

const Gpu = gpu_mod.Gpu;
const Buffer = gpu_mod.Buffer;

fn upload(gpu: *Gpu, t: Tensor) Error!Buffer {
    const src = try t.f32s();
    var buf = try gpu.allocShared(src.len * @sizeOf(f32));
    @memcpy(buf.f32s()[0..src.len], src);
    return buf;
}

fn download(buf: Buffer, t: Tensor) Error!void {
    const dst = try t.f32s();
    @memcpy(dst, buf.f32s()[0..dst.len]);
}

/// Collect Metal handles only when the Apple backend is compiled in.
/// On Linux `Buffer.handle` is `void`; that branch is not type-checked.
fn launchBufs(
    gpu: *Gpu,
    kernel: [:0]const u8,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    tg_x: u32,
    tg_y: u32,
    tg_z: u32,
    bufs: []const Buffer,
    params: []const u8,
) Error!void {
    try launchBufsOpts(gpu, kernel, grid_x, grid_y, grid_z, tg_x, tg_y, tg_z, bufs, params, .{});
}

fn launchBufsOpts(
    gpu: *Gpu,
    kernel: [:0]const u8,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    tg_x: u32,
    tg_y: u32,
    tg_z: u32,
    bufs: []const Buffer,
    params: []const u8,
    opts: gpu_mod.Gpu.LaunchOpts,
) Error!void {
    if (!have_apple) return error.AppleUnavailable;
    if (have_apple) {
        var tmp: [8]*gpu_mod.MtlBuffer = undefined;
        if (bufs.len > tmp.len) return error.Unsupported;
        for (bufs, 0..) |b, i| tmp[i] = b.handle;
        try gpu.launchOpts(kernel, grid_x, grid_y, grid_z, tg_x, tg_y, tg_z, tmp[0..bufs.len], params, opts);
    }
}

fn launch1d(gpu: *Gpu, kernel: [:0]const u8, n: u32, bufs: []const Buffer) Error!void {
    const tg = gpu.threadgroup1d();
    try launchBufs(gpu, kernel, n, 1, 1, tg, 1, 1, bufs, std.mem.asBytes(&n));
}

pub fn add(gpu: *Gpu, dst: Tensor, a: Tensor, b: Tensor) Error!void {
    const n: u32 = @intCast(try a.numel());
    var ab = try upload(gpu, a);
    defer ab.deinit();
    var bb = try upload(gpu, b);
    defer bb.deinit();
    var cb = try gpu.allocShared(dst.data.len);
    defer cb.deinit();
    try launch1d(gpu, "add_f32", n, &.{ ab, bb, cb });
    try download(cb, dst);
}

pub fn mul(gpu: *Gpu, dst: Tensor, a: Tensor, b: Tensor) Error!void {
    const n: u32 = @intCast(try a.numel());
    var ab = try upload(gpu, a);
    defer ab.deinit();
    var bb = try upload(gpu, b);
    defer bb.deinit();
    var cb = try gpu.allocShared(dst.data.len);
    defer cb.deinit();
    try launch1d(gpu, "mul_f32", n, &.{ ab, bb, cb });
    try download(cb, dst);
}

pub fn siluMul(gpu: *Gpu, dst: Tensor, gate: Tensor, up: Tensor) Error!void {
    const n: u32 = @intCast(try gate.numel());
    var gb = try upload(gpu, gate);
    defer gb.deinit();
    var ub = try upload(gpu, up);
    defer ub.deinit();
    var ob = try gpu.allocShared(dst.data.len);
    defer ob.deinit();
    try launch1d(gpu, "silu_mul_f32", n, &.{ gb, ub, ob });
    try download(ob, dst);
}

const RmsNormParams = extern struct {
    rows: u32,
    cols: u32,
    eps: f32,
};

pub fn rmsNorm(gpu: *Gpu, dst: Tensor, x: Tensor, weight: Tensor, eps: f32) Error!void {
    const cols: u32 = @intCast(x.shape[x.rank - 1]);
    const n: u32 = @intCast(try x.numel());
    const rows = n / cols;
    var xb = try upload(gpu, x);
    defer xb.deinit();
    var wb = try upload(gpu, weight);
    defer wb.deinit();
    var yb = try gpu.allocShared(dst.data.len);
    defer yb.deinit();
    const params = RmsNormParams{ .rows = rows, .cols = cols, .eps = eps };
    const tg = gpu.threadgroup1d();
    try launchBufs(gpu, "rmsnorm_f32", tg, rows, 1, tg, 1, 1, &.{ xb, wb, yb }, std.mem.asBytes(&params));
    try download(yb, dst);
}

const SoftmaxParams = extern struct {
    rows: u32,
    cols: u32,
};

pub fn softmax(gpu: *Gpu, dst: Tensor, x: Tensor) Error!void {
    const cols: u32 = @intCast(x.shape[x.rank - 1]);
    const n: u32 = @intCast(try x.numel());
    const rows = n / cols;
    var xb = try upload(gpu, x);
    defer xb.deinit();
    var yb = try gpu.allocShared(dst.data.len);
    defer yb.deinit();
    const params = SoftmaxParams{ .rows = rows, .cols = cols };
    const tg = gpu.threadgroup1d();
    try launchBufs(gpu, "softmax_f32", tg, rows, 1, tg, 1, 1, &.{ xb, yb }, std.mem.asBytes(&params));
    try download(yb, dst);
}

const MatmulParams = extern struct {
    m: u32,
    n: u32,
    k: u32,
};

pub const MatmulPath = enum {
    naive,
    simdgroup,
    simdgroup_x4,

    pub fn name(self: MatmulPath) []const u8 {
        return switch (self) {
            .naive => "matmul_f32",
            .simdgroup => "matmul_f32_simdgroup",
            .simdgroup_x4 => "matmul_f32_simdgroup_x4",
        };
    }
};

/// Last selected Metal matmul / matvec path names for diagnostics.
pub var last_matmul_path: []const u8 = "unset";
pub var last_matvec_path: []const u8 = "unset";
pub var last_q8_path: []const u8 = "unset";

/// Minimum M*N*K before single-SG simdgroup is preferred over naive.
pub const simdgroup_min_flops: usize = 64 * 64 * 64;
/// Prefer 4-SG threadgroups once the problem is large enough that occupancy wins.
/// Measured at 256³ on M1 Max: x4 was slower than 1-SG, so auto-select stays off
/// until a larger shape proves a win. Force with `ZYNFER_MATMUL_PATH=simdgroup_x4`.
pub const simdgroup_x4_min_flops: usize = 256 * 256 * 256;

/// Force matmul path via `ZYNFER_MATMUL_PATH=naive|simdgroup|simdgroup_x4`.
pub fn forcedMatmulPath() ?MatmulPath {
    if (comptime !have_apple) return null;
    const raw = std.c.getenv("ZYNFER_MATMUL_PATH") orelse return null;
    const v = std.mem.span(raw);
    if (std.mem.eql(u8, v, "naive")) return .naive;
    if (std.mem.eql(u8, v, "simdgroup")) return .simdgroup;
    if (std.mem.eql(u8, v, "simdgroup_x4") or std.mem.eql(u8, v, "x4")) return .simdgroup_x4;
    return null;
}

pub fn selectMatmulPath(gpu: *const Gpu, m: usize, n: usize, k: usize) MatmulPath {
    if (forcedMatmulPath()) |forced| {
        if ((forced == .simdgroup or forced == .simdgroup_x4) and !gpu.features.simdgroup_matrix_available)
            return .naive;
        return forced;
    }
    if (!gpu.features.simdgroup_matrix_available) return .naive;
    if (m < 8 or n < 8 or k < 8) return .naive;
    const flops = m * n * k;
    if (flops < simdgroup_min_flops) return .naive;
    // Keep `_x4` force-only; see `simdgroup_x4_min_flops` comment.
    return .simdgroup;
}

pub fn matmul(gpu: *Gpu, c: Tensor, a: Tensor, b: Tensor) Error!void {
    const path = selectMatmulPath(gpu, a.shape[0], b.shape[1], a.shape[1]);
    try matmulPath(gpu, c, a, b, path);
}

pub fn matmulPath(gpu: *Gpu, c: Tensor, a: Tensor, b: Tensor, path: MatmulPath) Error!void {
    const m: u32 = @intCast(a.shape[0]);
    const k: u32 = @intCast(a.shape[1]);
    const n: u32 = @intCast(b.shape[1]);
    var ab = try upload(gpu, a);
    defer ab.deinit();
    var bb = try upload(gpu, b);
    defer bb.deinit();
    var cb = try gpu.allocShared(c.data.len);
    defer cb.deinit();
    const params = MatmulParams{ .m = m, .n = n, .k = k };
    switch (path) {
        .naive => {
            last_matmul_path = "matmul_f32";
            const tg: u32 = 16;
            try launchBufs(gpu, "matmul_f32", n, m, 1, tg, tg, 1, &.{ ab, bb, cb }, std.mem.asBytes(&params));
        },
        .simdgroup => {
            if (!gpu.features.simdgroup_matrix_available) return error.Unsupported;
            last_matmul_path = "matmul_f32_simdgroup";
            const tile: u32 = 8;
            const tg_x = (n + tile - 1) / tile;
            const tg_y = (m + tile - 1) / tile;
            const tg_mem: u32 = 3 * 64 * 4;
            try launchBufsOpts(
                gpu,
                "matmul_f32_simdgroup",
                tg_x,
                tg_y,
                1,
                32,
                1,
                1,
                &.{ ab, bb, cb },
                std.mem.asBytes(&params),
                .{ .threadgroup_mem_bytes = tg_mem, .threadgroups = true },
            );
        },
        .simdgroup_x4 => {
            if (!gpu.features.simdgroup_matrix_available) return error.Unsupported;
            last_matmul_path = "matmul_f32_simdgroup_x4";
            const tile: u32 = 8;
            const sg: u32 = 4;
            const tg_x = (n + tile * sg - 1) / (tile * sg);
            const tg_y = (m + tile - 1) / tile;
            const tg_mem: u32 = sg * 3 * 64 * 4;
            try launchBufsOpts(
                gpu,
                "matmul_f32_simdgroup_x4",
                tg_x,
                tg_y,
                1,
                32 * sg,
                1,
                1,
                &.{ ab, bb, cb },
                std.mem.asBytes(&params),
                .{ .threadgroup_mem_bytes = tg_mem, .threadgroups = true },
            );
        },
    }
    try download(cb, c);
}

pub fn matvec(gpu: *Gpu, y: Tensor, a: Tensor, x: Tensor) Error!void {
    last_matvec_path = "matvec_f32";
    const m: u32 = @intCast(a.shape[0]);
    const k: u32 = @intCast(a.shape[1]);
    var ab = try upload(gpu, a);
    defer ab.deinit();
    var xb = try upload(gpu, x);
    defer xb.deinit();
    var yb = try gpu.allocShared(y.data.len);
    defer yb.deinit();
    const params = MatmulParams{ .m = m, .n = 1, .k = k };
    const tg = gpu.threadgroup1d();
    try launchBufs(gpu, "matvec_f32", m, 1, 1, tg, 1, 1, &.{ ab, xb, yb }, std.mem.asBytes(&params));
    try download(yb, y);
}

const MatmulQ8Params = extern struct {
    m: u32,
    n: u32,
    k: u32,
    scale_mode: u32,
};

/// C = dequant(W_q) @ B. Prefer this over f32 matmul when weights are already int8
/// and M·N·K is large enough that dequant-on-the-fly still wins on memory traffic
/// in a fair (pre-packed) bench — see `q8_matmul_min_flops`.
pub const q8_matmul_min_flops: usize = 128 * 128 * 128;

pub fn preferPackedQ8Matmul(m: usize, n: usize, k: usize) bool {
    return m * n * k >= q8_matmul_min_flops;
}

/// Persistent Metal int8 weights: upload once, reuse across matvec/matmul calls.
pub const Q8DeviceWeights = struct {
    q: Buffer,
    /// Always length `m` (per-tensor scale is broadcast at upload).
    scale: Buffer,
    m: usize,
    k: usize,
    mode: cpu.Q8ScaleMode,

    pub fn upload(
        gpu: *Gpu,
        w_q8: []const i8,
        scale: []const f32,
        m: usize,
        k: usize,
        mode: cpu.Q8ScaleMode,
    ) Error!Q8DeviceWeights {
        if (w_q8.len != m * k) return error.ShapeMismatch;
        switch (mode) {
            .per_row => if (scale.len != m) return error.ShapeMismatch,
            .per_tensor => if (scale.len != 1) return error.ShapeMismatch,
        }
        var q = try gpu.allocShared(w_q8.len);
        errdefer q.deinit();
        @memcpy(q.bytes[0..w_q8.len], @as([*]const u8, @ptrCast(w_q8.ptr))[0..w_q8.len]);

        var scale_buf = try gpu.allocShared(m * @sizeOf(f32));
        errdefer scale_buf.deinit();
        const ss = scale_buf.f32s();
        if (mode == .per_tensor) {
            @memset(ss[0..m], scale[0]);
        } else {
            @memcpy(ss[0..m], scale);
        }
        return .{ .q = q, .scale = scale_buf, .m = m, .k = k, .mode = mode };
    }

    pub fn deinit(self: *Q8DeviceWeights) void {
        self.q.deinit();
        self.scale.deinit();
        self.* = undefined;
    }
};

pub fn matvecQ8(
    gpu: *Gpu,
    y: Tensor,
    w_q8: []const i8,
    scale: []const f32,
    x: Tensor,
    mode: cpu.Q8ScaleMode,
) Error!void {
    var packed_w = try Q8DeviceWeights.upload(gpu, w_q8, scale, y.shape[0], x.shape[0], mode);
    defer packed_w.deinit();
    try matvecQ8Persistent(gpu, y, packed_w, x);
}

/// GEMV with weights already resident on the GPU (no weight re-upload).
pub fn matvecQ8Persistent(gpu: *Gpu, y: Tensor, w: Q8DeviceWeights, x: Tensor) Error!void {
    if (y.rank != 1 or x.rank != 1) return error.InvalidShape;
    const m = y.shape[0];
    const k = x.shape[0];
    if (w.m != m or w.k != k) return error.ShapeMismatch;

    last_q8_path = if (w.mode == .per_row) "matvec_q8_f32_persistent_per_row" else "matvec_q8_f32_persistent_per_tensor";

    var xb = try upload(gpu, x);
    defer xb.deinit();
    var yb = try gpu.allocShared(y.data.len);
    defer yb.deinit();
    try encodeMatvecQ8(gpu, yb, w.q, w.scale, xb, @intCast(m), @intCast(k));
    try download(yb, y);
}

pub fn matmulQ8(
    gpu: *Gpu,
    c: Tensor,
    w_q8: []const i8,
    scale: []const f32,
    b: Tensor,
    mode: cpu.Q8ScaleMode,
) Error!void {
    if (c.rank != 2 or b.rank != 2) return error.InvalidShape;
    const m = c.shape[0];
    const n = c.shape[1];
    const k = b.shape[0];
    if (b.shape[1] != n) return error.ShapeMismatch;
    var packed_w = try Q8DeviceWeights.upload(gpu, w_q8, scale, m, k, mode);
    defer packed_w.deinit();
    try matmulQ8Persistent(gpu, c, packed_w, b);
}

pub fn matmulQ8Persistent(gpu: *Gpu, c: Tensor, w: Q8DeviceWeights, b: Tensor) Error!void {
    if (c.rank != 2 or b.rank != 2) return error.InvalidShape;
    const m = c.shape[0];
    const n = c.shape[1];
    const k = b.shape[0];
    if (b.shape[1] != n or w.m != m or w.k != k) return error.ShapeMismatch;

    last_q8_path = if (w.mode == .per_row) "matmul_q8_f32_persistent_per_row" else "matmul_q8_f32_persistent_per_tensor";

    var bb = try upload(gpu, b);
    defer bb.deinit();
    var cb = try gpu.allocShared(c.data.len);
    defer cb.deinit();
    // Persistent path always stores per-row (broadcast) scales of length m.
    try encodeMatmulQ8(gpu, cb, w.q, w.scale, bb, @intCast(m), @intCast(n), @intCast(k), .per_row);
    try download(cb, c);
}

const RopeParams = extern struct {
    tokens: u32,
    n_heads: u32,
    head_dim: u32,
    pos0: u32,
    theta: f32,
};

pub fn rope(gpu: *Gpu, x: Tensor, pos0: usize, theta: f32) Error!void {
    const head_dim: u32 = @intCast(x.shape[x.rank - 1]);
    const n_heads: u32 = @intCast(x.shape[x.rank - 2]);
    const n: u32 = @intCast(try x.numel());
    const tokens = n / (n_heads * head_dim);
    var xb = try upload(gpu, x);
    defer xb.deinit();
    const params = RopeParams{
        .tokens = tokens,
        .n_heads = n_heads,
        .head_dim = head_dim,
        .pos0 = @intCast(pos0),
        .theta = theta,
    };
    const pairs = tokens * n_heads * (head_dim / 2);
    const tg = gpu.threadgroup1d();
    try launchBufs(gpu, "rope_f32", pairs, 1, 1, tg, 1, 1, &.{xb}, std.mem.asBytes(&params));
    try download(xb, x);
}

/// Thread-local softmax; `kv_len` must be <= `max_attention_kv`.
/// Stage 8 raised the cap from 64 → 256 (still thread-local scores).
pub const max_attention_kv: usize = 256;

const AttentionParams = extern struct {
    n_q: u32,
    n_kv: u32,
    q_len: u32,
    kv_len: u32,
    kv_stride: u32,
    head_dim: u32,
};

pub fn attention(
    gpu: *Gpu,
    out: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kv_len: usize,
    kv_stride: usize,
) Error!void {
    if (q.rank != 3 or k.rank != 3 or v.rank != 3 or out.rank != 3) return error.InvalidShape;
    const n_q = q.shape[0];
    const q_len = q.shape[1];
    const d = q.shape[2];
    const n_kv = k.shape[0];
    if (k.shape[1] != kv_stride or v.shape[1] != kv_stride) return error.ShapeMismatch;
    if (k.shape[2] != d or v.shape[0] != n_kv or v.shape[2] != d) return error.ShapeMismatch;
    if (out.shape[0] != n_q or out.shape[1] != q_len or out.shape[2] != d) return error.ShapeMismatch;
    if (n_q == 0 or n_kv == 0 or n_q % n_kv != 0) return error.InvalidShape;
    if (kv_len == 0 or q_len == 0 or kv_len > kv_stride or q_len > kv_len) return error.InvalidShape;
    if (kv_len > max_attention_kv) return error.Unsupported;

    var qb = try upload(gpu, q);
    defer qb.deinit();
    var kb = try upload(gpu, k);
    defer kb.deinit();
    var vb = try upload(gpu, v);
    defer vb.deinit();
    var ob = try gpu.allocShared(out.data.len);
    defer ob.deinit();
    const params = AttentionParams{
        .n_q = @intCast(n_q),
        .n_kv = @intCast(n_kv),
        .q_len = @intCast(q_len),
        .kv_len = @intCast(kv_len),
        .kv_stride = @intCast(kv_stride),
        .head_dim = @intCast(d),
    };
    try launchBufs(
        gpu,
        "attention_f32",
        @intCast(q_len),
        @intCast(n_q),
        1,
        1,
        1,
        1,
        &.{ qb, kb, vb, ob },
        std.mem.asBytes(&params),
    );
    try download(ob, out);
}

/// Stage 6: encode onto the current Gpu batch (or one-shot wait if no batch).
/// These take resident buffers — no host upload/download.
pub fn encodeAdd(gpu: *Gpu, dst: Buffer, a: Buffer, b: Buffer, n: u32) Error!void {
    try launch1d(gpu, "add_f32", n, &.{ a, b, dst });
}

pub fn encodeSiluMul(gpu: *Gpu, dst: Buffer, gate: Buffer, up: Buffer, n: u32) Error!void {
    try launch1d(gpu, "silu_mul_f32", n, &.{ gate, up, dst });
}

pub fn encodeRmsNorm(gpu: *Gpu, dst: Buffer, x: Buffer, weight: Buffer, rows: u32, cols: u32, eps: f32) Error!void {
    const params = RmsNormParams{ .rows = rows, .cols = cols, .eps = eps };
    const tg = gpu.threadgroup1d();
    try launchBufs(gpu, "rmsnorm_f32", tg, rows, 1, tg, 1, 1, &.{ x, weight, dst }, std.mem.asBytes(&params));
}

/// sum_out = x + residual; norm_out = rmsnorm(sum_out, w).
pub fn encodeAddRmsNorm(
    gpu: *Gpu,
    sum_out: Buffer,
    norm_out: Buffer,
    x: Buffer,
    residual: Buffer,
    weight: Buffer,
    rows: u32,
    cols: u32,
    eps: f32,
) Error!void {
    const params = RmsNormParams{ .rows = rows, .cols = cols, .eps = eps };
    const tg = gpu.threadgroup1d();
    try launchBufs(
        gpu,
        "add_rmsnorm_f32",
        tg,
        rows,
        1,
        tg,
        1,
        1,
        &.{ x, residual, weight, sum_out, norm_out },
        std.mem.asBytes(&params),
    );
}

pub fn encodeMatvecQ8(gpu: *Gpu, y: Buffer, w_q: Buffer, scale: Buffer, x: Buffer, m: u32, k: u32) Error!void {
    const params = MatmulParams{ .m = m, .n = 1, .k = k };
    const tg = gpu.threadgroup1d();
    try launchBufs(gpu, "matvec_q8_f32", m, 1, 1, tg, 1, 1, &.{ w_q, scale, x, y }, std.mem.asBytes(&params));
}

pub fn encodeMatmulQ8(
    gpu: *Gpu,
    c_buf: Buffer,
    w_q: Buffer,
    scale: Buffer,
    b: Buffer,
    m: u32,
    n: u32,
    k: u32,
    mode: cpu.Q8ScaleMode,
) Error!void {
    const params = MatmulQ8Params{
        .m = m,
        .n = n,
        .k = k,
        .scale_mode = if (mode == .per_tensor) 1 else 0,
    };
    const tg: u32 = 16;
    try launchBufs(gpu, "matmul_q8_f32", n, m, 1, tg, tg, 1, &.{ w_q, scale, b, c_buf }, std.mem.asBytes(&params));
}

pub fn encodeMatmulNaive(gpu: *Gpu, c_buf: Buffer, a: Buffer, b: Buffer, m: u32, n: u32, k: u32) Error!void {
    const params = MatmulParams{ .m = m, .n = n, .k = k };
    const tg: u32 = 16;
    try launchBufs(gpu, "matmul_f32", n, m, 1, tg, tg, 1, &.{ a, b, c_buf }, std.mem.asBytes(&params));
}

pub fn encodeRope(gpu: *Gpu, x: Buffer, tokens: u32, n_heads: u32, head_dim: u32, pos0: u32, theta: f32) Error!void {
    const params = RopeParams{
        .tokens = tokens,
        .n_heads = n_heads,
        .head_dim = head_dim,
        .pos0 = pos0,
        .theta = theta,
    };
    const pairs = tokens * n_heads * (head_dim / 2);
    const tg = gpu.threadgroup1d();
    try launchBufs(gpu, "rope_f32", pairs, 1, 1, tg, 1, 1, &.{x}, std.mem.asBytes(&params));
}

pub fn encodeAttention(
    gpu: *Gpu,
    out: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    n_q: u32,
    n_kv: u32,
    q_len: u32,
    kv_len: u32,
    kv_stride: u32,
    head_dim: u32,
) Error!void {
    if (kv_len > max_attention_kv) return error.Unsupported;
    const params = AttentionParams{
        .n_q = n_q,
        .n_kv = n_kv,
        .q_len = q_len,
        .kv_len = kv_len,
        .kv_stride = kv_stride,
        .head_dim = head_dim,
    };
    try launchBufs(gpu, "attention_f32", q_len, n_q, 1, 1, 1, 1, &.{ q, k, v, out }, std.mem.asBytes(&params));
}

const PermuteParams = extern struct {
    tokens: u32,
    n_heads: u32,
    head_dim: u32,
};

pub fn encodePermuteTokensHeads(gpu: *Gpu, out: Buffer, in_buf: Buffer, tokens: u32, n_heads: u32, head_dim: u32) Error!void {
    const params = PermuteParams{ .tokens = tokens, .n_heads = n_heads, .head_dim = head_dim };
    const n = tokens * n_heads * head_dim;
    const tg = gpu.threadgroup1d();
    try launchBufs(gpu, "permute_tokens_heads_f32", n, 1, 1, tg, 1, 1, &.{ in_buf, out }, std.mem.asBytes(&params));
}

pub fn encodePermuteHeadsTokens(gpu: *Gpu, out: Buffer, in_buf: Buffer, tokens: u32, n_heads: u32, head_dim: u32) Error!void {
    const params = PermuteParams{ .tokens = tokens, .n_heads = n_heads, .head_dim = head_dim };
    const n = tokens * n_heads * head_dim;
    const tg = gpu.threadgroup1d();
    try launchBufs(gpu, "permute_heads_tokens_f32", n, 1, 1, tg, 1, 1, &.{ in_buf, out }, std.mem.asBytes(&params));
}

const KvAppendParams = extern struct {
    n_kv: u32,
    t: u32,
    head_dim: u32,
    max_seq: u32,
    used: u32,
};

pub fn encodeKvAppend(
    gpu: *Gpu,
    k_new: Buffer,
    v_new: Buffer,
    k_cache: Buffer,
    v_cache: Buffer,
    n_kv: u32,
    t: u32,
    head_dim: u32,
    max_seq: u32,
    used: u32,
) Error!void {
    const params = KvAppendParams{
        .n_kv = n_kv,
        .t = t,
        .head_dim = head_dim,
        .max_seq = max_seq,
        .used = used,
    };
    const n = n_kv * t * head_dim;
    const tg = gpu.threadgroup1d();
    try launchBufs(gpu, "kv_append_f32", n, 1, 1, tg, 1, 1, &.{ k_new, v_new, k_cache, v_cache }, std.mem.asBytes(&params));
}

pub fn swigluResidual(
    gpu: *Gpu,
    out: Tensor,
    x: Tensor,
    wn: Tensor,
    wg: Tensor,
    wu: Tensor,
    wd: Tensor,
    eps: f32,
    allocator: std.mem.Allocator,
) Error!void {
    const tokens = x.shape[0];
    const hidden = x.shape[1];
    const inter = wg.shape[1];
    var normed = try Tensor.alloc(allocator, .f32, &.{ tokens, hidden });
    defer normed.deinit();
    var gate = try Tensor.alloc(allocator, .f32, &.{ tokens, inter });
    defer gate.deinit();
    var up = try Tensor.alloc(allocator, .f32, &.{ tokens, inter });
    defer up.deinit();
    var hidden_act = try Tensor.alloc(allocator, .f32, &.{ tokens, inter });
    defer hidden_act.deinit();
    var down = try Tensor.alloc(allocator, .f32, &.{ tokens, hidden });
    defer down.deinit();
    try rmsNorm(gpu, normed, x, wn, eps);
    try matmul(gpu, gate, normed, wg);
    try matmul(gpu, up, normed, wu);
    try siluMul(gpu, hidden_act, gate, up);
    try matmul(gpu, down, hidden_act, wd);
    try add(gpu, out, x, down);
}

fn fillIota(t: Tensor) !void {
    const xs = try t.f32s();
    for (xs, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.01;
}

test "Metal ops match CPU reference" {
    if (gpu_mod.skipAppleGpuTests()) return error.SkipZigTest;
    var gpu = try Gpu.init();
    defer gpu.deinit();
    const gpa = std.testing.allocator;

    var a = try Tensor.alloc(gpa, .f32, &.{16});
    defer a.deinit();
    var b = try Tensor.alloc(gpa, .f32, &.{16});
    defer b.deinit();
    var cpu_out = try Tensor.alloc(gpa, .f32, &.{16});
    defer cpu_out.deinit();
    var gpu_out = try Tensor.alloc(gpa, .f32, &.{16});
    defer gpu_out.deinit();
    try fillIota(a);
    try fillIota(b);
    for (try b.f32s()) |*v| v.* += 1;

    try cpu.add(cpu_out, a, b);
    try add(&gpu, gpu_out, a, b);
    try compare.expectClose(try cpu_out.f32s(), try gpu_out.f32s(), 1e-5, 1e-5);

    try cpu.siluMul(cpu_out, a, b);
    try siluMul(&gpu, gpu_out, a, b);
    try compare.expectClose(try cpu_out.f32s(), try gpu_out.f32s(), 1e-5, 1e-5);

    var x = try Tensor.alloc(gpa, .f32, &.{ 2, 8 });
    defer x.deinit();
    var w = try Tensor.alloc(gpa, .f32, &.{8});
    defer w.deinit();
    var yn = try Tensor.alloc(gpa, .f32, &.{ 2, 8 });
    defer yn.deinit();
    var gn = try Tensor.alloc(gpa, .f32, &.{ 2, 8 });
    defer gn.deinit();
    try fillIota(x);
    try w.fillF32(1);
    try cpu.rmsNorm(yn, x, w, 1e-6);
    try rmsNorm(&gpu, gn, x, w, 1e-6);
    try compare.expectClose(try yn.f32s(), try gn.f32s(), 1e-4, 1e-4);

    {
        var residual = try Tensor.alloc(gpa, .f32, &.{ 2, 8 });
        defer residual.deinit();
        try fillIota(residual);
        var sum_cpu = try Tensor.alloc(gpa, .f32, &.{ 2, 8 });
        defer sum_cpu.deinit();
        var norm_cpu = try Tensor.alloc(gpa, .f32, &.{ 2, 8 });
        defer norm_cpu.deinit();
        try cpu.add(sum_cpu, x, residual);
        try cpu.rmsNorm(norm_cpu, sum_cpu, w, 1e-6);

        var sum_gpu = try gpu.allocShared(2 * 8 * 4);
        defer sum_gpu.deinit();
        var norm_gpu = try gpu.allocShared(2 * 8 * 4);
        defer norm_gpu.deinit();
        var xb = try upload(&gpu, x);
        defer xb.deinit();
        var rb = try upload(&gpu, residual);
        defer rb.deinit();
        var wb = try upload(&gpu, w);
        defer wb.deinit();
        try encodeAddRmsNorm(&gpu, sum_gpu, norm_gpu, xb, rb, wb, 2, 8, 1e-6);
        try compare.expectClose(try sum_cpu.f32s(), sum_gpu.f32s()[0..16], 1e-4, 1e-4);
        try compare.expectClose(try norm_cpu.f32s(), norm_gpu.f32s()[0..16], 1e-4, 1e-4);
    }

    try cpu.softmax(yn, x);
    try softmax(&gpu, gn, x);
    try compare.expectClose(try yn.f32s(), try gn.f32s(), 1e-4, 1e-4);

    var A = try Tensor.alloc(gpa, .f32, &.{ 4, 3 });
    defer A.deinit();
    var B = try Tensor.alloc(gpa, .f32, &.{ 3, 5 });
    defer B.deinit();
    var Ccpu = try Tensor.alloc(gpa, .f32, &.{ 4, 5 });
    defer Ccpu.deinit();
    var Cgpu = try Tensor.alloc(gpa, .f32, &.{ 4, 5 });
    defer Cgpu.deinit();
    try fillIota(A);
    try fillIota(B);
    try cpu.matmul(Ccpu, A, B);
    try matmul(&gpu, Cgpu, A, B);
    try compare.expectClose(try Ccpu.f32s(), try Cgpu.f32s(), 1e-4, 1e-4);

    if (gpu.features.simdgroup_matrix_available) {
        var As = try Tensor.alloc(gpa, .f32, &.{ 64, 64 });
        defer As.deinit();
        var Bs = try Tensor.alloc(gpa, .f32, &.{ 64, 64 });
        defer Bs.deinit();
        var Ccpu_s = try Tensor.alloc(gpa, .f32, &.{ 64, 64 });
        defer Ccpu_s.deinit();
        var Cgpu_s = try Tensor.alloc(gpa, .f32, &.{ 64, 64 });
        defer Cgpu_s.deinit();
        try fillIota(As);
        try fillIota(Bs);
        try cpu.matmul(Ccpu_s, As, Bs);
        try matmulPath(&gpu, Cgpu_s, As, Bs, .simdgroup);
        try compare.expectClose(try Ccpu_s.f32s(), try Cgpu_s.f32s(), 2e-4, 2e-4);
    }

    var xv = try Tensor.alloc(gpa, .f32, &.{3});
    defer xv.deinit();
    var yv = try Tensor.alloc(gpa, .f32, &.{4});
    defer yv.deinit();
    var yvg = try Tensor.alloc(gpa, .f32, &.{4});
    defer yvg.deinit();
    try fillIota(xv);
    try cpu.matvec(yv, A, xv);
    try matvec(&gpu, yvg, A, xv);
    try compare.expectClose(try yv.f32s(), try yvg.f32s(), 1e-4, 1e-4);

    {
        const mq: usize = 32;
        const kq: usize = 64;
        var W = try Tensor.alloc(gpa, .f32, &.{ mq, kq });
        defer W.deinit();
        var xq = try Tensor.alloc(gpa, .f32, &.{kq});
        defer xq.deinit();
        var y_cpu = try Tensor.alloc(gpa, .f32, &.{mq});
        defer y_cpu.deinit();
        var y_gpu = try Tensor.alloc(gpa, .f32, &.{mq});
        defer y_gpu.deinit();
        try fillIota(W);
        try fillIota(xq);
        const q = try gpa.alloc(i8, mq * kq);
        defer gpa.free(q);
        const scale = try gpa.alloc(f32, mq);
        defer gpa.free(scale);
        try cpu.packRowQ8(try W.f32s(), mq, kq, q, scale);
        try cpu.matvecQ8(try y_cpu.f32s(), q, scale, try xq.f32s(), mq, kq, .per_row);
        try matvecQ8(&gpu, y_gpu, q, scale, xq, .per_row);
        try compare.expectClose(try y_cpu.f32s(), try y_gpu.f32s(), 2e-4, 2e-4);

        var persisted = try Q8DeviceWeights.upload(&gpu, q, scale, mq, kq, .per_row);
        defer persisted.deinit();
        var y_persist = try Tensor.alloc(gpa, .f32, &.{mq});
        defer y_persist.deinit();
        try matvecQ8Persistent(&gpu, y_persist, persisted, xq);
        try compare.expectClose(try y_cpu.f32s(), try y_persist.f32s(), 2e-4, 2e-4);
        // Second call must not need a re-upload of weights.
        try matvecQ8Persistent(&gpu, y_persist, persisted, xq);
        try compare.expectClose(try y_cpu.f32s(), try y_persist.f32s(), 2e-4, 2e-4);
    }

    if (gpu.features.simdgroup_matrix_available) {
        var Ar = try Tensor.alloc(gpa, .f32, &.{ 17, 23 });
        defer Ar.deinit();
        var Br = try Tensor.alloc(gpa, .f32, &.{ 23, 19 });
        defer Br.deinit();
        var Ccpu_r = try Tensor.alloc(gpa, .f32, &.{ 17, 19 });
        defer Ccpu_r.deinit();
        var Cgpu_r = try Tensor.alloc(gpa, .f32, &.{ 17, 19 });
        defer Cgpu_r.deinit();
        try fillIota(Ar);
        try fillIota(Br);
        try cpu.matmul(Ccpu_r, Ar, Br);
        try matmulPath(&gpu, Cgpu_r, Ar, Br, .simdgroup);
        try compare.expectClose(try Ccpu_r.f32s(), try Cgpu_r.f32s(), 3e-4, 3e-4);

        var Ax = try Tensor.alloc(gpa, .f32, &.{ 64, 64 });
        defer Ax.deinit();
        var Bx = try Tensor.alloc(gpa, .f32, &.{ 64, 64 });
        defer Bx.deinit();
        var Ccpu_x = try Tensor.alloc(gpa, .f32, &.{ 64, 64 });
        defer Ccpu_x.deinit();
        var Cgpu_x = try Tensor.alloc(gpa, .f32, &.{ 64, 64 });
        defer Cgpu_x.deinit();
        try fillIota(Ax);
        try fillIota(Bx);
        try cpu.matmul(Ccpu_x, Ax, Bx);
        try matmulPath(&gpu, Cgpu_x, Ax, Bx, .simdgroup_x4);
        try compare.expectClose(try Ccpu_x.f32s(), try Cgpu_x.f32s(), 2e-4, 2e-4);
    }

    {
        const mq: usize = 24;
        const kq: usize = 32;
        const nq: usize = 16;
        var Wq = try Tensor.alloc(gpa, .f32, &.{ mq, kq });
        defer Wq.deinit();
        var Bq = try Tensor.alloc(gpa, .f32, &.{ kq, nq });
        defer Bq.deinit();
        var Ccpu_q = try Tensor.alloc(gpa, .f32, &.{ mq, nq });
        defer Ccpu_q.deinit();
        var Cgpu_q = try Tensor.alloc(gpa, .f32, &.{ mq, nq });
        defer Cgpu_q.deinit();
        try fillIota(Wq);
        try fillIota(Bq);
        const q8 = try gpa.alloc(i8, mq * kq);
        defer gpa.free(q8);
        const scale_q = try gpa.alloc(f32, mq);
        defer gpa.free(scale_q);
        try cpu.packRowQ8(try Wq.f32s(), mq, kq, q8, scale_q);
        try cpu.matmulQ8(try Ccpu_q.f32s(), q8, scale_q, try Bq.f32s(), mq, nq, kq, .per_row);
        try matmulQ8(&gpu, Cgpu_q, q8, scale_q, Bq, .per_row);
        try compare.expectClose(try Ccpu_q.f32s(), try Cgpu_q.f32s(), 3e-4, 3e-4);
    }

    {
        const mq: usize = 20;
        const kq: usize = 24;
        const nq: usize = 12;
        var Wt = try Tensor.alloc(gpa, .f32, &.{ mq, kq });
        defer Wt.deinit();
        var Bt = try Tensor.alloc(gpa, .f32, &.{ kq, nq });
        defer Bt.deinit();
        var Ccpu_t = try Tensor.alloc(gpa, .f32, &.{ mq, nq });
        defer Ccpu_t.deinit();
        var Cgpu_t = try Tensor.alloc(gpa, .f32, &.{ mq, nq });
        defer Cgpu_t.deinit();
        try fillIota(Wt);
        try fillIota(Bt);
        const q8t = try gpa.alloc(i8, mq * kq);
        defer gpa.free(q8t);
        var scale_t: f32 = undefined;
        try cpu.packTensorQ8(try Wt.f32s(), mq, kq, q8t, &scale_t);
        try cpu.matmulQ8(try Ccpu_t.f32s(), q8t, &.{scale_t}, try Bt.f32s(), mq, nq, kq, .per_tensor);
        try matmulQ8(&gpu, Cgpu_t, q8t, &.{scale_t}, Bt, .per_tensor);
        try compare.expectClose(try Ccpu_t.f32s(), try Cgpu_t.f32s(), 3e-4, 3e-4);
    }

    if (gpu.features.simdgroup_matrix_available) {
        try std.testing.expectEqual(MatmulPath.simdgroup, selectMatmulPath(&gpu, 64, 64, 64));
        try std.testing.expectEqual(MatmulPath.simdgroup, selectMatmulPath(&gpu, 256, 256, 256));
        try std.testing.expectEqual(MatmulPath.naive, selectMatmulPath(&gpu, 4, 4, 4));
    }

    var r = try Tensor.alloc(gpa, .f32, &.{ 2, 2, 4 });
    defer r.deinit();
    var rc = try Tensor.alloc(gpa, .f32, &.{ 2, 2, 4 });
    defer rc.deinit();
    try fillIota(r);
    @memcpy(try rc.f32s(), try r.f32s());
    try cpu.rope(rc, 3, 10_000);
    try rope(&gpu, r, 3, 10_000);
    try compare.expectClose(try rc.f32s(), try r.f32s(), 1e-4, 1e-4);

    var q = try Tensor.alloc(gpa, .f32, &.{ 2, 3, 4 });
    defer q.deinit();
    var k = try Tensor.alloc(gpa, .f32, &.{ 1, 4, 4 });
    defer k.deinit();
    var v = try Tensor.alloc(gpa, .f32, &.{ 1, 4, 4 });
    defer v.deinit();
    var acpu = try Tensor.alloc(gpa, .f32, &.{ 2, 3, 4 });
    defer acpu.deinit();
    var agpu = try Tensor.alloc(gpa, .f32, &.{ 2, 3, 4 });
    defer agpu.deinit();
    try fillIota(q);
    try fillIota(k);
    try fillIota(v);
    var scores: [4]f32 = undefined;
    try cpu.attentionInto(acpu, q, k, v, 4, 4, &scores);
    try attention(&gpu, agpu, q, k, v, 4, 4);
    try compare.expectClose(try acpu.f32s(), try agpu.f32s(), 2e-4, 2e-4);
}

test "tiny SwiGLU residual matches CPU" {
    if (gpu_mod.skipAppleGpuTests()) return error.SkipZigTest;
    var gpu = try Gpu.init();
    defer gpu.deinit();
    const gpa = std.testing.allocator;
    const tokens: usize = 2;
    const hidden: usize = 8;
    const inter: usize = 16;
    var x = try Tensor.alloc(gpa, .f32, &.{ tokens, hidden });
    defer x.deinit();
    var wn = try Tensor.alloc(gpa, .f32, &.{hidden});
    defer wn.deinit();
    var wg = try Tensor.alloc(gpa, .f32, &.{ hidden, inter });
    defer wg.deinit();
    var wu = try Tensor.alloc(gpa, .f32, &.{ hidden, inter });
    defer wu.deinit();
    var wd = try Tensor.alloc(gpa, .f32, &.{ inter, hidden });
    defer wd.deinit();
    var cpu_out = try Tensor.alloc(gpa, .f32, &.{ tokens, hidden });
    defer cpu_out.deinit();
    var gpu_out = try Tensor.alloc(gpa, .f32, &.{ tokens, hidden });
    defer gpu_out.deinit();
    try fillIota(x);
    try wn.fillF32(1);
    try fillIota(wg);
    try fillIota(wu);
    try fillIota(wd);
    try cpu.swigluResidual(cpu_out, x, wn, wg, wu, wd, 1e-6, gpa);
    try swigluResidual(&gpu, gpu_out, x, wn, wg, wu, wd, 1e-6, gpa);
    try compare.expectClose(try cpu_out.f32s(), try gpu_out.f32s(), 2e-4, 2e-4);
}

test "forced Apple selection fails on a non-Apple build" {
    if (have_apple) return error.SkipZigTest;
    try std.testing.expectError(error.AppleUnavailable, Gpu.init());
}

/// Stage 8: fp16 Metal kernels are not retained. Callers must keep f32 or fail loudly.
pub fn matmulF16(gpu: *Gpu, c_out: Tensor, a: Tensor, b: Tensor) Error!void {
    _ = gpu;
    _ = c_out;
    _ = a;
    _ = b;
    return error.Unsupported;
}

pub fn matvecF16(gpu: *Gpu, y: Tensor, a: Tensor, x: Tensor) Error!void {
    _ = gpu;
    _ = y;
    _ = a;
    _ = x;
    return error.Unsupported;
}

test "fp16 Metal matmul/matvec are Unsupported" {
    if (gpu_mod.skipAppleGpuTests()) return error.SkipZigTest;
    var gpu = try Gpu.init();
    defer gpu.deinit();
    const gpa = std.testing.allocator;
    var a = try Tensor.alloc(gpa, .f32, &.{ 2, 2 });
    defer a.deinit();
    var b = try Tensor.alloc(gpa, .f32, &.{ 2, 2 });
    defer b.deinit();
    var c = try Tensor.alloc(gpa, .f32, &.{ 2, 2 });
    defer c.deinit();
    try std.testing.expectError(error.Unsupported, matmulF16(&gpu, c, a, b));
    var x = try Tensor.alloc(gpa, .f32, &.{2});
    defer x.deinit();
    var y = try Tensor.alloc(gpa, .f32, &.{2});
    defer y.deinit();
    try std.testing.expectError(error.Unsupported, matvecF16(&gpu, y, a, x));
}

test "Metal attention matches CPU at kv_len 96 (Stage 8 raised cap)" {
    if (gpu_mod.skipAppleGpuTests()) return error.SkipZigTest;
    var gpu = try Gpu.init();
    defer gpu.deinit();
    const gpa = std.testing.allocator;
    const kv_len: usize = 96;
    try std.testing.expect(kv_len <= max_attention_kv);
    var q = try Tensor.alloc(gpa, .f32, &.{ 2, 1, 4 });
    defer q.deinit();
    var k = try Tensor.alloc(gpa, .f32, &.{ 1, kv_len, 4 });
    defer k.deinit();
    var v = try Tensor.alloc(gpa, .f32, &.{ 1, kv_len, 4 });
    defer v.deinit();
    var acpu = try Tensor.alloc(gpa, .f32, &.{ 2, 1, 4 });
    defer acpu.deinit();
    var agpu = try Tensor.alloc(gpa, .f32, &.{ 2, 1, 4 });
    defer agpu.deinit();
    try fillIota(q);
    try fillIota(k);
    try fillIota(v);
    var scores: [96]f32 = undefined;
    try cpu.attentionInto(acpu, q, k, v, kv_len, kv_len, &scores);
    try attention(&gpu, agpu, q, k, v, kv_len, kv_len);
    try compare.expectClose(try acpu.f32s(), try agpu.f32s(), 2e-4, 2e-4);
}
