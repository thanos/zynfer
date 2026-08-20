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

    pub fn name(self: MatmulPath) []const u8 {
        return switch (self) {
            .naive => "matmul_f32",
            .simdgroup => "matmul_f32_simdgroup",
        };
    }
};

/// Minimum M*N*K before simdgroup is preferred over naive.
/// Below this, launch+sync dominate and naive is kept as the measured fallback.
pub const simdgroup_min_flops: usize = 64 * 64 * 64;

/// Force matmul path via `ZYNFER_MATMUL_PATH=naive|simdgroup`. Invalid values are ignored.
pub fn forcedMatmulPath() ?MatmulPath {
    if (comptime !have_apple) return null;
    const raw = std.c.getenv("ZYNFER_MATMUL_PATH") orelse return null;
    const v = std.mem.span(raw);
    if (std.mem.eql(u8, v, "naive")) return .naive;
    if (std.mem.eql(u8, v, "simdgroup")) return .simdgroup;
    return null;
}

pub fn selectMatmulPath(gpu: *const Gpu, m: usize, n: usize, k: usize) MatmulPath {
    if (forcedMatmulPath()) |forced| {
        if (forced == .simdgroup and !gpu.features.simdgroup_matrix_available) return .naive;
        return forced;
    }
    if (!gpu.features.simdgroup_matrix_available) return .naive;
    if (m < 8 or n < 8 or k < 8) return .naive;
    if (m * n * k < simdgroup_min_flops) return .naive;
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
            const tg: u32 = 16;
            try launchBufs(gpu, "matmul_f32", n, m, 1, tg, tg, 1, &.{ ab, bb, cb }, std.mem.asBytes(&params));
        },
        .simdgroup => {
            if (!gpu.features.simdgroup_matrix_available) return error.Unsupported;
            const tile: u32 = 8;
            const tg_x = (n + tile - 1) / tile;
            const tg_y = (m + tile - 1) / tile;
            // One simdgroup: A/B/C staging = 3 * 64 * 4 bytes.
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
    }
    try download(cb, c);
}

pub fn matvec(gpu: *Gpu, y: Tensor, a: Tensor, x: Tensor) Error!void {
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

/// y = (int8_weights * per_row_scale) @ x. Weights are row-major int8 [m,k].
pub fn matvecQ8(
    gpu: *Gpu,
    y: Tensor,
    w_q8: []const i8,
    scale: []const f32,
    x: Tensor,
) Error!void {
    if (y.rank != 1 or x.rank != 1) return error.InvalidShape;
    const m = y.shape[0];
    const k = x.shape[0];
    if (w_q8.len != m * k or scale.len != m) return error.ShapeMismatch;

    var wb = try gpu.allocShared(w_q8.len);
    defer wb.deinit();
    @memcpy(wb.bytes[0..w_q8.len], @as([*]const u8, @ptrCast(w_q8.ptr))[0..w_q8.len]);

    var sb = try gpu.allocShared(scale.len * @sizeOf(f32));
    defer sb.deinit();
    @memcpy(sb.f32s()[0..scale.len], scale);

    var xb = try upload(gpu, x);
    defer xb.deinit();
    var yb = try gpu.allocShared(y.data.len);
    defer yb.deinit();
    const params = MatmulParams{ .m = @intCast(m), .n = 1, .k = @intCast(k) };
    const tg = gpu.threadgroup1d();
    try launchBufs(gpu, "matvec_q8_f32", @intCast(m), 1, 1, tg, 1, 1, &.{ wb, sb, xb, yb }, std.mem.asBytes(&params));
    try download(yb, y);
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
pub const max_attention_kv: usize = 64;

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
        try cpu.matvecQ8(try y_cpu.f32s(), q, scale, try xq.f32s(), mq, kq);
        try matvecQ8(&gpu, y_gpu, q, scale, xq);
        try compare.expectClose(try y_cpu.f32s(), try y_gpu.f32s(), 2e-4, 2e-4);
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
