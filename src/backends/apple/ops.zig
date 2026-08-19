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

fn launch1d(gpu: *Gpu, kernel: [:0]const u8, n: u32, bufs: []const *gpu_mod.MtlBuffer) Error!void {
    const tg = gpu.threadgroup1d();
    try gpu.launch(kernel, n, 1, 1, tg, 1, 1, bufs, std.mem.asBytes(&n));
}

pub fn add(gpu: *Gpu, dst: Tensor, a: Tensor, b: Tensor) Error!void {
    const n: u32 = @intCast(try a.numel());
    var ab = try upload(gpu, a);
    defer ab.deinit();
    var bb = try upload(gpu, b);
    defer bb.deinit();
    var cb = try gpu.allocShared(dst.data.len);
    defer cb.deinit();
    try launch1d(gpu, "add_f32", n, &.{ ab.handle, bb.handle, cb.handle });
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
    try launch1d(gpu, "mul_f32", n, &.{ ab.handle, bb.handle, cb.handle });
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
    try launch1d(gpu, "silu_mul_f32", n, &.{ gb.handle, ub.handle, ob.handle });
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
    try gpu.launch("rmsnorm_f32", tg, rows, 1, tg, 1, 1, &.{ xb.handle, wb.handle, yb.handle }, std.mem.asBytes(&params));
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
    try gpu.launch("softmax_f32", tg, rows, 1, tg, 1, 1, &.{ xb.handle, yb.handle }, std.mem.asBytes(&params));
    try download(yb, dst);
}

const MatmulParams = extern struct {
    m: u32,
    n: u32,
    k: u32,
};

pub fn matmul(gpu: *Gpu, c: Tensor, a: Tensor, b: Tensor) Error!void {
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
    const tg: u32 = 16;
    try gpu.launch("matmul_f32", n, m, 1, tg, tg, 1, &.{ ab.handle, bb.handle, cb.handle }, std.mem.asBytes(&params));
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
    try gpu.launch("matvec_f32", m, 1, 1, tg, 1, 1, &.{ ab.handle, xb.handle, yb.handle }, std.mem.asBytes(&params));
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
    try gpu.launch("rope_f32", pairs, 1, 1, tg, 1, 1, &.{xb.handle}, std.mem.asBytes(&params));
    try download(xb, x);
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
    if (!have_apple) return error.SkipZigTest;
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

    var r = try Tensor.alloc(gpa, .f32, &.{ 2, 2, 4 });
    defer r.deinit();
    var rc = try Tensor.alloc(gpa, .f32, &.{ 2, 2, 4 });
    defer rc.deinit();
    try fillIota(r);
    @memcpy(try rc.f32s(), try r.f32s());
    try cpu.rope(rc, 3, 10_000);
    try rope(&gpu, r, 3, 10_000);
    try compare.expectClose(try rc.f32s(), try r.f32s(), 1e-4, 1e-4);
}

test "tiny SwiGLU residual matches CPU" {
    if (!have_apple) return error.SkipZigTest;
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
