//! Optional Accelerate (vDSP) matmul/matvec for large CPU shapes on Apple hosts.
//!
//! The scalar CPU oracle in `ops.zig` remains the correctness reference.
//! Paths are size-gated and differentially tested. Not used by the Metal
//! schedule; model code must not depend on Accelerate.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const Tensor = @import("../../runtime/tensor.zig").Tensor;
const TensorError = @import("../../runtime/tensor.zig").TensorError;
const compare = @import("../../runtime/compare.zig");
const cpu = @import("ops.zig");

pub const Error = TensorError || error{Unsupported};

pub const have_accelerate = build_options.have_apple and builtin.os.tag == .macos;

/// Prefer Accelerate matmul when M*N*K is at least this many FMAs.
pub const matmul_min_flops: usize = 64 * 64 * 64;
/// Prefer Accelerate matvec when M*K is at least this many FMAs.
/// Below this, scalar often wins once call overhead is counted.
pub const matvec_min_flops: usize = 256 * 256;

/// @deprecated use matmul_min_flops
pub const min_flops: usize = matmul_min_flops;

pub var last_path: []const u8 = "unset";

const c = if (have_accelerate) struct {
    pub extern "c" fn vDSP_mmul(
        __A: [*c]const f32,
        __IA: isize,
        __B: [*c]const f32,
        __IB: isize,
        __C: [*c]f32,
        __IC: isize,
        __M: usize,
        __N: usize,
        __P: usize,
    ) void;
} else struct {};

pub fn isUsefulMatmul(m: usize, n: usize, k: usize) bool {
    if (!have_accelerate) return false;
    return m * n * k >= matmul_min_flops;
}

pub fn isUsefulMatvec(m: usize, k: usize) bool {
    if (!have_accelerate) return false;
    return m * k >= matvec_min_flops;
}

pub fn isUseful(m: usize, n: usize, k: usize) bool {
    return isUsefulMatmul(m, n, k);
}

/// C[M,N] = A[M,K] * B[K,N] via vDSP when linked; otherwise error.Unsupported.
pub fn matmul(c_out: Tensor, a: Tensor, b: Tensor) Error!void {
    if (!have_accelerate) return error.Unsupported;
    if (a.rank != 2 or b.rank != 2 or c_out.rank != 2) return error.InvalidShape;
    const m = a.shape[0];
    const k = a.shape[1];
    const n = b.shape[1];
    if (b.shape[0] != k or c_out.shape[0] != m or c_out.shape[1] != n) return error.ShapeMismatch;
    const as = try a.f32s();
    const bs = try b.f32s();
    const cs = try c_out.f32s();
    last_path = "accelerate_vDSP_mmul";
    c.vDSP_mmul(as.ptr, 1, bs.ptr, 1, cs.ptr, 1, m, n, k);
}

/// y[M] = A[M,K] * x[K] via vDSP_mmul with N=1.
pub fn matvec(y: Tensor, a: Tensor, x: Tensor) Error!void {
    if (!have_accelerate) return error.Unsupported;
    if (a.rank != 2 or x.rank != 1 or y.rank != 1) return error.InvalidShape;
    const m = a.shape[0];
    const k = a.shape[1];
    if (x.shape[0] != k or y.shape[0] != m) return error.ShapeMismatch;
    const as = try a.f32s();
    const xs = try x.f32s();
    const ys = try y.f32s();
    last_path = "accelerate_vDSP_matvec";
    c.vDSP_mmul(as.ptr, 1, xs.ptr, 1, ys.ptr, 1, m, 1, k);
}

/// Selects Accelerate for large shapes; otherwise the scalar oracle.
pub fn matmulAuto(c_out: Tensor, a: Tensor, b: Tensor) Error!void {
    if (isUsefulMatmul(a.shape[0], b.shape[1], a.shape[1])) {
        try matmul(c_out, a, b);
    } else {
        last_path = "cpu_scalar_matmul";
        try cpu.matmul(c_out, a, b);
    }
}

pub fn matvecAuto(y: Tensor, a: Tensor, x: Tensor) Error!void {
    if (isUsefulMatvec(a.shape[0], a.shape[1])) {
        try matvec(y, a, x);
    } else {
        last_path = "cpu_scalar_matvec";
        try cpu.matvec(y, a, x);
    }
}

test "Accelerate matmul matches CPU oracle when available" {
    if (comptime !have_accelerate) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const m: usize = 32;
    const k: usize = 48;
    const n: usize = 40;
    var a = try Tensor.alloc(gpa, .f32, &.{ m, k });
    defer a.deinit();
    var b = try Tensor.alloc(gpa, .f32, &.{ k, n });
    defer b.deinit();
    var c_ref = try Tensor.alloc(gpa, .f32, &.{ m, n });
    defer c_ref.deinit();
    var c_acc = try Tensor.alloc(gpa, .f32, &.{ m, n });
    defer c_acc.deinit();
    const as = try a.f32s();
    const bs = try b.f32s();
    for (as, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 17)) * 0.01;
    for (bs, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 13)) * 0.02;
    try cpu.matmul(c_ref, a, b);
    try matmul(c_acc, a, b);
    try compare.expectClose(try c_ref.f32s(), try c_acc.f32s(), 1e-4, 1e-4);
}

test "Accelerate matvec matches CPU oracle when available" {
    if (comptime !have_accelerate) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    var a = try Tensor.alloc(gpa, .f32, &.{ 64, 48 });
    defer a.deinit();
    var x = try Tensor.alloc(gpa, .f32, &.{48});
    defer x.deinit();
    var y_ref = try Tensor.alloc(gpa, .f32, &.{64});
    defer y_ref.deinit();
    var y_acc = try Tensor.alloc(gpa, .f32, &.{64});
    defer y_acc.deinit();
    for (try a.f32s(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 11)) * 0.03;
    for (try x.f32s(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) * 0.04;
    try cpu.matvec(y_ref, a, x);
    try matvec(y_acc, a, x);
    try compare.expectClose(try y_ref.f32s(), try y_acc.f32s(), 1e-4, 1e-4);
}

test "Accelerate is unavailable on non-Apple builds" {
    if (comptime have_accelerate) return error.SkipZigTest;
    try std.testing.expect(!isUsefulMatmul(256, 256, 256));
    try std.testing.expect(!isUsefulMatvec(512, 512));
}
