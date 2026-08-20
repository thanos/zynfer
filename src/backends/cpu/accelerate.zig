//! Optional Accelerate (vDSP) matmul for large CPU shapes on Apple hosts.
//!
//! The scalar CPU oracle in `ops.zig` remains the correctness reference.
//! This path is size-gated and differentially tested. It is not used by the
//! Metal schedule; model code must not depend on Accelerate.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const Tensor = @import("../../runtime/tensor.zig").Tensor;
const TensorError = @import("../../runtime/tensor.zig").TensorError;
const compare = @import("../../runtime/compare.zig");
const cpu = @import("ops.zig");

pub const Error = TensorError || error{Unsupported};

pub const have_accelerate = build_options.have_apple and builtin.os.tag == .macos;

/// Prefer Accelerate when M*N*K is at least this many FMAs.
pub const min_flops: usize = 64 * 64 * 64;

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

pub fn isUseful(m: usize, n: usize, k: usize) bool {
    if (!have_accelerate) return false;
    return m * n * k >= min_flops;
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
    c.vDSP_mmul(as.ptr, 1, bs.ptr, 1, cs.ptr, 1, m, n, k);
}

/// Selects Accelerate for large shapes; otherwise the scalar oracle.
pub fn matmulAuto(c_out: Tensor, a: Tensor, b: Tensor) Error!void {
    if (isUseful(a.shape[0], b.shape[1], a.shape[1])) {
        try matmul(c_out, a, b);
    } else {
        try cpu.matmul(c_out, a, b);
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
