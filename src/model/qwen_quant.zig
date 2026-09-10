//! Stage M5 — per-row symmetric int8 weight quantization for Qwen projections.
//!
//! Scheme (chosen for Apple Metal + existing `matvec_q8_f32` / `matmul_aq8_f32`):
//! - **Layout:** HF / pack row-major `[out_features, in_features]`
//! - **Scale:** one f32 per output row: `scale[r] = max_abs(row) / 127` (symmetric, no ZP)
//! - **Quant:** `q = clamp(round(w / scale), -127, 127)` as i8
//! - **Group:** full row (= per-output-channel); no sub-row blocks in M5 v1
//! - **Tail:** none (exact `out*in` elements; row length = `in`)
//! - **Alignment:** i8 packed densely; scales contiguous f32
//! - **Dequant:** fused in GEMV/GEMM — never materialize full f32 weights on GPU
//!
//! Host Qwen tensors are stored `[in, out]` (transposed at load). Pack helpers
//! transpose to `[out, in]` before `packRowQ8`.

const std = @import("std");
const cpu = @import("../backends/cpu/ops.zig");
const Tensor = @import("../runtime/tensor.zig").Tensor;
const compare = @import("../runtime/compare.zig");
const bf16 = @import("../runtime/bf16.zig");

pub const Error = cpu.OpsError || std.mem.Allocator.Error || @import("../runtime/tensor.zig").TensorError;

pub const PackedQ8 = struct {
    q: []i8,
    scale: []f32,
};

/// Transpose `[rows, cols]` → `[cols, rows]` (f32).
pub fn transpose2d(dst: []f32, src: []const f32, rows: usize, cols: usize) void {
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        var c: usize = 0;
        while (c < cols) : (c += 1) {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

/// Pack host `[in, out]` f32 into row-quantized `[out, in]` i8 + scales.
pub fn packInOutToQ8(
    allocator: std.mem.Allocator,
    host_in_out: Tensor,
    out_dim: usize,
    in_dim: usize,
) Error!PackedQ8 {
    if (host_in_out.rank != 2) return error.InvalidShape;
    if (host_in_out.shape[0] != in_dim or host_in_out.shape[1] != out_dim) return error.ShapeMismatch;
    const src = try host_in_out.f32s();
    const tmp = try allocator.alloc(f32, out_dim * in_dim);
    defer allocator.free(tmp);
    transpose2d(tmp, src, in_dim, out_dim);
    return packRowsQ8(allocator, tmp, out_dim, in_dim);
}

/// Pack host already in HF `[out, in]` layout (e.g. tied embed / lm_head).
pub fn packOutInToQ8(
    allocator: std.mem.Allocator,
    host_out_in: Tensor,
    out_dim: usize,
    in_dim: usize,
) Error!PackedQ8 {
    if (host_out_in.rank != 2) return error.InvalidShape;
    if (host_out_in.shape[0] != out_dim or host_out_in.shape[1] != in_dim) return error.ShapeMismatch;
    const src = try host_out_in.f32s();
    return packRowsQ8(allocator, src, out_dim, in_dim);
}

/// Pack LE bf16 `[out, in]` bytes → row-quantized i8 (one f32 row scratch).
pub fn packBf16OutInToQ8(
    allocator: std.mem.Allocator,
    bf16_le: []const u8,
    out_dim: usize,
    in_dim: usize,
) Error!PackedQ8 {
    if (bf16_le.len != out_dim * in_dim * 2) return error.ShapeMismatch;
    const q = try allocator.alloc(i8, out_dim * in_dim);
    errdefer allocator.free(q);
    const scale = try allocator.alloc(f32, out_dim);
    errdefer allocator.free(scale);
    const row = try allocator.alloc(f32, in_dim);
    defer allocator.free(row);
    var r: usize = 0;
    while (r < out_dim) : (r += 1) {
        const off = r * in_dim * 2;
        bf16.decodeIntoF32(row, bf16_le[off..][0 .. in_dim * 2]);
        try cpu.packRowQ8(row, 1, in_dim, q[r * in_dim ..][0..in_dim], scale[r .. r + 1]);
    }
    return .{ .q = q, .scale = scale };
}

fn packRowsQ8(
    allocator: std.mem.Allocator,
    rows_out_in: []const f32,
    out_dim: usize,
    in_dim: usize,
) Error!PackedQ8 {
    const q = try allocator.alloc(i8, out_dim * in_dim);
    errdefer allocator.free(q);
    const scale = try allocator.alloc(f32, out_dim);
    errdefer allocator.free(scale);
    try cpu.packRowQ8(rows_out_in, out_dim, in_dim, q, scale);
    return .{ .q = q, .scale = scale };
}

/// CPU decode: dequant `[out,in]` i8 → f32 `[out,in]` (for artifact/converter checks).
pub fn dequantRowQ8(dst: []f32, q: []const i8, scale: []const f32, out_dim: usize, in_dim: usize) Error!void {
    if (dst.len != out_dim * in_dim or q.len != out_dim * in_dim or scale.len != out_dim) return error.ShapeMismatch;
    var r: usize = 0;
    while (r < out_dim) : (r += 1) {
        const s = scale[r];
        var c: usize = 0;
        while (c < in_dim) : (c += 1) {
            dst[r * in_dim + c] = @as(f32, @floatFromInt(q[r * in_dim + c])) * s;
        }
    }
}

test "packInOutToQ8 round-trips within packing error" {
    const gpa = std.testing.allocator;
    var w = try Tensor.alloc(gpa, .f32, &.{ 4, 3 }); // [in=4, out=3]
    defer w.deinit();
    const s = try w.f32s();
    for (s, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.1 - 0.5;
    const packed_w = try packInOutToQ8(gpa, w, 3, 4);
    defer gpa.free(packed_w.q);
    defer gpa.free(packed_w.scale);
    const recon = try gpa.alloc(f32, 12);
    defer gpa.free(recon);
    try dequantRowQ8(recon, packed_w.q, packed_w.scale, 3, 4);
    // Compare to transposed original
    const expect = try gpa.alloc(f32, 12);
    defer gpa.free(expect);
    transpose2d(expect, s, 4, 3);
    try compare.expectClose(expect, recon, 2e-2, 2e-2);
}

test "packBf16OutInToQ8 matches f32 pack within packing error" {
    const gpa = std.testing.allocator;
    const out_dim: usize = 3;
    const in_dim: usize = 4;
    var f32_host = try Tensor.alloc(gpa, .f32, &.{ out_dim, in_dim });
    defer f32_host.deinit();
    const s = try f32_host.f32s();
    for (s, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.1 - 0.5;
    var bf16_bytes: [3 * 4 * 2]u8 = undefined;
    bf16.encodeFromF32(&bf16_bytes, s);
    const from_bf16 = try packBf16OutInToQ8(gpa, &bf16_bytes, out_dim, in_dim);
    defer gpa.free(from_bf16.q);
    defer gpa.free(from_bf16.scale);
    const from_f32 = try packOutInToQ8(gpa, f32_host, out_dim, in_dim);
    defer gpa.free(from_f32.q);
    defer gpa.free(from_f32.scale);
    // Scales should match closely; i8 may differ by 1 on rounding edges.
    try compare.expectClose(from_f32.scale, from_bf16.scale, 1e-3, 1e-3);
}
