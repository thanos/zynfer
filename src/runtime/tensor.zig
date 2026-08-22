//! Contiguous host tensor used by the CPU reference and as a CPU-visible
//! staging view for accelerator tests.
//!
//! This is not NumPy. Rank is capped. Views with arbitrary strides are not
//! implemented; kernels that need a layout must receive a contiguous buffer
//! or an explicit stride pair they understand.

const std = @import("std");
const DType = @import("dtype.zig").DType;
const byteSize = @import("dtype.zig").byteSize;

pub const max_rank: usize = 4;

pub const TensorError = error{
    InvalidShape,
    Overflow,
    RankTooHigh,
    DTypeMismatch,
    ShapeMismatch,
    NotContiguous,
    NotF32,
    OutOfMemory,
};

pub const Tensor = struct {
    dtype: DType,
    rank: u8,
    shape: [max_rank]usize,
    /// Element strides in row-major contiguous layout.
    strides: [max_rank]usize,
    data: []u8,
    allocator: std.mem.Allocator,
    /// False for `viewAs` results. Only the owning tensor may `deinit`.
    owns: bool,

    pub fn alloc(allocator: std.mem.Allocator, dtype: DType, shape: []const usize) TensorError!Tensor {
        if (shape.len == 0 or shape.len > max_rank) return error.RankTooHigh;
        var t = Tensor{
            .dtype = dtype,
            .rank = @intCast(shape.len),
            .shape = .{ 0, 0, 0, 0 },
            .strides = .{ 0, 0, 0, 0 },
            .data = &.{},
            .allocator = allocator,
            .owns = true,
        };
        for (shape, 0..) |extent, i| {
            if (extent == 0) return error.InvalidShape;
            t.shape[i] = extent;
        }
        fillContiguousStrides(&t);
        const n = t.numel() catch return error.Overflow;
        const bytes = byteSize(dtype, n) catch return error.Overflow;
        t.data = allocator.alloc(u8, bytes) catch return error.OutOfMemory;
        @memset(t.data, 0);
        return t;
    }

    pub fn deinit(self: *Tensor) void {
        if (self.owns and self.data.len != 0) self.allocator.free(self.data);
        self.data = &.{};
        self.owns = false;
    }

    /// Packed row-major view of a prefix of `self`. Do not `deinit` the result.
    pub fn viewAs(self: Tensor, shape: []const usize) TensorError!Tensor {
        if (shape.len == 0 or shape.len > max_rank) return error.RankTooHigh;
        var t = self;
        t.owns = false;
        t.rank = @intCast(shape.len);
        t.shape = .{ 0, 0, 0, 0 };
        t.strides = .{ 0, 0, 0, 0 };
        for (shape, 0..) |extent, i| {
            if (extent == 0) return error.InvalidShape;
            t.shape[i] = extent;
        }
        fillContiguousStrides(&t);
        const n = t.numel() catch return error.Overflow;
        const bytes = byteSize(t.dtype, n) catch return error.Overflow;
        if (bytes > self.data.len) return error.InvalidShape;
        t.data = self.data[0..bytes];
        return t;
    }

    /// View the last row of a 2-D tensor `[rows, cols]` as `[1, cols]`.
    pub fn viewLastRow(self: Tensor) TensorError!Tensor {
        if (self.rank != 2) return error.InvalidShape;
        const rows = self.shape[0];
        const cols = self.shape[1];
        if (rows == 0) return error.InvalidShape;
        const row_elems = cols;
        const row_bytes = byteSize(self.dtype, row_elems) catch return error.Overflow;
        const offset = (rows - 1) * row_bytes;
        if (offset + row_bytes > self.data.len) return error.InvalidShape;
        var t = self;
        t.owns = false;
        t.rank = 2;
        t.shape = .{ 1, cols, 0, 0 };
        t.data = self.data[offset..][0..row_bytes];
        fillContiguousStrides(&t);
        return t;
    }

    pub fn numel(self: Tensor) error{Overflow}!usize {
        var n: usize = 1;
        var i: u8 = 0;
        while (i < self.rank) : (i += 1) {
            n = try std.math.mul(usize, n, self.shape[i]);
        }
        return n;
    }

    pub fn dim(self: Tensor, axis: usize) TensorError!usize {
        if (axis >= self.rank) return error.InvalidShape;
        return self.shape[axis];
    }

    pub fn isContiguous(self: Tensor) bool {
        var expected: usize = 1;
        var i: usize = self.rank;
        while (i > 0) {
            i -= 1;
            if (self.strides[i] != expected) return false;
            expected *= self.shape[i];
        }
        return true;
    }

    pub fn f32s(self: Tensor) TensorError![]f32 {
        if (self.dtype != .f32) return error.NotF32;
        if (!self.isContiguous()) return error.NotContiguous;
        const n = self.numel() catch return error.Overflow;
        return @as([*]f32, @ptrCast(@alignCast(self.data.ptr)))[0..n];
    }

    pub fn f32sConst(self: Tensor) TensorError![]const f32 {
        return self.f32s();
    }

    pub fn requireF32Contiguous(self: Tensor, expected: []const usize) TensorError![]f32 {
        if (self.dtype != .f32) return error.NotF32;
        if (!self.isContiguous()) return error.NotContiguous;
        if (self.rank != expected.len) return error.ShapeMismatch;
        for (expected, 0..) |d, i| {
            if (self.shape[i] != d) return error.ShapeMismatch;
        }
        return self.f32s();
    }

    pub fn fillF32(self: Tensor, value: f32) TensorError!void {
        const xs = try self.f32s();
        @memset(xs, value);
    }
};

fn fillContiguousStrides(t: *Tensor) void {
    var acc: usize = 1;
    var i: usize = t.rank;
    while (i > 0) {
        i -= 1;
        t.strides[i] = acc;
        acc *= t.shape[i];
    }
}

test "contiguous f32 tensor round-trip" {
    var t = try Tensor.alloc(std.testing.allocator, .f32, &.{ 2, 3 });
    defer t.deinit();
    try std.testing.expectEqual(@as(usize, 6), try t.numel());
    try std.testing.expect(t.isContiguous());
    const xs = try t.f32s();
    xs[5] = 1.5;
    try std.testing.expectEqual(@as(f32, 1.5), xs[5]);
}

test "zero dimension is rejected" {
    try std.testing.expectError(error.InvalidShape, Tensor.alloc(std.testing.allocator, .f32, &.{ 2, 0 }));
}

test "viewAs packed prefix does not own storage" {
    var t = try Tensor.alloc(std.testing.allocator, .f32, &.{ 4, 2 });
    defer t.deinit();
    try t.fillF32(3);
    const v = try t.viewAs(&.{ 1, 2 });
    try std.testing.expect(!v.owns);
    try std.testing.expectEqual(@as(usize, 2), try v.numel());
    try std.testing.expectEqual(@as(f32, 3), (try v.f32s())[0]);
}

test "viewLastRow selects final row not prefix" {
    var t = try Tensor.alloc(std.testing.allocator, .f32, &.{ 3, 2 });
    defer t.deinit();
    const xs = try t.f32s();
    xs[0] = 1;
    xs[1] = 2;
    xs[2] = 3;
    xs[3] = 4;
    xs[4] = 5;
    xs[5] = 6;
    const last = try t.viewLastRow();
    const ys = try last.f32s();
    try std.testing.expectEqual(@as(f32, 5), ys[0]);
    try std.testing.expectEqual(@as(f32, 6), ys[1]);
}
