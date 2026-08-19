//! Scalar and tensor element types used by the inference core.
//!
//! Model code should depend on this, not on Metal or HIP types.

const std = @import("std");

pub const DType = enum(u8) {
    f32,
    f16,
    bf16,

    pub fn sizeOf(self: DType) usize {
        return switch (self) {
            .f32 => 4,
            .f16, .bf16 => 2,
        };
    }

    pub fn name(self: DType) []const u8 {
        return switch (self) {
            .f32 => "f32",
            .f16 => "f16",
            .bf16 => "bf16",
        };
    }
};

pub fn byteSize(dtype: DType, n_elem: usize) error{Overflow}!usize {
    return std.math.mul(usize, n_elem, dtype.sizeOf());
}

test "byteSize overflow is reported" {
    try std.testing.expectError(error.Overflow, byteSize(.f32, std.math.maxInt(usize)));
}
