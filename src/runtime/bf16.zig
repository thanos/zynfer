//! BF16 ↔ F32 conversion for checkpoint weights stored as BF16 in `.zynfer`.

const std = @import("std");

/// Widen a little-endian BF16 bit pattern to f32.
pub fn toF32(w: u16) f32 {
    const bits: u32 = @as(u32, w) << 16;
    return @bitCast(bits);
}

/// Decode little-endian BF16 bytes into f32 (one value per 2 input bytes).
pub fn decodeIntoF32(dst: []f32, src: []const u8) void {
    std.debug.assert(dst.len * 2 <= src.len);
    var i: usize = 0;
    while (i < dst.len) : (i += 1) {
        const w = std.mem.readInt(u16, src[i * 2 ..][0..2], .little);
        dst[i] = toF32(w);
    }
}

test "bf16 decode round-trip known values" {
    var buf: [2]u8 = undefined;
    std.mem.writeInt(u16, &buf, 0x3f80, .little); // 1.0 in bf16
    var out: [1]f32 = undefined;
    decodeIntoF32(&out, &buf);
    try std.testing.expect(@abs(out[0] - 1.0) < 1e-3);
}
