//! Differential comparison helpers for CPU vs accelerated tensors.

const std = @import("std");

pub const Diff = struct {
    max_abs: f32,
    max_rel: f32,
    rms: f32,
    failing_index: usize,
    expected: f32,
    actual: f32,
};

pub fn diffF32(expected: []const f32, actual: []const f32) Diff {
    std.debug.assert(expected.len == actual.len);
    var out = Diff{
        .max_abs = 0,
        .max_rel = 0,
        .rms = 0,
        .failing_index = 0,
        .expected = 0,
        .actual = 0,
    };
    if (expected.len == 0) return out;
    var sum_sq: f64 = 0;
    for (expected, actual, 0..) |e, a, i| {
        const abs = @abs(a - e);
        const denom = @max(@abs(e), 1e-8);
        const rel = abs / denom;
        sum_sq += @as(f64, abs) * @as(f64, abs);
        if (abs >= out.max_abs) {
            out.max_abs = abs;
            out.max_rel = rel;
            out.failing_index = i;
            out.expected = e;
            out.actual = a;
        }
    }
    out.rms = @floatCast(@sqrt(sum_sq / @as(f64, @floatFromInt(expected.len))));
    return out;
}

pub fn matches(expected: []const f32, actual: []const f32, atol: f32, rtol: f32) bool {
    if (expected.len != actual.len) return false;
    const d = diffF32(expected, actual);
    return d.max_abs <= atol + rtol * @abs(d.expected);
}

pub fn expectClose(expected: []const f32, actual: []const f32, atol: f32, rtol: f32) !void {
    if (expected.len != actual.len) return error.LengthMismatch;
    const d = diffF32(expected, actual);
    if (d.max_abs > atol + rtol * @abs(d.expected)) {
        std.debug.print(
            "tensor mismatch: max_abs={d:.6} max_rel={d:.6} rms={d:.6} idx={d} expected={d:.8} actual={d:.8}\n",
            .{ d.max_abs, d.max_rel, d.rms, d.failing_index, d.expected, d.actual },
        );
        return error.NumericalMismatch;
    }
}

test "identical slices compare equal" {
    const a = [_]f32{ 1, 2, 3 };
    try expectClose(&a, &a, 0, 0);
}
