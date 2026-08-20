const std = @import("std");
const zynfer = @import("zynfer");

test "CPU and Apple numerical suites are wired" {
    _ = zynfer.cpu.ops;
    _ = zynfer.apple.ops;
    _ = zynfer.apple.block;
    _ = zynfer.tiny_block;
    _ = zynfer.compare;
}

test "unknown backend name does not fall back" {
    try std.testing.expectError(error.UnknownBackend, zynfer.backend.parseBackendKind("cuda"));
}

test "CPU rmsNorm is the oracle for a tiny row" {
    var x = try zynfer.Tensor.alloc(std.testing.allocator, .f32, &.{ 1, 4 });
    defer x.deinit();
    var w = try zynfer.Tensor.alloc(std.testing.allocator, .f32, &.{4});
    defer w.deinit();
    var y = try zynfer.Tensor.alloc(std.testing.allocator, .f32, &.{ 1, 4 });
    defer y.deinit();
    (try x.f32s())[0] = 1;
    (try x.f32s())[1] = -1;
    (try x.f32s())[2] = 1;
    (try x.f32s())[3] = -1;
    try w.fillF32(1);
    try zynfer.cpu.ops.rmsNorm(y, x, w, 1e-6);
    const ys = try y.f32s();
    try std.testing.expectApproxEqAbs(@as(f32, 1), ys[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1), ys[1], 1e-5);
}
