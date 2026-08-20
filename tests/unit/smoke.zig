const std = @import("std");
const zynfer = @import("zynfer");

test "library exports runtime and backends" {
    _ = zynfer.hip;
    _ = zynfer.device;
    _ = zynfer.env;
    _ = zynfer.Device;
    _ = zynfer.cpu.ops;
    _ = zynfer.apple.ops;
    _ = zynfer.apple.block;
    _ = zynfer.backend;
    _ = zynfer.tiny_block;
    _ = zynfer.kv_cache;
    _ = zynfer.artifact;
    _ = zynfer.qwen3;
}

test "HIP error formatting has no stale state initially" {
    var buf: [64]u8 = undefined;
    const text = zynfer.hip.formatLastError(&buf);
    try std.testing.expectEqualStrings("no HIP error", text);
}

test "cString helper is available to tests" {
    const raw = "gfx1201\x00tail";
    try std.testing.expectEqualStrings("gfx1201", zynfer.util.cString(raw));
}
