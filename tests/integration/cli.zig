//! CLI integration tests. These spawn the installed `zynfer` binary.
//!
//! `zig build integration` sets `ZYNFER_BIN`. If the variable is missing the
//! tests look for `zig-out/bin/zynfer` and skip when it is not there.

const std = @import("std");
const zynfer = @import("zynfer");

fn zynferBin() ![]const u8 {
    if (std.process.Environ.getPosix(std.testing.environ, "ZYNFER_BIN")) |path| {
        if (path.len > 0) return path;
    }
    const fallback = "zig-out/bin/zynfer";
    std.Io.Dir.cwd().access(std.testing.io, fallback, .{}) catch return error.SkipZigTest;
    return fallback;
}

fn run(args: []const []const u8) !zynfer.util.CommandOutput {
    const bin = try zynferBin();
    var argv_buf: [10][]const u8 = undefined;
    if (args.len + 1 > argv_buf.len) return error.SkipZigTest;
    argv_buf[0] = bin;
    for (args, 0..) |arg, i| argv_buf[i + 1] = arg;
    const host = zynfer.util.Host{
        .gpa = std.testing.allocator,
        .io = std.testing.io,
    };
    return zynfer.util.runCommand(host, argv_buf[0 .. args.len + 1]) orelse error.SpawnFailed;
}

fn expectExited(out: zynfer.util.CommandOutput, code: u8) !void {
    switch (out.term) {
        .exited => |got| try std.testing.expectEqual(code, got),
        else => return error.UnexpectedTerm,
    }
}

test "help prints usage and exits 0" {
    var out = try run(&.{"help"});
    defer out.deinit(std.testing.allocator);
    try expectExited(out, 0);
    try std.testing.expect(std.mem.indexOf(u8, out.stdout, "Usage:") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.stdout, "caps") != null);
}

test "env reports Zig version and compiled backends" {
    var out = try run(&.{"env"});
    defer out.deinit(std.testing.allocator);
    try expectExited(out, 0);
    try std.testing.expect(std.mem.indexOf(u8, out.stdout, "Zig version:") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.stdout, "CPU reference:") != null);
}

test "caps --backend cpu is the oracle path" {
    var out = try run(&.{ "caps", "--backend", "cpu" });
    defer out.deinit(std.testing.allocator);
    try expectExited(out, 0);
    try std.testing.expect(std.mem.indexOf(u8, out.stdout, "requested backend: cpu") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.stdout, "device architecture: generic-cpu") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.stdout, "disabled paths") != null);
}

test "unknown backend name does not fall back" {
    var out = try run(&.{ "caps", "--backend", "cuda" });
    defer out.deinit(std.testing.allocator);
    try expectExited(out, 2);
}

test "unknown command exits 2" {
    var out = try run(&.{"not-a-command"});
    defer out.deinit(std.testing.allocator);
    try expectExited(out, 2);
}

test "backends lists cpu as buildable" {
    var out = try run(&.{"backends"});
    defer out.deinit(std.testing.allocator);
    try expectExited(out, 0);
    try std.testing.expect(std.mem.indexOf(u8, out.stdout, "cpu") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.stdout, "buildable") != null);
}

test "cpu ops-bench emits json with cpu_ns" {
    var out = try run(&.{ "ops-bench", "--backend", "cpu" });
    defer out.deinit(std.testing.allocator);
    try expectExited(out, 0);
    try std.testing.expect(std.mem.indexOf(u8, out.stdout, "\"backend\":\"cpu\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.stdout, "\"cpu_ns\":") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.stdout, "add_f32_4096") != null);
}

test "stage7 prints retain/reject ledger" {
    var out = try run(&.{"stage7"});
    defer out.deinit(std.testing.allocator);
    try expectExited(out, 0);
    try std.testing.expect(std.mem.indexOf(u8, out.stdout, "Stage 7 decisions") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.stdout, "REJECT") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.stdout, "SME") != null);
}

test "stage8 prints hardening ledger" {
    var out = try run(&.{"stage8"});
    defer out.deinit(std.testing.allocator);
    try expectExited(out, 0);
    try std.testing.expect(std.mem.indexOf(u8, out.stdout, "Stage 8") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.stdout, "256") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.stdout, "REJECT") != null);
}

test "stage10 prints artifact ledger" {
    var out = try run(&.{"stage10"});
    defer out.deinit(std.testing.allocator);
    try expectExited(out, 0);
    try std.testing.expect(std.mem.indexOf(u8, out.stdout, "Stage 10") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.stdout, "ZYNF") != null);
}

test "artifact-compile then inspect round-trips" {
    const path = "zig-out/stage10-fixture.zynfer";
    var compile = try run(&.{ "artifact-compile", "--out", path });
    defer compile.deinit(std.testing.allocator);
    try expectExited(compile, 0);
    try std.testing.expect(std.mem.indexOf(u8, compile.stdout, "wrote") != null);

    var inspect = try run(&.{ "inspect", path });
    defer inspect.deinit(std.testing.allocator);
    try expectExited(inspect, 0);
    try std.testing.expect(std.mem.indexOf(u8, inspect.stdout, "qwen3-0.6b") != null);
    try std.testing.expect(std.mem.indexOf(u8, inspect.stdout, "fixture.embed") != null);
}
