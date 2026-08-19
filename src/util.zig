const std = @import("std");

pub const Host = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    environ: ?*const std.process.Environ.Map = null,
};

pub fn cString(buf: []const u8) []const u8 {
    return std.mem.sliceTo(buf, 0);
}

pub fn trimAscii(text: []const u8) []const u8 {
    return std.mem.trim(u8, text, " \t\r\n");
}

pub fn firstLine(text: []const u8) []const u8 {
    if (std.mem.indexOfScalar(u8, text, '\n')) |idx| {
        return trimAscii(text[0..idx]);
    }
    return trimAscii(text);
}

pub fn readFileOptional(host: Host, path: []const u8) ?[]u8 {
    return std.Io.Dir.cwd().readFileAlloc(host.io, path, host.gpa, .limited(64 * 1024)) catch null;
}

pub fn fileExists(io: std.Io, path: []const u8) bool {
    if (std.fs.path.isAbsolute(path)) {
        std.Io.Dir.accessAbsolute(io, path, .{}) catch return false;
        return true;
    }
    std.Io.Dir.cwd().access(io, path, .{}) catch return false;
    return true;
}

pub fn findOnPath(host: Host, name: []const u8) ?[]u8 {
    const environ = host.environ orelse return null;
    const path_env = environ.get("PATH") orelse return null;

    var it = std.mem.splitScalar(u8, path_env, ':');
    while (it.next()) |dir| {
        if (dir.len == 0) continue;
        const candidate = std.fs.path.join(host.gpa, &.{ dir, name }) catch continue;
        if (fileExists(host.io, candidate)) return candidate;
        host.gpa.free(candidate);
    }
    return null;
}

pub const CommandOutput = struct {
    stdout: []u8,
    stderr: []u8,
    term: std.process.Child.Term,

    pub fn deinit(self: CommandOutput, allocator: std.mem.Allocator) void {
        allocator.free(self.stdout);
        allocator.free(self.stderr);
    }

    pub fn exitedZero(self: CommandOutput) bool {
        return switch (self.term) {
            .exited => |code| code == 0,
            else => false,
        };
    }
};

pub fn runCommand(host: Host, argv: []const []const u8) ?CommandOutput {
    const result = std.process.run(host.gpa, host.io, .{
        .argv = argv,
        .stdout_limit = .limited(512 * 1024),
        .stderr_limit = .limited(512 * 1024),
    }) catch return null;
    return .{
        .stdout = result.stdout,
        .stderr = result.stderr,
        .term = result.term,
    };
}

pub fn runCommandStdout(host: Host, argv: []const []const u8) ?[]u8 {
    const result = runCommand(host, argv) orelse return null;
    defer host.gpa.free(result.stderr);
    if (!result.exitedZero()) {
        host.gpa.free(result.stdout);
        return null;
    }
    return result.stdout;
}

pub fn formatHipPackedVersion(buf: []u8, packed_version: i32) []const u8 {
    if (packed_version <= 0) {
        return std.fmt.bufPrint(buf, "unknown ({d})", .{packed_version}) catch buf[0..0];
    }
    const v: u32 = @intCast(packed_version);
    const major = v / 10_000_000;
    const minor = (v / 100_000) % 100;
    const patch = v % 100_000;
    return std.fmt.bufPrint(buf, "{d}.{d}.{d} ({d})", .{ major, minor, patch, packed_version }) catch buf[0..0];
}

pub fn formatBytes(buf: []u8, bytes: u64) []const u8 {
    const gib = @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0 * 1024.0);
    return std.fmt.bufPrint(buf, "{d} bytes ({d:.2} GiB)", .{ bytes, gib }) catch buf[0..0];
}

test "cString stops at NUL" {
    const raw = [_]u8{ 'g', 'f', 'x', 0, 'x' };
    try std.testing.expectEqualStrings("gfx", cString(&raw));
}

test "formatHipPackedVersion decodes ROCm-style packing" {
    var buf: [64]u8 = undefined;
    const text = formatHipPackedVersion(&buf, 60_241_134);
    try std.testing.expectEqualStrings("6.2.41134 (60241134)", text);
}
