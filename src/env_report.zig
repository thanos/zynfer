const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const util = @import("util.zig");

pub const Report = struct {
    arena: std.heap.ArenaAllocator,
    hostname: []const u8,
    os_name: []const u8,
    kernel: []const u8,
    machine: []const u8,
    zig_version: []const u8,
    zig_path: []const u8,
    compile_os: []const u8,
    compile_arch: []const u8,
    hip_linked: bool,
    apple_compiled: bool,
    hip_path: []const u8,
    rocm_path: []const u8,
    rocm_version: []const u8,
    hipcc_path: []const u8,
    hipcc_version: []const u8,
    rocminfo_path: []const u8,
    clang_path: []const u8,
    clang_version: []const u8,
    amdgpu_module: []const u8,
    amdgpu_version: []const u8,
    driver_state: []const u8,
    rocminfo_gpu: []const u8,
    rocminfo_arch: []const u8,
    rocminfo_vram: []const u8,

    pub fn deinit(self: *Report) void {
        self.arena.deinit();
    }
};

pub fn collect(host: util.Host) !Report {
    const gpa = host.gpa;
    var report = Report{
        .arena = std.heap.ArenaAllocator.init(gpa),
        .hostname = "unknown",
        .os_name = @tagName(builtin.os.tag),
        .kernel = "unknown",
        .machine = @tagName(builtin.cpu.arch),
        .zig_version = builtin.zig_version_string,
        .zig_path = "not found",
        .compile_os = @tagName(builtin.os.tag),
        .compile_arch = @tagName(builtin.cpu.arch),
        .hip_linked = build_options.have_hip,
        .apple_compiled = build_options.have_apple,
        .hip_path = if (build_options.hip_path.len == 0) "not found" else build_options.hip_path,
        .rocm_path = "not found",
        .rocm_version = "not found",
        .hipcc_path = "not found",
        .hipcc_version = "not found",
        .rocminfo_path = "not found",
        .clang_path = "not found",
        .clang_version = "not found",
        .amdgpu_module = "not found",
        .amdgpu_version = "not found",
        .driver_state = "unknown",
        .rocminfo_gpu = "not found",
        .rocminfo_arch = "not found",
        .rocminfo_vram = "not found",
    };
    errdefer report.deinit();
    const a = report.arena.allocator();

    if (unameInfo()) |uts| {
        report.os_name = try a.dupe(u8, util.cString(&uts.sysname));
        report.hostname = try a.dupe(u8, util.cString(&uts.nodename));
        report.kernel = try a.dupe(u8, util.cString(&uts.release));
        report.machine = try a.dupe(u8, util.cString(&uts.machine));
    }

    const arena_host = util.Host{
        .gpa = a,
        .io = host.io,
        .environ = host.environ,
    };

    if (util.findOnPath(arena_host, "zig")) |path| {
        report.zig_path = path;
    }

    if (host.environ) |environ| {
        if (environ.get("ROCM_PATH")) |path| {
            report.rocm_path = try a.dupe(u8, path);
        }
    }
    if (std.mem.eql(u8, report.rocm_path, "not found") and util.fileExists(host.io, "/opt/rocm")) {
        report.rocm_path = "/opt/rocm";
    } else if (std.mem.eql(u8, report.rocm_path, "not found") and build_options.hip_path.len != 0) {
        report.rocm_path = build_options.hip_path;
    }

    report.rocm_version = try firstExistingLine(arena_host, &.{
        "/opt/rocm/.info/version",
        "/opt/rocm/.info/version-dev",
        "/opt/rocm/share/doc/rocm-cmake/VERSION",
    });
    if (std.mem.eql(u8, report.rocm_version, "not found") and !std.mem.eql(u8, report.rocm_path, "not found")) {
        const version_path = try std.fs.path.join(a, &.{ report.rocm_path, ".info/version" });
        if (readFirstLine(arena_host, version_path)) |line| {
            report.rocm_version = line;
        }
    }

    if (util.findOnPath(arena_host, "hipcc")) |path| {
        report.hipcc_path = path;
        if (util.runCommandStdout(arena_host, &.{ path, "--version" })) |out| {
            report.hipcc_version = util.firstLine(out);
        }
    }

    if (util.findOnPath(arena_host, "rocminfo")) |path| {
        report.rocminfo_path = path;
        if (util.runCommandStdout(arena_host, &.{path})) |out| {
            report.rocminfo_gpu = try a.dupe(u8, parseRocminfoField(out, "Marketing Name:") orelse "not found");
            report.rocminfo_arch = try a.dupe(u8, parseRocminfoArch(out) orelse "not found");
            report.rocminfo_vram = try a.dupe(u8, parseRocminfoField(out, "Size:") orelse "not found");
        }
    }

    if (util.findOnPath(arena_host, "clang")) |path| {
        report.clang_path = path;
        if (util.runCommandStdout(arena_host, &.{ path, "--version" })) |out| {
            report.clang_version = util.firstLine(out);
        }
    }

    if (util.fileExists(host.io, "/sys/module/amdgpu")) {
        report.amdgpu_module = "loaded";
        report.driver_state = "amdgpu present";
        if (readFirstLine(arena_host, "/sys/module/amdgpu/version")) |line| {
            report.amdgpu_version = line;
        }
        if (readFirstLine(arena_host, "/sys/module/amdgpu/initstate")) |line| {
            report.driver_state = line;
        }
    } else if (builtin.os.tag == .linux) {
        report.amdgpu_module = "not loaded";
        report.driver_state = "amdgpu module not found";
    } else {
        report.amdgpu_module = "n/a";
        report.driver_state = "host is not Linux; AMD GPU execution is not expected here";
    }

    return report;
}

pub fn print(writer: *std.Io.Writer, report: Report) !void {
    try writer.print("zynfer environment report\n", .{});
    try writer.print("=========================\n\n", .{});
    try writer.print("host\n", .{});
    try writer.print("  hostname:         {s}\n", .{report.hostname});
    try writer.print("  OS:               {s}\n", .{report.os_name});
    try writer.print("  kernel:           {s}\n", .{report.kernel});
    try writer.print("  machine:          {s}\n", .{report.machine});
    try writer.print("  compile target:   {s}-{s}\n\n", .{ report.compile_arch, report.compile_os });

    try writer.print("toolchain\n", .{});
    try writer.print("  Zig version:      {s}\n", .{report.zig_version});
    try writer.print("  Zig path:         {s}\n", .{report.zig_path});
    try writer.print("  clang path:       {s}\n", .{report.clang_path});
    try writer.print("  clang version:    {s}\n\n", .{report.clang_version});

    try writer.print("backends compiled into this binary\n", .{});
    try writer.print("  CPU reference:    yes\n", .{});
    try writer.print("  Apple Metal:      {s}\n", .{if (report.apple_compiled) "yes" else "no"});
    try writer.print("  AMD HIP:          {s}\n\n", .{if (report.hip_linked) "yes (probe)" else "no"});

    try writer.print("ROCm / HIP\n", .{});
    try writer.print("  HIP linked:       {s}\n", .{if (report.hip_linked) "yes" else "no"});
    try writer.print("  HIP path:         {s}\n", .{report.hip_path});
    try writer.print("  ROCm path:        {s}\n", .{report.rocm_path});
    try writer.print("  ROCm version:     {s}\n", .{report.rocm_version});
    try writer.print("  hipcc path:       {s}\n", .{report.hipcc_path});
    try writer.print("  hipcc version:    {s}\n", .{report.hipcc_version});
    try writer.print("  rocminfo path:    {s}\n\n", .{report.rocminfo_path});

    try writer.print("driver\n", .{});
    try writer.print("  amdgpu module:    {s}\n", .{report.amdgpu_module});
    try writer.print("  amdgpu version:   {s}\n", .{report.amdgpu_version});
    try writer.print("  driver state:     {s}\n\n", .{report.driver_state});

    try writer.print("rocminfo GPU (if available)\n", .{});
    try writer.print("  marketing name:   {s}\n", .{report.rocminfo_gpu});
    try writer.print("  ISA / arch:       {s}\n", .{report.rocminfo_arch});
    try writer.print("  memory size:      {s}\n", .{report.rocminfo_vram});
}

fn unameInfo() ?std.posix.utsname {
    return switch (builtin.os.tag) {
        .linux, .macos, .freebsd, .netbsd, .openbsd, .dragonfly => std.posix.uname(),
        else => null,
    };
}

fn readFirstLine(host: util.Host, path: []const u8) ?[]u8 {
    const contents = util.readFileOptional(host, path) orelse return null;
    return host.gpa.dupe(u8, util.firstLine(contents)) catch null;
}

fn firstExistingLine(host: util.Host, paths: []const []const u8) ![]const u8 {
    for (paths) |path| {
        if (readFirstLine(host, path)) |line| return line;
    }
    return "not found";
}

fn parseRocminfoField(text: []const u8, label: []const u8) ?[]const u8 {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        const trimmed = util.trimAscii(line);
        if (std.mem.indexOf(u8, trimmed, label)) |idx| {
            return util.trimAscii(trimmed[idx + label.len ..]);
        }
    }
    return null;
}

fn parseRocminfoArch(text: []const u8) ?[]const u8 {
    if (parseRocminfoField(text, "Name:                    gfx")) |rest| {
        // rocminfo prints "Name: gfx1201" on the ISA agent; the helper above
        // already consumed "Name:" so restore the gfx prefix if needed.
        if (std.mem.startsWith(u8, rest, "gfx")) return rest;
    }
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        const trimmed = util.trimAscii(line);
        if (std.mem.indexOf(u8, trimmed, "gfx")) |idx| {
            const from = trimmed[idx..];
            var end: usize = 0;
            while (end < from.len and std.ascii.isAlphanumeric(from[end])) : (end += 1) {}
            if (end >= 4) return from[0..end];
        }
    }
    return null;
}

test "parse rocminfo marketing name" {
    const sample =
        \\  Marketing Name:          AMD Radeon AI PRO R9700
        \\  Name:                    gfx1201
        \\  Size:                    32 GB
    ;
    try std.testing.expectEqualStrings("AMD Radeon AI PRO R9700", parseRocminfoField(sample, "Marketing Name:").?);
    try std.testing.expectEqualStrings("gfx1201", parseRocminfoArch(sample).?);
}

test "collect produces a Zig version" {
    var report = try collect(.{
        .gpa = std.testing.allocator,
        .io = std.testing.io,
    });
    defer report.deinit();
    try std.testing.expect(report.zig_version.len > 0);
    try std.testing.expect(report.os_name.len > 0);
}
