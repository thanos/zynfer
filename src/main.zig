const std = @import("std");
const zynfer = @import("zynfer");

const usage =
    \\zynfer — AMD GPU LLM inference runtime (Stage 0)
    \\
    \\Usage:
    \\  zynfer              Print environment report and enumerate GPUs
    \\  zynfer env          Print the development-environment report
    \\  zynfer gpu          Enumerate HIP devices and print properties
    \\  zynfer bench        Time HIP device enumeration (warmup + measured)
    \\  zynfer help         Show this message
    \\
    \\Build:
    \\  zig build
    \\  zig build test
    \\  zig build run
    \\  zig build run -- gpu
    \\
    \\HIP is linked automatically when ROCm is found. Force it with:
    \\  zig build -Dhip=on -Dhip-path=/opt/rocm
    \\
;

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const host = zynfer.util.Host{
        .gpa = allocator,
        .io = io,
        .environ = init.environ_map,
    };

    var args_it = std.process.Args.Iterator.init(init.minimal.args);
    _ = args_it.skip();
    const command: []const u8 = args_it.next() orelse "all";

    var buf: [8192]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(io, &buf);
    const writer = &stdout_writer.interface;

    if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "-h") or std.mem.eql(u8, command, "--help")) {
        try writer.writeAll(usage);
        try writer.flush();
        return;
    }

    if (std.mem.eql(u8, command, "env")) {
        try printEnv(host, writer);
    } else if (std.mem.eql(u8, command, "gpu")) {
        try printGpu(writer);
    } else if (std.mem.eql(u8, command, "bench")) {
        try printEnv(host, writer);
        try writer.writeAll("\n");
        try printGpu(writer);
        try writer.writeAll("\n");
        try runBench(io, writer);
    } else if (std.mem.eql(u8, command, "all")) {
        try printEnv(host, writer);
        try writer.writeAll("\n");
        try printGpu(writer);
    } else {
        try writer.print("unknown command: {s}\n\n", .{command});
        try writer.writeAll(usage);
        try writer.flush();
        std.process.exit(2);
    }

    try writer.flush();
}

fn printEnv(host: zynfer.util.Host, writer: *std.Io.Writer) !void {
    var report = try zynfer.env.collect(host);
    defer report.deinit();
    try zynfer.env.print(writer, report);
}

fn printGpu(writer: *std.Io.Writer) !void {
    try writer.print("zynfer GPU report\n", .{});
    try writer.print("=================\n\n", .{});
    try zynfer.hip.printDevices(writer);
}

fn runBench(io: std.Io, writer: *std.Io.Writer) !void {
    try writer.print("zynfer Stage 0 benchmark\n", .{});
    try writer.print("========================\n\n", .{});

    if (!zynfer.hip.have_hip) {
        try writer.print("HIP is not linked. Enumeration latency cannot be measured on this host.\n", .{});
        try writer.print("Re-run on the Linux GPU machine after `zig build -Dhip=on`.\n", .{});
        return;
    }

    const warmup = 8;
    const iters = 32;

    var i: usize = 0;
    while (i < warmup) : (i += 1) {
        _ = zynfer.hip.deviceCount() catch |err| {
            try writer.print("warmup failed: {s}\n", .{@errorName(err)});
            return;
        };
    }

    const start = std.Io.Clock.awake.now(io);
    i = 0;
    var last_count: u32 = 0;
    while (i < iters) : (i += 1) {
        last_count = zynfer.hip.deviceCount() catch |err| {
            try writer.print("measured iteration failed: {s}\n", .{@errorName(err)});
            return;
        };
        if (last_count > 0) {
            _ = try zynfer.hip.describeDevice(0);
        }
    }
    const end = std.Io.Clock.awake.now(io);
    const elapsed_ns: u64 = @intCast(@max(@as(i96, 0), end.nanoseconds - start.nanoseconds));
    const avg_ns = elapsed_ns / iters;
    const avg_us = @as(f64, @floatFromInt(avg_ns)) / 1000.0;

    try writer.print("operation:      hipGetDeviceCount + hipGetDeviceProperties(0)\n", .{});
    try writer.print("warmup:         {d}\n", .{warmup});
    try writer.print("iterations:     {d}\n", .{iters});
    try writer.print("devices:        {d}\n", .{last_count});
    try writer.print("total:          {d} ns\n", .{elapsed_ns});
    try writer.print("avg latency:    {d:.1} us\n", .{avg_us});
    try writer.print("note:           this measures runtime query overhead, not kernel launch.\n", .{});
}
