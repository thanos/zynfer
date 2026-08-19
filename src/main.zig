const std = @import("std");
const zynfer = @import("zynfer");

const usage =
    \\zynfer — Zig LLM inference (CPU oracle + Apple Metal + AMD HIP probe)
    \\
    \\Usage:
    \\  zynfer              Environment report + GPU probe
    \\  zynfer env          Development-environment report
    \\  zynfer gpu          HIP device enumeration (AMD host)
    \\  zynfer caps         Backend/device capabilities and fallbacks
    \\  zynfer backends     List selectable backends
    \\  zynfer ops-bench    CPU vs Apple op microbenchmarks
    \\  zynfer bench        HIP query timing (AMD host)
    \\  zynfer help
    \\
    \\Force a backend (invalid choices fail; they do not fall back):
    \\  zynfer caps --backend cpu
    \\  zynfer caps --backend apple
    \\  ZYNFER_BACKEND=cpu zynfer caps
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
    var command: []const u8 = "all";
    var have_command = false;
    var forced_backend: ?[]const u8 = null;
    while (args_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--backend")) {
            forced_backend = args_it.next() orelse {
                std.debug.print("missing value for --backend\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.startsWith(u8, arg, "--backend=")) {
            forced_backend = arg["--backend=".len..];
        } else if (!have_command and !std.mem.startsWith(u8, arg, "-")) {
            command = arg;
            have_command = true;
        }
    }
    if (forced_backend == null) {
        if (host.environ) |env_map| {
            forced_backend = env_map.get("ZYNFER_BACKEND");
        }
    }

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
    } else if (std.mem.eql(u8, command, "caps")) {
        try printCaps(writer, forced_backend);
    } else if (std.mem.eql(u8, command, "backends")) {
        try printBackends(writer);
    } else if (std.mem.eql(u8, command, "ops-bench")) {
        try runOpsBench(allocator, io, writer, forced_backend);
    } else if (std.mem.eql(u8, command, "bench")) {
        try printEnv(host, writer);
        try writer.writeAll("\n");
        try printGpu(writer);
        try writer.writeAll("\n");
        try runHipBench(io, writer);
    } else if (std.mem.eql(u8, command, "all")) {
        try printEnv(host, writer);
        try writer.writeAll("\n");
        try printCaps(writer, forced_backend);
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
    try writer.print("zynfer HIP GPU report\n", .{});
    try writer.print("=====================\n\n", .{});
    try zynfer.hip.printDevices(writer);
}

fn printBackends(writer: *std.Io.Writer) !void {
    try writer.print("selectable backends\n", .{});
    try writer.print("-------------------\n", .{});
    for ([3]zynfer.BackendKind{ .cpu, .apple, .amd_hip }) |kind| {
        const status: []const u8 = if (zynfer.backend.isBackendBuildable(kind)) "buildable" else "not compiled";
        try writer.print("  {s: <10} {s}\n", .{ kind.name(), status });
    }
    try writer.print("\nDefault kind on this host: {s}\n", .{zynfer.backend.defaultKind().name()});
}

fn resolveKind(forced: ?[]const u8) !zynfer.BackendKind {
    if (forced) |name| {
        const kind = zynfer.backend.parseBackendKind(name) catch {
            std.debug.print("unknown backend '{s}'. use cpu, apple, or amd-hip.\n", .{name});
            std.process.exit(2);
        };
        zynfer.backend.requireBackend(kind) catch {
            std.debug.print("backend '{s}' is not available in this build.\n", .{kind.name()});
            std.process.exit(2);
        };
        return kind;
    }
    return zynfer.backend.defaultKind();
}

fn printCaps(writer: *std.Io.Writer, forced: ?[]const u8) !void {
    const kind = try resolveKind(forced);
    try writer.print("zynfer capabilities\n", .{});
    try writer.print("===================\n\n", .{});
    try writer.print("requested backend: {s}\n", .{kind.name()});

    const caps = switch (kind) {
        .cpu => zynfer.backend.cpuCapabilities(),
        .apple => zynfer.apple.gpu.capabilities(),
        .amd_hip => blk: {
            var c = zynfer.backend.cpuCapabilities();
            c.backend = .amd_hip;
            c.hip = zynfer.hip.have_hip;
            c.addDisabled("HIP backend is device enumeration only; transformer ops are not implemented here yet");
            break :blk c;
        },
    };

    try writer.print("device architecture: {s}\n", .{caps.arch.name()});
    try writer.print("unified memory: {s}\n", .{if (caps.unified_memory) "yes" else "no"});
    try writer.print("fp32: {s}  fp16: {s}  bf16: {s}\n", .{
        yn(caps.fp32),
        yn(caps.fp16),
        yn(caps.bf16),
    });
    try writer.print("simdgroup_matrix hardware: {s}\n", .{yn(caps.simdgroup_matrix)});
    try writer.print("Accelerate path: {s}\n", .{yn(caps.accelerate)});
    try writer.print("Core ML path: {s}\n", .{yn(caps.core_ml)});
    try writer.print("HIP linked: {s}\n", .{yn(caps.hip or zynfer.hip.have_hip)});

    switch (caps.arch) {
        .apple_m => |feat| {
            try writer.print("\nApple Metal device (label only; not used for kernel correctness)\n", .{});
            try writer.print("  name: {s}\n", .{feat.nameSlice()});
            try writer.print("  recommended working set: {d} bytes\n", .{feat.recommended_working_set_bytes});
            try writer.print("  max buffer: {d} bytes\n", .{feat.max_buffer_bytes});
            try writer.print("  max threads/threadgroup: {d}\n", .{feat.max_threads_per_threadgroup});
            try writer.print("  GPU family Apple7/8/9: {s}/{s}/{s}\n", .{
                yn(feat.gpu_family_apple7),
                yn(feat.gpu_family_apple8),
                yn(feat.gpu_family_apple9),
            });
            try writer.print("  chosen kernels: naive f32 Metal (add, mul, silu_mul, rmsnorm, softmax, matmul, matvec, rope)\n", .{});
        },
        else => {},
    }

    try writer.print("\ndisabled paths\n", .{});
    var i: usize = 0;
    while (i < caps.disabled_len) : (i += 1) {
        try writer.print("  - {s}\n", .{caps.disabled[i]});
    }
}

fn yn(v: bool) []const u8 {
    return if (v) "yes" else "no";
}

fn runOpsBench(gpa: std.mem.Allocator, io: std.Io, writer: *std.Io.Writer, forced: ?[]const u8) !void {
    const kind = try resolveKind(forced);
    try writer.print("zynfer ops-bench\n", .{});
    try writer.print("================\n", .{});
    try writer.print("backend={s}  (cpu always runs as oracle)\n", .{kind.name()});
    try writer.print("note: Apple times include per-op shared-buffer fill + encode_and_wait.\n", .{});
    try writer.print("      That is the current baseline, not a fused production decode path.\n\n", .{});

    const warmup = 2;
    const iters = 8;
    var metal_init_ns: ?u64 = null;
    var gpu_storage: zynfer.apple.gpu.Gpu = undefined;
    var gpu_ptr: ?*zynfer.apple.gpu.Gpu = null;
    if (kind == .apple and zynfer.apple.gpu.have_apple) {
        const t0 = std.Io.Clock.awake.now(io);
        gpu_storage = zynfer.apple.gpu.Gpu.init() catch |err| {
            try writer.print("Apple Metal init failed: {s}\n", .{@errorName(err)});
            try writer.print("If shaders fail to compile, install the Metal Toolchain:\n", .{});
            try writer.print("  xcodebuild -downloadComponent MetalToolchain\n", .{});
            return;
        };
        gpu_ptr = &gpu_storage;
        const t1 = std.Io.Clock.awake.now(io);
        metal_init_ns = @intCast(@max(@as(i96, 0), t1.nanoseconds - t0.nanoseconds));
        try writer.print("metal_device_create_plus_shader_compile_ns={d}\n\n", .{metal_init_ns.?});
    }
    defer if (gpu_ptr) |g| g.deinit();

    var rows: [4]BenchRow = undefined;
    rows[0] = try benchNamed(gpa, io, writer, gpu_ptr, "add_f32_4096", benchAdd, warmup, iters);
    rows[1] = try benchNamed(gpa, io, writer, gpu_ptr, "silu_mul_f32_4096", benchSiluMul, warmup, iters);
    rows[2] = try benchNamed(gpa, io, writer, gpu_ptr, "matvec_f32_256x256", benchMatvec, warmup, iters);
    rows[3] = try benchNamed(gpa, io, writer, gpu_ptr, "matmul_f32_32x64x64", benchMatmul, warmup, iters);

    try writer.print("\njson\n", .{});
    try writer.print("{{\"backend\":\"{s}\",\"zig\":\"{s}\",\"warmup\":{d},\"iters\":{d}", .{
        kind.name(),
        @import("builtin").zig_version_string,
        warmup,
        iters,
    });
    if (metal_init_ns) |ns| {
        try writer.print(",\"metal_init_ns\":{d}", .{ns});
    } else {
        try writer.print(",\"metal_init_ns\":null", .{});
    }
    try writer.print(",\"ops\":[", .{});
    for (rows, 0..) |row, i| {
        if (i != 0) try writer.print(",", .{});
        try writer.print("{{\"name\":\"{s}\",\"cpu_ns\":{d},", .{ row.name, row.cpu_ns });
        if (row.apple_ns) |ns| {
            try writer.print("\"apple_metal_ns\":{d}}}", .{ns});
        } else {
            try writer.print("\"apple_metal_ns\":null}}", .{});
        }
    }
    try writer.print("]}}\n", .{});
}

const BenchRow = struct {
    name: []const u8,
    cpu_ns: u64,
    apple_ns: ?u64,
};

const BenchFn = *const fn (gpa: std.mem.Allocator, gpu: ?*zynfer.apple.gpu.Gpu) anyerror!void;

fn benchNamed(
    gpa: std.mem.Allocator,
    io: std.Io,
    writer: *std.Io.Writer,
    gpu: ?*zynfer.apple.gpu.Gpu,
    name: []const u8,
    func: BenchFn,
    warmup: usize,
    iters: usize,
) !BenchRow {
    var i: usize = 0;
    while (i < warmup) : (i += 1) try func(gpa, null);
    const cpu_ns = try timeIters(io, iters, func, gpa, null);
    try writer.print("{s} cpu_ns={d} iters={d}\n", .{ name, cpu_ns / iters, iters });

    var apple_ns: ?u64 = null;
    if (gpu) |g| {
        i = 0;
        while (i < warmup) : (i += 1) try func(gpa, g);
        const total = try timeIters(io, iters, func, gpa, g);
        apple_ns = total / iters;
        try writer.print("{s} apple_metal_ns={d} iters={d}\n", .{ name, apple_ns.?, iters });
    } else {
        try writer.print("{s} apple_metal_ns=N/A\n", .{name});
    }
    return .{ .name = name, .cpu_ns = cpu_ns / iters, .apple_ns = apple_ns };
}

fn timeIters(io: std.Io, iters: usize, func: BenchFn, gpa: std.mem.Allocator, gpu: ?*zynfer.apple.gpu.Gpu) !u64 {
    const start = std.Io.Clock.awake.now(io);
    var i: usize = 0;
    while (i < iters) : (i += 1) try func(gpa, gpu);
    const end = std.Io.Clock.awake.now(io);
    return @intCast(@max(@as(i96, 0), end.nanoseconds - start.nanoseconds));
}

fn benchAdd(gpa: std.mem.Allocator, gpu: ?*zynfer.apple.gpu.Gpu) !void {
    var a = try zynfer.Tensor.alloc(gpa, .f32, &.{4096});
    defer a.deinit();
    var b = try zynfer.Tensor.alloc(gpa, .f32, &.{4096});
    defer b.deinit();
    var o = try zynfer.Tensor.alloc(gpa, .f32, &.{4096});
    defer o.deinit();
    try a.fillF32(1);
    try b.fillF32(2);
    if (gpu) |g| {
        try zynfer.apple.ops.add(g, o, a, b);
    } else {
        try zynfer.cpu.ops.add(o, a, b);
    }
}

fn benchSiluMul(gpa: std.mem.Allocator, gpu: ?*zynfer.apple.gpu.Gpu) !void {
    var a = try zynfer.Tensor.alloc(gpa, .f32, &.{4096});
    defer a.deinit();
    var b = try zynfer.Tensor.alloc(gpa, .f32, &.{4096});
    defer b.deinit();
    var o = try zynfer.Tensor.alloc(gpa, .f32, &.{4096});
    defer o.deinit();
    try a.fillF32(0.5);
    try b.fillF32(1.5);
    if (gpu) |g| {
        try zynfer.apple.ops.siluMul(g, o, a, b);
    } else {
        try zynfer.cpu.ops.siluMul(o, a, b);
    }
}

fn benchMatvec(gpa: std.mem.Allocator, gpu: ?*zynfer.apple.gpu.Gpu) !void {
    var a = try zynfer.Tensor.alloc(gpa, .f32, &.{ 256, 256 });
    defer a.deinit();
    var x = try zynfer.Tensor.alloc(gpa, .f32, &.{256});
    defer x.deinit();
    var y = try zynfer.Tensor.alloc(gpa, .f32, &.{256});
    defer y.deinit();
    try a.fillF32(0.01);
    try x.fillF32(0.02);
    if (gpu) |g| {
        try zynfer.apple.ops.matvec(g, y, a, x);
    } else {
        try zynfer.cpu.ops.matvec(y, a, x);
    }
}

fn benchMatmul(gpa: std.mem.Allocator, gpu: ?*zynfer.apple.gpu.Gpu) !void {
    var a = try zynfer.Tensor.alloc(gpa, .f32, &.{ 32, 64 });
    defer a.deinit();
    var b = try zynfer.Tensor.alloc(gpa, .f32, &.{ 64, 64 });
    defer b.deinit();
    var c = try zynfer.Tensor.alloc(gpa, .f32, &.{ 32, 64 });
    defer c.deinit();
    try a.fillF32(0.01);
    try b.fillF32(0.02);
    if (gpu) |g| {
        try zynfer.apple.ops.matmul(g, c, a, b);
    } else {
        try zynfer.cpu.ops.matmul(c, a, b);
    }
}

fn runHipBench(io: std.Io, writer: *std.Io.Writer) !void {
    try writer.print("zynfer HIP probe benchmark\n", .{});
    try writer.print("==========================\n\n", .{});

    if (!zynfer.hip.have_hip) {
        try writer.print("HIP is not linked. Enumeration latency cannot be measured on this host.\n", .{});
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
        last_count = try zynfer.hip.deviceCount();
        if (last_count > 0) _ = try zynfer.hip.describeDevice(0);
    }
    const elapsed_ns: u64 = @intCast(@max(@as(i96, 0), std.Io.Clock.awake.now(io).nanoseconds - start.nanoseconds));
    try writer.print("avg hip query: {d} ns  devices={d}\n", .{ elapsed_ns / iters, last_count });
}
