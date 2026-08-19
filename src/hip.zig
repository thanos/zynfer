const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const util = @import("util.zig");

pub const have_hip = build_options.have_hip;

pub const GpuInfo = extern struct {
    index: i32,
    name: [256]u8,
    gcn_arch: [256]u8,
    total_mem_bytes: u64,
    multiprocessor_count: i32,
    warp_size: i32,
    max_threads_per_block: i32,
    clock_rate_khz: i32,
    memory_clock_rate_khz: i32,
    memory_bus_width_bits: i32,
    l2_cache_bytes: i32,
    shared_mem_per_block: i32,
    regs_per_block: i32,
    pci_domain: i32,
    pci_bus: i32,
    pci_device: i32,
    integrated: i32,
    can_map_host_memory: i32,
    concurrent_kernels: i32,

    pub fn nameSlice(self: *const GpuInfo) []const u8 {
        return util.cString(&self.name);
    }

    pub fn archSlice(self: *const GpuInfo) []const u8 {
        return util.cString(&self.gcn_arch);
    }

    pub fn isTargetGpu(self: *const GpuInfo) bool {
        const arch = self.archSlice();
        if (std.mem.startsWith(u8, arch, "gfx1201")) return true;
        const name = self.nameSlice();
        return std.ascii.indexOfIgnoreCase(name, "r9700") != null;
    }
};

pub const Error = error{
    HipUnavailable,
    InvalidValue,
    OutOfMemory,
    NotInitialized,
    Deinitialized,
    NoDevice,
    InvalidDevice,
    NotSupported,
    Unknown,
};

pub const LastError = struct {
    op: []const u8 = "",
    status: i32 = 0,
    message: []const u8 = "no HIP error",
};

var last_error_buf: [256]u8 = undefined;
var last_error: LastError = .{};

pub fn lastHipError() LastError {
    return last_error;
}

pub fn formatLastError(buf: []u8) []const u8 {
    if (last_error.status == 0 and last_error.op.len == 0) {
        return "no HIP error";
    }
    return std.fmt.bufPrint(
        buf,
        "HIP {s} failed: {s} ({d})",
        .{ last_error.op, last_error.message, last_error.status },
    ) catch "HIP error";
}

const hip_c = if (have_hip) struct {
    pub extern fn zynfer_hip_runtime_version(out: *c_int) c_int;
    pub extern fn zynfer_hip_driver_version(out: *c_int) c_int;
    pub extern fn zynfer_hip_device_count(out: *c_int) c_int;
    pub extern fn zynfer_hip_describe_device(device: c_int, out: *GpuInfo) c_int;
    pub extern fn zynfer_hip_error_string(err: c_int) [*:0]const u8;
} else struct {};

fn recordError(status: c_int, op: []const u8) Error {
    last_error.op = op;
    last_error.status = status;
    if (have_hip) {
        const msg = std.mem.span(hip_c.zynfer_hip_error_string(status));
        const n = @min(msg.len, last_error_buf.len);
        @memcpy(last_error_buf[0..n], msg[0..n]);
        last_error.message = last_error_buf[0..n];
    } else {
        last_error.message = "HIP was not linked into this binary";
    }
    return statusToError(status);
}

fn statusToError(status: c_int) Error {
    return switch (status) {
        0 => unreachable,
        1 => error.InvalidValue,
        2 => error.OutOfMemory,
        3 => error.NotInitialized,
        4 => error.Deinitialized,
        100 => error.NoDevice,
        101 => error.InvalidDevice,
        801 => error.NotSupported,
        else => error.Unknown,
    };
}

fn check(status: c_int, op: []const u8) Error!void {
    if (status == 0) return;
    return recordError(status, op);
}

pub fn runtimeVersion() Error!i32 {
    if (!have_hip) return error.HipUnavailable;
    var version: c_int = 0;
    try check(hip_c.zynfer_hip_runtime_version(&version), "hipRuntimeGetVersion");
    return version;
}

pub fn driverVersion() Error!i32 {
    if (!have_hip) return error.HipUnavailable;
    var version: c_int = 0;
    try check(hip_c.zynfer_hip_driver_version(&version), "hipDriverGetVersion");
    return version;
}

pub fn deviceCount() Error!u32 {
    if (!have_hip) return error.HipUnavailable;
    var count: c_int = 0;
    try check(hip_c.zynfer_hip_device_count(&count), "hipGetDeviceCount");
    if (count < 0) return error.InvalidValue;
    return @intCast(count);
}

pub fn describeDevice(index: i32) Error!GpuInfo {
    if (!have_hip) return error.HipUnavailable;
    var info: GpuInfo = std.mem.zeroes(GpuInfo);
    try check(hip_c.zynfer_hip_describe_device(index, &info), "hipGetDeviceProperties");
    info.index = index;
    return info;
}

pub fn printDevices(writer: *std.Io.Writer) !void {
    if (!have_hip) {
        try writer.print("HIP:            not linked (built without ROCm/HIP)\n", .{});
        try writer.print("              rebuild on the Linux GPU host, or pass -Dhip=on\n", .{});
        return;
    }

    const runtime_v = runtimeVersion() catch |err| blk: {
        try writer.print("HIP runtime:    error {s}\n", .{@errorName(err)});
        break :blk @as(?i32, null);
    };
    const driver_v = driverVersion() catch |err| blk: {
        try writer.print("HIP driver API: error {s}\n", .{@errorName(err)});
        break :blk @as(?i32, null);
    };

    var ver_buf: [64]u8 = undefined;
    if (runtime_v) |v| {
        try writer.print("HIP runtime:    {s}\n", .{util.formatHipPackedVersion(&ver_buf, v)});
    }
    if (driver_v) |v| {
        try writer.print("HIP driver API: {s}\n", .{util.formatHipPackedVersion(&ver_buf, v)});
    }

    const count = deviceCount() catch |err| {
        var err_buf: [256]u8 = undefined;
        try writer.print("HIP devices:    error {s} ({s})\n", .{ @errorName(err), formatLastError(&err_buf) });
        return;
    };

    try writer.print("HIP devices:    {d}\n", .{count});
    if (count == 0) {
        try writer.print("              no HIP devices visible to this process\n", .{});
        return;
    }

    var i: u32 = 0;
    while (i < count) : (i += 1) {
        const info = describeDevice(@intCast(i)) catch |err| {
            var err_buf: [256]u8 = undefined;
            try writer.print("\nGPU {d}: error {s} ({s})\n", .{ i, @errorName(err), formatLastError(&err_buf) });
            continue;
        };
        try printDevice(writer, info);
    }
}

pub fn printDevice(writer: *std.Io.Writer, info: GpuInfo) !void {
    var bytes_buf: [64]u8 = undefined;
    const marker = if (info.isTargetGpu()) "  [target match]" else "";
    try writer.print("\nGPU {d}{s}\n", .{ info.index, marker });
    try writer.print("  name:              {s}\n", .{info.nameSlice()});
    try writer.print("  LLVM target:       {s}\n", .{info.archSlice()});
    try writer.print("  VRAM:              {s}\n", .{util.formatBytes(&bytes_buf, info.total_mem_bytes)});
    try writer.print("  compute units:     {d}\n", .{info.multiprocessor_count});
    try writer.print("  wave/warp size:    {d}\n", .{info.warp_size});
    try writer.print("  max threads/block: {d}\n", .{info.max_threads_per_block});
    try writer.print("  clock:             {d} kHz\n", .{info.clock_rate_khz});
    try writer.print("  memory clock:      {d} kHz\n", .{info.memory_clock_rate_khz});
    try writer.print("  memory bus:        {d} bits\n", .{info.memory_bus_width_bits});
    try writer.print("  L2 cache:          {d} bytes\n", .{info.l2_cache_bytes});
    try writer.print("  LDS/block:         {d} bytes\n", .{info.shared_mem_per_block});
    try writer.print("  regs/block:        {d}\n", .{info.regs_per_block});
    try writer.print("  PCI:               {d}:{d}:{d}\n", .{ info.pci_domain, info.pci_bus, info.pci_device });
    try writer.print("  integrated:        {s}\n", .{yesNo(info.integrated)});
    try writer.print("  map host memory:   {s}\n", .{yesNo(info.can_map_host_memory)});
    try writer.print("  concurrent kernels: {s}\n", .{yesNo(info.concurrent_kernels)});
}

fn yesNo(value: i32) []const u8 {
    return if (value != 0) "yes" else "no";
}

test "GpuInfo target detection" {
    var info = std.mem.zeroes(GpuInfo);
    @memcpy(info.gcn_arch[0..7], "gfx1201");
    try std.testing.expect(info.isTargetGpu());
    @memcpy(info.gcn_arch[0..7], "gfx1100");
    @memcpy(info.name[0..20], "AMD Radeon AI PRO R9");
    info.name[20] = '7';
    info.name[21] = '0';
    info.name[22] = '0';
    try std.testing.expect(info.isTargetGpu());
}

test "HIP unavailable path" {
    if (have_hip) return error.SkipZigTest;
    try std.testing.expectError(error.HipUnavailable, deviceCount());
    try std.testing.expectError(error.HipUnavailable, runtimeVersion());
    try std.testing.expectError(error.HipUnavailable, describeDevice(0));
}

test "HIP enumerates devices when linked" {
    if (!have_hip) return error.SkipZigTest;
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    const count = deviceCount() catch |err| {
        std.debug.print("HIP deviceCount failed: {s}\n", .{@errorName(err)});
        return err;
    };
    try std.testing.expect(count >= 1);
    const info = try describeDevice(0);
    try std.testing.expect(info.nameSlice().len > 0);
    try std.testing.expect(info.archSlice().len > 0);
    try std.testing.expect(info.total_mem_bytes > 0);
}
