//! Apple GPU device: Metal device, shared buffers, pipeline cache, capabilities.
//!
//! Zig owns policy. The Objective-C bridge owns retain/release. Model code
//! must not import this module.

const std = @import("std");
const build_options = @import("build_options");
const backend_mod = @import("../../runtime/backend.zig");

pub const have_apple = build_options.have_apple;

/// CI macOS runners may lack the Metal shader toolchain. Tests skip when this
/// is set; production `Gpu.init` still attempts a real device.
pub fn skipAppleGpuTests() bool {
    if (!have_apple) return true;
    const v = std.process.Environ.getPosix(std.testing.environ, "ZYNFER_SKIP_APPLE_GPU") orelse return false;
    return std.mem.eql(u8, v, "1") or std.mem.eql(u8, v, "true");
}

pub const Error = error{
    AppleUnavailable,
    NoDevice,
    CompileFailed,
    PipelineFailed,
    OutOfMemory,
    EncodeFailed,
    Unsupported,
    Invalid,
};

pub const MtlDevice = opaque {};
pub const MtlBuffer = opaque {};

pub const CapsC = extern struct {
    device_name: [256]u8,
    recommended_working_set_bytes: u64,
    max_buffer_bytes: u64,
    max_threads_per_threadgroup: u32,
    unified_memory: c_int,
    gpu_family_apple7: c_int,
    gpu_family_apple8: c_int,
    gpu_family_apple9: c_int,
    simdgroup_matrix_available: c_int,
};

const c = if (have_apple) struct {
    pub extern fn zynfer_mtl_device_create(out: *?*MtlDevice) c_int;
    pub extern fn zynfer_mtl_device_destroy(dev: *MtlDevice) void;
    pub extern fn zynfer_mtl_caps(dev: *const MtlDevice, out: *CapsC) c_int;
    pub extern fn zynfer_mtl_compile_library(dev: *MtlDevice, source: [*:0]const u8) c_int;
    pub extern fn zynfer_mtl_last_error(dev: *const MtlDevice) [*:0]const u8;
    pub extern fn zynfer_mtl_buffer_create(dev: *MtlDevice, bytes: usize, out: *?*MtlBuffer) c_int;
    pub extern fn zynfer_mtl_buffer_destroy(buf: *MtlBuffer) void;
    pub extern fn zynfer_mtl_buffer_contents(buf: *MtlBuffer) ?*anyopaque;
    pub extern fn zynfer_mtl_buffer_length(buf: *const MtlBuffer) usize;
    pub extern fn zynfer_mtl_encode_and_wait(
        dev: *MtlDevice,
        kernel: [*:0]const u8,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        tg_x: u32,
        tg_y: u32,
        tg_z: u32,
        bufs: [*]*MtlBuffer,
        nbufs: u32,
        params: ?*const anyopaque,
        params_len: u32,
        threadgroup_mem_bytes: u32,
        dispatch_threadgroups: c_int,
    ) c_int;
    pub extern fn zynfer_mtl_batch_begin(dev: *MtlDevice) c_int;
    pub extern fn zynfer_mtl_batch_encode(
        dev: *MtlDevice,
        kernel: [*:0]const u8,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        tg_x: u32,
        tg_y: u32,
        tg_z: u32,
        bufs: [*]*MtlBuffer,
        nbufs: u32,
        params: ?*const anyopaque,
        params_len: u32,
        threadgroup_mem_bytes: u32,
        dispatch_threadgroups: c_int,
    ) c_int;
    pub extern fn zynfer_mtl_batch_commit_and_wait(dev: *MtlDevice) c_int;
    pub extern fn zynfer_mtl_batch_abort(dev: *MtlDevice) void;
    pub extern fn zynfer_signpost_interval_begin(name: [*:0]const u8) u64;
    pub extern fn zynfer_signpost_interval_end(id: u64, name: [*:0]const u8) void;
} else struct {};

/// Opt-in Instruments interval (`ZYNFER_SIGNPOSTS=1`). No-op when disabled.
pub fn signpostBegin(name: [*:0]const u8) u64 {
    if (comptime !have_apple) return 0;
    return c.zynfer_signpost_interval_begin(name);
}

pub fn signpostEnd(id: u64, name: [*:0]const u8) void {
    if (comptime !have_apple) return;
    if (id == 0) return;
    c.zynfer_signpost_interval_end(id, name);
}

const kernels_source: [:0]const u8 = if (have_apple) @embedFile("kernels.metal") else "";

fn statusToError(status: c_int) Error {
    return switch (status) {
        0 => unreachable,
        1 => error.NoDevice,
        2 => error.CompileFailed,
        3 => error.PipelineFailed,
        4 => error.OutOfMemory,
        5 => error.EncodeFailed,
        6 => error.Unsupported,
        else => error.Invalid,
    };
}

pub const Buffer = struct {
    handle: if (have_apple) *MtlBuffer else void,
    bytes: []u8,

    pub fn deinit(self: *Buffer) void {
        if (!have_apple) return;
        c.zynfer_mtl_buffer_destroy(self.handle);
        self.* = undefined;
    }

    pub fn f32s(self: Buffer) []f32 {
        const n = self.bytes.len / 4;
        return @as([*]f32, @ptrCast(@alignCast(self.bytes.ptr)))[0..n];
    }
};

pub const Gpu = struct {
    handle: if (have_apple) *MtlDevice else void,
    features: backend_mod.AppleMFeatures = .{},
    last_error_buf: [512]u8 = [_]u8{0} ** 512,
    /// True between `batchBegin` and `batchCommit`/`batchAbort`.
    batch_active: bool = false,
    /// Dispatches encoded in the current (or last completed) batch.
    last_batch_encodes: u32 = 0,

    pub fn init() Error!Gpu {
        if (!have_apple) return error.AppleUnavailable;
        var handle: ?*MtlDevice = null;
        const st = c.zynfer_mtl_device_create(&handle);
        if (st != 0 or handle == null) return statusToError(if (st == 0) 1 else st);
        var gpu = Gpu{ .handle = handle.? };
        const compile_st = c.zynfer_mtl_compile_library(gpu.handle, kernels_source.ptr);
        if (compile_st != 0) {
            gpu.captureError();
            std.log.err("Metal library compile failed: {s}", .{gpu.lastError()});
            c.zynfer_mtl_device_destroy(gpu.handle);
            return statusToError(compile_st);
        }
        try gpu.loadCaps();
        return gpu;
    }

    pub fn deinit(self: *Gpu) void {
        if (!have_apple) return;
        if (self.batch_active) self.batchAbort();
        c.zynfer_mtl_device_destroy(self.handle);
        self.* = undefined;
    }

    pub fn lastError(self: *const Gpu) []const u8 {
        return std.mem.sliceTo(&self.last_error_buf, 0);
    }

    fn captureError(self: *Gpu) void {
        if (!have_apple) return;
        const msg = std.mem.span(c.zynfer_mtl_last_error(self.handle));
        const n = @min(msg.len, self.last_error_buf.len - 1);
        @memcpy(self.last_error_buf[0..n], msg[0..n]);
        self.last_error_buf[n] = 0;
    }

    fn loadCaps(self: *Gpu) Error!void {
        if (!have_apple) return error.AppleUnavailable;
        var raw: CapsC = std.mem.zeroes(CapsC);
        const st = c.zynfer_mtl_caps(self.handle, &raw);
        if (st != 0) return statusToError(st);
        self.features.device_name = raw.device_name;
        self.features.recommended_working_set_bytes = raw.recommended_working_set_bytes;
        self.features.max_buffer_bytes = raw.max_buffer_bytes;
        self.features.max_threads_per_threadgroup = raw.max_threads_per_threadgroup;
        self.features.unified_memory = raw.unified_memory != 0;
        self.features.gpu_family_apple7 = raw.gpu_family_apple7 != 0;
        self.features.gpu_family_apple8 = raw.gpu_family_apple8 != 0;
        self.features.gpu_family_apple9 = raw.gpu_family_apple9 != 0;
        self.features.simdgroup_matrix_available = raw.simdgroup_matrix_available != 0;
        self.features.accelerate_available = true; // linked on Apple builds; path is size-gated
        const sme_p = @import("../cpu/sme.zig").probe();
        self.features.sme_available = sme_p.feat_sme;
        self.features.sme2_available = sme_p.feat_sme2;
        const cm_p = @import("coreml.zig").probe();
        self.features.core_ml_available = cm_p.framework_linked;
    }

    pub fn capabilities(self: *const Gpu) backend_mod.Capabilities {
        const sme_p = @import("../cpu/sme.zig").probe();
        const cm_p = @import("coreml.zig").probe();
        var caps = backend_mod.Capabilities{
            .backend = .apple,
            .arch = .{ .apple_m = self.features },
            .unified_memory = self.features.unified_memory,
            .fp32 = true,
            .fp16 = false,
            .bf16 = false,
            .simdgroup_matrix = self.features.simdgroup_matrix_available,
            .accelerate = self.features.accelerate_available,
            .core_ml = cm_p.path_retained,
            .sme = sme_p.path_retained,
        };
        caps.addDisabled("fp16/bf16 Metal kernels not implemented");
        caps.addDisabled(sme_p.detail);
        caps.addDisabled(cm_p.detail);
        caps.addDisabled("Core ML/ANE inference path not retained (Stage 7: no measured end-to-end subgraph)");
        caps.addDisabled("AMX is not a public contract; CPU matrix wins attribute to Accelerate only");
        caps.addDisabled("No tokenizer/sampling; Stage 10 loads .zynfer metadata/tensors only");
        caps.addDisabled("Full Qwen forward is Stage 11; TTFT/tok/s need Stage 12");
        caps.addDisabled("Metal attention_f32 caps kv_len at 256 (thread-local scores); larger returns Unsupported");
        caps.addDisabled("Per-op apple.ops path still waits each kernel; Stage 6 batch+resident KV is the tiny-block schedule");
        caps.addDisabled("Each Gpu is single-owner; concurrent sessions need one Gpu (and buffers) per thread");
        if (!self.features.simdgroup_matrix_available) {
            caps.addDisabled("simdgroup_matrix hardware not available; matmul uses naive f32 only");
        }
        return caps;
    }

    /// Open a multi-dispatch command buffer. `launch`/`launchOpts` encode onto it
    /// until `batchCommit` (one wait) or `batchAbort`.
    pub fn batchBegin(self: *Gpu) Error!void {
        if (!have_apple) return error.AppleUnavailable;
        if (self.batch_active) return error.Invalid;
        const st = c.zynfer_mtl_batch_begin(self.handle);
        if (st != 0) {
            self.captureError();
            return statusToError(st);
        }
        self.batch_active = true;
        self.last_batch_encodes = 0;
    }

    pub fn batchCommit(self: *Gpu) Error!void {
        if (!have_apple) return error.AppleUnavailable;
        if (!self.batch_active) return error.Invalid;
        const st = c.zynfer_mtl_batch_commit_and_wait(self.handle);
        self.batch_active = false;
        if (st != 0) {
            self.captureError();
            return statusToError(st);
        }
    }

    pub fn batchAbort(self: *Gpu) void {
        if (!have_apple) return;
        if (!self.batch_active) return;
        c.zynfer_mtl_batch_abort(self.handle);
        self.batch_active = false;
    }

    pub fn allocShared(self: *Gpu, bytes: usize) Error!Buffer {
        if (!have_apple) return error.AppleUnavailable;
        var handle: ?*MtlBuffer = null;
        const st = c.zynfer_mtl_buffer_create(self.handle, bytes, &handle);
        if (st != 0 or handle == null) {
            self.captureError();
            return statusToError(if (st == 0) 4 else st);
        }
        const ptr = c.zynfer_mtl_buffer_contents(handle.?);
        const len = c.zynfer_mtl_buffer_length(handle.?);
        if (ptr == null or len < bytes) return error.OutOfMemory;
        return .{
            .handle = handle.?,
            .bytes = @as([*]u8, @ptrCast(ptr))[0..bytes],
        };
    }

    pub const LaunchOpts = struct {
        /// Bytes of `[[threadgroup(0)]]` memory. 0 means none.
        threadgroup_mem_bytes: u32 = 0,
        /// If true, grid_* are threadgroup counts (`dispatchThreadgroups`).
        threadgroups: bool = false,
    };

    pub fn launch(
        self: *Gpu,
        kernel: [:0]const u8,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        tg_x: u32,
        tg_y: u32,
        tg_z: u32,
        bufs: []const *MtlBuffer,
        params: []const u8,
    ) Error!void {
        try self.launchOpts(kernel, grid_x, grid_y, grid_z, tg_x, tg_y, tg_z, bufs, params, .{});
    }

    pub fn launchOpts(
        self: *Gpu,
        kernel: [:0]const u8,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        tg_x: u32,
        tg_y: u32,
        tg_z: u32,
        bufs: []const *MtlBuffer,
        params: []const u8,
        opts: LaunchOpts,
    ) Error!void {
        if (!have_apple) return error.AppleUnavailable;
        const st = if (self.batch_active)
            c.zynfer_mtl_batch_encode(
                self.handle,
                kernel.ptr,
                grid_x,
                grid_y,
                grid_z,
                tg_x,
                tg_y,
                tg_z,
                @constCast(bufs.ptr),
                @intCast(bufs.len),
                if (params.len == 0) null else params.ptr,
                @intCast(params.len),
                opts.threadgroup_mem_bytes,
                if (opts.threadgroups) 1 else 0,
            )
        else
            c.zynfer_mtl_encode_and_wait(
                self.handle,
                kernel.ptr,
                grid_x,
                grid_y,
                grid_z,
                tg_x,
                tg_y,
                tg_z,
                @constCast(bufs.ptr),
                @intCast(bufs.len),
                if (params.len == 0) null else params.ptr,
                @intCast(params.len),
                opts.threadgroup_mem_bytes,
                if (opts.threadgroups) 1 else 0,
            );
        if (st != 0) {
            self.captureError();
            if (self.batch_active) self.batchAbort();
            return statusToError(st);
        }
        if (self.batch_active) self.last_batch_encodes += 1;
    }

    pub fn threadgroup1d(self: *const Gpu) u32 {
        const max_tg = self.features.max_threads_per_threadgroup;
        if (max_tg >= 256) return 256;
        if (max_tg >= 128) return 128;
        if (max_tg >= 64) return 64;
        return 32;
    }
};

pub fn capabilities() backend_mod.Capabilities {
    if (!have_apple) {
        var caps = backend_mod.cpuCapabilities();
        caps.backend = .apple;
        caps.addDisabled("Apple backend was not compiled (host is not macOS)");
        return caps;
    }
    var gpu = Gpu.init() catch {
        var caps = backend_mod.cpuCapabilities();
        caps.backend = .apple;
        caps.addDisabled("Metal device create or shader compile failed; on recent Xcode install the Metal Toolchain with: xcodebuild -downloadComponent MetalToolchain");
        return caps;
    };
    defer gpu.deinit();
    return gpu.capabilities();
}

test "Apple GPU create/run/destroy add kernel" {
    if (comptime !have_apple) return error.SkipZigTest;
    if (skipAppleGpuTests()) return error.SkipZigTest;
    var gpu = try Gpu.init();
    defer gpu.deinit();
    try std.testing.expect(gpu.features.unified_memory);
    try std.testing.expect(gpu.features.device_name[0] != 0);

    const n: usize = 8;
    var a = try gpu.allocShared(n * 4);
    defer a.deinit();
    var b = try gpu.allocShared(n * 4);
    defer b.deinit();
    var cbuf = try gpu.allocShared(n * 4);
    defer cbuf.deinit();
    const as = a.f32s();
    const bs = b.f32s();
    const cs = cbuf.f32s();
    for (as, 0..) |*v, i| v.* = @floatFromInt(i);
    for (bs) |*v| v.* = 1;
    const count: u32 = @intCast(n);
    if (have_apple) {
        try gpu.launch("add_f32", count, 1, 1, 8, 1, 1, &.{ a.handle, b.handle, cbuf.handle }, std.mem.asBytes(&count));
    }
    try std.testing.expectEqual(@as(f32, 1), cs[0]);
    try std.testing.expectEqual(@as(f32, 8), cs[7]);
}

test "two Gpu instances run concurrently on separate threads" {
    if (comptime !have_apple) return error.SkipZigTest;
    if (skipAppleGpuTests()) return error.SkipZigTest;

    const Worker = struct {
        fn run(ok: *std.atomic.Value(bool)) void {
            var gpu = Gpu.init() catch {
                ok.store(false, .seq_cst);
                return;
            };
            defer gpu.deinit();
            const n: usize = 64;
            var a = gpu.allocShared(n * 4) catch {
                ok.store(false, .seq_cst);
                return;
            };
            defer a.deinit();
            var b = gpu.allocShared(n * 4) catch {
                ok.store(false, .seq_cst);
                return;
            };
            defer b.deinit();
            var out = gpu.allocShared(n * 4) catch {
                ok.store(false, .seq_cst);
                return;
            };
            defer out.deinit();
            for (a.f32s(), 0..) |*v, i| v.* = @floatFromInt(i);
            for (b.f32s()) |*v| v.* = 2;
            const count: u32 = @intCast(n);
            gpu.launch("add_f32", count, 1, 1, 8, 1, 1, &.{ a.handle, b.handle, out.handle }, std.mem.asBytes(&count)) catch {
                ok.store(false, .seq_cst);
                return;
            };
            if (out.f32s()[0] != 2 or out.f32s()[63] != 65) {
                ok.store(false, .seq_cst);
                return;
            }
            ok.store(true, .seq_cst);
        }
    };

    var ok0 = std.atomic.Value(bool).init(false);
    var ok1 = std.atomic.Value(bool).init(false);
    const t0 = try std.Thread.spawn(.{}, Worker.run, .{&ok0});
    const t1 = try std.Thread.spawn(.{}, Worker.run, .{&ok1});
    t0.join();
    t1.join();
    try std.testing.expect(ok0.load(.seq_cst));
    try std.testing.expect(ok1.load(.seq_cst));
}
