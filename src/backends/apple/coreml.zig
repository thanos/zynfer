//! Core ML / ANE Stage 7 probe and explicit non-path.
//!
//! ANE is only reachable through Core ML. Stage 7 links the framework and
//! probes compute-unit configuration, but does **not** retain an inference
//! path: there is no stable compiled subgraph (needs Stages 10–12), and ANE
//! placement is unverified. Force with `ZYNFER_FORCE_COREML=1` for a loud fail.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const Tensor = @import("../../runtime/tensor.zig").Tensor;
const TensorError = @import("../../runtime/tensor.zig").TensorError;

pub const Error = TensorError || error{Unsupported};

pub const have_coreml = build_options.have_apple and builtin.os.tag == .macos;

pub const Probe = struct {
    framework_linked: bool = false,
    configuration_ok: bool = false,
    compute_units_all_ok: bool = false,
    compute_units_cpu_and_ane_ok: bool = false,
    /// Always false unless Instruments-confirmed (never set by this probe).
    ane_execution_verified: bool = false,
    /// Inference path retained after Stage 7 gate. False = disabled.
    path_retained: bool = false,
    detail: []const u8 = "Core ML not probed",
};

pub var last_path: []const u8 = "unset";

const c = if (have_coreml) struct {
    pub const ProbeC = extern struct {
        framework_linked: c_int,
        configuration_ok: c_int,
        compute_units_all_ok: c_int,
        compute_units_cpu_and_ane_ok: c_int,
        ane_execution_verified: c_int,
        detail: [512]u8,
    };
    pub extern fn zynfer_coreml_probe(out: *ProbeC) c_int;
} else struct {};

var detail_buf: [512]u8 = undefined;

pub fn probe() Probe {
    if (comptime !have_coreml) {
        return .{
            .detail = "Core ML bridge not compiled (non-macOS / Apple off)",
        };
    }
    var raw: c.ProbeC = std.mem.zeroes(c.ProbeC);
    const st = c.zynfer_coreml_probe(&raw);
    const detail_z = std.mem.sliceTo(&raw.detail, 0);
    const n = @min(detail_z.len, detail_buf.len);
    @memcpy(detail_buf[0..n], detail_z[0..n]);
    if (n < detail_buf.len) detail_buf[n] = 0;

    return .{
        .framework_linked = raw.framework_linked != 0 and st == 0,
        .configuration_ok = raw.configuration_ok != 0,
        .compute_units_all_ok = raw.compute_units_all_ok != 0,
        .compute_units_cpu_and_ane_ok = raw.compute_units_cpu_and_ane_ok != 0,
        .ane_execution_verified = raw.ane_execution_verified != 0,
        .path_retained = false,
        .detail = std.mem.sliceTo(&detail_buf, 0),
    };
}

pub fn forceRequested() bool {
    const raw = std.c.getenv("ZYNFER_FORCE_COREML") orelse return false;
    const v = std.mem.span(raw);
    return std.mem.eql(u8, v, "1") or std.mem.eql(u8, v, "true") or std.mem.eql(u8, v, "on");
}

/// No Core ML op path is retained. Always Unsupported.
pub fn matmul(c_out: Tensor, a: Tensor, b: Tensor) Error!void {
    _ = c_out;
    _ = a;
    _ = b;
    last_path = "coreml_unsupported";
    return error.Unsupported;
}

test "Core ML probe does not claim ANE execution" {
    if (comptime !have_coreml) return error.SkipZigTest;
    const p = probe();
    try std.testing.expect(p.framework_linked);
    try std.testing.expect(p.configuration_ok);
    try std.testing.expect(p.compute_units_all_ok);
    try std.testing.expect(p.compute_units_cpu_and_ane_ok);
    try std.testing.expect(!p.ane_execution_verified);
    try std.testing.expect(!p.path_retained);
}

test "Core ML matmul is Unsupported" {
    const gpa = std.testing.allocator;
    var a = try Tensor.alloc(gpa, .f32, &.{ 2, 2 });
    defer a.deinit();
    var b = try Tensor.alloc(gpa, .f32, &.{ 2, 2 });
    defer b.deinit();
    var c_out = try Tensor.alloc(gpa, .f32, &.{ 2, 2 });
    defer c_out.deinit();
    try std.testing.expectError(error.Unsupported, matmul(c_out, a, b));
    try std.testing.expectEqualStrings("coreml_unsupported", last_path);
}
