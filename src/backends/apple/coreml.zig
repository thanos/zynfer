//! Core ML / ANE probe and explicit non-path (Stage 7 → Stage M7 final).
//!
//! ANE is only reachable through Core ML. There is no public ANE ISA.
//! Stage 7 linked the framework and probed compute units. Stage M7 reopens
//! the question at Qwen scale (MLState / fixed-shape prefill / compression)
//! and **closes it**: no retained inference path without Instruments-confirmed
//! ANE placement **and** an e2e win over M6 Metal.
//!
//! Force with `ZYNFER_FORCE_COREML=1` for a loud fail (exit 2).

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
    /// macOS 15+ / Core ML stateful API class present.
    ml_state_available: bool = false,
    macos_major: u32 = 0,
    model_artifact_present: bool = false,
    /// Inference path retained after Stage M7 gate. False = disabled.
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
        ml_state_available: c_int,
        macos_major: c_int,
        model_artifact_present: c_int,
        detail: [768]u8,
    };
    pub const SmokeC = extern struct {
        load_ok: c_int,
        predict_ok: c_int,
        compute_units: [64]u8,
        y0: f32,
        y1: f32,
        y2: f32,
        y3: f32,
        detail: [768]u8,
    };
    pub extern fn zynfer_coreml_probe(out: *ProbeC) c_int;
    pub extern fn zynfer_coreml_smoke(path: [*:0]const u8, out: *SmokeC) c_int;
} else struct {};

var detail_buf: [768]u8 = undefined;
var smoke_detail_buf: [768]u8 = undefined;
var smoke_units_buf: [64]u8 = undefined;

pub const Smoke = struct {
    load_ok: bool = false,
    predict_ok: bool = false,
    compute_units: []const u8 = "",
    y: [4]f32 = .{ 0, 0, 0, 0 },
    detail: []const u8 = "Core ML smoke not run",
    /// Bridge status: 0 = ok, else load/predict failure.
    status: c_int = 1,
};

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
        .ml_state_available = raw.ml_state_available != 0,
        .macos_major = if (raw.macos_major > 0) @intCast(raw.macos_major) else 0,
        .model_artifact_present = raw.model_artifact_present != 0,
        .path_retained = false,
        .detail = std.mem.sliceTo(&detail_buf, 0),
    };
}

pub fn forceRequested() bool {
    const raw = std.c.getenv("ZYNFER_FORCE_COREML") orelse return false;
    const v = std.mem.span(raw);
    return std.mem.eql(u8, v, "1") or std.mem.eql(u8, v, "true") or std.mem.eql(u8, v, "on");
}

/// Load a tiny .mlpackage and run one predict. Does **not** set
/// `ane_execution_verified` — Instruments is still required for ANE claims.
pub fn smoke(path: []const u8) Smoke {
    if (comptime !have_coreml) {
        return .{
            .detail = "Core ML bridge not compiled (non-macOS / Apple off)",
            .status = 1,
        };
    }
    var path_z_buf: [4096]u8 = undefined;
    if (path.len >= path_z_buf.len) {
        return .{
            .detail = "model path too long",
            .status = 2,
        };
    }
    @memcpy(path_z_buf[0..path.len], path);
    path_z_buf[path.len] = 0;

    var raw: c.SmokeC = std.mem.zeroes(c.SmokeC);
    const st = c.zynfer_coreml_smoke(path_z_buf[0..path.len :0].ptr, &raw);

    const detail_z = std.mem.sliceTo(&raw.detail, 0);
    const dn = @min(detail_z.len, smoke_detail_buf.len);
    @memcpy(smoke_detail_buf[0..dn], detail_z[0..dn]);
    if (dn < smoke_detail_buf.len) smoke_detail_buf[dn] = 0;

    const units_z = std.mem.sliceTo(&raw.compute_units, 0);
    const un = @min(units_z.len, smoke_units_buf.len);
    @memcpy(smoke_units_buf[0..un], units_z[0..un]);
    if (un < smoke_units_buf.len) smoke_units_buf[un] = 0;

    return .{
        .load_ok = raw.load_ok != 0,
        .predict_ok = raw.predict_ok != 0,
        .compute_units = std.mem.sliceTo(&smoke_units_buf, 0),
        .y = .{ raw.y0, raw.y1, raw.y2, raw.y3 },
        .detail = std.mem.sliceTo(&smoke_detail_buf, 0),
        .status = st,
    };
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
    try std.testing.expect(!p.model_artifact_present);
    // Darwin 24+ (macOS 15) typically exposes MLState; older hosts may not.
    if (p.macos_major >= 24) {
        try std.testing.expect(p.ml_state_available);
    }
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

test "Stage M7: path_retained stays false after Qwen-scale gate" {
    if (comptime !have_coreml) return error.SkipZigTest;
    const p = probe();
    try std.testing.expect(!p.path_retained);
    try std.testing.expect(!p.ane_execution_verified);
    try std.testing.expect(std.mem.indexOf(u8, p.detail, "Stage M7") != null);
}

test "Stage M7: toy Core ML load smoke" {
    if (comptime !have_coreml) return error.SkipZigTest;
    const path = "tools/fixtures/coreml_toy.mlpackage";
    std.Io.Dir.cwd().access(std.testing.io, path, .{}) catch return error.SkipZigTest;
    const s = smoke(path);
    try std.testing.expectEqual(@as(c_int, 0), s.status);
    try std.testing.expect(s.load_ok);
    try std.testing.expect(s.predict_ok);
    // Ones input → known toy output from make_coreml_toy.py
    try std.testing.expect(@abs(s.y[0] - 1.0205078) < 1e-3);
    try std.testing.expect(@abs(s.y[3] - 1.4589844) < 1e-3);
    // Smoke never upgrades ANE verification.
    try std.testing.expect(!probe().ane_execution_verified);
}
