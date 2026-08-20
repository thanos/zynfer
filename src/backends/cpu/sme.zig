//! SME/SME2 hardware detection and explicit non-path.
//!
//! Stage 7: feature-gate SME from sysctl / Zig target features. Do not claim
//! SME execution. Direct SME kernels are not retained with the supported
//! Zig/Clang toolchain (no maintainable public intrinsic path in-tree).
//! Force with `ZYNFER_FORCE_SME=1` to verify the failure is loud.

const std = @import("std");
const builtin = @import("builtin");
const Tensor = @import("../../runtime/tensor.zig").Tensor;
const TensorError = @import("../../runtime/tensor.zig").TensorError;

pub const Error = TensorError || error{Unsupported};

pub const Probe = struct {
    /// Runtime host reports `hw.optional.arm.FEAT_SME`.
    feat_sme: bool = false,
    /// Runtime host reports `hw.optional.arm.FEAT_SME2`.
    feat_sme2: bool = false,
    /// Compile target enables Zig `aarch64.feature.sme` (may differ from host).
    target_sme: bool = false,
    /// Inference path retained (kernels implemented and measured). Always false today.
    path_retained: bool = false,
    detail: []const u8 = "SME/SME2 kernels not implemented; detection only",
};

pub var last_path: []const u8 = "unset";

fn targetHasSme() bool {
    if (comptime builtin.cpu.arch != .aarch64) return false;
    return comptime builtin.cpu.features.isEnabled(@intFromEnum(std.Target.aarch64.Feature.sme));
}

fn sysctlBool(name: [*:0]const u8) bool {
    if (comptime builtin.os.tag != .macos) return false;
    var val: c_int = 0;
    var len: usize = @sizeOf(c_int);
    if (std.c.sysctlbyname(name, &val, &len, null, 0) != 0) return false;
    return val != 0;
}

/// Probe host + target. Never implies kernels will run.
pub fn probe() Probe {
    var p = Probe{
        .feat_sme = sysctlBool("hw.optional.arm.FEAT_SME"),
        .feat_sme2 = sysctlBool("hw.optional.arm.FEAT_SME2"),
        .target_sme = targetHasSme(),
        .path_retained = false,
    };
    if (p.feat_sme or p.feat_sme2) {
        p.detail = "SME hardware present; kernels still not implemented (no retained Zig/Clang SME path)";
    } else if (comptime builtin.os.tag == .macos and builtin.cpu.arch == .aarch64) {
        p.detail = "No FEAT_SME/SME2 on this host (expected on M1–M3); Accelerate remains the measured CPU matrix path";
    } else if (comptime builtin.os.tag != .macos) {
        p.detail = "SME probe is macOS/arm64-only; host is not Apple Silicon";
    }
    return p;
}

pub fn hardwareAvailable() bool {
    const p = probe();
    return p.feat_sme or p.feat_sme2;
}

/// True when the user forced SME and must receive a hard error.
pub fn forceRequested() bool {
    const raw = std.c.getenv("ZYNFER_FORCE_SME") orelse return false;
    const v = std.mem.span(raw);
    return std.mem.eql(u8, v, "1") or std.mem.eql(u8, v, "true") or std.mem.eql(u8, v, "on");
}

/// Stage 7 gate: no SME matmul is retained. Always Unsupported.
pub fn matmul(c_out: Tensor, a: Tensor, b: Tensor) Error!void {
    _ = c_out;
    _ = a;
    _ = b;
    last_path = "sme_unsupported";
    return error.Unsupported;
}

pub fn matvec(y: Tensor, a: Tensor, x: Tensor) Error!void {
    _ = y;
    _ = a;
    _ = x;
    last_path = "sme_unsupported";
    return error.Unsupported;
}

test "SME probe runs without claiming a path" {
    const p = probe();
    try std.testing.expect(!p.path_retained);
    if (comptime builtin.os.tag == .macos and builtin.cpu.arch == .aarch64) {
        // M1–M3 class hosts report FEAT_SME=0; M4+ may report 1.
        _ = p.feat_sme;
        _ = p.feat_sme2;
    } else {
        try std.testing.expect(!p.feat_sme);
        try std.testing.expect(!p.feat_sme2);
    }
}

test "SME matmul is Unsupported" {
    const gpa = std.testing.allocator;
    var a = try Tensor.alloc(gpa, .f32, &.{ 2, 2 });
    defer a.deinit();
    var b = try Tensor.alloc(gpa, .f32, &.{ 2, 2 });
    defer b.deinit();
    var c = try Tensor.alloc(gpa, .f32, &.{ 2, 2 });
    defer c.deinit();
    try std.testing.expectError(error.Unsupported, matmul(c, a, b));
    try std.testing.expectEqualStrings("sme_unsupported", last_path);
}
