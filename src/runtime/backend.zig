//! Backend identity vs device architecture, plus capability records.
//!
//! `BackendKind` is the execution API family. `DeviceArchitecture` is the
//! hardware family used for kernel selection. They are not the same enum.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

pub const BackendKind = enum {
    cpu,
    apple,
    amd_hip,

    pub fn name(self: BackendKind) []const u8 {
        return switch (self) {
            .cpu => "cpu",
            .apple => "apple",
            .amd_hip => "amd-hip",
        };
    }
};

pub const AppleExecutionPath = enum {
    metal_baseline,
    // accelerate / core_ml are discovered as capabilities, not silently used.
};

pub const AppleMFeatures = struct {
    /// Metal device name, for reports only. Kernel selection must not parse this.
    device_name: [256]u8 = [_]u8{0} ** 256,
    recommended_working_set_bytes: u64 = 0,
    max_buffer_bytes: u64 = 0,
    max_threads_per_threadgroup: u32 = 0,
    unified_memory: bool = true,
    gpu_family_apple7: bool = false,
    gpu_family_apple8: bool = false,
    gpu_family_apple9: bool = false,
    /// True when the device claims SIMD-group matrix support (Apple7+).
    /// Presence of the feature is not proof that a simdgroup_matrix kernel ran.
    simdgroup_matrix_available: bool = false,
    accelerate_available: bool = false,
    core_ml_available: bool = false,
    sme_available: bool = false,

    pub fn nameSlice(self: *const AppleMFeatures) []const u8 {
        return std.mem.sliceTo(&self.device_name, 0);
    }
};

pub const AmdRdnaFeatures = struct {
    llvm_target: [32]u8 = [_]u8{0} ** 32,
    total_mem_bytes: u64 = 0,
};

pub const DeviceArchitecture = union(enum) {
    generic_cpu,
    apple_m: AppleMFeatures,
    amd_rdna: AmdRdnaFeatures,

    pub fn name(self: DeviceArchitecture) []const u8 {
        return switch (self) {
            .generic_cpu => "generic-cpu",
            .apple_m => "apple-m",
            .amd_rdna => "amd-rdna",
        };
    }
};

pub const Capabilities = struct {
    backend: BackendKind,
    arch: DeviceArchitecture,
    unified_memory: bool = false,
    fp32: bool = true,
    fp16: bool = false,
    bf16: bool = false,
    simdgroup_matrix: bool = false,
    accelerate: bool = false,
    core_ml: bool = false,
    hip: bool = false,
    /// Human-readable reasons that optional paths are disabled.
    disabled: [8][]const u8 = [_][]const u8{""} ** 8,
    disabled_len: usize = 0,

    pub fn addDisabled(self: *Capabilities, reason: []const u8) void {
        if (self.disabled_len >= self.disabled.len) return;
        self.disabled[self.disabled_len] = reason;
        self.disabled_len += 1;
    }
};

pub const SelectionError = error{
    UnknownBackend,
    BackendUnavailable,
};

/// Parse a forced backend name. Invalid names fail. Missing implementations
/// fail; they do not silently fall back.
pub fn parseBackendKind(text: []const u8) SelectionError!BackendKind {
    if (std.mem.eql(u8, text, "cpu")) return .cpu;
    if (std.mem.eql(u8, text, "apple") or std.mem.eql(u8, text, "metal")) return .apple;
    if (std.mem.eql(u8, text, "amd") or std.mem.eql(u8, text, "hip") or std.mem.eql(u8, text, "amd-hip"))
        return .amd_hip;
    return error.UnknownBackend;
}

pub fn isBackendBuildable(kind: BackendKind) bool {
    return switch (kind) {
        .cpu => true,
        .apple => build_options.have_apple,
        .amd_hip => build_options.have_hip,
    };
}

pub fn requireBackend(kind: BackendKind) SelectionError!void {
    if (!isBackendBuildable(kind)) return error.BackendUnavailable;
}

pub fn defaultKind() BackendKind {
    if (build_options.have_apple) return .apple;
    if (build_options.have_hip) return .amd_hip;
    return .cpu;
}

pub fn cpuCapabilities() Capabilities {
    var caps = Capabilities{
        .backend = .cpu,
        .arch = .generic_cpu,
        .unified_memory = true,
        .fp32 = true,
        .fp16 = false,
        .bf16 = false,
        .accelerate = build_options.have_apple,
    };
    caps.addDisabled("fp16 CPU path not implemented (f32 oracle only)");
    if (!build_options.have_apple) {
        caps.addDisabled("Accelerate/BNNS not available on this host; CPU path is the scalar reference");
    } else {
        caps.addDisabled("Accelerate vDSP matmul is size-gated (M*N*K>=262144); scalar remains the oracle");
    }
    caps.addDisabled("SME/SME2 not implemented");
    caps.addDisabled("Core ML/ANE not implemented");
    return caps;
}

test "parseBackendKind accepts metal alias and rejects cuda" {
    try std.testing.expectError(error.UnknownBackend, parseBackendKind("cuda"));
    try std.testing.expectEqual(BackendKind.cpu, try parseBackendKind("cpu"));
    try std.testing.expectEqual(BackendKind.apple, try parseBackendKind("metal"));
    try std.testing.expectEqual(BackendKind.amd_hip, try parseBackendKind("hip"));
}

test "cpu backend is always buildable" {
    try std.testing.expect(isBackendBuildable(.cpu));
    try requireBackend(.cpu);
}
