//! Zynfer library root.
//!
//! Inference math lives in backend-neutral runtime types plus CPU/Apple/AMD
//! adapters. Stage 0 HIP diagnostics remain for the AMD host path.

pub const hip = @import("hip.zig");
pub const device = @import("device.zig");
pub const env = @import("env_report.zig");
pub const util = @import("util.zig");

pub const dtype = @import("runtime/dtype.zig");
pub const tensor = @import("runtime/tensor.zig");
pub const backend = @import("runtime/backend.zig");
pub const compare = @import("runtime/compare.zig");

pub const cpu = struct {
    pub const ops = @import("backends/cpu/ops.zig");
};

pub const apple = struct {
    pub const gpu = @import("backends/apple/gpu.zig");
    pub const ops = @import("backends/apple/ops.zig");
};

pub const Device = device.Device;
pub const Tensor = tensor.Tensor;
pub const DType = dtype.DType;
pub const BackendKind = backend.BackendKind;

test {
    _ = hip;
    _ = device;
    _ = env;
    _ = util;
    _ = dtype;
    _ = tensor;
    _ = backend;
    _ = compare;
    _ = cpu.ops;
    _ = apple.gpu;
    _ = apple.ops;
}
