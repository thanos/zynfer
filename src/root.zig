//! Zynfer library root.
//!
//! Stage 0 exposes only environment diagnostics, HIP device enumeration, and a
//! thin Device handle. Transformer code is intentionally absent.

pub const hip = @import("hip.zig");
pub const device = @import("device.zig");
pub const env = @import("env_report.zig");
pub const util = @import("util.zig");

pub const Device = device.Device;

test {
    _ = hip;
    _ = device;
    _ = env;
    _ = util;
}
