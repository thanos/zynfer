const std = @import("std");
const hip = @import("hip.zig");

/// Thin wrapper around a HIP device index.
///
/// Stage 0 only enumerates devices and prints properties. Allocation, copies,
/// and streams arrive in Stage 1.
pub const Device = struct {
    index: i32,
    info: hip.GpuInfo,

    pub fn init(index: i32) hip.Error!Device {
        return .{
            .index = index,
            .info = try hip.describeDevice(index),
        };
    }

    pub fn initFirst() hip.Error!Device {
        const count = try hip.deviceCount();
        if (count == 0) return error.NoDevice;
        return init(0);
    }

    pub fn name(self: Device) []const u8 {
        return self.info.nameSlice();
    }

    pub fn arch(self: Device) []const u8 {
        return self.info.archSlice();
    }
};

test "Device.initFirst is unavailable without HIP" {
    if (hip.have_hip) return error.SkipZigTest;
    try std.testing.expectError(error.HipUnavailable, Device.initFirst());
}
