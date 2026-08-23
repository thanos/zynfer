//! Stage M2 — per-family wall-time profile for one decode token + roofline helpers.

const std = @import("std");

pub const Family = enum {
    rmsnorm,
    qkv,
    rope,
    attention,
    o_proj,
    mlp,
    host_layout,
    embed,
    lm_head,
    sampling,

    pub fn label(self: Family) []const u8 {
        return switch (self) {
            .rmsnorm => "RMSNorm",
            .qkv => "QKV projection",
            .rope => "RoPE",
            .attention => "attention",
            .o_proj => "output projection",
            .mlp => "MLP",
            .host_layout => "host layout / KV append",
            .embed => "embedding",
            .lm_head => "LM head + final norm",
            .sampling => "sampling",
        };
    }
};

pub const family_count = @typeInfo(Family).@"enum".fields.len;

pub const FamilyNs = struct {
    family: Family,
    ns: u64,
};

pub const Accumulators = struct {
    ns: [family_count]u64 = .{0} ** family_count,
    metal_encodes: u64 = 0,
    metal_waits: u64 = 0,
    wall_ns: u64 = 0,
    kv_len: usize = 0,

    pub fn add(self: *Accumulators, family: Family, delta_ns: u64) void {
        self.ns[@intFromEnum(family)] += delta_ns;
    }

    pub fn sumFamilies(self: Accumulators) u64 {
        var s: u64 = 0;
        for (self.ns) |v| s += v;
        return s;
    }

    pub fn top3(self: Accumulators) [3]FamilyNs {
        var ranked: [family_count]FamilyNs = undefined;
        for (0..family_count) |i| {
            ranked[i] = .{ .family = @enumFromInt(i), .ns = self.ns[i] };
        }
        std.mem.sort(FamilyNs, &ranked, {}, struct {
            fn less(_: void, a: FamilyNs, b: FamilyNs) bool {
                return a.ns > b.ns;
            }
        }.less);
        return .{ ranked[0], ranked[1], ranked[2] };
    }
};

pub const Bandwidth = struct {
    bytes_moved: u64,
    elapsed_ns: u64,
    /// GiB/s using decimal-ish GB (1e9) as in STREAM-style benches.
    pub fn gbps(self: Bandwidth) f64 {
        if (self.elapsed_ns == 0) return 0;
        return (@as(f64, @floatFromInt(self.bytes_moved)) / 1e9) /
            (@as(f64, @floatFromInt(self.elapsed_ns)) / 1e9);
    }
};

pub const Roofline = struct {
    bytes_per_tok: u64,
    bandwidth_gbps: f64,
    /// Ideal upper bound tok/s ≈ BW / bytes_per_tok.
    ideal_tok_s: f64,
    measured_tok_s: f64,
    fraction: f64,
};

pub fn roofline(bytes_per_tok: u64, bandwidth_gbps: f64, decode_wall_ns: u64) Roofline {
    const bw_bytes_s = bandwidth_gbps * 1e9;
    const ideal = if (bytes_per_tok == 0) 0 else bw_bytes_s / @as(f64, @floatFromInt(bytes_per_tok));
    const measured = if (decode_wall_ns == 0) 0 else 1e9 / @as(f64, @floatFromInt(decode_wall_ns));
    const frac = if (ideal <= 0) 0 else measured / ideal;
    return .{
        .bytes_per_tok = bytes_per_tok,
        .bandwidth_gbps = bandwidth_gbps,
        .ideal_tok_s = ideal,
        .measured_tok_s = measured,
        .fraction = frac,
    };
}

pub fn emptyLaunchOverheadNs(encodes: u64, empty_launch_ns: u64) u64 {
    return encodes *% empty_launch_ns;
}

test "top3 picks largest families" {
    var a = Accumulators{};
    a.add(.mlp, 300);
    a.add(.qkv, 200);
    a.add(.lm_head, 100);
    a.add(.rope, 10);
    const t = a.top3();
    try std.testing.expectEqual(Family.mlp, t[0].family);
    try std.testing.expectEqual(Family.qkv, t[1].family);
    try std.testing.expectEqual(Family.lm_head, t[2].family);
}

test "roofline fraction is measured over ideal" {
    const r = roofline(1_000_000_000, 100.0, 1_000_000_000); // 1 GB/tok, 100 GB/s, 1s/tok
    try std.testing.expectApproxEqAbs(@as(f64, 100.0), r.ideal_tok_s, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), r.measured_tok_s, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 0.01), r.fraction, 1e-6);
}
