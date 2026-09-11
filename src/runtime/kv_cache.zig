//! KV cache for one attention layer.
//!
//! Layout is dense `[n_kv_heads, max_seq, head_dim]` so a single head's used
//! prefix is contiguous. Attention reads `kv_len = used` with
//! `kv_stride = max_seq`. Append writes at `used` and advances it.
//!
//! Logical tree (per layer):
//! ```text
//! layer
//!   └── sequence position
//!        └── KV head
//!             └── head dimension
//! ```
//! Physical storage packs heads outermost: `[n_kv, max_seq, head_dim]`.
//!
//! Stage 13 layout bake-off (`LayoutKind` + `benchLayouts`) compared this to
//! `[max_seq, n_kv, head_dim]`. Decode attention scans favor contiguous
//! per-head prefixes; append of one token is slightly cheaper on the seq-outer
//! layout, but attention dominates decode wall time, so heads-outer is retained.
//!
//! This is host storage. Backends may copy into device buffers; they must not
//! invent a second layout without a test against this one.

const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const TensorError = @import("tensor.zig").TensorError;

pub const KvCache = struct {
    n_kv: usize,
    max_seq: usize,
    head_dim: usize,
    used: usize,
    k: Tensor,
    v: Tensor,
    /// When false, only `used` is tracked (Metal owns the real KV buffers).
    owned: bool = true,

    pub fn init(allocator: std.mem.Allocator, n_kv: usize, max_seq: usize, head_dim: usize) TensorError!KvCache {
        if (n_kv == 0 or max_seq == 0 or head_dim == 0) return error.InvalidShape;
        return .{
            .n_kv = n_kv,
            .max_seq = max_seq,
            .head_dim = head_dim,
            .used = 0,
            .k = try Tensor.alloc(allocator, .f32, &.{ n_kv, max_seq, head_dim }),
            .v = try Tensor.alloc(allocator, .f32, &.{ n_kv, max_seq, head_dim }),
            .owned = true,
        };
    }

    /// Stage M6: host mirror for Metal-resident KV — no tensor storage.
    pub fn initMirror(n_kv: usize, max_seq: usize, head_dim: usize) TensorError!KvCache {
        if (n_kv == 0 or max_seq == 0 or head_dim == 0) return error.InvalidShape;
        return .{
            .n_kv = n_kv,
            .max_seq = max_seq,
            .head_dim = head_dim,
            .used = 0,
            .k = undefined,
            .v = undefined,
            .owned = false,
        };
    }

    pub fn deinit(self: *KvCache) void {
        if (self.owned) {
            self.k.deinit();
            self.v.deinit();
        }
        self.* = undefined;
    }

    pub fn reset(self: *KvCache) void {
        self.used = 0;
    }

    /// Shrink the live prefix to `n` tokens (`n <= used`). Storage beyond `n` is
    /// left in place and ignored — dense contiguous policy (Stage S2).
    pub fn truncateTo(self: *KvCache, n: usize) TensorError!void {
        if (n > self.used) return error.InvalidShape;
        self.used = n;
    }

    pub fn remaining(self: KvCache) usize {
        return self.max_seq - self.used;
    }

    /// Allocated K+V bytes for one layer (full capacity, f32). Mirror → 0.
    pub fn bytesCapacity(self: KvCache) u64 {
        if (!self.owned) return 0;
        return estimateLayerBytes(self.n_kv, self.max_seq, self.head_dim);
    }

    /// Bytes of the used K+V prefix (f32), not counting unused capacity. Mirror → 0.
    pub fn bytesUsed(self: KvCache) u64 {
        if (!self.owned) return 0;
        return estimateLayerBytes(self.n_kv, self.used, self.head_dim);
    }

    /// `k_new`/`v_new` are `[n_kv, t, head_dim]`.
    pub fn append(self: *KvCache, k_new: Tensor, v_new: Tensor) TensorError!void {
        if (!self.owned) return error.InvalidShape;
        if (k_new.rank != 3 or v_new.rank != 3) return error.InvalidShape;
        if (k_new.shape[0] != self.n_kv or v_new.shape[0] != self.n_kv) return error.ShapeMismatch;
        if (k_new.shape[2] != self.head_dim or v_new.shape[2] != self.head_dim) return error.ShapeMismatch;
        const t = k_new.shape[1];
        if (v_new.shape[1] != t) return error.ShapeMismatch;
        if (t > self.remaining()) return error.InvalidShape;

        const ks = try self.k.f32s();
        const vs = try self.v.f32s();
        const kn = try k_new.f32s();
        const vn = try v_new.f32s();
        const d = self.head_dim;
        var h: usize = 0;
        while (h < self.n_kv) : (h += 1) {
            var i: usize = 0;
            while (i < t) : (i += 1) {
                const dst = ((h * self.max_seq) + (self.used + i)) * d;
                const src = ((h * t) + i) * d;
                @memcpy(ks[dst..][0..d], kn[src..][0..d]);
                @memcpy(vs[dst..][0..d], vn[src..][0..d]);
            }
        }
        self.used += t;
    }
};

/// Physical layouts considered for host KV storage (Stage 13 bake-off).
pub const LayoutKind = enum {
    /// Retained: `[n_kv, max_seq, head_dim]` — contiguous used prefix per head.
    heads_seq_dim,
    /// Alternate: `[max_seq, n_kv, head_dim]` — contiguous append across heads;
    /// attention walks with stride `n_kv * head_dim`.
    seq_heads_dim,

    pub fn name(self: LayoutKind) []const u8 {
        return switch (self) {
            .heads_seq_dim => "[n_kv, max_seq, head_dim]",
            .seq_heads_dim => "[max_seq, n_kv, head_dim]",
        };
    }

    pub fn retained(self: LayoutKind) bool {
        return self == .heads_seq_dim;
    }
};

pub const LayoutBenchConfig = struct {
    n_kv: usize = 8,
    n_q: usize = 16,
    max_seq: usize = 2048,
    kv_len: usize = 1024,
    head_dim: usize = 128,
    warmup: usize = 3,
    iters: usize = 25,
};

pub const LayoutBenchRow = struct {
    layout: LayoutKind,
    attn_ns: u64,
    append_ns: u64,
};

pub const LayoutBenchReport = struct {
    cfg: LayoutBenchConfig,
    rows: [2]LayoutBenchRow,

    pub fn retainedWinsAttention(self: LayoutBenchReport) bool {
        return self.rows[0].attn_ns <= self.rows[1].attn_ns;
    }
};

/// Microbench decode attention K/V gather + single-token append for both layouts.
pub fn benchLayouts(allocator: std.mem.Allocator, io: std.Io, cfg: LayoutBenchConfig) !LayoutBenchReport {
    if (cfg.kv_len == 0 or cfg.kv_len > cfg.max_seq) return error.InvalidShape;
    if (cfg.n_q % cfg.n_kv != 0) return error.InvalidShape;

    var rows: [2]LayoutBenchRow = undefined;
    for ([_]LayoutKind{ .heads_seq_dim, .seq_heads_dim }, 0..) |layout, i| {
        rows[i] = .{
            .layout = layout,
            .attn_ns = try benchAttentionScan(allocator, io, layout, cfg),
            .append_ns = try benchAppendOne(allocator, io, layout, cfg),
        };
    }
    return .{ .cfg = cfg, .rows = rows };
}

fn elems(cfg: LayoutBenchConfig) usize {
    return cfg.n_kv * cfg.max_seq * cfg.head_dim;
}

fn kIndex(layout: LayoutKind, h: usize, t: usize, d: usize, cfg: LayoutBenchConfig) usize {
    return switch (layout) {
        .heads_seq_dim => (h * cfg.max_seq + t) * cfg.head_dim + d,
        .seq_heads_dim => (t * cfg.n_kv + h) * cfg.head_dim + d,
    };
}

fn benchAttentionScan(
    allocator: std.mem.Allocator,
    io: std.Io,
    layout: LayoutKind,
    cfg: LayoutBenchConfig,
) !u64 {
    const n = elems(cfg);
    const k = try allocator.alloc(f32, n);
    defer allocator.free(k);
    const v = try allocator.alloc(f32, n);
    defer allocator.free(v);
    const q = try allocator.alloc(f32, cfg.n_q * cfg.head_dim);
    defer allocator.free(q);
    const out = try allocator.alloc(f32, cfg.n_q * cfg.head_dim);
    defer allocator.free(out);
    const scores = try allocator.alloc(f32, cfg.kv_len);
    defer allocator.free(scores);

    for (k, 0..) |*x, i| x.* = @floatFromInt((i % 17) + 1);
    for (v, 0..) |*x, i| x.* = @floatFromInt((i % 13) + 1);
    for (q, 0..) |*x, i| x.* = @floatFromInt((i % 11) + 1);

    const group = cfg.n_q / cfg.n_kv;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));

    var w: usize = 0;
    while (w < cfg.warmup) : (w += 1) {
        attentionScanOnce(layout, cfg, k, v, q, out, scores, group, scale);
    }

    const t0 = std.Io.Clock.awake.now(io);
    var i: usize = 0;
    while (i < cfg.iters) : (i += 1) {
        attentionScanOnce(layout, cfg, k, v, q, out, scores, group, scale);
    }
    const t1 = std.Io.Clock.awake.now(io);
    const total = @as(u64, @intCast(@max(@as(i96, 0), t1.nanoseconds - t0.nanoseconds)));
    return total / cfg.iters;
}

fn attentionScanOnce(
    layout: LayoutKind,
    cfg: LayoutBenchConfig,
    k: []const f32,
    v: []const f32,
    q: []const f32,
    out: []f32,
    scores: []f32,
    group: usize,
    scale: f32,
) void {
    var h: usize = 0;
    while (h < cfg.n_q) : (h += 1) {
        const kv_h = h / group;
        const qrow = q[h * cfg.head_dim ..][0..cfg.head_dim];
        var max_s: f32 = -std.math.inf(f32);
        var tk: usize = 0;
        while (tk < cfg.kv_len) : (tk += 1) {
            var dot: f32 = 0;
            switch (layout) {
                // Production path: contiguous head_dim row per (head, pos).
                .heads_seq_dim => {
                    const krow = k[(kv_h * cfg.max_seq + tk) * cfg.head_dim ..][0..cfg.head_dim];
                    for (qrow, krow) |qv, kv| dot += qv * kv;
                },
                // Seq-outer: same head at successive positions strides by n_kv*head_dim.
                .seq_heads_dim => {
                    var d: usize = 0;
                    while (d < cfg.head_dim) : (d += 1) {
                        dot += qrow[d] * k[kIndex(.seq_heads_dim, kv_h, tk, d, cfg)];
                    }
                },
            }
            scores[tk] = dot * scale;
            max_s = @max(max_s, scores[tk]);
        }
        var sum: f32 = 0;
        for (scores[0..cfg.kv_len]) |*sc| {
            sc.* = @exp(sc.* - max_s);
            sum += sc.*;
        }
        const inv = if (sum == 0) 0 else 1.0 / sum;
        const orow = out[h * cfg.head_dim ..][0..cfg.head_dim];
        @memset(orow, 0);
        tk = 0;
        while (tk < cfg.kv_len) : (tk += 1) {
            const weight = scores[tk] * inv;
            switch (layout) {
                .heads_seq_dim => {
                    const vrow = v[(kv_h * cfg.max_seq + tk) * cfg.head_dim ..][0..cfg.head_dim];
                    for (orow, vrow) |*o, vv| o.* += weight * vv;
                },
                .seq_heads_dim => {
                    var d: usize = 0;
                    while (d < cfg.head_dim) : (d += 1) {
                        orow[d] += weight * v[kIndex(.seq_heads_dim, kv_h, tk, d, cfg)];
                    }
                },
            }
        }
    }
}

fn benchAppendOne(
    allocator: std.mem.Allocator,
    io: std.Io,
    layout: LayoutKind,
    cfg: LayoutBenchConfig,
) !u64 {
    const n = elems(cfg);
    const k = try allocator.alloc(f32, n);
    defer allocator.free(k);
    const v = try allocator.alloc(f32, n);
    defer allocator.free(v);
    const kn = try allocator.alloc(f32, cfg.n_kv * cfg.head_dim);
    defer allocator.free(kn);
    const vn = try allocator.alloc(f32, cfg.n_kv * cfg.head_dim);
    defer allocator.free(vn);
    @memset(k, 0);
    @memset(v, 0);
    for (kn, 0..) |*x, i| x.* = @floatFromInt(i + 1);
    for (vn, 0..) |*x, i| x.* = @floatFromInt(i + 2);

    const used: usize = cfg.kv_len / 2;
    if (used >= cfg.max_seq) return error.InvalidShape;

    var w: usize = 0;
    while (w < cfg.warmup) : (w += 1) {
        appendOneOnce(layout, cfg, k, v, kn, vn, used);
    }

    const t0 = std.Io.Clock.awake.now(io);
    var i: usize = 0;
    while (i < cfg.iters) : (i += 1) {
        appendOneOnce(layout, cfg, k, v, kn, vn, used);
    }
    const t1 = std.Io.Clock.awake.now(io);
    const total = @as(u64, @intCast(@max(@as(i96, 0), t1.nanoseconds - t0.nanoseconds)));
    return total / cfg.iters;
}

fn appendOneOnce(
    layout: LayoutKind,
    cfg: LayoutBenchConfig,
    k: []f32,
    v: []f32,
    kn: []const f32,
    vn: []const f32,
    used: usize,
) void {
    switch (layout) {
        .heads_seq_dim => {
            var h: usize = 0;
            while (h < cfg.n_kv) : (h += 1) {
                const dst = (h * cfg.max_seq + used) * cfg.head_dim;
                const src = h * cfg.head_dim;
                @memcpy(k[dst..][0..cfg.head_dim], kn[src..][0..cfg.head_dim]);
                @memcpy(v[dst..][0..cfg.head_dim], vn[src..][0..cfg.head_dim]);
            }
        },
        .seq_heads_dim => {
            // One token: n_kv contiguous head rows at position `used`.
            const dst = used * cfg.n_kv * cfg.head_dim;
            @memcpy(k[dst..][0 .. cfg.n_kv * cfg.head_dim], kn);
            @memcpy(v[dst..][0 .. cfg.n_kv * cfg.head_dim], vn);
        },
    }
}

/// K+V bytes for one layer at the given sequence length (f32).
pub fn estimateLayerBytes(n_kv: usize, seq: usize, head_dim: usize) u64 {
    const elems_n = @as(u64, @intCast(n_kv)) * @as(u64, @intCast(seq)) * @as(u64, @intCast(head_dim));
    return elems_n * 2 * @sizeOf(f32);
}

/// Full model KV capacity: `num_layers` × one layer of `[n_kv, max_seq, head_dim]` K and V.
pub fn estimateModelBytes(num_layers: usize, n_kv: usize, max_seq: usize, head_dim: usize) u64 {
    return @as(u64, @intCast(num_layers)) * estimateLayerBytes(n_kv, max_seq, head_dim);
}

test "append then used length matches" {
    var cache = try KvCache.init(std.testing.allocator, 1, 4, 2);
    defer cache.deinit();
    var kn = try Tensor.alloc(std.testing.allocator, .f32, &.{ 1, 2, 2 });
    defer kn.deinit();
    var vn = try Tensor.alloc(std.testing.allocator, .f32, &.{ 1, 2, 2 });
    defer vn.deinit();
    (try kn.f32s())[0] = 1;
    (try kn.f32s())[1] = 2;
    (try kn.f32s())[2] = 3;
    (try kn.f32s())[3] = 4;
    try vn.fillF32(0);
    try cache.append(kn, vn);
    try std.testing.expectEqual(@as(usize, 2), cache.used);
    const ks = try cache.k.f32s();
    try std.testing.expectEqual(@as(f32, 1), ks[0]);
    try std.testing.expectEqual(@as(f32, 3), ks[2]);
    try std.testing.expectEqual(@as(u64, 1 * 2 * 2 * 2 * 4), cache.bytesUsed());
    try std.testing.expectEqual(@as(u64, 1 * 4 * 2 * 2 * 4), cache.bytesCapacity());
}

test "append past max_seq is rejected" {
    var cache = try KvCache.init(std.testing.allocator, 1, 2, 2);
    defer cache.deinit();
    var kn = try Tensor.alloc(std.testing.allocator, .f32, &.{ 1, 2, 2 });
    defer kn.deinit();
    var vn = try Tensor.alloc(std.testing.allocator, .f32, &.{ 1, 2, 2 });
    defer vn.deinit();
    try kn.fillF32(1);
    try vn.fillF32(2);
    try cache.append(kn, vn);
    try std.testing.expectError(error.InvalidShape, cache.append(kn, vn));
}

test "Qwen3-0.6B KV capacity formula" {
    const bytes = estimateModelBytes(28, 8, 40960, 128);
    try std.testing.expectEqual(@as(u64, 28 * 8 * 40960 * 128 * 2 * 4), bytes);
}

test "layout index: heads-outer packs contiguous seq prefix" {
    const cfg = LayoutBenchConfig{ .n_kv = 2, .max_seq = 4, .head_dim = 2 };
    // Head 0 positions 0..1 occupy indices 0..3 before head 1.
    try std.testing.expectEqual(@as(usize, 0), kIndex(.heads_seq_dim, 0, 0, 0, cfg));
    try std.testing.expectEqual(@as(usize, 2), kIndex(.heads_seq_dim, 0, 1, 0, cfg));
    try std.testing.expectEqual(@as(usize, 8), kIndex(.heads_seq_dim, 1, 0, 0, cfg));
    // Seq-outer: same position, heads are adjacent.
    try std.testing.expectEqual(@as(usize, 0), kIndex(.seq_heads_dim, 0, 0, 0, cfg));
    try std.testing.expectEqual(@as(usize, 2), kIndex(.seq_heads_dim, 1, 0, 0, cfg));
    try std.testing.expectEqual(@as(usize, 4), kIndex(.seq_heads_dim, 0, 1, 0, cfg));
}

test "layout bake-off microbench runs both layouts" {
    const io = std.testing.io;
    const report = try benchLayouts(std.testing.allocator, io, .{
        .n_kv = 8,
        .n_q = 16,
        .max_seq = 512,
        .kv_len = 256,
        .head_dim = 64,
        .warmup = 1,
        .iters = 3,
    });
    try std.testing.expect(report.rows[0].layout == .heads_seq_dim);
    try std.testing.expect(report.rows[1].layout == .seq_heads_dim);
    try std.testing.expect(report.rows[0].attn_ns > 0);
    try std.testing.expect(report.rows[1].attn_ns > 0);
    try std.testing.expect(report.rows[0].append_ns > 0);
    try std.testing.expect(report.rows[1].append_ns > 0);
}
