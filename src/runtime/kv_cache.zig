//! KV cache for one attention layer.
//!
//! Layout is dense `[n_kv_heads, max_seq, head_dim]` so a single head's used
//! prefix is contiguous. Attention reads `kv_len = used` with
//! `kv_stride = max_seq`. Append writes at `used` and advances it.
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

    pub fn init(allocator: std.mem.Allocator, n_kv: usize, max_seq: usize, head_dim: usize) TensorError!KvCache {
        if (n_kv == 0 or max_seq == 0 or head_dim == 0) return error.InvalidShape;
        return .{
            .n_kv = n_kv,
            .max_seq = max_seq,
            .head_dim = head_dim,
            .used = 0,
            .k = try Tensor.alloc(allocator, .f32, &.{ n_kv, max_seq, head_dim }),
            .v = try Tensor.alloc(allocator, .f32, &.{ n_kv, max_seq, head_dim }),
        };
    }

    pub fn deinit(self: *KvCache) void {
        self.k.deinit();
        self.v.deinit();
        self.* = undefined;
    }

    pub fn reset(self: *KvCache) void {
        self.used = 0;
    }

    pub fn remaining(self: KvCache) usize {
        return self.max_seq - self.used;
    }

    /// `k_new`/`v_new` are `[n_kv, t, head_dim]`.
    pub fn append(self: *KvCache, k_new: Tensor, v_new: Tensor) TensorError!void {
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
