//! Deterministic f32 CPU reference for the operations the LLM actually uses.
//!
//! Readable scalar loops on purpose. This is the correctness oracle, not the
//! production CPU path. Accelerate/SME candidates, if added later, must be
//! checked against these functions.

const std = @import("std");
const Tensor = @import("../../runtime/tensor.zig").Tensor;
pub const OpsError = @import("../../runtime/tensor.zig").TensorError;
const compare = @import("../../runtime/compare.zig");

pub fn add(dst: Tensor, a: Tensor, b: Tensor) OpsError!void {
    const x = try a.f32s();
    const y = try b.f32s();
    const z = try dst.f32s();
    if (x.len != y.len or x.len != z.len) return error.ShapeMismatch;
    for (z, x, y) |*o, p, q| o.* = p + q;
}

pub fn mul(dst: Tensor, a: Tensor, b: Tensor) OpsError!void {
    const x = try a.f32s();
    const y = try b.f32s();
    const z = try dst.f32s();
    if (x.len != y.len or x.len != z.len) return error.ShapeMismatch;
    for (z, x, y) |*o, p, q| o.* = p * q;
}

pub fn silu(dst: Tensor, x: Tensor) OpsError!void {
    const in = try x.f32s();
    const out = try dst.f32s();
    if (in.len != out.len) return error.ShapeMismatch;
    for (out, in) |*o, v| o.* = siluScalar(v);
}

pub fn siluMul(dst: Tensor, gate: Tensor, up: Tensor) OpsError!void {
    const g = try gate.f32s();
    const u = try up.f32s();
    const o = try dst.f32s();
    if (g.len != u.len or g.len != o.len) return error.ShapeMismatch;
    for (o, g, u) |*out, gv, uv| out.* = siluScalar(gv) * uv;
}

pub fn siluScalar(x: f32) f32 {
    return x * (1.0 / (1.0 + @exp(-x)));
}

/// y = weight * x / sqrt(mean(x^2) + eps) along the last axis.
pub fn rmsNorm(dst: Tensor, x: Tensor, weight: Tensor, eps: f32) OpsError!void {
    if (x.rank < 1 or dst.rank != x.rank) return error.ShapeMismatch;
    const cols = x.shape[x.rank - 1];
    if (weight.rank != 1 or weight.shape[0] != cols) return error.ShapeMismatch;
    const xs = try x.f32s();
    const ws = try weight.f32s();
    const ys = try dst.f32s();
    if (xs.len != ys.len) return error.ShapeMismatch;
    const rows = xs.len / cols;
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const row = xs[r * cols ..][0..cols];
        const out = ys[r * cols ..][0..cols];
        var acc: f32 = 0;
        for (row) |v| acc += v * v;
        const inv = 1.0 / @sqrt(acc / @as(f32, @floatFromInt(cols)) + eps);
        for (out, row, ws) |*o, v, w| o.* = w * v * inv;
    }
}

/// Softmax along the last axis, numerically stable.
pub fn softmax(dst: Tensor, x: Tensor) OpsError!void {
    if (x.rank < 1 or dst.rank != x.rank) return error.ShapeMismatch;
    const cols = x.shape[x.rank - 1];
    const xs = try x.f32s();
    const ys = try dst.f32s();
    if (xs.len != ys.len) return error.ShapeMismatch;
    const rows = xs.len / cols;
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const row = xs[r * cols ..][0..cols];
        const out = ys[r * cols ..][0..cols];
        var max_v = row[0];
        for (row[1..]) |v| max_v = @max(max_v, v);
        var sum: f32 = 0;
        for (out, row) |*o, v| {
            o.* = @exp(v - max_v);
            sum += o.*;
        }
        const inv = 1.0 / sum;
        for (out) |*o| o.* *= inv;
    }
}

/// C[M,N] = A[M,K] * B[K,N], row-major.
pub fn matmul(c: Tensor, a: Tensor, b: Tensor) OpsError!void {
    if (a.rank != 2 or b.rank != 2 or c.rank != 2) return error.InvalidShape;
    const m = a.shape[0];
    const k = a.shape[1];
    const n = b.shape[1];
    if (b.shape[0] != k or c.shape[0] != m or c.shape[1] != n) return error.ShapeMismatch;
    const as = try a.f32s();
    const bs = try b.f32s();
    const cs = try c.f32s();
    matmulF32(cs, as, bs, m, n, k);
}

/// y[M] = A[M,K] * x[K].
pub fn matvec(y: Tensor, a: Tensor, x: Tensor) OpsError!void {
    if (a.rank != 2 or x.rank != 1 or y.rank != 1) return error.InvalidShape;
    const m = a.shape[0];
    const k = a.shape[1];
    if (x.shape[0] != k or y.shape[0] != m) return error.ShapeMismatch;
    const as = try a.f32s();
    const xs = try x.f32s();
    const ys = try y.f32s();
    matvecF32(ys, as, xs, m, k);
}

pub fn matmulF32(c: []f32, a: []const f32, b: []const f32, m: usize, n: usize, k: usize) void {
    @memset(c, 0);
    var i: usize = 0;
    while (i < m) : (i += 1) {
        var p: usize = 0;
        while (p < k) : (p += 1) {
            const av = a[i * k + p];
            var j: usize = 0;
            while (j < n) : (j += 1) {
                c[i * n + j] += av * b[p * n + j];
            }
        }
    }
}

pub fn matvecF32(y: []f32, a: []const f32, x: []const f32, m: usize, k: usize) void {
    var i: usize = 0;
    while (i < m) : (i += 1) {
        var acc: f32 = 0;
        var p: usize = 0;
        while (p < k) : (p += 1) acc += a[i * k + p] * x[p];
        y[i] = acc;
    }
}

/// Qwen3-style RoPE: rotate the first and second halves of the last axis.
/// `x` is [tokens, n_heads, head_dim] or [n_heads, head_dim] with positions
/// starting at `pos0`.
pub fn rope(x: Tensor, pos0: usize, theta: f32) OpsError!void {
    if (x.rank < 2) return error.InvalidShape;
    const head_dim = x.shape[x.rank - 1];
    if (head_dim % 2 != 0) return error.InvalidShape;
    const half = head_dim / 2;
    const xs = try x.f32s();
    const heads_and_tokens = xs.len / head_dim;
    const n_heads = x.shape[x.rank - 2];
    const tokens = heads_and_tokens / n_heads;
    var t: usize = 0;
    while (t < tokens) : (t += 1) {
        const pos = pos0 + t;
        var h: usize = 0;
        while (h < n_heads) : (h += 1) {
            const base = (t * n_heads + h) * head_dim;
            var i: usize = 0;
            while (i < half) : (i += 1) {
                const freq = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(half)));
                const angle = @as(f32, @floatFromInt(pos)) * freq;
                const c = @cos(angle);
                const s = @sin(angle);
                const a = xs[base + i];
                const b = xs[base + i + half];
                xs[base + i] = a * c - b * s;
                xs[base + i + half] = a * s + b * c;
            }
        }
    }
}

/// Causal grouped-query attention.
/// q: [n_q_heads, q_len, d]
/// k,v: [n_kv_heads, kv_len, d]
/// out: [n_q_heads, q_len, d]
pub fn attention(
    out: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    allocator: std.mem.Allocator,
) OpsError!void {
    if (q.rank != 3 or k.rank != 3 or v.rank != 3 or out.rank != 3) return error.InvalidShape;
    const n_q = q.shape[0];
    const q_len = q.shape[1];
    const d = q.shape[2];
    const n_kv = k.shape[0];
    const kv_len = k.shape[1];
    if (k.shape[2] != d or v.shape[0] != n_kv or v.shape[1] != kv_len or v.shape[2] != d)
        return error.ShapeMismatch;
    if (out.shape[0] != n_q or out.shape[1] != q_len or out.shape[2] != d) return error.ShapeMismatch;
    if (n_q % n_kv != 0) return error.InvalidShape;

    const qs = try q.f32s();
    const ks = try k.f32s();
    const vs = try v.f32s();
    const os = try out.f32s();
    const group = n_q / n_kv;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));

    const scores = allocator.alloc(f32, kv_len) catch return error.OutOfMemory;
    defer allocator.free(scores);

    var h: usize = 0;
    while (h < n_q) : (h += 1) {
        const kv_h = h / group;
        var tq: usize = 0;
        while (tq < q_len) : (tq += 1) {
            const qrow = qs[(h * q_len + tq) * d ..][0..d];
            var max_s: f32 = -std.math.inf(f32);
            var tk: usize = 0;
            while (tk < kv_len) : (tk += 1) {
                const causal_ok = (kv_len - q_len + tq) >= tk;
                if (!causal_ok) {
                    scores[tk] = -std.math.inf(f32);
                    continue;
                }
                const krow = ks[(kv_h * kv_len + tk) * d ..][0..d];
                var dot: f32 = 0;
                for (qrow, krow) |qv, kv| dot += qv * kv;
                scores[tk] = dot * scale;
                max_s = @max(max_s, scores[tk]);
            }
            var sum: f32 = 0;
            for (scores) |*sc| {
                if (std.math.isInf(sc.*)) {
                    sc.* = 0;
                } else {
                    sc.* = @exp(sc.* - max_s);
                    sum += sc.*;
                }
            }
            const inv = if (sum == 0) 0 else 1.0 / sum;
            const orow = os[(h * q_len + tq) * d ..][0..d];
            @memset(orow, 0);
            tk = 0;
            while (tk < kv_len) : (tk += 1) {
                const w = scores[tk] * inv;
                if (w == 0) continue;
                const vrow = vs[(kv_h * kv_len + tk) * d ..][0..d];
                for (orow, vrow) |*o, vv| o.* += w * vv;
            }
        }
    }
}

/// Residual SwiGLU used as a tiny end-to-end fixture:
///   n = rmsnorm(x, wn)
///   h = silu(n @ wg) * (n @ wu)
///   out = x + h @ wd
pub fn swigluResidual(
    out: Tensor,
    x: Tensor,
    wn: Tensor,
    wg: Tensor,
    wu: Tensor,
    wd: Tensor,
    eps: f32,
    allocator: std.mem.Allocator,
) OpsError!void {
    if (x.rank != 2) return error.InvalidShape;
    const tokens = x.shape[0];
    const hidden = x.shape[1];
    const inter = wg.shape[1];
    if (wn.rank != 1 or wn.shape[0] != hidden) return error.ShapeMismatch;
    if (wg.shape[0] != hidden or wu.shape[0] != hidden or wu.shape[1] != inter) return error.ShapeMismatch;
    if (wd.shape[0] != inter or wd.shape[1] != hidden) return error.ShapeMismatch;

    var normed = Tensor.alloc(allocator, .f32, &.{ tokens, hidden }) catch return error.OutOfMemory;
    defer normed.deinit();
    var gate = Tensor.alloc(allocator, .f32, &.{ tokens, inter }) catch return error.OutOfMemory;
    defer gate.deinit();
    var up = Tensor.alloc(allocator, .f32, &.{ tokens, inter }) catch return error.OutOfMemory;
    defer up.deinit();
    var hidden_act = Tensor.alloc(allocator, .f32, &.{ tokens, inter }) catch return error.OutOfMemory;
    defer hidden_act.deinit();
    var down = Tensor.alloc(allocator, .f32, &.{ tokens, hidden }) catch return error.OutOfMemory;
    defer down.deinit();

    try rmsNorm(normed, x, wn, eps);
    try matmul(gate, normed, wg);
    try matmul(up, normed, wu);
    try siluMul(hidden_act, gate, up);
    try matmul(down, hidden_act, wd);
    try add(out, x, down);
}

fn iotaFill(t: Tensor, start: f32, step: f32) !void {
    const xs = try t.f32s();
    for (xs, 0..) |*v, i| v.* = start + step * @as(f32, @floatFromInt(i));
}

test "add and siluMul tiny cases" {
    var a = try Tensor.alloc(std.testing.allocator, .f32, &.{2});
    defer a.deinit();
    var b = try Tensor.alloc(std.testing.allocator, .f32, &.{2});
    defer b.deinit();
    var c = try Tensor.alloc(std.testing.allocator, .f32, &.{2});
    defer c.deinit();
    (try a.f32s())[0] = 1;
    (try a.f32s())[1] = 2;
    (try b.f32s())[0] = 3;
    (try b.f32s())[1] = 4;
    try add(c, a, b);
    try std.testing.expectEqual(@as(f32, 4), (try c.f32s())[0]);
    try std.testing.expectEqual(@as(f32, 6), (try c.f32s())[1]);

    try siluMul(c, a, b);
    const expected0 = siluScalar(1) * 3;
    try compare.expectClose(&.{expected0}, (try c.f32s())[0..1], 1e-6, 1e-6);
}

test "softmax sums to one" {
    var x = try Tensor.alloc(std.testing.allocator, .f32, &.{ 1, 4 });
    defer x.deinit();
    var y = try Tensor.alloc(std.testing.allocator, .f32, &.{ 1, 4 });
    defer y.deinit();
    (try x.f32s())[0] = 1;
    (try x.f32s())[1] = 2;
    (try x.f32s())[2] = 3;
    (try x.f32s())[3] = 4;
    try softmax(y, x);
    var sum: f32 = 0;
    for (try y.f32s()) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1), sum, 1e-6);
}

test "matmul identity" {
    var a = try Tensor.alloc(std.testing.allocator, .f32, &.{ 2, 2 });
    defer a.deinit();
    var i = try Tensor.alloc(std.testing.allocator, .f32, &.{ 2, 2 });
    defer i.deinit();
    var c = try Tensor.alloc(std.testing.allocator, .f32, &.{ 2, 2 });
    defer c.deinit();
    try iotaFill(a, 1, 1);
    (try i.f32s())[0] = 1;
    (try i.f32s())[1] = 0;
    (try i.f32s())[2] = 0;
    (try i.f32s())[3] = 1;
    try matmul(c, a, i);
    try compare.expectClose(try a.f32s(), try c.f32s(), 1e-6, 0);
}

test "RoPE at position 0 is identity" {
    var x = try Tensor.alloc(std.testing.allocator, .f32, &.{ 1, 2, 4 });
    defer x.deinit();
    try iotaFill(x, 0.25, 0.25);
    var clone = try Tensor.alloc(std.testing.allocator, .f32, &.{ 1, 2, 4 });
    defer clone.deinit();
    @memcpy(try clone.f32s(), try x.f32s());
    try rope(x, 0, 10_000);
    try compare.expectClose(try clone.f32s(), try x.f32s(), 1e-6, 0);
}

test "RoPE d=2 pos=1 matches hand values" {
    var x = try Tensor.alloc(std.testing.allocator, .f32, &.{ 1, 1, 2 });
    defer x.deinit();
    (try x.f32s())[0] = 1;
    (try x.f32s())[1] = 0;
    try rope(x, 1, 10_000);
    const c = @cos(@as(f32, 1));
    const s = @sin(@as(f32, 1));
    try compare.expectClose(&.{ c, s }, try x.f32s(), 1e-6, 1e-6);
}

test "causal attention masks the future" {
    var q = try Tensor.alloc(std.testing.allocator, .f32, &.{ 1, 2, 2 });
    defer q.deinit();
    var k = try Tensor.alloc(std.testing.allocator, .f32, &.{ 1, 2, 2 });
    defer k.deinit();
    var v = try Tensor.alloc(std.testing.allocator, .f32, &.{ 1, 2, 2 });
    defer v.deinit();
    var o = try Tensor.alloc(std.testing.allocator, .f32, &.{ 1, 2, 2 });
    defer o.deinit();
    @memset(try q.f32s(), 0);
    @memset(try k.f32s(), 0);
    (try v.f32s())[0] = 1;
    (try v.f32s())[1] = 0;
    (try v.f32s())[2] = 0;
    (try v.f32s())[3] = 1;
    (try q.f32s())[0] = 1;
    (try q.f32s())[2] = 1;
    try attention(o, q, k, v, std.testing.allocator);
    // first query can only see the first key/value
    try std.testing.expectApproxEqAbs(@as(f32, 1), (try o.f32s())[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0), (try o.f32s())[1], 1e-5);
}

test "randomized matvec matches matmul with N=1 seed 7" {
    const seed: u64 = 7;
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    const m: usize = 5;
    const k: usize = 3;
    var a = try Tensor.alloc(std.testing.allocator, .f32, &.{ m, k });
    defer a.deinit();
    var x = try Tensor.alloc(std.testing.allocator, .f32, &.{k});
    defer x.deinit();
    var y = try Tensor.alloc(std.testing.allocator, .f32, &.{m});
    defer y.deinit();
    var b = try Tensor.alloc(std.testing.allocator, .f32, &.{ k, 1 });
    defer b.deinit();
    var c = try Tensor.alloc(std.testing.allocator, .f32, &.{ m, 1 });
    defer c.deinit();
    for (try a.f32s()) |*v| v.* = random.float(f32) * 2 - 1;
    for (try x.f32s()) |*v| v.* = random.float(f32) * 2 - 1;
    @memcpy(try b.f32s(), try x.f32s());
    try matvec(y, a, x);
    try matmul(c, a, b);
    try compare.expectClose(try y.f32s(), try c.f32s(), 1e-5, 1e-5);
}
