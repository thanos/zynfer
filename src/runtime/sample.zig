//! Logit sampling: greedy, temperature, top-k, top-p, seeded RNG.

const std = @import("std");

pub const Error = error{
    InvalidShape,
};

pub const Config = struct {
    /// <= 0 → greedy (argmax). >0 → softmax(logits / temperature).
    temperature: f32 = 0,
    /// 0 = disabled. Keep only the k largest probs before nucleus / draw.
    top_k: u32 = 0,
    /// 1.0 = disabled. Nucleus sampling cumulative probability mass.
    top_p: f32 = 1.0,
    seed: u64 = 0,
};

pub fn argmax(logits: []const f32) Error!u32 {
    if (logits.len == 0) return error.InvalidShape;
    var best_i: usize = 0;
    var best_v = logits[0];
    var i: usize = 1;
    while (i < logits.len) : (i += 1) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best_i = i;
        }
    }
    return @intCast(best_i);
}

/// `probs` and `idx` must each be at least `logits.len`.
pub fn sampleWithScratch(
    logits: []const f32,
    cfg: Config,
    probs: []f32,
    idx: []u32,
    rng: *std.Random.DefaultPrng,
) Error!u32 {
    if (logits.len == 0 or probs.len < logits.len or idx.len < logits.len) return error.InvalidShape;

    if (cfg.temperature <= 0) {
        return argmax(logits);
    }

    const n = logits.len;
    const inv_t = 1.0 / cfg.temperature;
    var i: usize = 0;
    while (i < n) : (i += 1) probs[i] = logits[i] * inv_t;

    var max_v = probs[0];
    i = 1;
    while (i < n) : (i += 1) max_v = @max(max_v, probs[i]);
    var sum: f32 = 0;
    i = 0;
    while (i < n) : (i += 1) {
        probs[i] = @exp(probs[i] - max_v);
        sum += probs[i];
    }
    const inv_sum = 1.0 / sum;
    i = 0;
    while (i < n) : (i += 1) {
        probs[i] *= inv_sum;
        idx[i] = @intCast(i);
    }

    sortIdxByProbDesc(idx[0..n], probs);

    var k_limit: usize = n;
    if (cfg.top_k > 0) k_limit = @min(k_limit, @as(usize, @intCast(cfg.top_k)));

    var cutoff: usize = k_limit;
    if (cfg.top_p < 1.0 - 1e-6) {
        var cum: f32 = 0;
        var j: usize = 0;
        while (j < k_limit) : (j += 1) {
            cum += probs[idx[j]];
            if (cum >= cfg.top_p) {
                cutoff = j + 1;
                break;
            }
        }
    }
    if (cutoff == 0) cutoff = 1;

    sum = 0;
    i = 0;
    while (i < cutoff) : (i += 1) sum += probs[idx[i]];
    if (sum <= 0) return idx[0];

    const r = rng.random().float(f32) * sum;
    var cdf: f32 = 0;
    i = 0;
    while (i < cutoff) : (i += 1) {
        cdf += probs[idx[i]];
        if (r <= cdf) return idx[i];
    }
    return idx[cutoff - 1];
}

fn sortIdxByProbDesc(idx: []u32, probs: []const f32) void {
    const n = idx.len;
    if (n <= 1) return;
    var start = n / 2;
    while (true) {
        siftDown(idx, probs, start, n);
        if (start == 0) break;
        start -= 1;
    }
    var end = n;
    while (end > 1) {
        end -= 1;
        const tmp = idx[0];
        idx[0] = idx[end];
        idx[end] = tmp;
        siftDown(idx, probs, 0, end);
    }
    var i: usize = 0;
    while (i < n / 2) : (i += 1) {
        const tmp = idx[i];
        idx[i] = idx[n - 1 - i];
        idx[n - 1 - i] = tmp;
    }
}

fn siftDown(idx: []u32, probs: []const f32, start: usize, end: usize) void {
    var root = start;
    while (true) {
        var child = 2 * root + 1;
        if (child >= end) return;
        if (child + 1 < end and probs[idx[child]] < probs[idx[child + 1]]) child += 1;
        if (probs[idx[root]] >= probs[idx[child]]) return;
        const tmp = idx[root];
        idx[root] = idx[child];
        idx[child] = tmp;
        root = child;
    }
}

test "argmax picks the peak" {
    const logits = [_]f32{ 0.1, 3.0, 0.2 };
    try std.testing.expectEqual(@as(u32, 1), try argmax(&logits));
}

test "greedy temperature zero is argmax" {
    var rng = std.Random.DefaultPrng.init(42);
    var probs: [3]f32 = undefined;
    var idx: [3]u32 = undefined;
    const logits = [_]f32{ 0.1, 3.0, 0.2 };
    const id = try sampleWithScratch(&logits, .{ .temperature = 0 }, &probs, &idx, &rng);
    try std.testing.expectEqual(@as(u32, 1), id);
}

test "seeded sample is deterministic" {
    var probs: [8]f32 = undefined;
    var idx: [8]u32 = undefined;
    const logits = [_]f32{ 1, 1.1, 0.9, 1.05, 0.5, 0.4, 0.3, 0.2 };
    var rng_a = std.Random.DefaultPrng.init(7);
    var rng_b = std.Random.DefaultPrng.init(7);
    const a = try sampleWithScratch(&logits, .{ .temperature = 0.8, .top_k = 4, .top_p = 0.9 }, &probs, &idx, &rng_a);
    const b = try sampleWithScratch(&logits, .{ .temperature = 0.8, .top_k = 4, .top_p = 0.9 }, &probs, &idx, &rng_b);
    try std.testing.expectEqual(a, b);
}
