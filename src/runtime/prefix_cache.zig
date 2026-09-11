//! Stage S2 — exact-prefix registry + cold/warm prefill A/B.
//!
//! Dense contiguous KV reuse (truncate + suffix continue). Not paged/block KV,
//! not packed continuous batching, not HTTP.

const std = @import("std");
const artifact = @import("../model/artifact.zig");
const qwen3 = @import("../model/qwen3.zig");
const qwen_forward = @import("../model/qwen_forward.zig");
const backend_mod = @import("backend.zig");

pub const Error = qwen_forward.Error || error{
    EmptyPrefix,
    EmptyTrials,
    CacheFull,
};

/// Longest shared token prefix length of `a` and `b`.
pub fn longestCommonPrefix(a: []const u32, b: []const u32) usize {
    const n = @min(a.len, b.len);
    var i: usize = 0;
    while (i < n and a[i] == b[i]) : (i += 1) {}
    return i;
}

const Entry = struct {
    token_ids: []u32,
    hits: u64 = 0,
    last_used: u64 = 0,
};

/// Exact-match prefix table with LRU eviction (educational cache management).
pub const PrefixCache = struct {
    allocator: std.mem.Allocator,
    max_entries: usize,
    entries: std.ArrayList(Entry),
    clock: u64 = 0,

    pub fn init(allocator: std.mem.Allocator, max_entries: usize) PrefixCache {
        return .{
            .allocator = allocator,
            .max_entries = if (max_entries == 0) 1 else max_entries,
            .entries = .empty,
        };
    }

    pub fn deinit(self: *PrefixCache) void {
        for (self.entries.items) |e| self.allocator.free(e.token_ids);
        self.entries.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn len(self: *const PrefixCache) usize {
        return self.entries.items.len;
    }

    /// Longest registered prefix that is an exact prefix of `prompt`.
    pub fn lookupLongest(self: *PrefixCache, prompt: []const u32) ?struct { index: usize, prefix_len: usize } {
        var best_i: ?usize = null;
        var best_len: usize = 0;
        for (self.entries.items, 0..) |e, i| {
            if (e.token_ids.len == 0 or e.token_ids.len > prompt.len) continue;
            if (!std.mem.eql(u32, e.token_ids, prompt[0..e.token_ids.len])) continue;
            if (e.token_ids.len > best_len) {
                best_len = e.token_ids.len;
                best_i = i;
            }
        }
        const idx = best_i orelse return null;
        self.clock += 1;
        self.entries.items[idx].hits += 1;
        self.entries.items[idx].last_used = self.clock;
        return .{ .index = idx, .prefix_len = best_len };
    }

    /// Insert an exact prefix identity. Evicts LRU when full.
    pub fn insert(self: *PrefixCache, prefix: []const u32) Error!void {
        if (prefix.len == 0) return error.EmptyPrefix;
        // Replace identical entry (refresh).
        for (self.entries.items, 0..) |*e, i| {
            if (std.mem.eql(u32, e.token_ids, prefix)) {
                self.clock += 1;
                e.last_used = self.clock;
                e.hits += 1;
                _ = i;
                return;
            }
        }
        while (self.entries.items.len >= self.max_entries) {
            try self.evictLru();
        }
        const owned = try self.allocator.dupe(u32, prefix);
        errdefer self.allocator.free(owned);
        self.clock += 1;
        try self.entries.append(self.allocator, .{
            .token_ids = owned,
            .hits = 0,
            .last_used = self.clock,
        });
    }

    fn evictLru(self: *PrefixCache) Error!void {
        if (self.entries.items.len == 0) return error.CacheFull;
        var victim: usize = 0;
        var oldest = self.entries.items[0].last_used;
        for (self.entries.items, 0..) |e, i| {
            if (e.last_used < oldest) {
                oldest = e.last_used;
                victim = i;
            }
        }
        const removed = self.entries.orderedRemove(victim);
        self.allocator.free(removed.token_ids);
    }
};

pub const PrefillTrial = struct {
    cold_ns: u64,
    warm_ns: u64,
    /// Tokens processed on cold path (prefix + suffix).
    cold_tokens: usize,
    /// Tokens processed on warm continuation (suffix only).
    warm_tokens: usize,
    logits_match: bool,
};

pub const PrefillReuseReport = struct {
    n_trials: usize,
    prefix_len: usize,
    cold_prefill_ns: u64,
    warm_prefill_ns: u64,
    /// One-time warm prefix prefill (excluded from warm_prefill_ns sum of trials).
    warm_prefix_ns: u64,
    savings_ratio: f64,
    all_logits_match: bool,
    trials: []PrefillTrial,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *PrefillReuseReport) void {
        self.allocator.free(self.trials);
        self.* = undefined;
    }
};

fn maxAbsDiff(a: []const f32, b: []const f32) f32 {
    var m: f32 = 0;
    const n = @min(a.len, b.len);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const d = @abs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

fn nsDelta(start: std.Io.Timestamp, end: std.Io.Timestamp) u64 {
    return @intCast(@max(@as(i96, 0), end.nanoseconds - start.nanoseconds));
}

/// Cold = full `prefill(P‖S)` each trial; warm = `prefill(P)` once then
/// `truncateTo(|P|)` + `prefillContinue(S)` per trial.
pub fn runColdWarmPrefill(
    allocator: std.mem.Allocator,
    io: std.Io,
    art: *const artifact.Artifact,
    arch: qwen3.Arch,
    kind: backend_mod.BackendKind,
    max_seq: usize,
    prefix: []const u32,
    suffixes: []const []const u32,
    logits_atol: f32,
) Error!PrefillReuseReport {
    if (prefix.len == 0) return error.EmptyPrefix;
    if (suffixes.len == 0) return error.EmptyTrials;
    try backend_mod.requireBackend(kind);

    var max_full: usize = prefix.len;
    for (suffixes) |s| max_full = @max(max_full, prefix.len + s.len);
    if (max_full == 0 or max_full > max_seq or max_full > arch.max_position_embeddings) {
        return error.InvalidShape;
    }

    const trials = try allocator.alloc(PrefillTrial, suffixes.len);
    errdefer allocator.free(trials);

    var cold_sess = try qwen_forward.Session.initWithBackend(allocator, art, arch, max_seq, kind);
    defer cold_sess.deinit();
    var warm_sess = try qwen_forward.Session.initWithBackend(allocator, art, arch, max_seq, kind);
    defer warm_sess.deinit();

    const vocab = arch.vocab_size;
    const cold_logits = try allocator.alloc(f32, vocab);
    defer allocator.free(cold_logits);
    const warm_logits = try allocator.alloc(f32, vocab);
    defer allocator.free(warm_logits);
    const full_buf = try allocator.alloc(u32, max_full);
    defer allocator.free(full_buf);

    // Warm: prime shared prefix once.
    const tp0 = std.Io.Clock.awake.now(io);
    try warm_sess.prefillLastLogits(prefix, warm_logits);
    const warm_prefix_ns = nsDelta(tp0, std.Io.Clock.awake.now(io));
    if (warm_sess.kvLen() != prefix.len) return error.InvalidShape;

    var cold_sum: u64 = 0;
    var warm_sum: u64 = 0;
    var all_ok = true;

    for (suffixes, 0..) |suffix, i| {
        if (suffix.len == 0) return error.InvalidShape;
        @memcpy(full_buf[0..prefix.len], prefix);
        @memcpy(full_buf[prefix.len..][0..suffix.len], suffix);
        const full = full_buf[0 .. prefix.len + suffix.len];

        const tc0 = std.Io.Clock.awake.now(io);
        try cold_sess.prefillLastLogits(full, cold_logits);
        const cold_ns = nsDelta(tc0, std.Io.Clock.awake.now(io));

        try warm_sess.truncateTo(prefix.len);
        const tw0 = std.Io.Clock.awake.now(io);
        try warm_sess.prefillContinue(suffix, warm_logits);
        const warm_ns = nsDelta(tw0, std.Io.Clock.awake.now(io));

        const match = maxAbsDiff(cold_logits, warm_logits) <= logits_atol;
        if (!match) all_ok = false;

        trials[i] = .{
            .cold_ns = cold_ns,
            .warm_ns = warm_ns,
            .cold_tokens = full.len,
            .warm_tokens = suffix.len,
            .logits_match = match,
        };
        cold_sum += cold_ns;
        warm_sum += warm_ns;
    }

    const savings = if (cold_sum == 0)
        0
    else
        1.0 - (@as(f64, @floatFromInt(warm_sum)) / @as(f64, @floatFromInt(cold_sum)));

    return .{
        .n_trials = suffixes.len,
        .prefix_len = prefix.len,
        .cold_prefill_ns = cold_sum,
        .warm_prefill_ns = warm_sum,
        .warm_prefix_ns = warm_prefix_ns,
        .savings_ratio = savings,
        .all_logits_match = all_ok,
        .trials = trials,
        .allocator = allocator,
    };
}

test "longestCommonPrefix" {
    try std.testing.expectEqual(@as(usize, 0), longestCommonPrefix(&.{}, &.{1}));
    try std.testing.expectEqual(@as(usize, 2), longestCommonPrefix(&.{ 1, 2, 3 }, &.{ 1, 2, 9 }));
    try std.testing.expectEqual(@as(usize, 1), longestCommonPrefix(&.{ 1, 2 }, &.{ 1, 8, 9 }));
}

test "PrefixCache lookup insert and LRU eviction" {
    const gpa = std.testing.allocator;
    var cache = PrefixCache.init(gpa, 2);
    defer cache.deinit();

    const a = [_]u32{ 1, 2, 3 };
    const b = [_]u32{ 1, 2, 9 };
    const c = [_]u32{ 7, 8 };

    try cache.insert(&a);
    try cache.insert(b[0..2]); // {1,2}
    try std.testing.expectEqual(@as(usize, 2), cache.len());

    const hit = cache.lookupLongest(&a) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(usize, 3), hit.prefix_len);

    try cache.insert(&c); // evict LRU
    try std.testing.expectEqual(@as(usize, 2), cache.len());
    try std.testing.expect(cache.lookupLongest(&c) != null);
}

test "cold vs warm suffix prefill logits match on mini" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const bytes = try qwen_forward.buildMiniArtifact(gpa);
    defer gpa.free(bytes);
    var art = try artifact.Artifact.loadOwned(gpa, try gpa.dupe(u8, bytes));
    defer art.deinit();
    const arch = qwen3.stage11_mini;

    const prefix = [_]u32{ 2, 3, 4, 5 };
    const s0 = [_]u32{ 6, 7 };
    const s1 = [_]u32{ 1, 2 };
    const suffixes = [_][]const u32{ &s0, &s1 };

    var report = try runColdWarmPrefill(gpa, io, &art, arch, .cpu, 32, &prefix, &suffixes, 1e-4);
    defer report.deinit();

    try std.testing.expect(report.all_logits_match);
    try std.testing.expect(report.warm_prefill_ns < report.cold_prefill_ns);
    try std.testing.expect(report.savings_ratio > 0);
    try std.testing.expectEqual(@as(usize, 4), report.prefix_len);
}
