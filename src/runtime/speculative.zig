//! Stage S3 — speculative decoding (n-gram draft + target verify).
//!
//! Registered Qwen3-0.6B/4B are plain CausalLM (no MTP heads). Draft is a
//! cheap n-gram proposal from the committed token history; the target model
//! verifies with greedy argmax before each `decodeToken`. On mismatch the
//! true argmax is committed instead (no KV rollback needed — verify precedes
//! decode). Report acceptance / tokens-per-round / committed tok/s; never
//! report raw proposed throughput as useful inference throughput.

const std = @import("std");
const artifact = @import("../model/artifact.zig");
const qwen3 = @import("../model/qwen3.zig");
const qwen_forward = @import("../model/qwen_forward.zig");
const sample_mod = @import("sample.zig");
const backend_mod = @import("backend.zig");

pub const Error = qwen_forward.Error || error{
    EmptyPrompt,
    InvalidDepth,
};

pub const Config = struct {
    max_new_tokens: u32 = 64,
    /// Max draft tokens proposed per round (proposal depth K).
    proposal_depth: u32 = 4,
    /// N-gram order (2 = bigram continuation). Must be >= 2.
    ngram_order: u32 = 2,
    stop_ids: []const u32 = &.{},
};

pub const SpecStats = struct {
    prompt_tokens: usize = 0,
    generated_tokens: usize = 0,
    prefill_ns: u64 = 0,
    decode_ns: u64 = 0,
    wall_ns: u64 = 0,
    proposal_depth: u32 = 0,
    /// Sum of draft lengths offered across rounds.
    proposed_tokens: usize = 0,
    /// Draft tokens that matched target greedy.
    accepted_draft_tokens: usize = 0,
    /// Speculative rounds (each ends by reject, full-accept+bonus, or empty draft).
    rounds: usize = 0,
    /// accepted_draft / proposed (0 if nothing proposed).
    acceptance_rate: f64 = 0,
    /// generated / rounds.
    tokens_per_round: f64 = 0,
    /// generated / wall (committed throughput — not proposed throughput).
    committed_tok_s: f64 = 0,
};

fn isStop(id: u32, stop_ids: []const u32) bool {
    for (stop_ids) |s| if (s == id) return true;
    return false;
}

fn nsDelta(start: std.Io.Timestamp, end: std.Io.Timestamp) u64 {
    return @intCast(@max(@as(i96, 0), end.nanoseconds - start.nanoseconds));
}

fn argmaxId(logits: []const f32) Error!u32 {
    return sample_mod.argmax(logits);
}

/// Propose up to `max_propose` tokens by finding the latest n-gram match in `ctx`
/// and copying the tokens that historically followed it.
pub fn ngramPropose(
    ctx: []const u32,
    order: u32,
    max_propose: u32,
    out: []u32,
) usize {
    if (order < 2 or max_propose == 0 or out.len == 0) return 0;
    const ord: usize = order;
    if (ctx.len < ord) return 0;
    const needle = ctx[ctx.len - (ord - 1) ..];
    // Search earlier windows (exclude the needle at the end).
    const search_end = ctx.len - (ord - 1);
    var best_pos: ?usize = null;
    var i: usize = 0;
    while (i + (ord - 1) <= search_end) : (i += 1) {
        // Do not match the needle occupying the very end (no follow tokens there).
        if (i == search_end) break;
        if (std.mem.eql(u32, ctx[i .. i + (ord - 1)], needle)) {
            const follow = i + (ord - 1);
            if (follow < ctx.len) best_pos = follow;
        }
    }
    const start = best_pos orelse return 0;
    var n: usize = 0;
    while (n < max_propose and n < out.len and start + n < ctx.len) : (n += 1) {
        out[n] = ctx[start + n];
    }
    return n;
}

/// Greedy speculative generate: n-gram draft, target verify, commit.
pub fn generateSpeculative(
    sess: *qwen_forward.Session,
    io: std.Io,
    prompt_ids: []const u32,
    out_ids: *std.ArrayList(u32),
    cfg: Config,
) Error!SpecStats {
    if (prompt_ids.len == 0) return error.EmptyPrompt;
    if (cfg.proposal_depth == 0) return error.InvalidDepth;
    if (cfg.ngram_order < 2) return error.InvalidDepth;

    const logits = try sess.logits.f32s();
    try out_ids.ensureTotalCapacity(sess.allocator, out_ids.items.len + cfg.max_new_tokens);

    var context: std.ArrayList(u32) = .empty;
    defer context.deinit(sess.allocator);
    try context.ensureTotalCapacity(sess.allocator, prompt_ids.len + cfg.max_new_tokens);
    try context.appendSlice(sess.allocator, prompt_ids);

    const t0 = std.Io.Clock.awake.now(io);
    try sess.prefillLastLogits(prompt_ids, logits);
    const t_prefill = std.Io.Clock.awake.now(io);

    var generated: usize = 0;
    var decode_ns: u64 = 0;
    var proposed_sum: usize = 0;
    var accepted_sum: usize = 0;
    var rounds: usize = 0;

    var draft_buf: [32]u32 = undefined;
    const depth = @min(cfg.proposal_depth, draft_buf.len);

    while (generated < cfg.max_new_tokens) {
        if (sess.kvLen() >= sess.max_seq) break;
        rounds += 1;

        const n_prop = ngramPropose(context.items, cfg.ngram_order, depth, draft_buf[0..depth]);
        proposed_sum += n_prop;

        var stop_hit = false;

        if (n_prop == 0) {
            // No draft — ordinary greedy step.
            const next = try argmaxId(logits);
            out_ids.appendAssumeCapacity(next);
            try context.append(sess.allocator, next);
            generated += 1;
            if (isStop(next, cfg.stop_ids)) break;
            if (generated >= cfg.max_new_tokens or sess.kvLen() >= sess.max_seq) break;
            const td0 = std.Io.Clock.awake.now(io);
            try sess.decodeToken(next, logits);
            decode_ns += nsDelta(td0, std.Io.Clock.awake.now(io));
            continue;
        }

        var accepted_round: usize = 0;
        var rejected = false;
        var di: usize = 0;
        while (di < n_prop and generated < cfg.max_new_tokens) : (di += 1) {
            if (sess.kvLen() >= sess.max_seq) break;
            const target = try argmaxId(logits);
            const draft = draft_buf[di];
            if (draft != target) {
                // Reject: commit target truth, advance once, end round.
                out_ids.appendAssumeCapacity(target);
                try context.append(sess.allocator, target);
                generated += 1;
                rejected = true;
                if (isStop(target, cfg.stop_ids)) {
                    stop_hit = true;
                    break;
                }
                if (generated >= cfg.max_new_tokens or sess.kvLen() >= sess.max_seq) break;
                const td0 = std.Io.Clock.awake.now(io);
                try sess.decodeToken(target, logits);
                decode_ns += nsDelta(td0, std.Io.Clock.awake.now(io));
                break;
            }
            // Accept draft.
            out_ids.appendAssumeCapacity(draft);
            try context.append(sess.allocator, draft);
            generated += 1;
            accepted_round += 1;
            accepted_sum += 1;
            if (isStop(draft, cfg.stop_ids)) {
                rejected = true;
                stop_hit = true;
                break;
            }
            if (generated >= cfg.max_new_tokens or sess.kvLen() >= sess.max_seq) break;
            const td0 = std.Io.Clock.awake.now(io);
            try sess.decodeToken(draft, logits);
            decode_ns += nsDelta(td0, std.Io.Clock.awake.now(io));
        }

        if (stop_hit) break;

        // Full accept of all proposals → one bonus token from final logits.
        if (!rejected and accepted_round == n_prop and generated < cfg.max_new_tokens and sess.kvLen() < sess.max_seq) {
            const bonus = try argmaxId(logits);
            out_ids.appendAssumeCapacity(bonus);
            try context.append(sess.allocator, bonus);
            generated += 1;
            if (isStop(bonus, cfg.stop_ids)) break;
            if (generated < cfg.max_new_tokens and sess.kvLen() < sess.max_seq) {
                const td0 = std.Io.Clock.awake.now(io);
                try sess.decodeToken(bonus, logits);
                decode_ns += nsDelta(td0, std.Io.Clock.awake.now(io));
            }
        }
    }

    const wall = nsDelta(t0, std.Io.Clock.awake.now(io));
    const accept_rate = if (proposed_sum == 0)
        0
    else
        @as(f64, @floatFromInt(accepted_sum)) / @as(f64, @floatFromInt(proposed_sum));
    const tpr = if (rounds == 0) 0 else @as(f64, @floatFromInt(generated)) / @as(f64, @floatFromInt(rounds));
    const tok_s = if (wall == 0) 0 else @as(f64, @floatFromInt(generated)) / (@as(f64, @floatFromInt(wall)) * 1e-9);

    return .{
        .prompt_tokens = prompt_ids.len,
        .generated_tokens = generated,
        .prefill_ns = nsDelta(t0, t_prefill),
        .decode_ns = decode_ns,
        .wall_ns = wall,
        .proposal_depth = depth,
        .proposed_tokens = proposed_sum,
        .accepted_draft_tokens = accepted_sum,
        .rounds = rounds,
        .acceptance_rate = accept_rate,
        .tokens_per_round = tpr,
        .committed_tok_s = tok_s,
    };
}

/// Baseline greedy generate for A/B (temperature 0).
pub fn generateBaseline(
    sess: *qwen_forward.Session,
    io: std.Io,
    prompt_ids: []const u32,
    out_ids: *std.ArrayList(u32),
    max_new_tokens: u32,
    stop_ids: []const u32,
) Error!qwen_forward.Session.GenerateStats {
    var rng = std.Random.DefaultPrng.init(0);
    return sess.generate(io, prompt_ids, out_ids, .{
        .max_new_tokens = max_new_tokens,
        .sample = .{ .temperature = 0 },
        .stop_ids = stop_ids,
        .use_kv_cache = true,
    }, &rng);
}

pub const SpecBenchReport = struct {
    proposal_depth: u32,
    ngram_order: u32,
    token_parity: bool,
    baseline_wall_ns: u64,
    baseline_tok_s: f64,
    baseline_generated: usize,
    speculative: SpecStats,
    allocator: std.mem.Allocator,
    baseline_ids: []u32,
    speculative_ids: []u32,

    pub fn deinit(self: *SpecBenchReport) void {
        self.allocator.free(self.baseline_ids);
        self.allocator.free(self.speculative_ids);
        self.* = undefined;
    }
};

pub fn runSpecBench(
    allocator: std.mem.Allocator,
    io: std.Io,
    art: *const artifact.Artifact,
    arch: qwen3.Arch,
    kind: backend_mod.BackendKind,
    max_seq: usize,
    prompt_ids: []const u32,
    max_new: u32,
    proposal_depth: u32,
    ngram_order: u32,
) Error!SpecBenchReport {
    try backend_mod.requireBackend(kind);
    if (prompt_ids.len == 0) return error.EmptyPrompt;

    var base_sess = try qwen_forward.Session.initWithBackend(allocator, art, arch, max_seq, kind);
    defer base_sess.deinit();
    var base_out: std.ArrayList(u32) = .empty;
    defer base_out.deinit(allocator);
    const base_stats = try generateBaseline(&base_sess, io, prompt_ids, &base_out, max_new, &.{});

    var spec_sess = try qwen_forward.Session.initWithBackend(allocator, art, arch, max_seq, kind);
    defer spec_sess.deinit();
    var spec_out: std.ArrayList(u32) = .empty;
    defer spec_out.deinit(allocator);
    const spec_stats = try generateSpeculative(&spec_sess, io, prompt_ids, &spec_out, .{
        .max_new_tokens = max_new,
        .proposal_depth = proposal_depth,
        .ngram_order = ngram_order,
    });

    const parity = std.mem.eql(u32, base_out.items, spec_out.items);
    const base_ids = try allocator.dupe(u32, base_out.items);
    errdefer allocator.free(base_ids);
    const spec_ids = try allocator.dupe(u32, spec_out.items);

    const base_tok_s = if (base_stats.prefill_ns + base_stats.decode_ns == 0)
        0
    else
        @as(f64, @floatFromInt(base_stats.generated_tokens)) /
            (@as(f64, @floatFromInt(base_stats.prefill_ns + base_stats.decode_ns)) * 1e-9);

    return .{
        .proposal_depth = proposal_depth,
        .ngram_order = ngram_order,
        .token_parity = parity,
        .baseline_wall_ns = base_stats.prefill_ns + base_stats.decode_ns,
        .baseline_tok_s = base_tok_s,
        .baseline_generated = base_stats.generated_tokens,
        .speculative = spec_stats,
        .allocator = allocator,
        .baseline_ids = base_ids,
        .speculative_ids = spec_ids,
    };
}

test "ngramPropose finds continuation" {
    const ctx = [_]u32{ 1, 2, 3, 4, 2, 3 };
    var out: [4]u32 = undefined;
    // needle = last (order-1)=[2,3] for order 3; earlier match at index 1 → follow starts at 3 → [4,2,3]
    const n = ngramPropose(&ctx, 3, 4, &out);
    try std.testing.expect(n >= 1);
    try std.testing.expectEqual(@as(u32, 4), out[0]);
}

test "speculative greedy matches baseline on mini" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const bytes = try qwen_forward.buildMiniArtifact(gpa);
    defer gpa.free(bytes);
    var art = try artifact.Artifact.loadOwned(gpa, try gpa.dupe(u8, bytes));
    defer art.deinit();
    const arch = qwen3.stage11_mini;
    const prompt = [_]u32{ 2, 3, 2, 3, 4, 2, 3 };
    var report = try runSpecBench(gpa, io, &art, arch, .cpu, 32, &prompt, 8, 4, 2);
    defer report.deinit();
    try std.testing.expect(report.token_parity);
    try std.testing.expectEqual(report.baseline_generated, report.speculative.generated_tokens);
    try std.testing.expect(report.speculative.rounds >= 1);
    try std.testing.expect(report.speculative.proposal_depth == 4);
}
