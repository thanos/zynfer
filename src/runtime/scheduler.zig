//! Stage S1 — multi-request queue + scheduling over independent Sessions.
//!
//! This is **request-level** scheduling (FIFO admit + `max_inflight` +
//! round-robin decode), not Metal command-buffer packing (M3) and not packed
//! continuous batching in one Metal forward (deferred). Each in-flight
//! request owns its own `Session` / KV.

const std = @import("std");
const artifact = @import("../model/artifact.zig");
const qwen3 = @import("../model/qwen3.zig");
const qwen_forward = @import("../model/qwen_forward.zig");
const sample_mod = @import("sample.zig");
const backend_mod = @import("backend.zig");

pub const Error = qwen_forward.Error || error{
    EmptyBatch,
    InvalidInflight,
    SchedulerStall,
};

pub const RequestState = enum {
    queued,
    prefilling,
    decoding,
    done,
    aborted,
};

pub const RequestStats = struct {
    id: u32 = 0,
    prompt_tokens: usize = 0,
    generated_tokens: usize = 0,
    /// Wall from enqueue to first sampled token.
    ttft_ns: u64 = 0,
    /// Wall from enqueue to completion.
    e2e_ns: u64 = 0,
    /// Time spent waiting in queue before first prefill work.
    queue_wait_ns: u64 = 0,
    prefill_ns: u64 = 0,
    decode_ns: u64 = 0,
    /// Mean inter-token latency (tokens after the first); 0 if <2 tokens.
    mean_itl_ns: u64 = 0,
    /// Generated token ids (owned by BatchReport.allocator).
    token_ids: []u32 = &.{},
};

pub const BatchReport = struct {
    mode: []const u8,
    max_inflight: usize,
    n_requests: usize,
    wall_ns: u64,
    total_generated: usize,
    /// Aggregate decode throughput: total_generated / wall (includes queue/prefill).
    aggregate_tok_s: f64,
    requests: []RequestStats,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *BatchReport) void {
        for (self.requests) |*r| {
            if (r.token_ids.len != 0) self.allocator.free(r.token_ids);
        }
        self.allocator.free(self.requests);
        self.* = undefined;
    }
};

const Slot = struct {
    sess: qwen_forward.Session,
    req_index: usize,
    logits: []f32,
    out_ids: std.ArrayList(u32),
    rng: std.Random.DefaultPrng,
    generated: usize = 0,
    started: std.Io.Timestamp = undefined,
    prefill_done: std.Io.Timestamp = undefined,
    last_emit: std.Io.Timestamp = undefined,
    itl_sum_ns: u64 = 0,
    itl_count: usize = 0,
    decode_ns: u64 = 0,
    alive: bool = false,

    fn deinit(self: *Slot, allocator: std.mem.Allocator) void {
        if (self.alive) {
            self.out_ids.deinit(allocator);
            allocator.free(self.logits);
            self.sess.deinit();
            self.alive = false;
        }
    }
};

pub const Job = struct {
    prompt_ids: []const u32,
    max_new_tokens: u32,
    /// Optional stop ids (borrowed).
    stop_ids: []const u32 = &.{},
    sample: sample_mod.Config = .{ .temperature = 0 },
    seed: u64 = 0,
};

fn isStop(id: u32, stop_ids: []const u32) bool {
    for (stop_ids) |s| if (s == id) return true;
    return false;
}

fn nsDelta(start: std.Io.Timestamp, end: std.Io.Timestamp) u64 {
    return @intCast(@max(@as(i96, 0), end.nanoseconds - start.nanoseconds));
}

fn meanItl(sum_ns: u64, count: usize) u64 {
    if (count == 0) return 0;
    return sum_ns / count;
}

/// Run `jobs` serially: one Session at a time (baseline for S1 A/B).
pub fn runSerial(
    allocator: std.mem.Allocator,
    io: std.Io,
    art: *const artifact.Artifact,
    arch: qwen3.Arch,
    kind: backend_mod.BackendKind,
    max_seq: usize,
    jobs: []const Job,
) Error!BatchReport {
    if (jobs.len == 0) return error.EmptyBatch;
    try backend_mod.requireBackend(kind);

    const t0 = std.Io.Clock.awake.now(io);
    const stats = try allocator.alloc(RequestStats, jobs.len);
    errdefer {
        for (stats) |*r| {
            if (r.token_ids.len != 0) allocator.free(r.token_ids);
        }
        allocator.free(stats);
    }
    @memset(stats, .{});

    var total_gen: usize = 0;
    for (jobs, 0..) |job, i| {
        const enqueue = std.Io.Clock.awake.now(io);
        var sess = try qwen_forward.Session.initWithBackend(allocator, art, arch, max_seq, kind);
        defer sess.deinit();

        var out: std.ArrayList(u32) = .empty;
        defer out.deinit(allocator);
        var rng = std.Random.DefaultPrng.init(job.seed);

        var itl_buf: [256]u64 = undefined;
        const itl_cap = @min(itl_buf.len, job.max_new_tokens);
        const itl_slice = itl_buf[0..itl_cap];

        const start = std.Io.Clock.awake.now(io);
        const gs = try sess.generate(io, job.prompt_ids, &out, .{
            .max_new_tokens = job.max_new_tokens,
            .sample = job.sample,
            .stop_ids = job.stop_ids,
            .use_kv_cache = true,
            .itl_ns_out = itl_slice,
        }, &rng);
        const done = std.Io.Clock.awake.now(io);

        var itl_sum: u64 = 0;
        var itl_n: usize = 0;
        while (itl_n < gs.itl_count) : (itl_n += 1) itl_sum += itl_slice[itl_n];

        const tokens = try allocator.dupe(u32, out.items);
        stats[i] = .{
            .id = @intCast(i),
            .prompt_tokens = gs.prompt_tokens,
            .generated_tokens = gs.generated_tokens,
            .ttft_ns = gs.ttft_ns + nsDelta(enqueue, start),
            .e2e_ns = nsDelta(enqueue, done),
            .queue_wait_ns = nsDelta(enqueue, start),
            .prefill_ns = gs.prefill_ns,
            .decode_ns = gs.decode_ns,
            .mean_itl_ns = meanItl(itl_sum, gs.itl_count),
            .token_ids = tokens,
        };
        total_gen += gs.generated_tokens;
    }

    const wall = nsDelta(t0, std.Io.Clock.awake.now(io));
    const tok_s = if (wall == 0) 0 else @as(f64, @floatFromInt(total_gen)) / (@as(f64, @floatFromInt(wall)) * 1e-9);
    return .{
        .mode = "serial",
        .max_inflight = 1,
        .n_requests = jobs.len,
        .wall_ns = wall,
        .total_generated = total_gen,
        .aggregate_tok_s = tok_s,
        .requests = stats,
        .allocator = allocator,
    };
}

/// FIFO queue with up to `max_inflight` concurrent Sessions.
/// Prefill when a slot opens; round-robin `decodeToken` across active slots.
pub fn runScheduled(
    allocator: std.mem.Allocator,
    io: std.Io,
    art: *const artifact.Artifact,
    arch: qwen3.Arch,
    kind: backend_mod.BackendKind,
    max_seq: usize,
    jobs: []const Job,
    max_inflight: usize,
) Error!BatchReport {
    if (jobs.len == 0) return error.EmptyBatch;
    if (max_inflight == 0) return error.InvalidInflight;
    if (max_inflight == 1) return runSerial(allocator, io, art, arch, kind, max_seq, jobs);
    try backend_mod.requireBackend(kind);

    const inflight = @min(max_inflight, jobs.len);
    const t0 = std.Io.Clock.awake.now(io);

    const stats = try allocator.alloc(RequestStats, jobs.len);
    errdefer {
        for (stats) |*r| {
            if (r.token_ids.len != 0) allocator.free(r.token_ids);
        }
        allocator.free(stats);
    }
    @memset(stats, .{});

    const enqueue_ts = try allocator.alloc(std.Io.Timestamp, jobs.len);
    defer allocator.free(enqueue_ts);
    @memset(enqueue_ts, t0);

    var slots = try allocator.alloc(Slot, inflight);
    defer {
        for (slots) |*s| s.deinit(allocator);
        allocator.free(slots);
    }
    @memset(slots, .{
        .sess = undefined,
        .req_index = 0,
        .logits = &.{},
        .out_ids = .empty,
        .rng = undefined,
        .alive = false,
    });

    var next_job: usize = 0;
    var completed: usize = 0;
    var total_gen: usize = 0;
    var rr: usize = 0;

    while (completed < jobs.len) {
        // Each iteration: optionally admit one FIFO job into a free slot
        // (prefill), then take one round-robin decode step if any slot is
        // alive. That grows concurrency up to max_inflight without stalling
        // first-token latency behind a full prefill wave.
        var admitted = false;
        for (slots) |*slot| {
            if (slot.alive) continue;
            if (next_job >= jobs.len) break;
            const job = jobs[next_job];
            const req_i = next_job;
            next_job += 1;

            const start = std.Io.Clock.awake.now(io);
            var sess = try qwen_forward.Session.initWithBackend(allocator, art, arch, max_seq, kind);
            errdefer sess.deinit();
            const logits = try allocator.alloc(f32, arch.vocab_size);
            errdefer allocator.free(logits);
            var out: std.ArrayList(u32) = .empty;
            errdefer out.deinit(allocator);
            try out.ensureTotalCapacity(allocator, job.max_new_tokens);

            try sess.prefillLastLogits(job.prompt_ids, logits);
            const after_prefill = std.Io.Clock.awake.now(io);

            slot.* = .{
                .sess = sess,
                .req_index = req_i,
                .logits = logits,
                .out_ids = out,
                .rng = std.Random.DefaultPrng.init(job.seed),
                .generated = 0,
                .started = start,
                .prefill_done = after_prefill,
                .last_emit = after_prefill,
                .itl_sum_ns = 0,
                .itl_count = 0,
                .decode_ns = 0,
                .alive = true,
            };
            stats[req_i].id = @intCast(req_i);
            stats[req_i].prompt_tokens = job.prompt_ids.len;
            stats[req_i].queue_wait_ns = nsDelta(enqueue_ts[req_i], start);
            stats[req_i].prefill_ns = nsDelta(start, after_prefill);
            admitted = true;
            break;
        }

        var stepped = false;
        var attempts: usize = 0;
        while (attempts < slots.len) : (attempts += 1) {
            const si = (rr + attempts) % slots.len;
            var slot = &slots[si];
            if (!slot.alive) continue;

            const job = jobs[slot.req_index];

            // Match Session.generateCached: no sample when KV is full.
            if (slot.sess.blocks[0].cache.used >= slot.sess.max_seq) {
                const done = std.Io.Clock.awake.now(io);
                const tokens = try allocator.dupe(u32, slot.out_ids.items);
                stats[slot.req_index].generated_tokens = slot.generated;
                stats[slot.req_index].decode_ns = slot.decode_ns;
                stats[slot.req_index].mean_itl_ns = meanItl(slot.itl_sum_ns, slot.itl_count);
                stats[slot.req_index].e2e_ns = nsDelta(enqueue_ts[slot.req_index], done);
                stats[slot.req_index].token_ids = tokens;
                total_gen += slot.generated;
                completed += 1;
                slot.deinit(allocator);
                rr = (si + 1) % slots.len;
                stepped = true;
                break;
            }

            const next = try sample_mod.sampleWithScratch(
                slot.logits,
                job.sample,
                slot.sess.sample_probs,
                slot.sess.sample_idx,
                &slot.rng,
            );
            slot.out_ids.appendAssumeCapacity(next);
            slot.generated += 1;
            const now = std.Io.Clock.awake.now(io);
            if (slot.generated == 1) {
                stats[slot.req_index].ttft_ns = nsDelta(enqueue_ts[slot.req_index], now);
            } else {
                slot.itl_sum_ns += nsDelta(slot.last_emit, now);
                slot.itl_count += 1;
            }
            slot.last_emit = now;

            const finished = slot.generated >= job.max_new_tokens or isStop(next, job.stop_ids);

            if (!finished) {
                const td0 = std.Io.Clock.awake.now(io);
                try slot.sess.decodeToken(next, slot.logits);
                slot.decode_ns += nsDelta(td0, std.Io.Clock.awake.now(io));
            }

            if (finished) {
                const done = std.Io.Clock.awake.now(io);
                const tokens = try allocator.dupe(u32, slot.out_ids.items);
                stats[slot.req_index].generated_tokens = slot.generated;
                stats[slot.req_index].decode_ns = slot.decode_ns;
                stats[slot.req_index].mean_itl_ns = meanItl(slot.itl_sum_ns, slot.itl_count);
                stats[slot.req_index].e2e_ns = nsDelta(enqueue_ts[slot.req_index], done);
                stats[slot.req_index].token_ids = tokens;
                total_gen += slot.generated;
                completed += 1;
                slot.deinit(allocator);
            }

            rr = (si + 1) % slots.len;
            stepped = true;
            break;
        }

        if (!admitted and !stepped) {
            if (next_job >= jobs.len and completed >= jobs.len) break;
            return error.SchedulerStall;
        }
    }

    const wall = nsDelta(t0, std.Io.Clock.awake.now(io));
    const tok_s = if (wall == 0) 0 else @as(f64, @floatFromInt(total_gen)) / (@as(f64, @floatFromInt(wall)) * 1e-9);
    return .{
        .mode = "scheduled",
        .max_inflight = inflight,
        .n_requests = jobs.len,
        .wall_ns = wall,
        .total_generated = total_gen,
        .aggregate_tok_s = tok_s,
        .requests = stats,
        .allocator = allocator,
    };
}

test "scheduler serial and scheduled match greedy tokens on mini" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const bytes = try qwen_forward.buildMiniArtifact(gpa);
    defer gpa.free(bytes);
    var art = try artifact.Artifact.loadOwned(gpa, try gpa.dupe(u8, bytes));
    defer art.deinit();
    const arch = qwen3.stage11_mini;

    const prompt_a = [_]u32{ 2, 3 };
    const prompt_b = [_]u32{ 2, 4 };
    const jobs = [_]Job{
        .{ .prompt_ids = &prompt_a, .max_new_tokens = 3, .seed = 1 },
        .{ .prompt_ids = &prompt_b, .max_new_tokens = 3, .seed = 2 },
    };
    const max_seq: usize = 16;

    var serial = try runSerial(gpa, io, &art, arch, .cpu, max_seq, &jobs);
    defer serial.deinit();
    var scheduled = try runScheduled(gpa, io, &art, arch, .cpu, max_seq, &jobs, 2);
    defer scheduled.deinit();

    try std.testing.expectEqual(@as(usize, 2), serial.n_requests);
    try std.testing.expectEqual(@as(usize, 2), scheduled.n_requests);
    try std.testing.expectEqual(serial.total_generated, scheduled.total_generated);
    try std.testing.expectEqualSlices(u32, serial.requests[0].token_ids, scheduled.requests[0].token_ids);
    try std.testing.expectEqualSlices(u32, serial.requests[1].token_ids, scheduled.requests[1].token_ids);
    try std.testing.expect(scheduled.max_inflight == 2);
}

test "scheduler rejects empty batch and zero inflight" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const bytes = try qwen_forward.buildMiniArtifact(gpa);
    defer gpa.free(bytes);
    var art = try artifact.Artifact.loadOwned(gpa, try gpa.dupe(u8, bytes));
    defer art.deinit();
    const arch = qwen3.stage11_mini;
    const jobs = [_]Job{};
    try std.testing.expectError(error.EmptyBatch, runSerial(gpa, io, &art, arch, .cpu, 8, &jobs));
    const one = [_]Job{.{ .prompt_ids = &.{2}, .max_new_tokens = 1 }};
    try std.testing.expectError(error.InvalidInflight, runScheduled(gpa, io, &art, arch, .cpu, 8, &one, 0));
}
