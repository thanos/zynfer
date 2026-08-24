//! Full Qwen3 forward: embed → blocks → final norm → LM head → logits.
//! CPU is the correctness oracle; Metal routing is Stage M0.

const std = @import("std");
const artifact = @import("artifact.zig");
const qwen3 = @import("qwen3.zig");
const qwen_weights = @import("qwen_weights.zig");
const qwen_block = @import("qwen_block.zig");
const decode_profile = @import("decode_profile.zig");
const cpu = @import("../backends/cpu/ops.zig");
const backend_mod = @import("../runtime/backend.zig");
const apple = @import("../backends/apple/qwen_adapter.zig");
const apple_gpu = @import("../backends/apple/gpu.zig");
const apple_schedule = @import("../backends/apple/qwen_schedule.zig");
const Tensor = @import("../runtime/tensor.zig").Tensor;
const compare = @import("../runtime/compare.zig");
const sample_mod = @import("../runtime/sample.zig");
const tokenizer = @import("tokenizer.zig");

pub const BackendKind = backend_mod.BackendKind;

pub const Error = qwen_weights.Error || qwen_block.Error || sample_mod.Error || backend_mod.SelectionError || std.mem.Allocator.Error;

pub const TopK = struct {
    id: u32,
    logit: f32,
};

pub const DumpHook = *const fn (ctx: ?*anyopaque, name: []const u8, data: []const f32) void;

pub const Session = struct {
    arch: qwen3.Arch,
    weights: qwen_weights.Weights,
    blocks: []qwen_block.BlockSession,
    max_seq: usize,
    hidden_a: Tensor,
    hidden_b: Tensor,
    normed: Tensor,
    logits: Tensor,
    allocator: std.mem.Allocator,
    backend: BackendKind,
    gpu: ?*apple_gpu.Gpu,
    /// Stage M3 batched schedule (null on CPU or baseline Metal).
    metal_stack: ?*apple_schedule.MetalStack,

    pub fn init(
        allocator: std.mem.Allocator,
        art: *const artifact.Artifact,
        arch: qwen3.Arch,
        max_seq: usize,
    ) Error!Session {
        return initWithBackend(allocator, art, arch, max_seq, .cpu);
    }

    pub fn initWithBackend(
        allocator: std.mem.Allocator,
        art: *const artifact.Artifact,
        arch: qwen3.Arch,
        max_seq: usize,
        kind: BackendKind,
    ) Error!Session {
        try backend_mod.requireBackend(kind);
        if (max_seq == 0 or max_seq > arch.max_position_embeddings) return error.InvalidShape;
        var weights = try qwen_weights.Weights.load(allocator, art, arch);
        errdefer weights.deinit();

        const blocks = try allocator.alloc(qwen_block.BlockSession, arch.num_layers);
        errdefer allocator.free(blocks);
        @memset(blocks, undefined);

        var layer: u32 = 0;
        while (layer < arch.num_layers) : (layer += 1) {
            blocks[layer] = try qwen_block.BlockSession.init(
                allocator,
                arch,
                &weights.layers[layer],
                max_seq,
            );
            errdefer blocks[layer].deinit();
        }

        const hidden: usize = @intCast(arch.hidden_size);
        const vocab: usize = @intCast(arch.vocab_size);

        var gpu: ?*apple_gpu.Gpu = null;
        var metal_stack: ?*apple_schedule.MetalStack = null;
        if (kind == .apple) {
            const g = try allocator.create(apple_gpu.Gpu);
            errdefer allocator.destroy(g);
            g.* = try apple_gpu.Gpu.init();
            gpu = g;
            if (!apple_schedule.useBaselinePath()) {
                const ms = try allocator.create(apple_schedule.MetalStack);
                errdefer allocator.destroy(ms);
                ms.* = try apple_schedule.MetalStack.init(
                    allocator,
                    g,
                    arch,
                    max_seq,
                    &weights,
                    apple_schedule.useHalfPath(),
                    apple_schedule.useQ8Path(),
                    art,
                );
                metal_stack = ms;
            }
        }

        return .{
            .arch = arch,
            .weights = weights,
            .blocks = blocks,
            .max_seq = max_seq,
            .hidden_a = try Tensor.alloc(allocator, .f32, &.{ max_seq, hidden }),
            .hidden_b = try Tensor.alloc(allocator, .f32, &.{ max_seq, hidden }),
            .normed = try Tensor.alloc(allocator, .f32, &.{ max_seq, hidden }),
            .logits = try Tensor.alloc(allocator, .f32, &.{vocab}),
            .allocator = allocator,
            .backend = kind,
            .gpu = gpu,
            .metal_stack = metal_stack,
        };
    }

    pub fn deinit(self: *Session) void {
        if (self.metal_stack) |ms| {
            ms.deinit();
            self.allocator.destroy(ms);
        }
        if (self.gpu) |g| {
            g.deinit();
            self.allocator.destroy(g);
        }
        self.hidden_a.deinit();
        self.hidden_b.deinit();
        self.normed.deinit();
        self.logits.deinit();
        for (self.blocks) |*b| b.deinit();
        self.allocator.free(self.blocks);
        self.weights.deinit();
        self.* = undefined;
    }

    pub fn reset(self: *Session) void {
        for (self.blocks) |*b| b.reset();
        if (self.metal_stack) |ms| ms.reset();
    }

    fn syncHostKvUsed(self: *Session) void {
        if (self.metal_stack) |ms| {
            const used = ms.layers_kv[0].used;
            for (self.blocks) |*b| b.cache.used = used;
        }
    }

    fn forwardBlock(
        self: *Session,
        layer: u32,
        in_view: Tensor,
        out_view: Tensor,
    ) Error!void {
        const pos = self.blocks[layer].cache.used;
        switch (self.backend) {
            .cpu => try qwen_block.forward(
                qwen_block.CpuAdapter{},
                &self.blocks[layer],
                in_view,
                pos,
                out_view,
            ),
            .apple => try qwen_block.forward(
                apple.Adapter{ .gpu = self.gpu.? },
                &self.blocks[layer],
                in_view,
                pos,
                out_view,
            ),
            .amd_hip => return error.BackendUnavailable,
        }
    }

    fn forwardBlockProfiled(
        self: *Session,
        layer: u32,
        in_view: Tensor,
        out_view: Tensor,
        buckets: *decode_profile.Accumulators,
        io: std.Io,
        enable_signposts: bool,
    ) Error!void {
        const pos = self.blocks[layer].cache.used;
        switch (self.backend) {
            .cpu => try qwen_block.forward(
                qwen_block.ProfilingCpuAdapter{ .buckets = buckets, .io = io, .enable_signposts = enable_signposts },
                &self.blocks[layer],
                in_view,
                pos,
                out_view,
            ),
            .apple => try qwen_block.forward(
                apple.ProfilingAdapter{
                    .gpu = self.gpu.?,
                    .buckets = buckets,
                    .io = io,
                    .enable_signposts = enable_signposts,
                },
                &self.blocks[layer],
                in_view,
                pos,
                out_view,
            ),
            .amd_hip => return error.BackendUnavailable,
        }
    }

    pub fn backendName(self: Session) []const u8 {
        return self.backend.name();
    }

    /// Prefill `token_ids` and write logits for the **last** token into `logits_out`.
    pub fn prefillLastLogits(self: *Session, token_ids: []const u32, logits_out: []f32) Error!void {
        try self.prefillLastLogitsDump(token_ids, logits_out, null, null);
    }

    /// Same as `prefillLastLogits`, optionally invoking `hook(ctx, name, data)` for debug dumps.
    pub fn prefillLastLogitsDump(
        self: *Session,
        token_ids: []const u32,
        logits_out: []f32,
        hook: ?DumpHook,
        hook_ctx: ?*anyopaque,
    ) Error!void {
        if (token_ids.len == 0 or token_ids.len > self.max_seq) return error.InvalidShape;
        if (logits_out.len != self.arch.vocab_size) return error.ShapeMismatch;

        const dump = struct {
            fn call(h: ?DumpHook, ctx: ?*anyopaque, name: []const u8, data: []const f32) void {
                if (h) |f| f(ctx, name, data);
            }
        }.call;

        self.reset();
        const t = token_ids.len;
        const hidden: usize = @intCast(self.arch.hidden_size);

        const embed_view = try self.hidden_a.viewAs(&.{ t, hidden });
        try cpu.embeddingGather(embed_view, self.weights.embed, token_ids);
        if (hook) |_| {
            const embed = try embed_view.f32s();
            dump(hook, hook_ctx, "embed_last", embed[(t - 1) * hidden ..][0..hidden]);
        }

        if (self.metal_stack) |ms| {
            try ms.forwardLastLogits(embed_view, logits_out);
            self.syncHostKvUsed();
            dump(hook, hook_ctx, "logits", logits_out);
            return;
        }

        var in_buf = self.hidden_a;
        var out_buf = self.hidden_b;
        var layer: u32 = 0;
        while (layer < self.arch.num_layers) : (layer += 1) {
            const in_view = try in_buf.viewAs(&.{ t, hidden });
            const out_view = try out_buf.viewAs(&.{ t, hidden });
            try self.forwardBlock(layer, in_view, out_view);
            if (hook) |_| {
                const out = try out_view.f32s();
                var name_buf: [32]u8 = undefined;
                const name = std.fmt.bufPrint(&name_buf, "layer{d:0>2}", .{layer}) catch unreachable;
                dump(hook, hook_ctx, name, out[(t - 1) * hidden ..][0..hidden]);
            }
            const tmp = in_buf;
            in_buf = out_buf;
            out_buf = tmp;
        }

        const last_in = try in_buf.viewAs(&.{ t, hidden });
        const last_row = try last_in.viewLastRow();
        const normed_row = try self.normed.viewAs(&.{ 1, hidden });
        try cpu.rmsNorm(normed_row, last_row, self.weights.final_norm, self.arch.rms_norm_eps);

        const normed_only = try normed_row.viewAs(&.{hidden});
        dump(hook, hook_ctx, "normed", try normed_only.f32s());
        if (self.weights.lm_head_tied) {
            try lmHeadTied(logits_out, try normed_only.f32s(), self.weights.embed);
        } else {
            var logits_t = self.logits;
            try cpu.matvec(logits_t, self.weights.lm_head, normed_only);
            @memcpy(logits_out, try logits_t.f32s());
        }
        dump(hook, hook_ctx, "logits", logits_out);
    }

    /// Decode one new token using the KV cache filled by a prior prefill/decode.
    pub fn decodeToken(self: *Session, token_id: u32, logits_out: []f32) Error!void {
        if (logits_out.len != self.arch.vocab_size) return error.ShapeMismatch;
        if (self.blocks[0].cache.used >= self.max_seq) return error.InvalidShape;

        const hidden: usize = @intCast(self.arch.hidden_size);
        const t: usize = 1;
        const token_ids = [_]u32{token_id};

        const embed_view = try self.hidden_a.viewAs(&.{ t, hidden });
        try cpu.embeddingGather(embed_view, self.weights.embed, &token_ids);

        if (self.metal_stack) |ms| {
            try ms.forwardLastLogits(embed_view, logits_out);
            self.syncHostKvUsed();
            return;
        }

        var in_buf = self.hidden_a;
        var out_buf = self.hidden_b;
        var layer: u32 = 0;
        while (layer < self.arch.num_layers) : (layer += 1) {
            const in_view = try in_buf.viewAs(&.{ t, hidden });
            const out_view = try out_buf.viewAs(&.{ t, hidden });
            try self.forwardBlock(layer, in_view, out_view);
            const tmp = in_buf;
            in_buf = out_buf;
            out_buf = tmp;
        }

        const last_in = try in_buf.viewAs(&.{ t, hidden });
        const last_row = try last_in.viewLastRow();
        const normed_row = try self.normed.viewAs(&.{ 1, hidden });
        try cpu.rmsNorm(normed_row, last_row, self.weights.final_norm, self.arch.rms_norm_eps);
        const normed_only = try normed_row.viewAs(&.{hidden});
        if (self.weights.lm_head_tied) {
            try lmHeadTied(logits_out, try normed_only.f32s(), self.weights.embed);
        } else {
            var logits_t = self.logits;
            try cpu.matvec(logits_t, self.weights.lm_head, normed_only);
            @memcpy(logits_out, try logits_t.f32s());
        }
    }

    /// Stage M2: one decode token with per-family wall buckets (+ optional sample).
    /// Caller must have already run prefill (KV primed). Does not mutate `buckets` wall until done.
    pub fn profileDecodeToken(
        self: *Session,
        io: std.Io,
        token_id: u32,
        logits_out: []f32,
        buckets: *decode_profile.Accumulators,
        enable_signposts: bool,
        do_sample: bool,
        sample_cfg: sample_mod.Config,
        rng: *std.Random.DefaultPrng,
    ) Error!u32 {
        if (logits_out.len != self.arch.vocab_size) return error.ShapeMismatch;
        if (self.blocks[0].cache.used >= self.max_seq) return error.InvalidShape;

        // M3 batched path: one timed forward (family split is M0-baseline-oriented).
        if (self.metal_stack != null) {
            self.resetMetalLaunchCounters();
            const wall0 = std.Io.Clock.awake.now(io);
            try self.decodeToken(token_id, logits_out);
            buckets.wall_ns = nsDelta(wall0, std.Io.Clock.awake.now(io));
            buckets.add(.mlp, buckets.wall_ns); // whole stack under one row for M3 profile
            buckets.metal_encodes = apple_schedule.last_qwen_encodes;
            buckets.metal_waits = apple_schedule.last_qwen_waits;
            buckets.kv_len = self.blocks[0].cache.used;
            var sampled: u32 = 0;
            if (do_sample) {
                const probs = try self.allocator.alloc(f32, self.arch.vocab_size);
                defer self.allocator.free(probs);
                const idx = try self.allocator.alloc(u32, self.arch.vocab_size);
                defer self.allocator.free(idx);
                const t0 = std.Io.Clock.awake.now(io);
                sampled = try sample_mod.sampleWithScratch(logits_out, sample_cfg, probs, idx, rng);
                buckets.add(.sampling, nsDelta(t0, std.Io.Clock.awake.now(io)));
            }
            return sampled;
        }

        self.resetMetalLaunchCounters();
        const wall0 = std.Io.Clock.awake.now(io);

        const hidden: usize = @intCast(self.arch.hidden_size);
        const t: usize = 1;
        const token_ids = [_]u32{token_id};

        const embed_view = try self.hidden_a.viewAs(&.{ t, hidden });
        {
            const t0 = std.Io.Clock.awake.now(io);
            try cpu.embeddingGather(embed_view, self.weights.embed, &token_ids);
            buckets.add(.embed, nsDelta(t0, std.Io.Clock.awake.now(io)));
        }

        var in_buf = self.hidden_a;
        var out_buf = self.hidden_b;
        var layer: u32 = 0;
        while (layer < self.arch.num_layers) : (layer += 1) {
            const in_view = try in_buf.viewAs(&.{ t, hidden });
            const out_view = try out_buf.viewAs(&.{ t, hidden });
            try self.forwardBlockProfiled(layer, in_view, out_view, buckets, io, enable_signposts);
            const tmp = in_buf;
            in_buf = out_buf;
            out_buf = tmp;
        }

        {
            const t0 = std.Io.Clock.awake.now(io);
            const last_in = try in_buf.viewAs(&.{ t, hidden });
            const last_row = try last_in.viewLastRow();
            const normed_row = try self.normed.viewAs(&.{ 1, hidden });
            try cpu.rmsNorm(normed_row, last_row, self.weights.final_norm, self.arch.rms_norm_eps);
            const normed_only = try normed_row.viewAs(&.{hidden});
            if (self.weights.lm_head_tied) {
                try lmHeadTied(logits_out, try normed_only.f32s(), self.weights.embed);
            } else {
                var logits_t = self.logits;
                try cpu.matvec(logits_t, self.weights.lm_head, normed_only);
                @memcpy(logits_out, try logits_t.f32s());
            }
            buckets.add(.lm_head, nsDelta(t0, std.Io.Clock.awake.now(io)));
        }

        var sampled: u32 = 0;
        if (do_sample) {
            const probs = try self.allocator.alloc(f32, self.arch.vocab_size);
            defer self.allocator.free(probs);
            const idx = try self.allocator.alloc(u32, self.arch.vocab_size);
            defer self.allocator.free(idx);
            const t0 = std.Io.Clock.awake.now(io);
            sampled = try sample_mod.sampleWithScratch(logits_out, sample_cfg, probs, idx, rng);
            buckets.add(.sampling, nsDelta(t0, std.Io.Clock.awake.now(io)));
        }

        const snap = self.metalLaunchSnapshot();
        buckets.metal_encodes = snap.encodes;
        buckets.metal_waits = snap.waits;
        buckets.kv_len = self.blocks[0].cache.used;
        buckets.wall_ns = nsDelta(wall0, std.Io.Clock.awake.now(io));
        return sampled;
    }

    pub const GenerateConfig = struct {
        max_new_tokens: u32 = 64,
        sample: @import("../runtime/sample.zig").Config = .{},
        /// Stop when sampling these ids (typically eos / im_end / endoftext).
        stop_ids: []const u32 = &.{},
        /// Optional per-token wall intervals after the first token (ITL), length >= max_new_tokens.
        itl_ns_out: ?[]u64 = null,
        /// Called after each new token id is appended (for streaming decode).
        on_token: ?*const fn (ctx: ?*anyopaque, token_id: u32) void = null,
        on_token_ctx: ?*anyopaque = null,
        /// When false, each step recomputes the full prefix (Stage 13 baseline).
        use_kv_cache: bool = true,
    };

    pub const GenerateStats = struct {
        prompt_tokens: usize,
        generated_tokens: usize,
        prefill_ns: u64,
        /// Wall time from start of prefill to first generated token sampled.
        ttft_ns: u64,
        decode_ns: u64,
        /// Number of ITL samples written to `itl_ns_out` (generated_tokens - 1 when streaming intervals).
        itl_count: usize = 0,
        use_kv_cache: bool = true,
        /// Measured Metal kernel encodes during prefill (0 on CPU).
        metal_encodes_prefill: u64 = 0,
        metal_waits_prefill: u64 = 0,
        /// Measured Metal encodes/waits across all `decodeToken` calls.
        metal_encodes_decode: u64 = 0,
        metal_waits_decode: u64 = 0,
        /// Number of `decodeToken` calls (usually generated_tokens - 1 when no early stop before last).
        decode_steps: usize = 0,
    };

    fn resetMetalLaunchCounters(self: *Session) void {
        if (self.gpu) |g| g.resetLaunchCounters();
    }

    fn metalLaunchSnapshot(self: *const Session) struct { encodes: u64, waits: u64 } {
        if (self.gpu) |g| return .{ .encodes = g.total_encodes, .waits = g.total_waits };
        return .{ .encodes = 0, .waits = 0 };
    }

    /// Prefill `prompt_ids`, then autoregressively sample up to `max_new_tokens`.
    /// Appends generated ids to `out_ids` (caller provides ArrayList).
    ///
    /// With `use_kv_cache=true` (default): one prefill, then incremental decode.
    /// With `use_kv_cache=false`: each step resets and recomputes the entire
    /// prefix (intentionally O(n²) — Stage 13 educational baseline).
    pub fn generate(
        self: *Session,
        io: std.Io,
        prompt_ids: []const u32,
        out_ids: *std.ArrayList(u32),
        cfg: GenerateConfig,
        rng: *std.Random.DefaultPrng,
    ) Error!GenerateStats {
        if (cfg.use_kv_cache) {
            return self.generateCached(io, prompt_ids, out_ids, cfg, rng);
        }
        return self.generateUncached(io, prompt_ids, out_ids, cfg, rng);
    }

    fn generateCached(
        self: *Session,
        io: std.Io,
        prompt_ids: []const u32,
        out_ids: *std.ArrayList(u32),
        cfg: GenerateConfig,
        rng: *std.Random.DefaultPrng,
    ) Error!GenerateStats {
        if (prompt_ids.len == 0) return error.InvalidShape;

        const logits = try self.allocator.alloc(f32, self.arch.vocab_size);
        defer self.allocator.free(logits);
        const probs = try self.allocator.alloc(f32, self.arch.vocab_size);
        defer self.allocator.free(probs);
        const idx = try self.allocator.alloc(u32, self.arch.vocab_size);
        defer self.allocator.free(idx);

        const t0 = std.Io.Clock.awake.now(io);
        self.resetMetalLaunchCounters();
        try self.prefillLastLogits(prompt_ids, logits);
        const t_prefill = std.Io.Clock.awake.now(io);
        const metal_prefill = self.metalLaunchSnapshot();

        var generated: usize = 0;
        var decode_ns: u64 = 0;
        var ttft_ns: u64 = 0;
        var itl_count: usize = 0;
        var last_emit = t_prefill;
        var decode_steps: usize = 0;
        var metal_encodes_decode: u64 = 0;
        var metal_waits_decode: u64 = 0;

        while (generated < cfg.max_new_tokens) {
            if (self.blocks[0].cache.used >= self.max_seq) break;

            const next = try sample_mod.sampleWithScratch(logits, cfg.sample, probs, idx, rng);
            try out_ids.append(self.allocator, next);
            generated += 1;

            const t_emit = std.Io.Clock.awake.now(io);
            if (generated == 1) {
                ttft_ns = nsDelta(t0, t_emit);
            } else if (cfg.itl_ns_out) |itl| {
                if (itl_count < itl.len) {
                    itl[itl_count] = nsDelta(last_emit, t_emit);
                    itl_count += 1;
                }
            }
            last_emit = t_emit;

            if (cfg.on_token) |cb| cb(cfg.on_token_ctx, next);
            if (isStop(next, cfg.stop_ids)) break;

            self.resetMetalLaunchCounters();
            const td0 = std.Io.Clock.awake.now(io);
            try self.decodeToken(next, logits);
            decode_ns += nsDelta(td0, std.Io.Clock.awake.now(io));
            const snap = self.metalLaunchSnapshot();
            metal_encodes_decode += snap.encodes;
            metal_waits_decode += snap.waits;
            decode_steps += 1;
        }

        if (ttft_ns == 0) ttft_ns = nsDelta(t0, std.Io.Clock.awake.now(io));

        return .{
            .prompt_tokens = prompt_ids.len,
            .generated_tokens = generated,
            .prefill_ns = nsDelta(t0, t_prefill),
            .ttft_ns = ttft_ns,
            .decode_ns = decode_ns,
            .itl_count = itl_count,
            .use_kv_cache = true,
            .metal_encodes_prefill = metal_prefill.encodes,
            .metal_waits_prefill = metal_prefill.waits,
            .metal_encodes_decode = metal_encodes_decode,
            .metal_waits_decode = metal_waits_decode,
            .decode_steps = decode_steps,
        };
    }

    /// Intentionally inefficient: every step resets KV and re-runs the full prefix.
    fn generateUncached(
        self: *Session,
        io: std.Io,
        prompt_ids: []const u32,
        out_ids: *std.ArrayList(u32),
        cfg: GenerateConfig,
        rng: *std.Random.DefaultPrng,
    ) Error!GenerateStats {
        if (prompt_ids.len == 0) return error.InvalidShape;

        const logits = try self.allocator.alloc(f32, self.arch.vocab_size);
        defer self.allocator.free(logits);
        const probs = try self.allocator.alloc(f32, self.arch.vocab_size);
        defer self.allocator.free(probs);
        const idx = try self.allocator.alloc(u32, self.arch.vocab_size);
        defer self.allocator.free(idx);

        var prefix: std.ArrayList(u32) = .empty;
        defer prefix.deinit(self.allocator);
        try prefix.appendSlice(self.allocator, prompt_ids);

        const t0 = std.Io.Clock.awake.now(io);
        try self.prefillLastLogits(prefix.items, logits);
        const t_prefill = std.Io.Clock.awake.now(io);

        var generated: usize = 0;
        var decode_ns: u64 = 0;
        var ttft_ns: u64 = 0;
        var itl_count: usize = 0;
        var last_emit = t_prefill;

        while (generated < cfg.max_new_tokens) {
            if (prefix.items.len >= self.max_seq) break;

            const next = try sample_mod.sampleWithScratch(logits, cfg.sample, probs, idx, rng);
            try out_ids.append(self.allocator, next);
            try prefix.append(self.allocator, next);
            generated += 1;

            const t_emit = std.Io.Clock.awake.now(io);
            if (generated == 1) {
                ttft_ns = nsDelta(t0, t_emit);
            } else if (cfg.itl_ns_out) |itl| {
                if (itl_count < itl.len) {
                    itl[itl_count] = nsDelta(last_emit, t_emit);
                    itl_count += 1;
                }
            }
            last_emit = t_emit;

            if (cfg.on_token) |cb| cb(cfg.on_token_ctx, next);
            if (isStop(next, cfg.stop_ids)) break;

            const td0 = std.Io.Clock.awake.now(io);
            try self.prefillLastLogits(prefix.items, logits);
            decode_ns += nsDelta(td0, std.Io.Clock.awake.now(io));
        }

        if (ttft_ns == 0) ttft_ns = nsDelta(t0, std.Io.Clock.awake.now(io));

        return .{
            .prompt_tokens = prompt_ids.len,
            .generated_tokens = generated,
            .prefill_ns = nsDelta(t0, t_prefill),
            .ttft_ns = ttft_ns,
            .decode_ns = decode_ns,
            .itl_count = itl_count,
            .use_kv_cache = false,
        };
    }

    /// Total allocated KV capacity across all layers (f32 K+V).
    pub fn kvBytesCapacity(self: *const Session) u64 {
        if (self.blocks.len == 0) return 0;
        return @as(u64, @intCast(self.blocks.len)) * self.blocks[0].cache.bytesCapacity();
    }

    pub fn kvBytesUsed(self: *const Session) u64 {
        var sum: u64 = 0;
        for (self.blocks) |b| sum += b.cache.bytesUsed();
        return sum;
    }
};

fn isStop(id: u32, stop_ids: []const u32) bool {
    for (stop_ids) |s| if (s == id) return true;
    return false;
}

fn nsDelta(start: std.Io.Timestamp, end: std.Io.Timestamp) u64 {
    return @intCast(@max(@as(i96, 0), end.nanoseconds - start.nanoseconds));
}

fn lmHeadTied(logits: []f32, hidden: []const f32, embed: Tensor) Error!void {
    if (embed.rank != 2) return error.InvalidShape;
    const vocab = embed.shape[0];
    const hidden_dim = embed.shape[1];
    if (logits.len != vocab or hidden.len != hidden_dim) return error.ShapeMismatch;
    const tab = try embed.f32s();
    for (0..vocab) |v| {
        const row = tab[v * hidden_dim ..][0..hidden_dim];
        var dot: f32 = 0;
        for (hidden, row) |h, e| dot += h * e;
        logits[v] = dot;
    }
}

pub fn topK(logits: []const f32, k: usize, out: []TopK) void {
    const n = @min(k, out.len);
    for (0..n) |i| {
        out[i] = .{ .id = 0, .logit = -std.math.inf(f32) };
    }
    for (logits, 0..) |logit, id| {
        var i: usize = 0;
        while (i < n) : (i += 1) {
            if (logit > out[i].logit) {
                var j = n - 1;
                while (j > i) : (j -= 1) out[j] = out[j - 1];
                out[i] = .{ .id = @intCast(id), .logit = logit };
                break;
            }
        }
    }
}

/// Deterministic Stage 11 CI artifact: 1-layer mini Qwen with f32 weights.
pub fn buildMiniArtifact(allocator: std.mem.Allocator) Error![]u8 {
    const arch = qwen3.stage11_mini;
    const meta = artifact.Meta.fromArch(arch);
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var specs: std.ArrayList(artifact.TensorSpec) = .empty;
    defer specs.deinit(a);

    const hidden = arch.hidden_size;
    const vocab = arch.vocab_size;

    try appendF32Matrix(a, &specs, qwen3.embed_tokens_name, vocab, hidden, 1);
    try appendF32Vec(a, &specs, qwen3.final_norm_name, hidden, 100);
    try appendLayerMini(a, &specs, 0, arch, 200);

    return artifact.build(allocator, meta, specs.items);
}

fn appendLayerMini(
    a: std.mem.Allocator,
    specs: *std.ArrayList(artifact.TensorSpec),
    layer: u32,
    arch: qwen3.Arch,
    seed: u32,
) Error!void {
    var buf: [96]u8 = undefined;
    const hidden = arch.hidden_size;
    const qd = arch.qDim();
    const kvd = arch.kvDim();
    const inter = arch.intermediate_size;
    const hd = arch.head_dim;
    const s = seed;

    try appendF32Vec(a, specs, qwen3.layerInputNormName(layer, &buf), hidden, s + 1);
    try appendF32Vec(a, specs, qwen3.layerQNormName(layer, &buf), hd, s + 2);
    try appendF32Vec(a, specs, qwen3.layerKNormName(layer, &buf), hd, s + 3);
    try appendF32Matrix(a, specs, qwen3.layerQProjName(layer, &buf), qd, hidden, s + 4);
    try appendF32Matrix(a, specs, qwen3.layerKProjName(layer, &buf), kvd, hidden, s + 5);
    try appendF32Matrix(a, specs, qwen3.layerVProjName(layer, &buf), kvd, hidden, s + 6);
    try appendF32Matrix(a, specs, qwen3.layerOProjName(layer, &buf), hidden, qd, s + 7);
    try appendF32Vec(a, specs, qwen3.layerPostAttnNormName(layer, &buf), hidden, s + 8);
    try appendF32Matrix(a, specs, qwen3.layerGateProjName(layer, &buf), inter, hidden, s + 9);
    try appendF32Matrix(a, specs, qwen3.layerUpProjName(layer, &buf), inter, hidden, s + 10);
    try appendF32Matrix(a, specs, qwen3.layerDownProjName(layer, &buf), hidden, inter, s + 11);
}

fn appendF32Vec(
    a: std.mem.Allocator,
    specs: *std.ArrayList(artifact.TensorSpec),
    name: []const u8,
    n: u32,
    seed: u32,
) Error!void {
    const data = try a.alloc(f32, n);
    fillVec(data, seed);
    const shape = try a.dupe(u32, &.{n});
    const owned = try a.dupe(u8, name);
    try specs.append(a, .{
        .name = owned,
        .tensor_id = 0,
        .dtype = .f32,
        .shape = shape,
        .bytes = std.mem.sliceAsBytes(data),
    });
}

fn appendF32Matrix(
    a: std.mem.Allocator,
    specs: *std.ArrayList(artifact.TensorSpec),
    name: []const u8,
    rows: u32,
    cols: u32,
    seed: u32,
) Error!void {
    const n = @as(usize, rows) * @as(usize, cols);
    const data = try a.alloc(f32, n);
    fillMatrixHF(data, rows, cols, seed);
    const shape = try a.dupe(u32, &.{ rows, cols });
    const owned = try a.dupe(u8, name);
    try specs.append(a, .{
        .name = owned,
        .tensor_id = 0,
        .dtype = .f32,
        .shape = shape,
        .bytes = std.mem.sliceAsBytes(data),
    });
}

fn fillVec(out: []f32, seed: u32) void {
    for (out, 0..) |*v, i| v.* = @as(f32, @floatFromInt(seed)) * 0.01 + @as(f32, @floatFromInt(i)) * 0.001;
}

/// HF layout `[rows, cols]` (out × in).
fn fillMatrixHF(out: []f32, rows: u32, cols: u32, seed: u32) void {
    const r: usize = @intCast(rows);
    const c: usize = @intCast(cols);
    var i: usize = 0;
    while (i < r) : (i += 1) {
        var j: usize = 0;
        while (j < c) : (j += 1) {
            out[i * c + j] = @as(f32, @floatFromInt(seed)) * 0.002 + @as(f32, @floatFromInt(i + j)) * 0.0003;
        }
    }
}

test "mini artifact forward produces deterministic logits" {
    const gpa = std.testing.allocator;
    const bytes = try buildMiniArtifact(gpa);
    defer gpa.free(bytes);

    var art = try artifact.Artifact.loadOwned(gpa, try gpa.dupe(u8, bytes));
    defer art.deinit();

    const arch = qwen3.stage11_mini;
    var sess = try Session.init(gpa, &art, arch, 8);
    defer sess.deinit();

    const token_ids = [_]u32{ 2, 3 };
    const logits = try gpa.alloc(f32, arch.vocab_size);
    defer gpa.free(logits);
    try sess.prefillLastLogits(&token_ids, logits);

    var sess2 = try Session.init(gpa, &art, arch, 8);
    defer sess2.deinit();
    const logits2 = try gpa.alloc(f32, arch.vocab_size);
    defer gpa.free(logits2);
    try sess2.prefillLastLogits(&token_ids, logits2);
    try compare.expectClose(logits, logits2, 0, 0);

    var top: [3]TopK = undefined;
    topK(logits, 3, &top);
    try std.testing.expect(top[0].logit >= top[1].logit);
}

test "mini forward is non-zero" {
    const gpa = std.testing.allocator;
    const bytes = try buildMiniArtifact(gpa);
    defer gpa.free(bytes);
    var art = try artifact.Artifact.loadOwned(gpa, try gpa.dupe(u8, bytes));
    defer art.deinit();
    var sess = try Session.init(gpa, &art, qwen3.stage11_mini, 8);
    defer sess.deinit();
    const token_ids = [_]u32{ 1, 2, 3 };
    const logits = try gpa.alloc(f32, qwen3.stage11_mini.vocab_size);
    defer gpa.free(logits);
    try sess.prefillLastLogits(&token_ids, logits);
    var sum: f32 = 0;
    for (logits) |v| sum += @abs(v);
    try std.testing.expect(sum > 0);
}

test "Stage 13: cached generate matches uncached greedy tokens" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const bytes = try buildMiniArtifact(gpa);
    defer gpa.free(bytes);
    var art = try artifact.Artifact.loadOwned(gpa, try gpa.dupe(u8, bytes));
    defer art.deinit();

    const arch = qwen3.stage11_mini;
    const prompt = [_]u32{ 2, 3 };
    const max_new: u32 = 4;

    var cached_ids: std.ArrayList(u32) = .empty;
    defer cached_ids.deinit(gpa);
    var uncached_ids: std.ArrayList(u32) = .empty;
    defer uncached_ids.deinit(gpa);

    var sess_c = try Session.init(gpa, &art, arch, prompt.len + max_new);
    defer sess_c.deinit();
    var rng_c = std.Random.DefaultPrng.init(0);
    _ = try sess_c.generate(io, &prompt, &cached_ids, .{
        .max_new_tokens = max_new,
        .sample = .{ .temperature = 0 },
        .use_kv_cache = true,
    }, &rng_c);

    var sess_u = try Session.init(gpa, &art, arch, prompt.len + max_new);
    defer sess_u.deinit();
    var rng_u = std.Random.DefaultPrng.init(0);
    _ = try sess_u.generate(io, &prompt, &uncached_ids, .{
        .max_new_tokens = max_new,
        .sample = .{ .temperature = 0 },
        .use_kv_cache = false,
    }, &rng_u);

    try std.testing.expectEqualSlices(u32, cached_ids.items, uncached_ids.items);
    try std.testing.expect(cached_ids.items.len == max_new);
    try std.testing.expect(sess_c.kvBytesUsed() > 0);
    try std.testing.expect(sess_c.kvBytesCapacity() >= sess_c.kvBytesUsed());
}

test "Stage M0: mini Metal forward matches CPU logits" {
    if (apple_gpu.skipAppleGpuTests()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const bytes = try buildMiniArtifact(gpa);
    defer gpa.free(bytes);
    var art = try artifact.Artifact.loadOwned(gpa, try gpa.dupe(u8, bytes));
    defer art.deinit();

    const arch = qwen3.stage11_mini;
    const token_ids = [_]u32{ 2, 3 };

    var cpu_sess = try Session.init(gpa, &art, arch, 8);
    defer cpu_sess.deinit();
    var metal_sess = try Session.initWithBackend(gpa, &art, arch, 8, .apple);
    defer metal_sess.deinit();

    const cpu_logits = try gpa.alloc(f32, arch.vocab_size);
    defer gpa.free(cpu_logits);
    const metal_logits = try gpa.alloc(f32, arch.vocab_size);
    defer gpa.free(metal_logits);

    try cpu_sess.prefillLastLogits(&token_ids, cpu_logits);
    try metal_sess.prefillLastLogits(&token_ids, metal_logits);
    try compare.expectClose(cpu_logits, metal_logits, 3e-3, 3e-3);
}

test "Stage M1: Metal generate reports measured encode/wait counts" {
    if (apple_gpu.skipAppleGpuTests()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const bytes = try buildMiniArtifact(gpa);
    defer gpa.free(bytes);
    var art = try artifact.Artifact.loadOwned(gpa, try gpa.dupe(u8, bytes));
    defer art.deinit();

    const arch = qwen3.stage11_mini;
    const prompt = [_]u32{ 2, 3 };
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(gpa);

    var sess = try Session.initWithBackend(gpa, &art, arch, prompt.len + 3, .apple);
    defer sess.deinit();
    var rng = std.Random.DefaultPrng.init(0);
    const stats = try sess.generate(io, &prompt, &out, .{
        .max_new_tokens = 3,
        .sample = .{ .temperature = 0 },
        .use_kv_cache = true,
    }, &rng);

    try std.testing.expect(stats.metal_encodes_prefill > 0);
    try std.testing.expect(stats.metal_waits_prefill > 0);
    try std.testing.expect(stats.decode_steps >= 1);
    try std.testing.expect(stats.metal_encodes_decode > 0);
    try std.testing.expect(stats.metal_waits_decode > 0);
    if (sess.metal_stack == null) {
        // M0 per-op path: every encode waits.
        try std.testing.expectEqual(stats.metal_encodes_prefill, stats.metal_waits_prefill);
        try std.testing.expectEqual(stats.metal_encodes_decode, stats.metal_waits_decode);
    } else {
        // M3 batched: waits ≪ encodes (one CB for stack + one for LM head per forward).
        try std.testing.expect(stats.metal_waits_prefill < stats.metal_encodes_prefill);
        try std.testing.expect(stats.metal_waits_decode < stats.metal_encodes_decode);
    }
}

test "Stage M2: profileDecodeToken fills family buckets" {
    if (apple_gpu.skipAppleGpuTests()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const bytes = try buildMiniArtifact(gpa);
    defer gpa.free(bytes);
    var art = try artifact.Artifact.loadOwned(gpa, try gpa.dupe(u8, bytes));
    defer art.deinit();

    const arch = qwen3.stage11_mini;
    const prompt = [_]u32{ 2, 3 };
    var sess = try Session.initWithBackend(gpa, &art, arch, prompt.len + 2, .apple);
    defer sess.deinit();

    const logits = try gpa.alloc(f32, arch.vocab_size);
    defer gpa.free(logits);
    try sess.prefillLastLogits(&prompt, logits);

    var buckets = decode_profile.Accumulators{};
    var rng = std.Random.DefaultPrng.init(0);
    const io = std.testing.io;
    _ = try sess.profileDecodeToken(io, 1, logits, &buckets, false, true, .{ .temperature = 0 }, &rng);

    try std.testing.expect(buckets.wall_ns > 0);
    try std.testing.expect(buckets.sumFamilies() > 0);
    try std.testing.expect(buckets.metal_encodes > 0);
    try std.testing.expect(buckets.metal_waits > 0);
    if (sess.metal_stack == null) {
        try std.testing.expectEqual(buckets.metal_encodes, buckets.metal_waits);
    } else {
        try std.testing.expect(buckets.metal_waits < buckets.metal_encodes);
        try std.testing.expectEqual(@as(u32, 2), apple_schedule.last_qwen_waits);
    }
    const top = buckets.top3();
    try std.testing.expect(top[0].ns >= top[1].ns);
}

test "Stage M3: batched Metal mini matches CPU logits and collapses waits" {
    if (apple_gpu.skipAppleGpuTests()) return error.SkipZigTest;
    defer apple_schedule.force_baseline_path = null;

    const gpa = std.testing.allocator;
    const bytes = try buildMiniArtifact(gpa);
    defer gpa.free(bytes);
    var art = try artifact.Artifact.loadOwned(gpa, try gpa.dupe(u8, bytes));
    defer art.deinit();

    const arch = qwen3.stage11_mini;
    const token_ids = [_]u32{ 2, 3 };

    apple_schedule.force_baseline_path = false;
    var batched = try Session.initWithBackend(gpa, &art, arch, 8, .apple);
    defer batched.deinit();
    try std.testing.expect(batched.metal_stack != null);

    apple_schedule.force_baseline_path = true;
    var baseline = try Session.initWithBackend(gpa, &art, arch, 8, .apple);
    defer baseline.deinit();
    try std.testing.expect(baseline.metal_stack == null);

    var cpu_sess = try Session.init(gpa, &art, arch, 8);
    defer cpu_sess.deinit();

    const cpu_logits = try gpa.alloc(f32, arch.vocab_size);
    defer gpa.free(cpu_logits);
    const bat_logits = try gpa.alloc(f32, arch.vocab_size);
    defer gpa.free(bat_logits);
    const base_logits = try gpa.alloc(f32, arch.vocab_size);
    defer gpa.free(base_logits);

    try cpu_sess.prefillLastLogits(&token_ids, cpu_logits);
    try batched.prefillLastLogits(&token_ids, bat_logits);
    try baseline.prefillLastLogits(&token_ids, base_logits);

    try compare.expectClose(cpu_logits, bat_logits, 3e-3, 3e-3);
    try compare.expectClose(cpu_logits, base_logits, 3e-3, 3e-3);
    try std.testing.expectEqualStrings(apple_schedule.path_batched, apple_schedule.last_qwen_path);
    try std.testing.expectEqual(@as(u32, 2), apple_schedule.last_qwen_waits);
    try std.testing.expect(apple_schedule.last_qwen_encodes > 10);
}

test "Stage M4: batched Metal bf16 matches CPU logits within half tolerance" {
    if (apple_gpu.skipAppleGpuTests()) return error.SkipZigTest;
    defer apple_schedule.force_baseline_path = null;
    defer apple_schedule.force_half_path = null;

    const gpa = std.testing.allocator;
    const bytes = try buildMiniArtifact(gpa);
    defer gpa.free(bytes);
    var art = try artifact.Artifact.loadOwned(gpa, try gpa.dupe(u8, bytes));
    defer art.deinit();

    const arch = qwen3.stage11_mini;
    const token_ids = [_]u32{ 2, 3 };

    apple_schedule.force_baseline_path = false;
    apple_schedule.force_half_path = true;
    var half = try Session.initWithBackend(gpa, &art, arch, 8, .apple);
    defer half.deinit();
    try std.testing.expect(half.metal_stack != null);
    try std.testing.expect(half.metal_stack.?.half_mode);

    apple_schedule.force_half_path = false;
    var f32_stack = try Session.initWithBackend(gpa, &art, arch, 8, .apple);
    defer f32_stack.deinit();

    var cpu_sess = try Session.init(gpa, &art, arch, 8);
    defer cpu_sess.deinit();

    const cpu_logits = try gpa.alloc(f32, arch.vocab_size);
    defer gpa.free(cpu_logits);
    const half_logits = try gpa.alloc(f32, arch.vocab_size);
    defer gpa.free(half_logits);
    const f32_logits = try gpa.alloc(f32, arch.vocab_size);
    defer gpa.free(f32_logits);

    try cpu_sess.prefillLastLogits(&token_ids, cpu_logits);
    try half.prefillLastLogits(&token_ids, half_logits);
    try std.testing.expectEqualStrings(apple_schedule.path_bf16, apple_schedule.last_qwen_path);
    try f32_stack.prefillLastLogits(&token_ids, f32_logits);

    // BF16 weights+KV: ~3–4 decimal digits; 5e-3 atol is dtype-justified vs f32 CPU oracle.
    try compare.expectClose(cpu_logits, half_logits, 5e-3, 5e-3);
    try compare.expectClose(cpu_logits, f32_logits, 3e-3, 3e-3);
    try std.testing.expectEqual(@as(u32, 2), apple_schedule.last_qwen_waits);
}

test "Stage M4: greedy tokens match CPU (mini, bf16 batched)" {
    if (apple_gpu.skipAppleGpuTests()) return error.SkipZigTest;
    defer apple_schedule.force_baseline_path = null;
    defer apple_schedule.force_half_path = null;

    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const bytes = try buildMiniArtifact(gpa);
    defer gpa.free(bytes);
    var art = try artifact.Artifact.loadOwned(gpa, try gpa.dupe(u8, bytes));
    defer art.deinit();

    const arch = qwen3.stage11_mini;
    const prompt = [_]u32{ 2, 3 };
    const max_new: u32 = 4;

    var cpu_ids: std.ArrayList(u32) = .empty;
    defer cpu_ids.deinit(gpa);
    var half_ids: std.ArrayList(u32) = .empty;
    defer half_ids.deinit(gpa);

    var cpu_sess = try Session.init(gpa, &art, arch, prompt.len + max_new);
    defer cpu_sess.deinit();
    var rng_cpu = std.Random.DefaultPrng.init(42);
    _ = try cpu_sess.generate(io, &prompt, &cpu_ids, .{
        .max_new_tokens = max_new,
        .sample = .{ .temperature = 0 },
        .use_kv_cache = true,
    }, &rng_cpu);

    apple_schedule.force_baseline_path = false;
    apple_schedule.force_half_path = true;
    var half_sess = try Session.initWithBackend(gpa, &art, arch, prompt.len + max_new, .apple);
    defer half_sess.deinit();
    var rng_half = std.Random.DefaultPrng.init(42);
    _ = try half_sess.generate(io, &prompt, &half_ids, .{
        .max_new_tokens = max_new,
        .sample = .{ .temperature = 0 },
        .use_kv_cache = true,
    }, &rng_half);

    try std.testing.expectEqualSlices(u32, cpu_ids.items, half_ids.items);
}

test "Stage M4: greedy tokens match CPU (full model when artifact present)" {
    if (apple_gpu.skipAppleGpuTests()) return error.SkipZigTest;
    const full = std.process.Environ.getPosix(std.testing.environ, "ZYNFER_FULL_MODEL_TESTS") orelse "";
    if (!(std.mem.eql(u8, full, "1") or std.mem.eql(u8, full, "true"))) return error.SkipZigTest;
    defer apple_schedule.force_baseline_path = null;
    defer apple_schedule.force_half_path = null;

    const path = "models/qwen3-0.6b.zynfer";
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var art = try artifact.Artifact.loadFile(gpa, io, path);
    defer art.deinit();
    const arch = try art.meta.toArch();

    var tok = try tokenizer.Tokenizer.loadHfDir(gpa, io, "models/Qwen3-0.6B");
    defer tok.deinit();
    const prompt_text = "Explain gravity simply.";
    const wrapped = try tok.applyChatTemplate(gpa, prompt_text);
    defer gpa.free(wrapped);
    const prompt_ids = try tok.encode(gpa, wrapped);
    defer gpa.free(prompt_ids);

    const max_new: u32 = 2;
    const max_seq = prompt_ids.len + max_new;
    if (max_seq > arch.max_position_embeddings) return error.SkipZigTest;

    var cpu_ids: std.ArrayList(u32) = .empty;
    defer cpu_ids.deinit(gpa);
    var half_ids: std.ArrayList(u32) = .empty;
    defer half_ids.deinit(gpa);

    var cpu_sess = try Session.init(gpa, &art, arch, max_seq);
    defer cpu_sess.deinit();
    var rng_cpu = std.Random.DefaultPrng.init(0);
    _ = try cpu_sess.generate(io, prompt_ids, &cpu_ids, .{
        .max_new_tokens = max_new,
        .sample = .{ .temperature = 0 },
        .use_kv_cache = true,
    }, &rng_cpu);

    apple_schedule.force_baseline_path = false;
    apple_schedule.force_half_path = true;
    var half_sess = try Session.initWithBackend(gpa, &art, arch, max_seq, .apple);
    defer half_sess.deinit();
    var rng_half = std.Random.DefaultPrng.init(0);
    _ = try half_sess.generate(io, prompt_ids, &half_ids, .{
        .max_new_tokens = max_new,
        .sample = .{ .temperature = 0 },
        .use_kv_cache = true,
    }, &rng_half);

    try std.testing.expectEqualSlices(u32, cpu_ids.items, half_ids.items);
}

test "Stage M5: batched Metal int8 matches CPU logits within quant tolerance" {
    if (apple_gpu.skipAppleGpuTests()) return error.SkipZigTest;
    defer apple_schedule.force_baseline_path = null;
    defer apple_schedule.force_half_path = null;
    defer apple_schedule.force_q8_path = null;

    const gpa = std.testing.allocator;
    const bytes = try buildMiniArtifact(gpa);
    defer gpa.free(bytes);
    var art = try artifact.Artifact.loadOwned(gpa, try gpa.dupe(u8, bytes));
    defer art.deinit();

    const arch = qwen3.stage11_mini;
    const token_ids = [_]u32{ 2, 3 };

    apple_schedule.force_baseline_path = false;
    apple_schedule.force_half_path = false;
    apple_schedule.force_q8_path = true;
    var q8 = try Session.initWithBackend(gpa, &art, arch, 8, .apple);
    defer q8.deinit();
    try std.testing.expect(q8.metal_stack != null);
    try std.testing.expect(q8.metal_stack.?.q8_mode);

    var cpu_sess = try Session.init(gpa, &art, arch, 8);
    defer cpu_sess.deinit();

    const cpu_logits = try gpa.alloc(f32, arch.vocab_size);
    defer gpa.free(cpu_logits);
    const q8_logits = try gpa.alloc(f32, arch.vocab_size);
    defer gpa.free(q8_logits);

    try cpu_sess.prefillLastLogits(&token_ids, cpu_logits);
    try q8.prefillLastLogits(&token_ids, q8_logits);

    // Per-row int8 packing error; 5e-2 atol is dtype-justified vs f32 CPU.
    try compare.expectClose(cpu_logits, q8_logits, 5e-2, 5e-2);
    try std.testing.expectEqualStrings(apple_schedule.path_q8, apple_schedule.last_qwen_path);
    try std.testing.expectEqual(@as(u32, 2), apple_schedule.last_qwen_waits);
}

test "Stage M5: greedy tokens match CPU (full model when artifact present)" {
    if (apple_gpu.skipAppleGpuTests()) return error.SkipZigTest;
    const full = std.process.Environ.getPosix(std.testing.environ, "ZYNFER_FULL_MODEL_TESTS") orelse "";
    if (!(std.mem.eql(u8, full, "1") or std.mem.eql(u8, full, "true"))) return error.SkipZigTest;
    defer apple_schedule.force_baseline_path = null;
    defer apple_schedule.force_half_path = null;
    defer apple_schedule.force_q8_path = null;

    const path = "models/qwen3-0.6b.zynfer";
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var art = try artifact.Artifact.loadFile(gpa, io, path);
    defer art.deinit();
    const arch = try art.meta.toArch();

    var tok = try tokenizer.Tokenizer.loadHfDir(gpa, io, "models/Qwen3-0.6B");
    defer tok.deinit();
    const prompt_text = "Explain gravity simply.";
    const wrapped = try tok.applyChatTemplate(gpa, prompt_text);
    defer gpa.free(wrapped);
    const prompt_ids = try tok.encode(gpa, wrapped);
    defer gpa.free(prompt_ids);

    const max_new: u32 = 2;
    const max_seq = prompt_ids.len + max_new;
    if (max_seq > arch.max_position_embeddings) return error.SkipZigTest;

    var cpu_ids: std.ArrayList(u32) = .empty;
    defer cpu_ids.deinit(gpa);
    var q8_ids: std.ArrayList(u32) = .empty;
    defer q8_ids.deinit(gpa);

    var cpu_sess = try Session.init(gpa, &art, arch, max_seq);
    defer cpu_sess.deinit();
    var rng_cpu = std.Random.DefaultPrng.init(0);
    _ = try cpu_sess.generate(io, prompt_ids, &cpu_ids, .{
        .max_new_tokens = max_new,
        .sample = .{ .temperature = 0 },
        .use_kv_cache = true,
    }, &rng_cpu);

    apple_schedule.force_baseline_path = false;
    apple_schedule.force_half_path = false;
    apple_schedule.force_q8_path = true;
    var q8_sess = try Session.initWithBackend(gpa, &art, arch, max_seq, .apple);
    defer q8_sess.deinit();
    try std.testing.expect(q8_sess.metal_stack != null);
    try std.testing.expect(q8_sess.metal_stack.?.q8_mode);
    var rng_q8 = std.Random.DefaultPrng.init(0);
    _ = try q8_sess.generate(io, prompt_ids, &q8_ids, .{
        .max_new_tokens = max_new,
        .sample = .{ .temperature = 0 },
        .use_kv_cache = true,
    }, &rng_q8);

    try std.testing.expectEqualSlices(u32, cpu_ids.items, q8_ids.items);
    try std.testing.expectEqualStrings(apple_schedule.path_q8, apple_schedule.last_qwen_path);
}
