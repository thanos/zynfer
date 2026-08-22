//! Full Qwen3 CPU forward: embed → blocks → final norm → LM head → logits.

const std = @import("std");
const artifact = @import("artifact.zig");
const qwen3 = @import("qwen3.zig");
const qwen_weights = @import("qwen_weights.zig");
const qwen_block = @import("qwen_block.zig");
const cpu = @import("../backends/cpu/ops.zig");
const Tensor = @import("../runtime/tensor.zig").Tensor;
const compare = @import("../runtime/compare.zig");

pub const Error = qwen_weights.Error || qwen_block.Error;

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

    pub fn init(
        allocator: std.mem.Allocator,
        art: *const artifact.Artifact,
        arch: qwen3.Arch,
        max_seq: usize,
    ) Error!Session {
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
        };
    }

    pub fn deinit(self: *Session) void {
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

        var in_buf = self.hidden_a;
        var out_buf = self.hidden_b;
        var layer: u32 = 0;
        while (layer < self.arch.num_layers) : (layer += 1) {
            const in_view = try in_buf.viewAs(&.{ t, hidden });
            const out_view = try out_buf.viewAs(&.{ t, hidden });
            try qwen_block.forward(
                qwen_block.CpuAdapter{},
                &self.blocks[layer],
                in_view,
                self.blocks[layer].cache.used,
                out_view,
            );
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
};

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
