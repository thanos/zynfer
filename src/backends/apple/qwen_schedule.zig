//! Stage M3 — batched Metal schedule for Qwen blocks.
//!
//! Default path: resident per-layer weights + Metal KV, one command buffer /
//! one `waitUntilCompleted` for the full stack (all layers), retained fusions
//! `silu_mul` + `add_rmsnorm_f32` (Stage 6 lesson at Qwen scale).
//!
//! `ZYNFER_QWEN_METAL=baseline` forces the M0 per-op Adapter path for A/B.

const std = @import("std");
const Tensor = @import("../../runtime/tensor.zig").Tensor;
const qwen3 = @import("../../model/qwen3.zig");
const qwen_weights = @import("../../model/qwen_weights.zig");
const gpu_mod = @import("gpu.zig");
const apple_ops = @import("ops.zig");

const Gpu = gpu_mod.Gpu;
const Buffer = gpu_mod.Buffer;

pub const path_baseline = "baseline_per_op";
pub const path_batched = "batched_resident_kv_fused";

pub var last_qwen_path: []const u8 = "unset";
pub var last_qwen_encodes: u32 = 0;
pub var last_qwen_waits: u32 = 0;

/// Test override. `null` honors `ZYNFER_QWEN_METAL`. `true` → baseline.
pub var force_baseline_path: ?bool = null;

pub fn useBaselinePath() bool {
    if (force_baseline_path) |forced| return forced;
    if (comptime !gpu_mod.have_apple) return true;
    const raw = std.c.getenv("ZYNFER_QWEN_METAL") orelse return false;
    const v = std.mem.span(raw);
    return std.mem.eql(u8, v, "baseline") or std.mem.eql(u8, v, "per-op");
}

fn f32Bytes(n: usize) usize {
    return n * @sizeOf(f32);
}

fn copyTensorToBuf(buf: Buffer, t: Tensor) !void {
    const src = try t.f32s();
    @memcpy(buf.f32s()[0..src.len], src);
}

const LayerDeviceWeights = struct {
    input_ln: Buffer,
    q_norm: Buffer,
    k_norm: Buffer,
    wq: Buffer,
    wk: Buffer,
    wv: Buffer,
    wo: Buffer,
    post_attn_ln: Buffer,
    wg: Buffer,
    wu: Buffer,
    wd: Buffer,

    fn init(gpu: *Gpu, arch: qwen3.Arch) !LayerDeviceWeights {
        const h: usize = @intCast(arch.hidden_size);
        const qd: usize = @intCast(arch.qDim());
        const kvd: usize = @intCast(arch.kvDim());
        const inter: usize = @intCast(arch.intermediate_size);
        const d: usize = @intCast(arch.head_dim);
        return .{
            .input_ln = try gpu.allocShared(f32Bytes(h)),
            .q_norm = try gpu.allocShared(f32Bytes(d)),
            .k_norm = try gpu.allocShared(f32Bytes(d)),
            .wq = try gpu.allocShared(f32Bytes(h * qd)),
            .wk = try gpu.allocShared(f32Bytes(h * kvd)),
            .wv = try gpu.allocShared(f32Bytes(h * kvd)),
            .wo = try gpu.allocShared(f32Bytes(qd * h)),
            .post_attn_ln = try gpu.allocShared(f32Bytes(h)),
            .wg = try gpu.allocShared(f32Bytes(h * inter)),
            .wu = try gpu.allocShared(f32Bytes(h * inter)),
            .wd = try gpu.allocShared(f32Bytes(inter * h)),
        };
    }

    fn deinit(self: *LayerDeviceWeights) void {
        self.input_ln.deinit();
        self.q_norm.deinit();
        self.k_norm.deinit();
        self.wq.deinit();
        self.wk.deinit();
        self.wv.deinit();
        self.wo.deinit();
        self.post_attn_ln.deinit();
        self.wg.deinit();
        self.wu.deinit();
        self.wd.deinit();
        self.* = undefined;
    }

    fn uploadFrom(self: *LayerDeviceWeights, w: qwen_weights.LayerWeights) !void {
        try copyTensorToBuf(self.input_ln, w.input_ln);
        try copyTensorToBuf(self.q_norm, w.q_norm);
        try copyTensorToBuf(self.k_norm, w.k_norm);
        try copyTensorToBuf(self.wq, w.wq);
        try copyTensorToBuf(self.wk, w.wk);
        try copyTensorToBuf(self.wv, w.wv);
        try copyTensorToBuf(self.wo, w.wo);
        try copyTensorToBuf(self.post_attn_ln, w.post_attn_ln);
        try copyTensorToBuf(self.wg, w.wg);
        try copyTensorToBuf(self.wu, w.wu);
        try copyTensorToBuf(self.wd, w.wd);
    }
};

const LayerKv = struct {
    k_cache: Buffer,
    v_cache: Buffer,
    max_seq: usize,
    used: usize = 0,

    fn init(gpu: *Gpu, arch: qwen3.Arch, max_seq: usize) !LayerKv {
        const n_kv: usize = @intCast(arch.num_key_value_heads);
        const d: usize = @intCast(arch.head_dim);
        return .{
            .k_cache = try gpu.allocShared(f32Bytes(n_kv * max_seq * d)),
            .v_cache = try gpu.allocShared(f32Bytes(n_kv * max_seq * d)),
            .max_seq = max_seq,
        };
    }

    fn deinit(self: *LayerKv) void {
        self.k_cache.deinit();
        self.v_cache.deinit();
        self.* = undefined;
    }

    fn reset(self: *LayerKv) void {
        self.used = 0;
    }
};

const Scratch = struct {
    /// Ping-pong activations between layers `[max_seq, hidden]`.
    act_a: Buffer,
    act_b: Buffer,
    xn: Buffer,
    q_lin: Buffer,
    k_lin: Buffer,
    v_lin: Buffer,
    q_htd: Buffer,
    k_htd: Buffer,
    v_htd: Buffer,
    attn_htd: Buffer,
    attn_lin: Buffer,
    ao: Buffer,
    x1: Buffer,
    mlp_n: Buffer,
    gate: Buffer,
    up: Buffer,
    hidden_act: Buffer,
    down: Buffer,
    /// Final-norm row + logits (decode / last-prefill token).
    normed: Buffer,
    logits: Buffer,
    /// Optional long-KV attention scores: n_q * max_seq * max_attention_kv.
    scores: Buffer,
    embed: Buffer,
    final_norm: Buffer,
    lm_head: Buffer,
    lm_head_tied: bool,

    fn init(gpu: *Gpu, arch: qwen3.Arch, max_seq: usize, weights: *const qwen_weights.Weights) !Scratch {
        const h: usize = @intCast(arch.hidden_size);
        const qd: usize = @intCast(arch.qDim());
        const kvd: usize = @intCast(arch.kvDim());
        const inter: usize = @intCast(arch.intermediate_size);
        const n_q: usize = @intCast(arch.num_attention_heads);
        const n_kv: usize = @intCast(arch.num_key_value_heads);
        const d: usize = @intCast(arch.head_dim);
        const vocab: usize = @intCast(arch.vocab_size);
        const score_kv = @min(max_seq, apple_ops.max_attention_kv);
        const scores_n = n_q * max_seq * score_kv;

        var s = Scratch{
            .act_a = try gpu.allocShared(f32Bytes(max_seq * h)),
            .act_b = try gpu.allocShared(f32Bytes(max_seq * h)),
            .xn = try gpu.allocShared(f32Bytes(max_seq * h)),
            .q_lin = try gpu.allocShared(f32Bytes(max_seq * qd)),
            .k_lin = try gpu.allocShared(f32Bytes(max_seq * kvd)),
            .v_lin = try gpu.allocShared(f32Bytes(max_seq * kvd)),
            .q_htd = try gpu.allocShared(f32Bytes(n_q * max_seq * d)),
            .k_htd = try gpu.allocShared(f32Bytes(n_kv * max_seq * d)),
            .v_htd = try gpu.allocShared(f32Bytes(n_kv * max_seq * d)),
            .attn_htd = try gpu.allocShared(f32Bytes(n_q * max_seq * d)),
            .attn_lin = try gpu.allocShared(f32Bytes(max_seq * qd)),
            .ao = try gpu.allocShared(f32Bytes(max_seq * h)),
            .x1 = try gpu.allocShared(f32Bytes(max_seq * h)),
            .mlp_n = try gpu.allocShared(f32Bytes(max_seq * h)),
            .gate = try gpu.allocShared(f32Bytes(max_seq * inter)),
            .up = try gpu.allocShared(f32Bytes(max_seq * inter)),
            .hidden_act = try gpu.allocShared(f32Bytes(max_seq * inter)),
            .down = try gpu.allocShared(f32Bytes(max_seq * h)),
            .normed = try gpu.allocShared(f32Bytes(h)),
            .logits = try gpu.allocShared(f32Bytes(vocab)),
            .scores = try gpu.allocShared(f32Bytes(scores_n)),
            .embed = try gpu.allocShared(f32Bytes(vocab * h)),
            .final_norm = try gpu.allocShared(f32Bytes(h)),
            .lm_head = undefined,
            .lm_head_tied = weights.lm_head_tied,
        };
        errdefer s.deinitPartialBeforeLmHead();
        try copyTensorToBuf(s.embed, weights.embed);
        try copyTensorToBuf(s.final_norm, weights.final_norm);
        if (weights.lm_head_tied) {
            s.lm_head = s.embed;
        } else {
            s.lm_head = try gpu.allocShared(f32Bytes(vocab * h));
            try copyTensorToBuf(s.lm_head, weights.lm_head);
        }
        return s;
    }

    fn deinitPartialBeforeLmHead(self: *Scratch) void {
        self.act_a.deinit();
        self.act_b.deinit();
        self.xn.deinit();
        self.q_lin.deinit();
        self.k_lin.deinit();
        self.v_lin.deinit();
        self.q_htd.deinit();
        self.k_htd.deinit();
        self.v_htd.deinit();
        self.attn_htd.deinit();
        self.attn_lin.deinit();
        self.ao.deinit();
        self.x1.deinit();
        self.mlp_n.deinit();
        self.gate.deinit();
        self.up.deinit();
        self.hidden_act.deinit();
        self.down.deinit();
        self.normed.deinit();
        self.logits.deinit();
        self.scores.deinit();
        self.embed.deinit();
        self.final_norm.deinit();
    }

    fn deinit(self: *Scratch) void {
        if (!self.lm_head_tied) self.lm_head.deinit();
        self.deinitPartialBeforeLmHead();
        self.* = undefined;
    }
};

pub const MetalStack = struct {
    gpu: *Gpu,
    arch: qwen3.Arch,
    max_seq: usize,
    layers_w: []LayerDeviceWeights,
    layers_kv: []LayerKv,
    scratch: Scratch,
    allocator: std.mem.Allocator,

    pub fn init(
        allocator: std.mem.Allocator,
        gpu: *Gpu,
        arch: qwen3.Arch,
        max_seq: usize,
        weights: *const qwen_weights.Weights,
    ) !MetalStack {
        if (max_seq == 0 or max_seq > arch.max_position_embeddings) return error.InvalidShape;
        if (max_seq > apple_ops.max_attention_kv) return error.Unsupported;

        const n_layers: usize = @intCast(arch.num_layers);
        const layers_w = try allocator.alloc(LayerDeviceWeights, n_layers);
        errdefer allocator.free(layers_w);
        @memset(layers_w, undefined);
        var i: usize = 0;
        errdefer while (i > 0) {
            i -= 1;
            layers_w[i].deinit();
        };
        while (i < n_layers) : (i += 1) {
            layers_w[i] = try LayerDeviceWeights.init(gpu, arch);
            try layers_w[i].uploadFrom(weights.layers[i]);
        }

        const layers_kv = try allocator.alloc(LayerKv, n_layers);
        errdefer allocator.free(layers_kv);
        @memset(layers_kv, undefined);
        var j: usize = 0;
        errdefer while (j > 0) {
            j -= 1;
            layers_kv[j].deinit();
        };
        while (j < n_layers) : (j += 1) {
            layers_kv[j] = try LayerKv.init(gpu, arch, max_seq);
        }

        const scratch = try Scratch.init(gpu, arch, max_seq, weights);
        return .{
            .gpu = gpu,
            .arch = arch,
            .max_seq = max_seq,
            .layers_w = layers_w,
            .layers_kv = layers_kv,
            .scratch = scratch,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MetalStack) void {
        self.scratch.deinit();
        for (self.layers_kv) |*kv| kv.deinit();
        self.allocator.free(self.layers_kv);
        for (self.layers_w) |*w| w.deinit();
        self.allocator.free(self.layers_w);
        self.* = undefined;
    }

    pub fn reset(self: *MetalStack) void {
        for (self.layers_kv) |*kv| kv.reset();
    }

    /// Prefill or decode: `x_host` is `[t, hidden]` embeddings; writes last-token logits.
    pub fn forwardLastLogits(self: *MetalStack, x_host: Tensor, logits_out: []f32) !void {
        const arch = self.arch;
        const h: usize = @intCast(arch.hidden_size);
        if (x_host.rank != 2 or x_host.shape[1] != h) return error.ShapeMismatch;
        const t = x_host.shape[0];
        if (t == 0 or t > self.max_seq) return error.InvalidShape;
        if (logits_out.len != arch.vocab_size) return error.ShapeMismatch;

        const pos0 = self.layers_kv[0].used;
        if (pos0 + t > self.max_seq) return error.InvalidShape;
        if (pos0 + t > apple_ops.max_attention_kv) return error.Unsupported;

        // Sanity: all layers share the same used count.
        for (self.layers_kv) |kv| {
            if (kv.used != pos0) return error.InvalidShape;
        }

        @memcpy(self.scratch.act_a.f32s()[0 .. t * h], try x_host.f32s());

        try self.gpu.batchBegin();
        errdefer self.gpu.batchAbort();

        var in_buf = self.scratch.act_a;
        var out_buf = self.scratch.act_b;
        var layer: u32 = 0;
        while (layer < arch.num_layers) : (layer += 1) {
            try encodeLayer(
                self.gpu,
                &self.layers_w[layer],
                &self.layers_kv[layer],
                &self.scratch,
                arch,
                in_buf,
                out_buf,
                t,
                pos0,
            );
            const tmp = in_buf;
            in_buf = out_buf;
            out_buf = tmp;
        }

        // Final RMSNorm on last token row + LM head (Metal matvec).
        const last_row_offset = (t - 1) * h;
        // Copy last row into normed via a 1-row rmsnorm: use a tiny view by
        // encoding rmsnorm with rows=1 over a contiguous last-row alias.
        // act buffers are [t, H] row-major — last row starts at last_row_offset.
        // We don't have buffer views; use encodeRmsNorm on full `t` then read
        // last row, OR copy last row to `normed` on host after wait.
        // Prefer GPU: matmul/rmsnorm on 1 row — upload last row into `xn` slot.
        // Simplest correct approach inside the same CB: rmsnorm with rows=t,
        // then matvec from a dedicated 1×H buffer filled by… we need a gather.
        // Host path after commit for final norm+lm was M0; for M3 we download
        // last hidden, then a second tiny batch for norm+lm head.

        try self.gpu.batchCommit();

        last_qwen_path = path_batched;
        last_qwen_encodes = self.gpu.last_batch_encodes;
        last_qwen_waits = 1;

        for (self.layers_kv) |*kv| kv.used = pos0 + t;

        // Final norm + LM head: second CB (1 wait) — still ≪ 476 waits.
        const last_hidden = in_buf.f32s()[last_row_offset..][0..h];
        @memcpy(self.scratch.xn.f32s()[0..h], last_hidden);

        try self.gpu.batchBegin();
        errdefer self.gpu.batchAbort();
        const hu: u32 = @intCast(h);
        const vu: u32 = @intCast(arch.vocab_size);
        try apple_ops.encodeRmsNorm(self.gpu, self.scratch.normed, self.scratch.xn, self.scratch.final_norm, 1, hu, arch.rms_norm_eps);
        try apple_ops.encodeMatvec(self.gpu, self.scratch.logits, self.scratch.lm_head, self.scratch.normed, vu, hu);
        try self.gpu.batchCommit();
        last_qwen_encodes += self.gpu.last_batch_encodes;
        last_qwen_waits += 1;

        @memcpy(logits_out, self.scratch.logits.f32s()[0..logits_out.len]);
    }
};

fn encodeLayer(
    gpu: *Gpu,
    w: *const LayerDeviceWeights,
    kv: *LayerKv,
    s: *Scratch,
    arch: qwen3.Arch,
    x: Buffer,
    out: Buffer,
    t: usize,
    pos0: usize,
) !void {
    const tu: u32 = @intCast(t);
    const hu: u32 = @intCast(arch.hidden_size);
    const n_q: u32 = @intCast(arch.num_attention_heads);
    const n_kv: u32 = @intCast(arch.num_key_value_heads);
    const d: u32 = @intCast(arch.head_dim);
    const qd: u32 = @intCast(arch.qDim());
    const kvd: u32 = @intCast(arch.kvDim());
    const inter: u32 = @intCast(arch.intermediate_size);
    const max_seq: u32 = @intCast(kv.max_seq);
    const used_u: u32 = @intCast(pos0);
    const eps = arch.rms_norm_eps;
    const theta = arch.rope_theta;

    try apple_ops.encodeRmsNorm(gpu, s.xn, x, w.input_ln, tu, hu, eps);
    try apple_ops.encodeMatmulNaive(gpu, s.q_lin, s.xn, w.wq, tu, qd, hu);
    try apple_ops.encodeMatmulNaive(gpu, s.k_lin, s.xn, w.wk, tu, kvd, hu);
    try apple_ops.encodeMatmulNaive(gpu, s.v_lin, s.xn, w.wv, tu, kvd, hu);

    // QK-norm: treat [t, n_heads, d] as rows=t*n_heads, cols=d.
    try apple_ops.encodeRmsNorm(gpu, s.q_lin, s.q_lin, w.q_norm, tu * n_q, d, eps);
    try apple_ops.encodeRmsNorm(gpu, s.k_lin, s.k_lin, w.k_norm, tu * n_kv, d, eps);

    try apple_ops.encodeRope(gpu, s.q_lin, tu, n_q, d, used_u, theta);
    try apple_ops.encodeRope(gpu, s.k_lin, tu, n_kv, d, used_u, theta);

    try apple_ops.encodePermuteTokensHeads(gpu, s.q_htd, s.q_lin, tu, n_q, d);
    try apple_ops.encodePermuteTokensHeads(gpu, s.k_htd, s.k_lin, tu, n_kv, d);
    try apple_ops.encodePermuteTokensHeads(gpu, s.v_htd, s.v_lin, tu, n_kv, d);
    try apple_ops.encodeKvAppend(gpu, s.k_htd, s.v_htd, kv.k_cache, kv.v_cache, n_kv, tu, d, max_seq, used_u);

    const kv_len: u32 = used_u + tu;
    try apple_ops.encodeAttention(
        gpu,
        s.attn_htd,
        s.q_htd,
        kv.k_cache,
        kv.v_cache,
        n_q,
        n_kv,
        tu,
        kv_len,
        max_seq,
        d,
        s.scores,
    );
    try apple_ops.encodePermuteHeadsTokens(gpu, s.attn_lin, s.attn_htd, tu, n_q, d);
    try apple_ops.encodeMatmulNaive(gpu, s.ao, s.attn_lin, w.wo, tu, hu, qd);

    // Retained fusion: residual add + post-attn RMSNorm.
    try apple_ops.encodeAddRmsNorm(gpu, s.x1, s.mlp_n, x, s.ao, w.post_attn_ln, tu, hu, eps);

    try apple_ops.encodeMatmulNaive(gpu, s.gate, s.mlp_n, w.wg, tu, inter, hu);
    try apple_ops.encodeMatmulNaive(gpu, s.up, s.mlp_n, w.wu, tu, inter, hu);
    try apple_ops.encodeSiluMul(gpu, s.hidden_act, s.gate, s.up, tu * inter);
    try apple_ops.encodeMatmulNaive(gpu, s.down, s.hidden_act, w.wd, tu, hu, inter);
    try apple_ops.encodeAdd(gpu, out, s.x1, s.down, tu * hu);
}
