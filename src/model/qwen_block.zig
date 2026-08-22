//! One Qwen3 transformer block: RMSNorm → QKV → QK-norm → RoPE → GQA → SwiGLU.

const std = @import("std");
const Tensor = @import("../runtime/tensor.zig").Tensor;
const TensorError = @import("../runtime/tensor.zig").TensorError;
const KvCache = @import("../runtime/kv_cache.zig").KvCache;
const qwen3 = @import("qwen3.zig");
const qwen_weights = @import("qwen_weights.zig");
const cpu = @import("../backends/cpu/ops.zig");

pub const Error = TensorError;

pub const Scratch = struct {
    xn: Tensor,
    q_lin: Tensor,
    k_lin: Tensor,
    v_lin: Tensor,
    q_htd: Tensor,
    k_htd: Tensor,
    v_htd: Tensor,
    attn_htd: Tensor,
    attn_lin: Tensor,
    ao: Tensor,
    x1: Tensor,
    mlp_n: Tensor,
    gate: Tensor,
    up: Tensor,
    hidden_act: Tensor,
    down: Tensor,
    scores: []f32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, arch: qwen3.Arch, max_seq: usize) Error!Scratch {
        const hidden: usize = @intCast(arch.hidden_size);
        const qd: usize = @intCast(arch.qDim());
        const kvd: usize = @intCast(arch.kvDim());
        const inter: usize = @intCast(arch.intermediate_size);
        const n_q: usize = @intCast(arch.num_attention_heads);
        const n_kv: usize = @intCast(arch.num_key_value_heads);
        const d: usize = @intCast(arch.head_dim);

        const scores = allocator.alloc(f32, max_seq) catch return error.OutOfMemory;
        errdefer allocator.free(scores);
        return .{
            .xn = try Tensor.alloc(allocator, .f32, &.{ max_seq, hidden }),
            .q_lin = try Tensor.alloc(allocator, .f32, &.{ max_seq, qd }),
            .k_lin = try Tensor.alloc(allocator, .f32, &.{ max_seq, kvd }),
            .v_lin = try Tensor.alloc(allocator, .f32, &.{ max_seq, kvd }),
            .q_htd = try Tensor.alloc(allocator, .f32, &.{ n_q, max_seq, d }),
            .k_htd = try Tensor.alloc(allocator, .f32, &.{ n_kv, max_seq, d }),
            .v_htd = try Tensor.alloc(allocator, .f32, &.{ n_kv, max_seq, d }),
            .attn_htd = try Tensor.alloc(allocator, .f32, &.{ n_q, max_seq, d }),
            .attn_lin = try Tensor.alloc(allocator, .f32, &.{ max_seq, qd }),
            .ao = try Tensor.alloc(allocator, .f32, &.{ max_seq, hidden }),
            .x1 = try Tensor.alloc(allocator, .f32, &.{ max_seq, hidden }),
            .mlp_n = try Tensor.alloc(allocator, .f32, &.{ max_seq, hidden }),
            .gate = try Tensor.alloc(allocator, .f32, &.{ max_seq, inter }),
            .up = try Tensor.alloc(allocator, .f32, &.{ max_seq, inter }),
            .hidden_act = try Tensor.alloc(allocator, .f32, &.{ max_seq, inter }),
            .down = try Tensor.alloc(allocator, .f32, &.{ max_seq, hidden }),
            .scores = scores,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Scratch) void {
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
        self.allocator.free(self.scores);
        self.* = undefined;
    }
};

pub const BlockSession = struct {
    arch: qwen3.Arch,
    weights: *const qwen_weights.LayerWeights,
    cache: KvCache,
    scratch: Scratch,

    pub fn init(
        allocator: std.mem.Allocator,
        arch: qwen3.Arch,
        weights: *const qwen_weights.LayerWeights,
        max_seq: usize,
    ) Error!BlockSession {
        if (arch.num_attention_heads % arch.num_key_value_heads != 0) return error.InvalidShape;
        return .{
            .arch = arch,
            .weights = weights,
            .cache = try KvCache.init(
                allocator,
                @intCast(arch.num_key_value_heads),
                max_seq,
                @intCast(arch.head_dim),
            ),
            .scratch = try Scratch.init(allocator, arch, max_seq),
        };
    }

    pub fn deinit(self: *BlockSession) void {
        self.scratch.deinit();
        self.cache.deinit();
        self.* = undefined;
    }

    pub fn reset(self: *BlockSession) void {
        self.cache.reset();
    }
};

pub const CpuAdapter = struct {
    pub fn rmsNorm(_: CpuAdapter, dst: Tensor, x: Tensor, w: Tensor, eps: f32) Error!void {
        try cpu.rmsNorm(dst, x, w, eps);
    }

    pub fn matmul(_: CpuAdapter, c: Tensor, a: Tensor, b: Tensor) Error!void {
        try cpu.matmul(c, a, b);
    }

    pub fn add(_: CpuAdapter, dst: Tensor, a: Tensor, b: Tensor) Error!void {
        try cpu.add(dst, a, b);
    }

    pub fn siluMul(_: CpuAdapter, dst: Tensor, gate: Tensor, up: Tensor) Error!void {
        try cpu.siluMul(dst, gate, up);
    }

    pub fn rope(_: CpuAdapter, x: Tensor, pos0: usize, theta: f32) Error!void {
        try cpu.rope(x, pos0, theta);
    }

    pub fn attentionInto(
        _: CpuAdapter,
        out: Tensor,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_len: usize,
        kv_stride: usize,
        scores: []f32,
    ) Error!void {
        try cpu.attentionInto(out, q, k, v, kv_len, kv_stride, scores);
    }
};

/// Forward one block. `pos0` must equal `sess.cache.used` on entry.
pub fn forward(ops: anytype, sess: *BlockSession, x: Tensor, pos0: usize, out: Tensor) Error!void {
    const arch = sess.arch;
    const w = sess.weights;
    const hidden: usize = @intCast(arch.hidden_size);
    const qd: usize = @intCast(arch.qDim());
    const kvd: usize = @intCast(arch.kvDim());
    const inter: usize = @intCast(arch.intermediate_size);
    const n_q: usize = @intCast(arch.num_attention_heads);
    const n_kv: usize = @intCast(arch.num_key_value_heads);
    const d: usize = @intCast(arch.head_dim);

    if (x.rank != 2 or out.rank != 2) return error.InvalidShape;
    const t = x.shape[0];
    if (x.shape[1] != hidden or out.shape[0] != t or out.shape[1] != hidden) return error.ShapeMismatch;
    if (t == 0 or pos0 != sess.cache.used) return error.InvalidShape;
    if (t > sess.cache.remaining()) return error.InvalidShape;

    const xn = try sess.scratch.xn.viewAs(&.{ t, hidden });
    try ops.rmsNorm(xn, x, w.input_ln, arch.rms_norm_eps);

    const q_lin = try sess.scratch.q_lin.viewAs(&.{ t, qd });
    const k_lin = try sess.scratch.k_lin.viewAs(&.{ t, kvd });
    const v_lin = try sess.scratch.v_lin.viewAs(&.{ t, kvd });
    try ops.matmul(q_lin, xn, w.wq);
    try ops.matmul(k_lin, xn, w.wk);
    try ops.matmul(v_lin, xn, w.wv);

    const q_thd = try q_lin.viewAs(&.{ t, n_q, d });
    const k_thd = try k_lin.viewAs(&.{ t, n_kv, d });
    const v_thd = try v_lin.viewAs(&.{ t, n_kv, d });
    try ops.rmsNorm(q_thd, q_thd, w.q_norm, arch.rms_norm_eps);
    try ops.rmsNorm(k_thd, k_thd, w.k_norm, arch.rms_norm_eps);
    try ops.rope(q_thd, pos0, arch.rope_theta);
    try ops.rope(k_thd, pos0, arch.rope_theta);

    const q_htd = try sess.scratch.q_htd.viewAs(&.{ n_q, t, d });
    const k_htd = try sess.scratch.k_htd.viewAs(&.{ n_kv, t, d });
    const v_htd = try sess.scratch.v_htd.viewAs(&.{ n_kv, t, d });
    try cpu.tokensHeadsToHeadsTokens(q_htd, q_thd);
    try cpu.tokensHeadsToHeadsTokens(k_htd, k_thd);
    try cpu.tokensHeadsToHeadsTokens(v_htd, v_thd);

    try sess.cache.append(k_htd, v_htd);

    const attn_htd = try sess.scratch.attn_htd.viewAs(&.{ n_q, t, d });
    try ops.attentionInto(
        attn_htd,
        q_htd,
        sess.cache.k,
        sess.cache.v,
        sess.cache.used,
        sess.cache.max_seq,
        sess.scratch.scores,
    );

    const attn_thd = try sess.scratch.attn_lin.viewAs(&.{ t, n_q, d });
    try cpu.headsTokensToTokensHeads(attn_thd, attn_htd);
    const attn_flat = try attn_thd.viewAs(&.{ t, qd });

    const ao = try sess.scratch.ao.viewAs(&.{ t, hidden });
    try ops.matmul(ao, attn_flat, w.wo);
    const x1 = try sess.scratch.x1.viewAs(&.{ t, hidden });
    try ops.add(x1, x, ao);

    const mlp_n = try sess.scratch.mlp_n.viewAs(&.{ t, hidden });
    try ops.rmsNorm(mlp_n, x1, w.post_attn_ln, arch.rms_norm_eps);
    const gate = try sess.scratch.gate.viewAs(&.{ t, inter });
    const up = try sess.scratch.up.viewAs(&.{ t, inter });
    const hidden_act = try sess.scratch.hidden_act.viewAs(&.{ t, inter });
    const down = try sess.scratch.down.viewAs(&.{ t, hidden });
    try ops.matmul(gate, mlp_n, w.wg);
    try ops.matmul(up, mlp_n, w.wu);
    try ops.siluMul(hidden_act, gate, up);
    try ops.matmul(down, hidden_act, w.wd);
    try ops.add(out, x1, down);
}
