//! One tiny transformer block used as the prefill/decode + KV-cache fixture.
//!
//! This is not Qwen3-0.6B. Shapes are small and weights are synthetic. The
//! schedule is backend-neutral: adapters supply RMSNorm/matmul/RoPE/attention.
//! Model code does not import Metal or HIP.

const std = @import("std");
const Tensor = @import("../runtime/tensor.zig").Tensor;
const TensorError = @import("../runtime/tensor.zig").TensorError;
const KvCache = @import("../runtime/kv_cache.zig").KvCache;
const cpu = @import("../backends/cpu/ops.zig");
const compare = @import("../runtime/compare.zig");

pub const Error = TensorError;

pub const Spec = struct {
    hidden: usize = 8,
    n_q: usize = 2,
    n_kv: usize = 1,
    head_dim: usize = 4,
    inter: usize = 16,
    max_seq: usize = 32,
    eps: f32 = 1e-6,
    theta: f32 = 10_000,

    pub fn qDim(self: Spec) usize {
        return self.n_q * self.head_dim;
    }

    pub fn kvDim(self: Spec) usize {
        return self.n_kv * self.head_dim;
    }
};

pub const fixture_spec = Spec{};

pub const Weights = struct {
    attn_norm: Tensor,
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wo: Tensor,
    wn: Tensor,
    wg: Tensor,
    wu: Tensor,
    wd: Tensor,

    pub fn init(allocator: std.mem.Allocator, spec: Spec) Error!Weights {
        const qd = spec.qDim();
        const kvd = spec.kvDim();
        var w = Weights{
            .attn_norm = try Tensor.alloc(allocator, .f32, &.{spec.hidden}),
            .wq = undefined,
            .wk = undefined,
            .wv = undefined,
            .wo = undefined,
            .wn = undefined,
            .wg = undefined,
            .wu = undefined,
            .wd = undefined,
        };
        errdefer w.attn_norm.deinit();
        w.wq = try Tensor.alloc(allocator, .f32, &.{ spec.hidden, qd });
        errdefer w.wq.deinit();
        w.wk = try Tensor.alloc(allocator, .f32, &.{ spec.hidden, kvd });
        errdefer w.wk.deinit();
        w.wv = try Tensor.alloc(allocator, .f32, &.{ spec.hidden, kvd });
        errdefer w.wv.deinit();
        w.wo = try Tensor.alloc(allocator, .f32, &.{ qd, spec.hidden });
        errdefer w.wo.deinit();
        w.wn = try Tensor.alloc(allocator, .f32, &.{spec.hidden});
        errdefer w.wn.deinit();
        w.wg = try Tensor.alloc(allocator, .f32, &.{ spec.hidden, spec.inter });
        errdefer w.wg.deinit();
        w.wu = try Tensor.alloc(allocator, .f32, &.{ spec.hidden, spec.inter });
        errdefer w.wu.deinit();
        w.wd = try Tensor.alloc(allocator, .f32, &.{ spec.inter, spec.hidden });
        return w;
    }

    pub fn deinit(self: *Weights) void {
        self.attn_norm.deinit();
        self.wq.deinit();
        self.wk.deinit();
        self.wv.deinit();
        self.wo.deinit();
        self.wn.deinit();
        self.wg.deinit();
        self.wu.deinit();
        self.wd.deinit();
        self.* = undefined;
    }

    pub fn fillFixture(self: Weights) Error!void {
        try iotaFill(self.attn_norm, 0.5, 0.02);
        try iotaFill(self.wq, 0.01, 0.001);
        try iotaFill(self.wk, 0.02, 0.001);
        try iotaFill(self.wv, 0.03, 0.001);
        try iotaFill(self.wo, 0.04, 0.001);
        try iotaFill(self.wn, 0.6, 0.02);
        try iotaFill(self.wg, 0.05, 0.001);
        try iotaFill(self.wu, 0.06, 0.001);
        try iotaFill(self.wd, 0.07, 0.001);
    }

    pub fn copyFrom(self: Weights, src: Weights) Error!void {
        try copyF32(self.attn_norm, src.attn_norm);
        try copyF32(self.wq, src.wq);
        try copyF32(self.wk, src.wk);
        try copyF32(self.wv, src.wv);
        try copyF32(self.wo, src.wo);
        try copyF32(self.wn, src.wn);
        try copyF32(self.wg, src.wg);
        try copyF32(self.wu, src.wu);
        try copyF32(self.wd, src.wd);
    }
};

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

    pub fn init(allocator: std.mem.Allocator, spec: Spec) Error!Scratch {
        const qd = spec.qDim();
        const kvd = spec.kvDim();
        const scores = allocator.alloc(f32, spec.max_seq) catch return error.OutOfMemory;
        errdefer allocator.free(scores);
        return .{
            .xn = try Tensor.alloc(allocator, .f32, &.{ spec.max_seq, spec.hidden }),
            .q_lin = try Tensor.alloc(allocator, .f32, &.{ spec.max_seq, qd }),
            .k_lin = try Tensor.alloc(allocator, .f32, &.{ spec.max_seq, kvd }),
            .v_lin = try Tensor.alloc(allocator, .f32, &.{ spec.max_seq, kvd }),
            .q_htd = try Tensor.alloc(allocator, .f32, &.{ spec.n_q, spec.max_seq, spec.head_dim }),
            .k_htd = try Tensor.alloc(allocator, .f32, &.{ spec.n_kv, spec.max_seq, spec.head_dim }),
            .v_htd = try Tensor.alloc(allocator, .f32, &.{ spec.n_kv, spec.max_seq, spec.head_dim }),
            .attn_htd = try Tensor.alloc(allocator, .f32, &.{ spec.n_q, spec.max_seq, spec.head_dim }),
            .attn_lin = try Tensor.alloc(allocator, .f32, &.{ spec.max_seq, qd }),
            .ao = try Tensor.alloc(allocator, .f32, &.{ spec.max_seq, spec.hidden }),
            .x1 = try Tensor.alloc(allocator, .f32, &.{ spec.max_seq, spec.hidden }),
            .mlp_n = try Tensor.alloc(allocator, .f32, &.{ spec.max_seq, spec.hidden }),
            .gate = try Tensor.alloc(allocator, .f32, &.{ spec.max_seq, spec.inter }),
            .up = try Tensor.alloc(allocator, .f32, &.{ spec.max_seq, spec.inter }),
            .hidden_act = try Tensor.alloc(allocator, .f32, &.{ spec.max_seq, spec.inter }),
            .down = try Tensor.alloc(allocator, .f32, &.{ spec.max_seq, spec.hidden }),
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

pub const Session = struct {
    spec: Spec,
    weights: Weights,
    cache: KvCache,
    scratch: Scratch,

    pub fn init(allocator: std.mem.Allocator, spec: Spec) Error!Session {
        if (spec.n_q % spec.n_kv != 0) return error.InvalidShape;
        if (spec.head_dim % 2 != 0) return error.InvalidShape;
        var weights = try Weights.init(allocator, spec);
        errdefer weights.deinit();
        var cache = try KvCache.init(allocator, spec.n_kv, spec.max_seq, spec.head_dim);
        errdefer cache.deinit();
        const scratch = try Scratch.init(allocator, spec);
        return .{
            .spec = spec,
            .weights = weights,
            .cache = cache,
            .scratch = scratch,
        };
    }

    pub fn deinit(self: *Session) void {
        self.scratch.deinit();
        self.cache.deinit();
        self.weights.deinit();
        self.* = undefined;
    }

    pub fn reset(self: *Session) void {
        self.cache.reset();
    }

    pub fn prefill(self: *Session, x: Tensor, out: Tensor) !void {
        try forward(CpuAdapter{}, self, x, self.cache.used, out);
    }

    pub fn decode(self: *Session, x: Tensor, out: Tensor) !void {
        if (x.rank != 2 or x.shape[0] != 1) return error.InvalidShape;
        try forward(CpuAdapter{}, self, x, self.cache.used, out);
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

/// One block: RMSNorm → QKV → RoPE → KV append → causal GQA → O + residual → SwiGLU residual.
/// `pos0` must equal `sess.cache.used` on entry. Decode must not allocate host tensors.
pub fn forward(ops: anytype, sess: *Session, x: Tensor, pos0: usize, out: Tensor) !void {
    const spec = sess.spec;
    if (x.rank != 2 or out.rank != 2) return error.InvalidShape;
    const t = x.shape[0];
    const hidden = spec.hidden;
    if (x.shape[1] != hidden or out.shape[0] != t or out.shape[1] != hidden) return error.ShapeMismatch;
    if (t == 0 or t > spec.max_seq) return error.InvalidShape;
    if (pos0 != sess.cache.used) return error.InvalidShape;
    if (t > sess.cache.remaining()) return error.InvalidShape;

    const d = spec.head_dim;
    const qd = spec.qDim();
    const kvd = spec.kvDim();

    const xn = try sess.scratch.xn.viewAs(&.{ t, hidden });
    try ops.rmsNorm(xn, x, sess.weights.attn_norm, spec.eps);

    const q_lin = try sess.scratch.q_lin.viewAs(&.{ t, qd });
    const k_lin = try sess.scratch.k_lin.viewAs(&.{ t, kvd });
    const v_lin = try sess.scratch.v_lin.viewAs(&.{ t, kvd });
    try ops.matmul(q_lin, xn, sess.weights.wq);
    try ops.matmul(k_lin, xn, sess.weights.wk);
    try ops.matmul(v_lin, xn, sess.weights.wv);

    const q_thd = try q_lin.viewAs(&.{ t, spec.n_q, d });
    const k_thd = try k_lin.viewAs(&.{ t, spec.n_kv, d });
    const v_thd = try v_lin.viewAs(&.{ t, spec.n_kv, d });
    try ops.rope(q_thd, pos0, spec.theta);
    try ops.rope(k_thd, pos0, spec.theta);

    const q_htd = try sess.scratch.q_htd.viewAs(&.{ spec.n_q, t, d });
    const k_htd = try sess.scratch.k_htd.viewAs(&.{ spec.n_kv, t, d });
    const v_htd = try sess.scratch.v_htd.viewAs(&.{ spec.n_kv, t, d });
    try cpu.tokensHeadsToHeadsTokens(q_htd, q_thd);
    try cpu.tokensHeadsToHeadsTokens(k_htd, k_thd);
    try cpu.tokensHeadsToHeadsTokens(v_htd, v_thd);

    try sess.cache.append(k_htd, v_htd);

    const attn_htd = try sess.scratch.attn_htd.viewAs(&.{ spec.n_q, t, d });
    try ops.attentionInto(
        attn_htd,
        q_htd,
        sess.cache.k,
        sess.cache.v,
        sess.cache.used,
        sess.cache.max_seq,
        sess.scratch.scores,
    );

    const attn_thd = try sess.scratch.attn_lin.viewAs(&.{ t, spec.n_q, d });
    try cpu.headsTokensToTokensHeads(attn_thd, attn_htd);
    const attn_flat = try attn_thd.viewAs(&.{ t, qd });

    const ao = try sess.scratch.ao.viewAs(&.{ t, hidden });
    try ops.matmul(ao, attn_flat, sess.weights.wo);
    const x1 = try sess.scratch.x1.viewAs(&.{ t, hidden });
    try ops.add(x1, x, ao);

    const mlp_n = try sess.scratch.mlp_n.viewAs(&.{ t, hidden });
    try ops.rmsNorm(mlp_n, x1, sess.weights.wn, spec.eps);
    const gate = try sess.scratch.gate.viewAs(&.{ t, spec.inter });
    const up = try sess.scratch.up.viewAs(&.{ t, spec.inter });
    const hidden_act = try sess.scratch.hidden_act.viewAs(&.{ t, spec.inter });
    const down = try sess.scratch.down.viewAs(&.{ t, hidden });
    try ops.matmul(gate, mlp_n, sess.weights.wg);
    try ops.matmul(up, mlp_n, sess.weights.wu);
    try ops.siluMul(hidden_act, gate, up);
    try ops.matmul(down, hidden_act, sess.weights.wd);
    try ops.add(out, x1, down);
}

pub fn iotaFill(t: Tensor, start: f32, step: f32) Error!void {
    const xs = try t.f32s();
    for (xs, 0..) |*v, i| v.* = start + step * @as(f32, @floatFromInt(i));
}

fn copyF32(dst: Tensor, src: Tensor) Error!void {
    const a = try dst.f32s();
    const b = try src.f32s();
    if (a.len != b.len) return error.ShapeMismatch;
    @memcpy(a, b);
}

test "prefill matches stepwise decode" {
    const spec = fixture_spec;
    const gpa = std.testing.allocator;
    var sess = try Session.init(gpa, spec);
    defer sess.deinit();
    try sess.weights.fillFixture();

    const tokens: usize = 4;
    var x = try Tensor.alloc(gpa, .f32, &.{ tokens, spec.hidden });
    defer x.deinit();
    try iotaFill(x, 0.1, 0.05);
    var prefill_out = try Tensor.alloc(gpa, .f32, &.{ tokens, spec.hidden });
    defer prefill_out.deinit();
    try sess.prefill(x, prefill_out);

    sess.reset();
    var decode_out = try Tensor.alloc(gpa, .f32, &.{ tokens, spec.hidden });
    defer decode_out.deinit();
    var step_in = try Tensor.alloc(gpa, .f32, &.{ 1, spec.hidden });
    defer step_in.deinit();
    var step_out = try Tensor.alloc(gpa, .f32, &.{ 1, spec.hidden });
    defer step_out.deinit();
    const xs = try x.f32s();
    const ds = try decode_out.f32s();
    var i: usize = 0;
    while (i < tokens) : (i += 1) {
        @memcpy(try step_in.f32s(), xs[i * spec.hidden ..][0..spec.hidden]);
        try sess.decode(step_in, step_out);
        @memcpy(ds[i * spec.hidden ..][0..spec.hidden], try step_out.f32s());
    }
    try compare.expectClose(try prefill_out.f32s(), try decode_out.f32s(), 1e-5, 1e-5);
}

test "decode does not allocate host memory" {
    const spec = fixture_spec;
    var fa = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    var sess = try Session.init(fa.allocator(), spec);
    defer sess.deinit();
    try sess.weights.fillFixture();

    var x = try Tensor.alloc(std.testing.allocator, .f32, &.{ 1, spec.hidden });
    defer x.deinit();
    var out = try Tensor.alloc(std.testing.allocator, .f32, &.{ 1, spec.hidden });
    defer out.deinit();
    try iotaFill(x, 0.2, 0.1);

    const allocs_before = fa.allocations;
    try sess.decode(x, out);
    try std.testing.expectEqual(allocs_before, fa.allocations);
}

test "cache overflow is reported" {
    var spec = fixture_spec;
    spec.max_seq = 2;
    var sess = try Session.init(std.testing.allocator, spec);
    defer sess.deinit();
    try sess.weights.fillFixture();
    var x = try Tensor.alloc(std.testing.allocator, .f32, &.{ 2, spec.hidden });
    defer x.deinit();
    var out = try Tensor.alloc(std.testing.allocator, .f32, &.{ 2, spec.hidden });
    defer out.deinit();
    try iotaFill(x, 0.1, 0.1);
    try sess.prefill(x, out);
    var step_in = try Tensor.alloc(std.testing.allocator, .f32, &.{ 1, spec.hidden });
    defer step_in.deinit();
    var step_out = try Tensor.alloc(std.testing.allocator, .f32, &.{ 1, spec.hidden });
    defer step_out.deinit();
    try iotaFill(step_in, 0.3, 0.1);
    try std.testing.expectError(error.InvalidShape, sess.decode(step_in, step_out));
}
