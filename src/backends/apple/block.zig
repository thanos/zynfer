//! Apple Metal schedule for the tiny transformer block fixture.
//!
//! Stage 6 default path: persistent shared buffers, Metal-resident KV,
//! and one command buffer / one `waitUntilCompleted` per prefill or decode.
//!
//! Set `ZYNFER_APPLE_BLOCK=baseline` to force the Stage 4/5 per-op path
//! (upload + encode_and_wait per kernel) for A/B measurement.

const std = @import("std");
const Tensor = @import("../../runtime/tensor.zig").Tensor;
const compare = @import("../../runtime/compare.zig");
const tiny = @import("../../model/tiny_block.zig");
const gpu_mod = @import("gpu.zig");
const apple_ops = @import("ops.zig");

const Gpu = gpu_mod.Gpu;
const Buffer = gpu_mod.Buffer;

/// Stable labels reported by `block-bench` / `last_block_path`.
pub const path_baseline = "baseline_per_op";
pub const path_staged = "batched_resident_kv_fused";

pub var last_block_path: []const u8 = "unset";
pub var last_block_encodes: u32 = 0;
pub var last_block_waits: u32 = 0;

/// Test override for path selection. `null` honors `ZYNFER_APPLE_BLOCK`.
/// `true` → Stage 4/5 per-op; `false` → Stage 6 one-CB/wait path.
pub var force_baseline_path: ?bool = null;

fn useBaselinePath() bool {
    if (force_baseline_path) |forced| return forced;
    if (comptime !gpu_mod.have_apple) return true;
    const raw = std.c.getenv("ZYNFER_APPLE_BLOCK") orelse return false;
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

fn copyBufToTensor(t: Tensor, buf: Buffer) !void {
    const dst = try t.f32s();
    @memcpy(dst, buf.f32s()[0..dst.len]);
}

const DeviceWeights = struct {
    attn_norm: Buffer,
    wq: Buffer,
    wk: Buffer,
    wv: Buffer,
    wo: Buffer,
    wn: Buffer,
    wg: Buffer,
    wu: Buffer,
    wd: Buffer,

    fn init(gpu: *Gpu, spec: tiny.Spec) !DeviceWeights {
        const qd = spec.qDim();
        const kvd = spec.kvDim();
        const h = spec.hidden;
        const inter = spec.inter;
        return .{
            .attn_norm = try gpu.allocShared(f32Bytes(h)),
            .wq = try gpu.allocShared(f32Bytes(h * qd)),
            .wk = try gpu.allocShared(f32Bytes(h * kvd)),
            .wv = try gpu.allocShared(f32Bytes(h * kvd)),
            .wo = try gpu.allocShared(f32Bytes(qd * h)),
            .wn = try gpu.allocShared(f32Bytes(h)),
            .wg = try gpu.allocShared(f32Bytes(h * inter)),
            .wu = try gpu.allocShared(f32Bytes(h * inter)),
            .wd = try gpu.allocShared(f32Bytes(inter * h)),
        };
    }

    fn deinit(self: *DeviceWeights) void {
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

    fn uploadFrom(self: *DeviceWeights, w: tiny.Weights) !void {
        try copyTensorToBuf(self.attn_norm, w.attn_norm);
        try copyTensorToBuf(self.wq, w.wq);
        try copyTensorToBuf(self.wk, w.wk);
        try copyTensorToBuf(self.wv, w.wv);
        try copyTensorToBuf(self.wo, w.wo);
        try copyTensorToBuf(self.wn, w.wn);
        try copyTensorToBuf(self.wg, w.wg);
        try copyTensorToBuf(self.wu, w.wu);
        try copyTensorToBuf(self.wd, w.wd);
    }
};

const DeviceScratch = struct {
    x: Buffer,
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
    out: Buffer,
    k_cache: Buffer,
    v_cache: Buffer,

    fn init(gpu: *Gpu, spec: tiny.Spec) !DeviceScratch {
        const ms = spec.max_seq;
        const h = spec.hidden;
        const qd = spec.qDim();
        const kvd = spec.kvDim();
        const d = spec.head_dim;
        const inter = spec.inter;
        return .{
            .x = try gpu.allocShared(f32Bytes(ms * h)),
            .xn = try gpu.allocShared(f32Bytes(ms * h)),
            .q_lin = try gpu.allocShared(f32Bytes(ms * qd)),
            .k_lin = try gpu.allocShared(f32Bytes(ms * kvd)),
            .v_lin = try gpu.allocShared(f32Bytes(ms * kvd)),
            .q_htd = try gpu.allocShared(f32Bytes(spec.n_q * ms * d)),
            .k_htd = try gpu.allocShared(f32Bytes(spec.n_kv * ms * d)),
            .v_htd = try gpu.allocShared(f32Bytes(spec.n_kv * ms * d)),
            .attn_htd = try gpu.allocShared(f32Bytes(spec.n_q * ms * d)),
            .attn_lin = try gpu.allocShared(f32Bytes(ms * qd)),
            .ao = try gpu.allocShared(f32Bytes(ms * h)),
            .x1 = try gpu.allocShared(f32Bytes(ms * h)),
            .mlp_n = try gpu.allocShared(f32Bytes(ms * h)),
            .gate = try gpu.allocShared(f32Bytes(ms * inter)),
            .up = try gpu.allocShared(f32Bytes(ms * inter)),
            .hidden_act = try gpu.allocShared(f32Bytes(ms * inter)),
            .down = try gpu.allocShared(f32Bytes(ms * h)),
            .out = try gpu.allocShared(f32Bytes(ms * h)),
            .k_cache = try gpu.allocShared(f32Bytes(spec.n_kv * ms * d)),
            .v_cache = try gpu.allocShared(f32Bytes(spec.n_kv * ms * d)),
        };
    }

    fn deinit(self: *DeviceScratch) void {
        self.x.deinit();
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
        self.out.deinit();
        self.k_cache.deinit();
        self.v_cache.deinit();
        self.* = undefined;
    }
};

/// Baseline Adapter: one upload/launch/wait per op (Stage 4/5).
pub const Adapter = struct {
    gpu: *Gpu,

    pub fn rmsNorm(self: Adapter, dst: Tensor, x: Tensor, w: Tensor, eps: f32) !void {
        try apple_ops.rmsNorm(self.gpu, dst, x, w, eps);
    }

    pub fn matmul(self: Adapter, c: Tensor, a: Tensor, b: Tensor) !void {
        try apple_ops.matmul(self.gpu, c, a, b);
    }

    pub fn add(self: Adapter, dst: Tensor, a: Tensor, b: Tensor) !void {
        try apple_ops.add(self.gpu, dst, a, b);
    }

    pub fn siluMul(self: Adapter, dst: Tensor, gate: Tensor, up: Tensor) !void {
        try apple_ops.siluMul(self.gpu, dst, gate, up);
    }

    pub fn rope(self: Adapter, x: Tensor, pos0: usize, theta: f32) !void {
        try apple_ops.rope(self.gpu, x, pos0, theta);
    }

    pub fn attentionInto(
        self: Adapter,
        out: Tensor,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_len: usize,
        kv_stride: usize,
        _: []f32,
    ) !void {
        try apple_ops.attention(self.gpu, out, q, k, v, kv_len, kv_stride);
    }
};

pub const Session = struct {
    inner: tiny.Session,
    gpu: *Gpu,
    weights_gpu: DeviceWeights,
    scratch_gpu: DeviceScratch,
    weights_dirty: bool,

    pub fn init(allocator: std.mem.Allocator, gpu: *Gpu, spec: tiny.Spec) !Session {
        var inner = try tiny.Session.init(allocator, spec);
        errdefer inner.deinit();
        var weights_gpu = try DeviceWeights.init(gpu, spec);
        errdefer weights_gpu.deinit();
        var scratch_gpu = try DeviceScratch.init(gpu, spec);
        errdefer scratch_gpu.deinit();
        return .{
            .inner = inner,
            .gpu = gpu,
            .weights_gpu = weights_gpu,
            .scratch_gpu = scratch_gpu,
            .weights_dirty = true,
        };
    }

    pub fn deinit(self: *Session) void {
        self.scratch_gpu.deinit();
        self.weights_gpu.deinit();
        self.inner.deinit();
        self.* = undefined;
    }

    pub fn reset(self: *Session) void {
        self.inner.reset();
    }

    pub fn markWeightsDirty(self: *Session) void {
        self.weights_dirty = true;
    }

    pub fn prefill(self: *Session, x: Tensor, out: Tensor) !void {
        try self.forward(x, out);
    }

    pub fn decode(self: *Session, x: Tensor, out: Tensor) !void {
        if (x.rank != 2 or x.shape[0] != 1) return error.InvalidShape;
        try self.forward(x, out);
    }

    fn forward(self: *Session, x: Tensor, out: Tensor) !void {
        if (useBaselinePath()) {
            last_block_path = path_baseline;
            last_block_waits = 15;
            last_block_encodes = 15;
            try tiny.forward(Adapter{ .gpu = self.gpu }, &self.inner, x, self.inner.cache.used, out);
            return;
        }
        try self.forwardBatched(x, out);
    }

    fn forwardBatched(self: *Session, x: Tensor, out: Tensor) !void {
        const spec = self.inner.spec;
        if (x.rank != 2 or out.rank != 2) return error.InvalidShape;
        const t = x.shape[0];
        const hidden = spec.hidden;
        if (x.shape[1] != hidden or out.shape[0] != t or out.shape[1] != hidden) return error.ShapeMismatch;
        if (t == 0 or t > spec.max_seq) return error.InvalidShape;
        const pos0 = self.inner.cache.used;
        if (t > self.inner.cache.remaining()) return error.InvalidShape;
        if (pos0 + t > apple_ops.max_attention_kv) return error.Unsupported;

        if (self.weights_dirty) {
            try self.weights_gpu.uploadFrom(self.inner.weights);
            self.weights_dirty = false;
        }

        // Host → shared input (before encoding).
        @memcpy(self.scratch_gpu.x.f32s()[0 .. t * hidden], try x.f32s());

        try self.gpu.batchBegin();
        errdefer self.gpu.batchAbort();

        const tu: u32 = @intCast(t);
        const hu: u32 = @intCast(hidden);
        const n_q: u32 = @intCast(spec.n_q);
        const n_kv: u32 = @intCast(spec.n_kv);
        const d: u32 = @intCast(spec.head_dim);
        const qd: u32 = @intCast(spec.qDim());
        const kvd: u32 = @intCast(spec.kvDim());
        const inter: u32 = @intCast(spec.inter);
        const max_seq: u32 = @intCast(spec.max_seq);
        const used_u: u32 = @intCast(pos0);
        const s = &self.scratch_gpu;
        const w = &self.weights_gpu;

        try apple_ops.encodeRmsNorm(self.gpu, s.xn, s.x, w.attn_norm, tu, hu, spec.eps);
        try apple_ops.encodeMatmulNaive(self.gpu, s.q_lin, s.xn, w.wq, tu, qd, hu);
        try apple_ops.encodeMatmulNaive(self.gpu, s.k_lin, s.xn, w.wk, tu, kvd, hu);
        try apple_ops.encodeMatmulNaive(self.gpu, s.v_lin, s.xn, w.wv, tu, kvd, hu);
        try apple_ops.encodeRope(self.gpu, s.q_lin, tu, n_q, d, used_u, spec.theta);
        try apple_ops.encodeRope(self.gpu, s.k_lin, tu, n_kv, d, used_u, spec.theta);
        try apple_ops.encodePermuteTokensHeads(self.gpu, s.q_htd, s.q_lin, tu, n_q, d);
        try apple_ops.encodePermuteTokensHeads(self.gpu, s.k_htd, s.k_lin, tu, n_kv, d);
        try apple_ops.encodePermuteTokensHeads(self.gpu, s.v_htd, s.v_lin, tu, n_kv, d);
        try apple_ops.encodeKvAppend(self.gpu, s.k_htd, s.v_htd, s.k_cache, s.v_cache, n_kv, tu, d, max_seq, used_u);

        const kv_len: u32 = used_u + tu;
        try apple_ops.encodeAttention(self.gpu, s.attn_htd, s.q_htd, s.k_cache, s.v_cache, n_q, n_kv, tu, kv_len, max_seq, d);
        try apple_ops.encodePermuteHeadsTokens(self.gpu, s.attn_lin, s.attn_htd, tu, n_q, d);
        try apple_ops.encodeMatmulNaive(self.gpu, s.ao, s.attn_lin, w.wo, tu, hu, qd);
        try apple_ops.encodeAddRmsNorm(self.gpu, s.x1, s.mlp_n, s.x, s.ao, w.wn, tu, hu, spec.eps);
        try apple_ops.encodeMatmulNaive(self.gpu, s.gate, s.mlp_n, w.wg, tu, inter, hu);
        try apple_ops.encodeMatmulNaive(self.gpu, s.up, s.mlp_n, w.wu, tu, inter, hu);
        try apple_ops.encodeSiluMul(self.gpu, s.hidden_act, s.gate, s.up, tu * inter);
        try apple_ops.encodeMatmulNaive(self.gpu, s.down, s.hidden_act, w.wd, tu, hu, inter);
        try apple_ops.encodeAdd(self.gpu, s.out, s.x1, s.down, tu * hu);

        try self.gpu.batchCommit();

        last_block_path = path_staged;
        last_block_encodes = self.gpu.last_batch_encodes;
        last_block_waits = 1;

        @memcpy(try out.f32s(), s.out.f32s()[0 .. t * hidden]);
        self.inner.cache.used = pos0 + t;
    }
};

test "Metal Stage 6 path matches baseline path and CPU" {
    if (gpu_mod.skipAppleGpuTests()) return error.SkipZigTest;
    defer force_baseline_path = null;

    var gpu = try Gpu.init();
    defer gpu.deinit();
    const gpa = std.testing.allocator;
    const spec = tiny.fixture_spec;

    var cpu_sess = try tiny.Session.init(gpa, spec);
    defer cpu_sess.deinit();
    try cpu_sess.weights.fillFixture();

    force_baseline_path = false;
    var staged_sess = try Session.init(gpa, &gpu, spec);
    defer staged_sess.deinit();
    try staged_sess.inner.weights.copyFrom(cpu_sess.weights);
    staged_sess.markWeightsDirty();

    force_baseline_path = true;
    var baseline_sess = try Session.init(gpa, &gpu, spec);
    defer baseline_sess.deinit();
    try baseline_sess.inner.weights.copyFrom(cpu_sess.weights);
    baseline_sess.markWeightsDirty();

    const tokens: usize = 4;
    var x = try Tensor.alloc(gpa, .f32, &.{ tokens, spec.hidden });
    defer x.deinit();
    try tiny.iotaFill(x, 0.1, 0.05);

    var cpu_out = try Tensor.alloc(gpa, .f32, &.{ tokens, spec.hidden });
    defer cpu_out.deinit();
    var staged_out = try Tensor.alloc(gpa, .f32, &.{ tokens, spec.hidden });
    defer staged_out.deinit();
    var baseline_out = try Tensor.alloc(gpa, .f32, &.{ tokens, spec.hidden });
    defer baseline_out.deinit();

    try cpu_sess.prefill(x, cpu_out);

    force_baseline_path = false;
    try staged_sess.prefill(x, staged_out);
    try std.testing.expectEqualStrings(path_staged, last_block_path);
    try std.testing.expectEqual(@as(u32, 1), last_block_waits);
    try std.testing.expect(last_block_encodes >= 15);

    force_baseline_path = true;
    try baseline_sess.prefill(x, baseline_out);
    try std.testing.expectEqualStrings(path_baseline, last_block_path);
    try std.testing.expectEqual(@as(u32, 15), last_block_waits);

    try compare.expectClose(try cpu_out.f32s(), try staged_out.f32s(), 3e-4, 3e-4);
    try compare.expectClose(try cpu_out.f32s(), try baseline_out.f32s(), 3e-4, 3e-4);
    try compare.expectClose(try staged_out.f32s(), try baseline_out.f32s(), 3e-4, 3e-4);

    cpu_sess.reset();
    staged_sess.reset();
    baseline_sess.reset();

    var step_in = try Tensor.alloc(gpa, .f32, &.{ 1, spec.hidden });
    defer step_in.deinit();
    var cpu_step = try Tensor.alloc(gpa, .f32, &.{ 1, spec.hidden });
    defer cpu_step.deinit();
    var staged_step = try Tensor.alloc(gpa, .f32, &.{ 1, spec.hidden });
    defer staged_step.deinit();
    var baseline_step = try Tensor.alloc(gpa, .f32, &.{ 1, spec.hidden });
    defer baseline_step.deinit();
    const xs = try x.f32s();
    var i: usize = 0;
    while (i < tokens) : (i += 1) {
        @memcpy(try step_in.f32s(), xs[i * spec.hidden ..][0..spec.hidden]);
        try cpu_sess.decode(step_in, cpu_step);

        force_baseline_path = false;
        try staged_sess.decode(step_in, staged_step);
        try std.testing.expectEqualStrings(path_staged, last_block_path);

        force_baseline_path = true;
        try baseline_sess.decode(step_in, baseline_step);
        try std.testing.expectEqualStrings(path_baseline, last_block_path);

        try compare.expectClose(try cpu_step.f32s(), try staged_step.f32s(), 3e-4, 3e-4);
        try compare.expectClose(try cpu_step.f32s(), try baseline_step.f32s(), 3e-4, 3e-4);
        try compare.expectClose(try staged_step.f32s(), try baseline_step.f32s(), 3e-4, 3e-4);
    }
}

test "Metal tiny block matches CPU prefill and decode" {
    if (gpu_mod.skipAppleGpuTests()) return error.SkipZigTest;
    var gpu = try Gpu.init();
    defer gpu.deinit();
    const gpa = std.testing.allocator;
    const spec = tiny.fixture_spec;

    var cpu_sess = try tiny.Session.init(gpa, spec);
    defer cpu_sess.deinit();
    try cpu_sess.weights.fillFixture();

    var metal_sess = try Session.init(gpa, &gpu, spec);
    defer metal_sess.deinit();
    try metal_sess.inner.weights.copyFrom(cpu_sess.weights);
    metal_sess.markWeightsDirty();

    const tokens: usize = 4;
    var x = try Tensor.alloc(gpa, .f32, &.{ tokens, spec.hidden });
    defer x.deinit();
    try tiny.iotaFill(x, 0.1, 0.05);

    var cpu_prefill = try Tensor.alloc(gpa, .f32, &.{ tokens, spec.hidden });
    defer cpu_prefill.deinit();
    var metal_prefill = try Tensor.alloc(gpa, .f32, &.{ tokens, spec.hidden });
    defer metal_prefill.deinit();
    try cpu_sess.prefill(x, cpu_prefill);
    try metal_sess.prefill(x, metal_prefill);
    try compare.expectClose(try cpu_prefill.f32s(), try metal_prefill.f32s(), 3e-4, 3e-4);
    try std.testing.expectEqual(@as(usize, tokens), metal_sess.inner.cache.used);
    if (!useBaselinePath()) {
        try std.testing.expectEqual(@as(u32, 1), last_block_waits);
        try std.testing.expect(last_block_encodes >= 15);
    }

    cpu_sess.reset();
    metal_sess.reset();
    var step_in = try Tensor.alloc(gpa, .f32, &.{ 1, spec.hidden });
    defer step_in.deinit();
    var cpu_step = try Tensor.alloc(gpa, .f32, &.{ 1, spec.hidden });
    defer cpu_step.deinit();
    var metal_step = try Tensor.alloc(gpa, .f32, &.{ 1, spec.hidden });
    defer metal_step.deinit();
    const xs = try x.f32s();
    var i: usize = 0;
    while (i < tokens) : (i += 1) {
        @memcpy(try step_in.f32s(), xs[i * spec.hidden ..][0..spec.hidden]);
        try cpu_sess.decode(step_in, cpu_step);
        try metal_sess.decode(step_in, metal_step);
        try compare.expectClose(try cpu_step.f32s(), try metal_step.f32s(), 3e-4, 3e-4);
    }
}

test "Metal attention rejects kv_len above the thread-local cap" {
    if (gpu_mod.skipAppleGpuTests()) return error.SkipZigTest;
    var gpu = try Gpu.init();
    defer gpu.deinit();
    const gpa = std.testing.allocator;
    var q = try Tensor.alloc(gpa, .f32, &.{ 1, 1, 2 });
    defer q.deinit();
    var k = try Tensor.alloc(gpa, .f32, &.{ 1, 65, 2 });
    defer k.deinit();
    var v = try Tensor.alloc(gpa, .f32, &.{ 1, 65, 2 });
    defer v.deinit();
    var o = try Tensor.alloc(gpa, .f32, &.{ 1, 1, 2 });
    defer o.deinit();
    try std.testing.expectError(error.Unsupported, apple_ops.attention(&gpu, o, q, k, v, 65, 65));
}

test "Metal batch two adds matches sequential" {
    if (gpu_mod.skipAppleGpuTests()) return error.SkipZigTest;
    var gpu = try Gpu.init();
    defer gpu.deinit();
    const n: usize = 16;
    var a = try gpu.allocShared(f32Bytes(n));
    defer a.deinit();
    var b = try gpu.allocShared(f32Bytes(n));
    defer b.deinit();
    var mid = try gpu.allocShared(f32Bytes(n));
    defer mid.deinit();
    var out = try gpu.allocShared(f32Bytes(n));
    defer out.deinit();
    for (a.f32s(), 0..) |*v, i| v.* = @floatFromInt(i);
    for (b.f32s()) |*v| v.* = 1;
    try gpu.batchBegin();
    try apple_ops.encodeAdd(&gpu, mid, a, b, @intCast(n));
    try apple_ops.encodeAdd(&gpu, out, mid, b, @intCast(n));
    try gpu.batchCommit();
    try std.testing.expectEqual(@as(u32, 2), gpu.last_batch_encodes);
    try std.testing.expectEqual(@as(f32, 2), out.f32s()[0]);
    try std.testing.expectEqual(@as(f32, 17), out.f32s()[15]);
}
