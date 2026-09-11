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
const artifact = @import("../../model/artifact.zig");
const gpu_mod = @import("gpu.zig");
const apple_ops = @import("ops.zig");
const bf16 = @import("../../runtime/bf16.zig");
const qwen_quant = @import("../../model/qwen_quant.zig");

const Gpu = gpu_mod.Gpu;
const Buffer = gpu_mod.Buffer;

pub const path_baseline = "baseline_per_op";
pub const path_batched = "batched_resident_kv_fused";
pub const path_bf16 = "batched_resident_kv_bf16";
pub const path_q8 = "batched_resident_kv_q8";

pub var last_qwen_path: []const u8 = "unset";
pub var last_qwen_encodes: u32 = 0;
pub var last_qwen_waits: u32 = 0;

/// Test override. `null` honors `ZYNFER_QWEN_METAL`. `true` → baseline.
pub var force_baseline_path: ?bool = null;
/// Test override for M4 half path.
pub var force_half_path: ?bool = null;
/// Test override for M5 int8 path.
pub var force_q8_path: ?bool = null;

pub fn useBaselinePath() bool {
    if (force_baseline_path) |forced| return forced;
    if (comptime !gpu_mod.have_apple) return true;
    const raw = std.c.getenv("ZYNFER_QWEN_METAL") orelse return false;
    const v = std.mem.span(raw);
    return std.mem.eql(u8, v, "baseline") or std.mem.eql(u8, v, "per-op");
}

/// Stage M5: per-row int8 projections (`ZYNFER_QWEN_METAL=int8|q8`).
pub fn useQ8Path() bool {
    if (force_q8_path) |forced| return forced;
    if (useBaselinePath()) return false;
    if (comptime !gpu_mod.have_apple) return false;
    const raw = std.c.getenv("ZYNFER_QWEN_METAL") orelse return false;
    const v = std.mem.span(raw);
    return std.mem.eql(u8, v, "int8") or std.mem.eql(u8, v, "q8");
}

/// Stage M4: bf16 resident weights + bf16 KV (`ZYNFER_QWEN_METAL=bf16|half|fp16`).
pub fn useHalfPath() bool {
    if (force_half_path) |forced| return forced;
    if (useBaselinePath() or useQ8Path()) return false;
    if (comptime !gpu_mod.have_apple) return false;
    const raw = std.c.getenv("ZYNFER_QWEN_METAL") orelse return false;
    const v = std.mem.span(raw);
    return std.mem.eql(u8, v, "bf16") or std.mem.eql(u8, v, "half") or std.mem.eql(u8, v, "fp16");
}

fn f32Bytes(n: usize) usize {
    return n * @sizeOf(f32);
}

fn bf16Bytes(n: usize) usize {
    return n * 2;
}

fn weightBytes(n: usize, half: bool) usize {
    return if (half) bf16Bytes(n) else f32Bytes(n);
}

fn bufLen(b: Buffer) u64 {
    return @intCast(b.bytes.len);
}

fn q8Len(w: apple_ops.Q8DeviceWeights) u64 {
    return bufLen(w.q) + bufLen(w.scale);
}

pub const MetalResidentBytes = struct {
    weights: u64 = 0,
    kv: u64 = 0,
    scratch: u64 = 0,
};

fn copyTensorToBuf(buf: Buffer, t: Tensor) !void {
    const src = try t.f32s();
    @memcpy(buf.f32s()[0..src.len], src);
}

fn copyTensorToBufBf16(buf: Buffer, t: Tensor) !void {
    const src = try t.f32s();
    bf16.encodeFromF32(buf.bytes[0 .. src.len * 2], src);
}

/// Prefer native f16/bf16 artifact bytes; fall back to narrowing host f32.
fn copyWeightHalf(
    buf: Buffer,
    art: *const artifact.Artifact,
    name: []const u8,
    transpose: bool,
    host: Tensor,
) !void {
    if (try qwen_weights.copyArtifactToBf16(buf.bytes, art, name, transpose)) return;
    try copyTensorToBufBf16(buf, host);
}

fn copyEmbedBf16(
    buf: Buffer,
    art: *const artifact.Artifact,
    weights: *const qwen_weights.Weights,
) !void {
    if (try qwen_weights.copyArtifactToBf16(buf.bytes, art, qwen3.embed_tokens_name, false)) return;
    if (!weights.embed_resident) return error.InvalidDtype;
    try copyTensorToBufBf16(buf, weights.embed);
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

    fn init(gpu: *Gpu, arch: qwen3.Arch, half: bool) !LayerDeviceWeights {
        const h: usize = @intCast(arch.hidden_size);
        const qd: usize = @intCast(arch.qDim());
        const kvd: usize = @intCast(arch.kvDim());
        const inter: usize = @intCast(arch.intermediate_size);
        const d: usize = @intCast(arch.head_dim);
        return .{
            .input_ln = try gpu.allocShared(weightBytes(h, half)),
            .q_norm = try gpu.allocShared(weightBytes(d, half)),
            .k_norm = try gpu.allocShared(weightBytes(d, half)),
            .wq = try gpu.allocShared(weightBytes(h * qd, half)),
            .wk = try gpu.allocShared(weightBytes(h * kvd, half)),
            .wv = try gpu.allocShared(weightBytes(h * kvd, half)),
            .wo = try gpu.allocShared(weightBytes(qd * h, half)),
            .post_attn_ln = try gpu.allocShared(weightBytes(h, half)),
            .wg = try gpu.allocShared(weightBytes(h * inter, half)),
            .wu = try gpu.allocShared(weightBytes(h * inter, half)),
            .wd = try gpu.allocShared(weightBytes(inter * h, half)),
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

    fn uploadFrom(
        self: *LayerDeviceWeights,
        w: qwen_weights.LayerWeights,
        half: bool,
        art: *const artifact.Artifact,
        layer: u32,
    ) !void {
        var name_buf: [96]u8 = undefined;
        if (half) {
            try copyWeightHalf(self.input_ln, art, qwen3.layerInputNormName(layer, &name_buf), false, w.input_ln);
            try copyWeightHalf(self.q_norm, art, qwen3.layerQNormName(layer, &name_buf), false, w.q_norm);
            try copyWeightHalf(self.k_norm, art, qwen3.layerKNormName(layer, &name_buf), false, w.k_norm);
            try copyWeightHalf(self.wq, art, qwen3.layerQProjName(layer, &name_buf), true, w.wq);
            try copyWeightHalf(self.wk, art, qwen3.layerKProjName(layer, &name_buf), true, w.wk);
            try copyWeightHalf(self.wv, art, qwen3.layerVProjName(layer, &name_buf), true, w.wv);
            try copyWeightHalf(self.wo, art, qwen3.layerOProjName(layer, &name_buf), true, w.wo);
            try copyWeightHalf(self.post_attn_ln, art, qwen3.layerPostAttnNormName(layer, &name_buf), false, w.post_attn_ln);
            try copyWeightHalf(self.wg, art, qwen3.layerGateProjName(layer, &name_buf), true, w.wg);
            try copyWeightHalf(self.wu, art, qwen3.layerUpProjName(layer, &name_buf), true, w.wu);
            try copyWeightHalf(self.wd, art, qwen3.layerDownProjName(layer, &name_buf), true, w.wd);
            return;
        }
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

/// Stage M5: f32 norms + per-row int8 projections (HF [out,in] packing).
const LayerQ8Weights = struct {
    input_ln: Buffer,
    q_norm: Buffer,
    k_norm: Buffer,
    post_attn_ln: Buffer,
    wq: apple_ops.Q8DeviceWeights,
    wk: apple_ops.Q8DeviceWeights,
    wv: apple_ops.Q8DeviceWeights,
    wo: apple_ops.Q8DeviceWeights,
    wg: apple_ops.Q8DeviceWeights,
    wu: apple_ops.Q8DeviceWeights,
    wd: apple_ops.Q8DeviceWeights,

    fn packUpload(
        allocator: std.mem.Allocator,
        gpu: *Gpu,
        host: Tensor,
        out_dim: usize,
        in_dim: usize,
    ) !apple_ops.Q8DeviceWeights {
        const packed_w = try qwen_quant.packInOutToQ8(allocator, host, out_dim, in_dim);
        defer allocator.free(packed_w.q);
        defer allocator.free(packed_w.scale);
        return apple_ops.Q8DeviceWeights.upload(gpu, packed_w.q, packed_w.scale, out_dim, in_dim, .per_row);
    }

    /// Pack HF-layout `[out, in]` (tied embed / untied lm_head after load checks).
    fn packUploadOutIn(
        allocator: std.mem.Allocator,
        gpu: *Gpu,
        host: Tensor,
        out_dim: usize,
        in_dim: usize,
    ) !apple_ops.Q8DeviceWeights {
        const packed_w = try qwen_quant.packOutInToQ8(allocator, host, out_dim, in_dim);
        defer allocator.free(packed_w.q);
        defer allocator.free(packed_w.scale);
        return apple_ops.Q8DeviceWeights.upload(gpu, packed_w.q, packed_w.scale, out_dim, in_dim, .per_row);
    }

    /// Artifact i8 is already HF `[out, in]` — same layout Metal Q8 expects.
    fn uploadArtifactQ8(
        gpu: *Gpu,
        art: *const artifact.Artifact,
        name: []const u8,
        out_dim: usize,
        in_dim: usize,
    ) !apple_ops.Q8DeviceWeights {
        const entry = try art.findByName(name);
        const dt = try entry.dtypeTag();
        if (dt != .i8) return error.InvalidDtype;
        if (entry.rank != 2) return error.InvalidShape;
        if (entry.shape[0] != out_dim or entry.shape[1] != in_dim) return error.ShapeMismatch;
        const q_raw = try art.tensorBytesByName(name);
        if (q_raw.len != out_dim * in_dim) return error.ShapeMismatch;

        var scale_name_buf: [128]u8 = undefined;
        const scale_name = std.fmt.bufPrint(&scale_name_buf, "{s}.qscale", .{name}) catch return error.InvalidName;
        const scale_raw = try art.tensorBytesByName(scale_name);
        if (scale_raw.len != out_dim * @sizeOf(f32)) return error.ShapeMismatch;
        if (@intFromPtr(scale_raw.ptr) % @alignOf(f32) != 0) return error.BadAlignment;
        const scales: []const f32 = @as([*]const f32, @ptrCast(@alignCast(scale_raw.ptr)))[0..out_dim];
        const q = std.mem.bytesAsSlice(i8, q_raw);
        return apple_ops.Q8DeviceWeights.upload(gpu, q, scales, out_dim, in_dim, .per_row);
    }

    fn tryUploadArtifactQ8(
        gpu: *Gpu,
        art: *const artifact.Artifact,
        name: []const u8,
        out_dim: usize,
        in_dim: usize,
    ) !?apple_ops.Q8DeviceWeights {
        const entry = art.findByName(name) catch return null;
        const dt = entry.dtypeTag() catch return null;
        if (dt != .i8) return null;
        return try uploadArtifactQ8(gpu, art, name, out_dim, in_dim);
    }

    fn init(
        allocator: std.mem.Allocator,
        gpu: *Gpu,
        arch: qwen3.Arch,
        w: qwen_weights.LayerWeights,
        art: *const artifact.Artifact,
        layer: u32,
    ) !LayerQ8Weights {
        const h: usize = @intCast(arch.hidden_size);
        const qd: usize = @intCast(arch.qDim());
        const kvd: usize = @intCast(arch.kvDim());
        const inter: usize = @intCast(arch.intermediate_size);
        const d: usize = @intCast(arch.head_dim);

        var input_ln = try gpu.allocShared(f32Bytes(h));
        errdefer input_ln.deinit();
        var q_norm = try gpu.allocShared(f32Bytes(d));
        errdefer q_norm.deinit();
        var k_norm = try gpu.allocShared(f32Bytes(d));
        errdefer k_norm.deinit();
        var post_attn_ln = try gpu.allocShared(f32Bytes(h));
        errdefer post_attn_ln.deinit();
        try copyTensorToBuf(input_ln, w.input_ln);
        try copyTensorToBuf(q_norm, w.q_norm);
        try copyTensorToBuf(k_norm, w.k_norm);
        try copyTensorToBuf(post_attn_ln, w.post_attn_ln);

        var name_buf: [96]u8 = undefined;
        const q_name = qwen3.layerQProjName(layer, &name_buf);
        const use_artifact = blk: {
            const entry = art.findByName(q_name) catch break :blk false;
            const dt = entry.dtypeTag() catch break :blk false;
            break :blk dt == .i8;
        };

        var wq: apple_ops.Q8DeviceWeights = undefined;
        var wk: apple_ops.Q8DeviceWeights = undefined;
        var wv: apple_ops.Q8DeviceWeights = undefined;
        var wo: apple_ops.Q8DeviceWeights = undefined;
        var wg: apple_ops.Q8DeviceWeights = undefined;
        var wu: apple_ops.Q8DeviceWeights = undefined;
        var wd: apple_ops.Q8DeviceWeights = undefined;

        if (use_artifact) {
            wq = try uploadArtifactQ8(gpu, art, q_name, qd, h);
            errdefer wq.deinit();
            wk = try uploadArtifactQ8(gpu, art, qwen3.layerKProjName(layer, &name_buf), kvd, h);
            errdefer wk.deinit();
            wv = try uploadArtifactQ8(gpu, art, qwen3.layerVProjName(layer, &name_buf), kvd, h);
            errdefer wv.deinit();
            wo = try uploadArtifactQ8(gpu, art, qwen3.layerOProjName(layer, &name_buf), h, qd);
            errdefer wo.deinit();
            wg = try uploadArtifactQ8(gpu, art, qwen3.layerGateProjName(layer, &name_buf), inter, h);
            errdefer wg.deinit();
            wu = try uploadArtifactQ8(gpu, art, qwen3.layerUpProjName(layer, &name_buf), inter, h);
            errdefer wu.deinit();
            wd = try uploadArtifactQ8(gpu, art, qwen3.layerDownProjName(layer, &name_buf), h, inter);
            errdefer wd.deinit();
        } else {
            if (!w.projs_resident) return error.InvalidDtype;
            wq = try packUpload(allocator, gpu, w.wq, qd, h);
            errdefer wq.deinit();
            wk = try packUpload(allocator, gpu, w.wk, kvd, h);
            errdefer wk.deinit();
            wv = try packUpload(allocator, gpu, w.wv, kvd, h);
            errdefer wv.deinit();
            wo = try packUpload(allocator, gpu, w.wo, h, qd);
            errdefer wo.deinit();
            wg = try packUpload(allocator, gpu, w.wg, inter, h);
            errdefer wg.deinit();
            wu = try packUpload(allocator, gpu, w.wu, inter, h);
            errdefer wu.deinit();
            wd = try packUpload(allocator, gpu, w.wd, h, inter);
            errdefer wd.deinit();
        }

        return .{
            .input_ln = input_ln,
            .q_norm = q_norm,
            .k_norm = k_norm,
            .post_attn_ln = post_attn_ln,
            .wq = wq,
            .wk = wk,
            .wv = wv,
            .wo = wo,
            .wg = wg,
            .wu = wu,
            .wd = wd,
        };
    }

    fn deinit(self: *LayerQ8Weights) void {
        self.input_ln.deinit();
        self.q_norm.deinit();
        self.k_norm.deinit();
        self.post_attn_ln.deinit();
        self.wq.deinit();
        self.wk.deinit();
        self.wv.deinit();
        self.wo.deinit();
        self.wg.deinit();
        self.wu.deinit();
        self.wd.deinit();
        self.* = undefined;
    }
};

const LayerKv = struct {
    k_cache: Buffer,
    v_cache: Buffer,
    max_seq: usize,
    used: usize = 0,

    fn init(gpu: *Gpu, arch: qwen3.Arch, max_seq: usize, half: bool) !LayerKv {
        const n_kv: usize = @intCast(arch.num_key_value_heads);
        const d: usize = @intCast(arch.head_dim);
        const kb = weightBytes(n_kv * max_seq * d, half);
        return .{
            .k_cache = try gpu.allocShared(kb),
            .v_cache = try gpu.allocShared(kb),
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

    fn truncateTo(self: *LayerKv, n: usize) !void {
        if (n > self.used) return error.InvalidShape;
        self.used = n;
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
    /// Scratch.embed is bf16 (half path or Apple Q8 table).
    embed_bf16: bool,

    fn init(
        gpu: *Gpu,
        arch: qwen3.Arch,
        max_seq: usize,
        weights: *const qwen_weights.Weights,
        /// bf16 resident layer weights / final_norm / lm_head (non-Q8 half path).
        weight_half: bool,
        /// bf16 embedding table (half path or Q8).
        embed_bf16: bool,
        art: *const artifact.Artifact,
    ) !Scratch {
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
            .embed = try gpu.allocShared(weightBytes(vocab * h, embed_bf16)),
            .final_norm = try gpu.allocShared(weightBytes(h, weight_half)),
            .lm_head = undefined,
            .lm_head_tied = weights.lm_head_tied,
            .embed_bf16 = embed_bf16,
        };
        errdefer s.deinitPartialBeforeLmHead();
        if (embed_bf16) {
            try copyEmbedBf16(s.embed, art, weights);
        } else {
            if (!weights.embed_resident) return error.InvalidDtype;
            try copyTensorToBuf(s.embed, weights.embed);
        }
        if (weight_half) {
            try copyWeightHalf(s.final_norm, art, qwen3.final_norm_name, false, weights.final_norm);
        } else {
            try copyTensorToBuf(s.final_norm, weights.final_norm);
        }
        if (weights.lm_head_tied) {
            s.lm_head = s.embed;
        } else {
            s.lm_head = try gpu.allocShared(weightBytes(vocab * h, weight_half));
            if (weight_half) {
                if (!weights.embed_resident and weights.lm_head_tied) unreachable;
                if (weights.embed_resident) {
                    try copyWeightHalf(s.lm_head, art, qwen3.lm_head_name, true, weights.lm_head);
                } else {
                    if (!try qwen_weights.copyArtifactToBf16(s.lm_head.bytes, art, qwen3.lm_head_name, true))
                        return error.InvalidDtype;
                }
            } else {
                if (!weights.embed_resident) return error.InvalidDtype;
                try copyTensorToBuf(s.lm_head, weights.lm_head);
            }
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
    half_mode: bool,
    q8_mode: bool,
    /// bf16 K/V caches (half path and Q8 path).
    kv_bf16: bool,
    layers_w: []LayerDeviceWeights,
    layers_q8: []LayerQ8Weights,
    layers_kv: []LayerKv,
    scratch: Scratch,
    lm_head_q8: ?apple_ops.Q8DeviceWeights,
    allocator: std.mem.Allocator,

    pub fn init(
        allocator: std.mem.Allocator,
        gpu: *Gpu,
        arch: qwen3.Arch,
        max_seq: usize,
        weights: *const qwen_weights.Weights,
        half: bool,
        q8: bool,
        art: *const artifact.Artifact,
    ) !MetalStack {
        if (max_seq == 0 or max_seq > arch.max_position_embeddings) return error.InvalidShape;
        if (max_seq > apple_ops.max_attention_kv) return error.Unsupported;
        // Weight-half and Q8 projections are mutually exclusive schedules.
        if (half and q8) return error.Unsupported;

        const n_layers: usize = @intCast(arch.num_layers);
        var layers_w: []LayerDeviceWeights = &.{};
        var layers_q8: []LayerQ8Weights = &.{};
        var lm_head_q8: ?apple_ops.Q8DeviceWeights = null;
        const weight_half = half and !q8;
        const embed_bf16 = half or q8;
        // Q8 keeps i8 weights + f32 norms; KV is bf16 to cut decode traffic.
        const kv_bf16 = half or q8;

        if (q8) {
            layers_q8 = try allocator.alloc(LayerQ8Weights, n_layers);
            errdefer allocator.free(layers_q8);
            @memset(layers_q8, undefined);
            var i: usize = 0;
            errdefer while (i > 0) {
                i -= 1;
                layers_q8[i].deinit();
            };
            while (i < n_layers) : (i += 1) {
                layers_q8[i] = try LayerQ8Weights.init(
                    allocator,
                    gpu,
                    arch,
                    weights.layers[i],
                    art,
                    @intCast(i),
                );
            }
            const vocab: usize = @intCast(arch.vocab_size);
            const h: usize = @intCast(arch.hidden_size);
            // Prefer on-disk i8 lm_head. Tied embed falls back to bf16 matvec on
            // Scratch.embed (no host f32 twin, no init-time pack of vocab×hidden).
            lm_head_q8 = if (try LayerQ8Weights.tryUploadArtifactQ8(gpu, art, qwen3.lm_head_name, vocab, h)) |lh|
                lh
            else if (weights.embed_resident and !weights.lm_head_tied)
                try LayerQ8Weights.packUploadOutIn(allocator, gpu, weights.lm_head, vocab, h)
            else
                null;
            errdefer if (lm_head_q8) |*lh| lh.deinit();
        } else {
            layers_w = try allocator.alloc(LayerDeviceWeights, n_layers);
            errdefer allocator.free(layers_w);
            @memset(layers_w, undefined);
            var i: usize = 0;
            errdefer while (i > 0) {
                i -= 1;
                layers_w[i].deinit();
            };
            while (i < n_layers) : (i += 1) {
                layers_w[i] = try LayerDeviceWeights.init(gpu, arch, weight_half);
                try layers_w[i].uploadFrom(weights.layers[i], weight_half, art, @intCast(i));
            }
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
            layers_kv[j] = try LayerKv.init(gpu, arch, max_seq, kv_bf16);
        }

        const scratch = try Scratch.init(gpu, arch, max_seq, weights, weight_half, embed_bf16, art);
        return .{
            .gpu = gpu,
            .arch = arch,
            .max_seq = max_seq,
            .half_mode = weight_half,
            .q8_mode = q8,
            .kv_bf16 = kv_bf16,
            .layers_w = layers_w,
            .layers_q8 = layers_q8,
            .layers_kv = layers_kv,
            .scratch = scratch,
            .lm_head_q8 = lm_head_q8,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MetalStack) void {
        if (self.lm_head_q8) |*lh| lh.deinit();
        self.scratch.deinit();
        for (self.layers_kv) |*kv| kv.deinit();
        self.allocator.free(self.layers_kv);
        for (self.layers_q8) |*w| w.deinit();
        if (self.layers_q8.len != 0) self.allocator.free(self.layers_q8);
        for (self.layers_w) |*w| w.deinit();
        if (self.layers_w.len != 0) self.allocator.free(self.layers_w);
        self.* = undefined;
    }

    pub fn reset(self: *MetalStack) void {
        for (self.layers_kv) |*kv| kv.reset();
    }

    /// Shrink live KV length to `n` on every layer (`n <= used`).
    pub fn truncateTo(self: *MetalStack, n: usize) !void {
        for (self.layers_kv) |*kv| try kv.truncateTo(n);
    }

    pub fn kvBytesCapacity(self: *const MetalStack) u64 {
        var n: u64 = 0;
        for (self.layers_kv) |kv| n += @as(u64, @intCast(kv.k_cache.bytes.len + kv.v_cache.bytes.len));
        return n;
    }

    pub fn kvBytesUsed(self: *const MetalStack, used: usize) u64 {
        if (self.max_seq == 0 or used == 0) return 0;
        const cap = self.kvBytesCapacity();
        return cap * @as(u64, @intCast(used)) / @as(u64, @intCast(self.max_seq));
    }

    pub fn residentBytes(self: *const MetalStack) MetalResidentBytes {
        var weights: u64 = 0;
        for (self.layers_w) |w| {
            weights += bufLen(w.input_ln) + bufLen(w.q_norm) + bufLen(w.k_norm) + bufLen(w.post_attn_ln);
            weights += bufLen(w.wq) + bufLen(w.wk) + bufLen(w.wv) + bufLen(w.wo);
            weights += bufLen(w.wg) + bufLen(w.wu) + bufLen(w.wd);
        }
        for (self.layers_q8) |w| {
            weights += bufLen(w.input_ln) + bufLen(w.q_norm) + bufLen(w.k_norm) + bufLen(w.post_attn_ln);
            weights += q8Len(w.wq) + q8Len(w.wk) + q8Len(w.wv) + q8Len(w.wo);
            weights += q8Len(w.wg) + q8Len(w.wu) + q8Len(w.wd);
        }
        if (self.lm_head_q8) |lh| weights += q8Len(lh);

        const s = self.scratch;
        var scratch: u64 = 0;
        scratch += bufLen(s.act_a) + bufLen(s.act_b) + bufLen(s.xn);
        scratch += bufLen(s.q_lin) + bufLen(s.k_lin) + bufLen(s.v_lin);
        scratch += bufLen(s.q_htd) + bufLen(s.k_htd) + bufLen(s.v_htd) + bufLen(s.attn_htd);
        scratch += bufLen(s.attn_lin) + bufLen(s.ao) + bufLen(s.x1) + bufLen(s.mlp_n);
        scratch += bufLen(s.gate) + bufLen(s.up) + bufLen(s.hidden_act) + bufLen(s.down);
        scratch += bufLen(s.normed) + bufLen(s.logits) + bufLen(s.scores);
        scratch += bufLen(s.embed) + bufLen(s.final_norm);
        if (!s.lm_head_tied) scratch += bufLen(s.lm_head);
        // Embed/final_norm/(lm_head) are weight residents sitting in scratch struct.
        weights += bufLen(s.embed) + bufLen(s.final_norm);
        if (!s.lm_head_tied) weights += bufLen(s.lm_head);
        scratch -= bufLen(s.embed) + bufLen(s.final_norm);
        if (!s.lm_head_tied) scratch -= bufLen(s.lm_head);

        return .{
            .weights = weights,
            .kv = self.kvBytesCapacity(),
            .scratch = scratch,
        };
    }

    /// Prefill or decode from token ids (gather from Scratch.embed → act_a).
    pub fn forwardLastLogitsFromTokens(self: *MetalStack, token_ids: []const u32, logits_out: []f32) !void {
        const arch = self.arch;
        const h: usize = @intCast(arch.hidden_size);
        const vocab: usize = @intCast(arch.vocab_size);
        const t = token_ids.len;
        if (t == 0 or t > self.max_seq) return error.InvalidShape;
        if (logits_out.len != vocab) return error.ShapeMismatch;

        const out = self.scratch.act_a.f32s()[0 .. t * h];
        if (self.scratch.embed_bf16) {
            const tab = self.scratch.embed.bytes;
            if (tab.len < vocab * h * 2) return error.ShapeMismatch;
            for (token_ids, 0..) |tid, i| {
                if (tid >= vocab) return error.InvalidShape;
                const src = @as(usize, tid) * h * 2;
                bf16.decodeIntoF32(out[i * h ..][0..h], tab[src..][0 .. h * 2]);
            }
        } else {
            const tab = self.scratch.embed.f32s();
            if (tab.len < vocab * h) return error.ShapeMismatch;
            for (token_ids, 0..) |tid, i| {
                if (tid >= vocab) return error.InvalidShape;
                const src = @as(usize, tid) * h;
                @memcpy(out[i * h ..][0..h], tab[src..][0..h]);
            }
        }

        try self.forwardLastLogitsAct(t, logits_out);
    }

    /// Prefill or decode: `x_host` is `[t, hidden]` embeddings; writes last-token logits.
    pub fn forwardLastLogits(self: *MetalStack, x_host: Tensor, logits_out: []f32) !void {
        const arch = self.arch;
        const h: usize = @intCast(arch.hidden_size);
        if (x_host.rank != 2 or x_host.shape[1] != h) return error.ShapeMismatch;
        const t = x_host.shape[0];
        if (t == 0 or t > self.max_seq) return error.InvalidShape;
        if (logits_out.len != arch.vocab_size) return error.ShapeMismatch;
        @memcpy(self.scratch.act_a.f32s()[0 .. t * h], try x_host.f32s());
        try self.forwardLastLogitsAct(t, logits_out);
    }

    fn forwardLastLogitsAct(self: *MetalStack, t: usize, logits_out: []f32) !void {
        const arch = self.arch;
        const h: usize = @intCast(arch.hidden_size);

        const pos0 = self.layers_kv[0].used;
        if (pos0 + t > self.max_seq) return error.InvalidShape;
        if (pos0 + t > apple_ops.max_attention_kv) return error.Unsupported;

        for (self.layers_kv) |kv| {
            if (kv.used != pos0) return error.InvalidShape;
        }

        try self.gpu.batchBegin();
        errdefer self.gpu.batchAbort();

        var in_buf = self.scratch.act_a;
        var out_buf = self.scratch.act_b;
        var layer: u32 = 0;
        while (layer < arch.num_layers) : (layer += 1) {
            if (self.q8_mode) {
                try encodeLayerQ8(
                    self.gpu,
                    &self.layers_q8[layer],
                    &self.layers_kv[layer],
                    &self.scratch,
                    arch,
                    in_buf,
                    out_buf,
                    t,
                    pos0,
                    self.kv_bf16,
                );
            } else {
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
                    self.half_mode,
                );
            }
            const tmp = in_buf;
            in_buf = out_buf;
            out_buf = tmp;
        }

        try self.gpu.batchCommit();

        last_qwen_path = if (self.q8_mode) path_q8 else if (self.half_mode) path_bf16 else path_batched;
        last_qwen_encodes = self.gpu.last_batch_encodes;
        last_qwen_waits = 1;

        for (self.layers_kv) |*kv| kv.used = pos0 + t;

        const last_row_offset = (t - 1) * h;
        const last_hidden = in_buf.f32s()[last_row_offset..][0..h];
        @memcpy(self.scratch.xn.f32s()[0..h], last_hidden);

        try self.gpu.batchBegin();
        errdefer self.gpu.batchAbort();
        const hu: u32 = @intCast(h);
        const vu: u32 = @intCast(arch.vocab_size);
        if (self.q8_mode) {
            try apple_ops.encodeRmsNorm(self.gpu, self.scratch.normed, self.scratch.xn, self.scratch.final_norm, 1, hu, arch.rms_norm_eps);
            if (self.lm_head_q8) |lh| {
                try apple_ops.encodeMatvecQ8(self.gpu, self.scratch.logits, lh.q, lh.scale, self.scratch.normed, vu, hu);
            } else {
                // Tied word embeddings: Scratch.embed is bf16 `[vocab, hidden]`.
                if (!self.scratch.embed_bf16) return error.InvalidDtype;
                try apple_ops.encodeMatvecBf16(self.gpu, self.scratch.logits, self.scratch.embed, self.scratch.normed, vu, hu);
            }
        } else if (self.half_mode) {
            try apple_ops.encodeRmsNormBf16(self.gpu, self.scratch.normed, self.scratch.xn, self.scratch.final_norm, 1, hu, arch.rms_norm_eps);
            try apple_ops.encodeMatvecBf16(self.gpu, self.scratch.logits, self.scratch.lm_head, self.scratch.normed, vu, hu);
        } else {
            try apple_ops.encodeRmsNorm(self.gpu, self.scratch.normed, self.scratch.xn, self.scratch.final_norm, 1, hu, arch.rms_norm_eps);
            try apple_ops.encodeMatvec(self.gpu, self.scratch.logits, self.scratch.lm_head, self.scratch.normed, vu, hu);
        }
        try self.gpu.batchCommit();
        last_qwen_encodes += self.gpu.last_batch_encodes;
        last_qwen_waits += 1;

        @memcpy(logits_out, self.scratch.logits.f32s()[0..logits_out.len]);
    }
};

fn encodeLayerQ8(
    gpu: *Gpu,
    w: *const LayerQ8Weights,
    kv: *LayerKv,
    s: *Scratch,
    arch: qwen3.Arch,
    x: Buffer,
    out: Buffer,
    t: usize,
    pos0: usize,
    kv_bf16: bool,
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
    try apple_ops.encodeMatmulAq8(gpu, s.q_lin, s.xn, w.wq.q, w.wq.scale, tu, qd, hu);
    try apple_ops.encodeMatmulAq8(gpu, s.k_lin, s.xn, w.wk.q, w.wk.scale, tu, kvd, hu);
    try apple_ops.encodeMatmulAq8(gpu, s.v_lin, s.xn, w.wv.q, w.wv.scale, tu, kvd, hu);

    try apple_ops.encodeRmsNorm(gpu, s.q_lin, s.q_lin, w.q_norm, tu * n_q, d, eps);
    try apple_ops.encodeRmsNorm(gpu, s.k_lin, s.k_lin, w.k_norm, tu * n_kv, d, eps);

    try apple_ops.encodeRope(gpu, s.q_lin, tu, n_q, d, used_u, theta);
    try apple_ops.encodeRope(gpu, s.k_lin, tu, n_kv, d, used_u, theta);

    try apple_ops.encodePermuteTokensHeads(gpu, s.q_htd, s.q_lin, tu, n_q, d);
    try apple_ops.encodePermuteTokensHeads(gpu, s.k_htd, s.k_lin, tu, n_kv, d);
    try apple_ops.encodePermuteTokensHeads(gpu, s.v_htd, s.v_lin, tu, n_kv, d);
    if (kv_bf16) {
        try apple_ops.encodeKvAppendBf16(gpu, s.k_htd, s.v_htd, kv.k_cache, kv.v_cache, n_kv, tu, d, max_seq, used_u);
    } else {
        try apple_ops.encodeKvAppend(gpu, s.k_htd, s.v_htd, kv.k_cache, kv.v_cache, n_kv, tu, d, max_seq, used_u);
    }

    const kv_len: u32 = used_u + tu;
    if (kv_bf16) {
        try apple_ops.encodeAttentionBf16Kv(
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
    } else {
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
    }
    try apple_ops.encodePermuteHeadsTokens(gpu, s.attn_lin, s.attn_htd, tu, n_q, d);
    try apple_ops.encodeMatmulAq8(gpu, s.ao, s.attn_lin, w.wo.q, w.wo.scale, tu, hu, qd);

    try apple_ops.encodeAddRmsNorm(gpu, s.x1, s.mlp_n, x, s.ao, w.post_attn_ln, tu, hu, eps);

    try apple_ops.encodeMatmulAq8(gpu, s.gate, s.mlp_n, w.wg.q, w.wg.scale, tu, inter, hu);
    try apple_ops.encodeMatmulAq8(gpu, s.up, s.mlp_n, w.wu.q, w.wu.scale, tu, inter, hu);
    try apple_ops.encodeSiluMul(gpu, s.hidden_act, s.gate, s.up, tu * inter);
    try apple_ops.encodeMatmulAq8(gpu, s.down, s.hidden_act, w.wd.q, w.wd.scale, tu, hu, inter);
    try apple_ops.encodeAdd(gpu, out, s.x1, s.down, tu * hu);
}

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
    half: bool,
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

    if (half) {
        try apple_ops.encodeRmsNormBf16(gpu, s.xn, x, w.input_ln, tu, hu, eps);
        try apple_ops.encodeMatmulNaiveBf16(gpu, s.q_lin, s.xn, w.wq, tu, qd, hu);
        try apple_ops.encodeMatmulNaiveBf16(gpu, s.k_lin, s.xn, w.wk, tu, kvd, hu);
        try apple_ops.encodeMatmulNaiveBf16(gpu, s.v_lin, s.xn, w.wv, tu, kvd, hu);
        try apple_ops.encodeRmsNormBf16(gpu, s.q_lin, s.q_lin, w.q_norm, tu * n_q, d, eps);
        try apple_ops.encodeRmsNormBf16(gpu, s.k_lin, s.k_lin, w.k_norm, tu * n_kv, d, eps);
    } else {
        try apple_ops.encodeRmsNorm(gpu, s.xn, x, w.input_ln, tu, hu, eps);
        try apple_ops.encodeMatmulNaive(gpu, s.q_lin, s.xn, w.wq, tu, qd, hu);
        try apple_ops.encodeMatmulNaive(gpu, s.k_lin, s.xn, w.wk, tu, kvd, hu);
        try apple_ops.encodeMatmulNaive(gpu, s.v_lin, s.xn, w.wv, tu, kvd, hu);
        try apple_ops.encodeRmsNorm(gpu, s.q_lin, s.q_lin, w.q_norm, tu * n_q, d, eps);
        try apple_ops.encodeRmsNorm(gpu, s.k_lin, s.k_lin, w.k_norm, tu * n_kv, d, eps);
    }

    try apple_ops.encodeRope(gpu, s.q_lin, tu, n_q, d, used_u, theta);
    try apple_ops.encodeRope(gpu, s.k_lin, tu, n_kv, d, used_u, theta);

    try apple_ops.encodePermuteTokensHeads(gpu, s.q_htd, s.q_lin, tu, n_q, d);
    try apple_ops.encodePermuteTokensHeads(gpu, s.k_htd, s.k_lin, tu, n_kv, d);
    try apple_ops.encodePermuteTokensHeads(gpu, s.v_htd, s.v_lin, tu, n_kv, d);
    if (half) {
        try apple_ops.encodeKvAppendBf16(gpu, s.k_htd, s.v_htd, kv.k_cache, kv.v_cache, n_kv, tu, d, max_seq, used_u);
    } else {
        try apple_ops.encodeKvAppend(gpu, s.k_htd, s.v_htd, kv.k_cache, kv.v_cache, n_kv, tu, d, max_seq, used_u);
    }

    const kv_len: u32 = used_u + tu;
    if (half) {
        try apple_ops.encodeAttentionBf16Kv(
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
    } else {
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
    }
    try apple_ops.encodePermuteHeadsTokens(gpu, s.attn_lin, s.attn_htd, tu, n_q, d);
    if (half) {
        try apple_ops.encodeMatmulNaiveBf16(gpu, s.ao, s.attn_lin, w.wo, tu, hu, qd);
        try apple_ops.encodeAddRmsNormBf16(gpu, s.x1, s.mlp_n, x, s.ao, w.post_attn_ln, tu, hu, eps);
        try apple_ops.encodeMatmulNaiveBf16(gpu, s.gate, s.mlp_n, w.wg, tu, inter, hu);
        try apple_ops.encodeMatmulNaiveBf16(gpu, s.up, s.mlp_n, w.wu, tu, inter, hu);
    } else {
        try apple_ops.encodeMatmulNaive(gpu, s.ao, s.attn_lin, w.wo, tu, hu, qd);
        try apple_ops.encodeAddRmsNorm(gpu, s.x1, s.mlp_n, x, s.ao, w.post_attn_ln, tu, hu, eps);
        try apple_ops.encodeMatmulNaive(gpu, s.gate, s.mlp_n, w.wg, tu, inter, hu);
        try apple_ops.encodeMatmulNaive(gpu, s.up, s.mlp_n, w.wu, tu, inter, hu);
    }
    try apple_ops.encodeSiluMul(gpu, s.hidden_act, s.gate, s.up, tu * inter);
    if (half) {
        try apple_ops.encodeMatmulNaiveBf16(gpu, s.down, s.hidden_act, w.wd, tu, hu, inter);
    } else {
        try apple_ops.encodeMatmulNaive(gpu, s.down, s.hidden_act, w.wd, tu, hu, inter);
    }
    try apple_ops.encodeAdd(gpu, out, s.x1, s.down, tu * hu);
}
