//! Load Qwen3 weights from a validated `.zynfer` artifact into f32 tensors.
//!
//! Hugging Face stores linear weights as `[out_features, in_features]`. Zynfer
//! matmul uses `[in, out]`, so projections are transposed on load.

const std = @import("std");
const artifact = @import("artifact.zig");
const qwen3 = @import("qwen3.zig");
const bf16 = @import("../runtime/bf16.zig");
const float16 = @import("../runtime/float16.zig");
const qwen_quant = @import("qwen_quant.zig");
const Tensor = @import("../runtime/tensor.zig").Tensor;
const DType = @import("../runtime/dtype.zig").DType;
pub const Error = artifact.Error || @import("../runtime/tensor.zig").TensorError || qwen_quant.Error;

pub const LayerWeights = struct {
    input_ln: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wo: Tensor,
    post_attn_ln: Tensor,
    wg: Tensor,
    wu: Tensor,
    wd: Tensor,

    pub fn deinit(self: *LayerWeights) void {
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
};

pub const Weights = struct {
    arch: qwen3.Arch,
    embed: Tensor,
    final_norm: Tensor,
    lm_head: Tensor,
    lm_head_tied: bool,
    layers: []LayerWeights,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Weights) void {
        self.embed.deinit();
        self.final_norm.deinit();
        if (!self.lm_head_tied) self.lm_head.deinit();
        for (self.layers) |*layer| layer.deinit();
        self.allocator.free(self.layers);
        self.* = undefined;
    }

    pub fn load(allocator: std.mem.Allocator, art: *const artifact.Artifact, arch: qwen3.Arch) Error!Weights {
        const meta = try art.meta.toArch();
        if (meta.hidden_size != arch.hidden_size or meta.num_layers != arch.num_layers) {
            return error.ShapeMismatch;
        }
        var embed = try loadNamed(allocator, art, qwen3.embed_tokens_name, false);
        errdefer embed.deinit();
        try expectShape2(embed, arch.vocab_size, arch.hidden_size);

        var final_norm = try loadNamed(allocator, art, qwen3.final_norm_name, false);
        errdefer final_norm.deinit();
        try expectShape1(final_norm, arch.hidden_size);

        const lm_head_tied = arch.tie_word_embeddings;
        var lm_head = embed;
        if (!lm_head_tied) {
            lm_head = try loadNamed(allocator, art, qwen3.lm_head_name, true);
            errdefer lm_head.deinit();
            try expectShape2(lm_head, arch.vocab_size, arch.hidden_size);
        }

        const layers = try allocator.alloc(LayerWeights, arch.num_layers);
        errdefer allocator.free(layers);
        @memset(layers, undefined);

        var layer_buf: [96]u8 = undefined;
        var layer: u32 = 0;
        while (layer < arch.num_layers) : (layer += 1) {
            layers[layer] = try loadLayer(allocator, art, arch, layer, &layer_buf);
            errdefer layers[layer].deinit();
        }

        return .{
            .arch = arch,
            .embed = embed,
            .final_norm = final_norm,
            .lm_head = lm_head,
            .lm_head_tied = lm_head_tied,
            .layers = layers,
            .allocator = allocator,
        };
    }
};

fn loadLayer(
    allocator: std.mem.Allocator,
    art: *const artifact.Artifact,
    arch: qwen3.Arch,
    layer: u32,
    buf: *[96]u8,
) Error!LayerWeights {
    const hidden = arch.hidden_size;
    const qd = arch.qDim();
    const kvd = arch.kvDim();
    const inter = arch.intermediate_size;
    const hd = arch.head_dim;

    var input_ln = try loadNamed(allocator, art, qwen3.layerInputNormName(layer, buf), false);
    errdefer input_ln.deinit();
    try expectShape1(input_ln, hidden);

    var q_norm = try loadNamed(allocator, art, qwen3.layerQNormName(layer, buf), false);
    errdefer q_norm.deinit();
    try expectShape1(q_norm, hd);

    var k_norm = try loadNamed(allocator, art, qwen3.layerKNormName(layer, buf), false);
    errdefer k_norm.deinit();
    try expectShape1(k_norm, hd);

    var wq = try loadNamed(allocator, art, qwen3.layerQProjName(layer, buf), true);
    errdefer wq.deinit();
    try expectShape2(wq, hidden, qd);

    var wk = try loadNamed(allocator, art, qwen3.layerKProjName(layer, buf), true);
    errdefer wk.deinit();
    try expectShape2(wk, hidden, kvd);

    var wv = try loadNamed(allocator, art, qwen3.layerVProjName(layer, buf), true);
    errdefer wv.deinit();
    try expectShape2(wv, hidden, kvd);

    var wo = try loadNamed(allocator, art, qwen3.layerOProjName(layer, buf), true);
    errdefer wo.deinit();
    try expectShape2(wo, qd, hidden);

    var post_attn_ln = try loadNamed(allocator, art, qwen3.layerPostAttnNormName(layer, buf), false);
    errdefer post_attn_ln.deinit();
    try expectShape1(post_attn_ln, hidden);

    var wg = try loadNamed(allocator, art, qwen3.layerGateProjName(layer, buf), true);
    errdefer wg.deinit();
    try expectShape2(wg, hidden, inter);

    var wu = try loadNamed(allocator, art, qwen3.layerUpProjName(layer, buf), true);
    errdefer wu.deinit();
    try expectShape2(wu, hidden, inter);

    var wd = try loadNamed(allocator, art, qwen3.layerDownProjName(layer, buf), true);
    errdefer wd.deinit();
    try expectShape2(wd, inter, hidden);

    return .{
        .input_ln = input_ln,
        .q_norm = q_norm,
        .k_norm = k_norm,
        .wq = wq,
        .wk = wk,
        .wv = wv,
        .wo = wo,
        .post_attn_ln = post_attn_ln,
        .wg = wg,
        .wu = wu,
        .wd = wd,
    };
}

fn loadNamed(
    allocator: std.mem.Allocator,
    art: *const artifact.Artifact,
    name: []const u8,
    transpose: bool,
) Error!Tensor {
    const entry = try art.findByName(name);
    const dt = try entry.dtypeTag();
    if (dt == .i8) {
        return loadI8Dequant(allocator, art, entry, name, transpose);
    }
    const raw = try art.tensorBytesByName(name);
    return loadTensorF32(allocator, entry, raw, transpose);
}

/// Stage M8: on-disk int8 projections (HF `[out,in]` + `{name}.qscale`) → host f32.
fn loadI8Dequant(
    allocator: std.mem.Allocator,
    art: *const artifact.Artifact,
    entry: *const artifact.TensorEntry,
    name: []const u8,
    transpose: bool,
) Error!Tensor {
    if (entry.rank != 2) return error.InvalidShape;
    const out_dim: usize = entry.shape[0];
    const in_dim: usize = entry.shape[1];
    const q_raw = try art.tensorBytesByName(name);
    if (q_raw.len != out_dim * in_dim) return error.ShapeMismatch;

    var scale_buf: [128]u8 = undefined;
    const scale_name = std.fmt.bufPrint(&scale_buf, "{s}.qscale", .{name}) catch return error.InvalidName;
    const scale_raw = try art.tensorBytesByName(scale_name);
    if (scale_raw.len != out_dim * @sizeOf(f32)) return error.ShapeMismatch;

    const q = std.mem.bytesAsSlice(i8, q_raw);
    if (scale_raw.len % @sizeOf(f32) != 0) return error.ShapeMismatch;
    if (@intFromPtr(scale_raw.ptr) % @alignOf(f32) != 0) return error.BadAlignment;
    const scales: []const f32 = @as([*]const f32, @ptrCast(@alignCast(scale_raw.ptr)))[0 .. scale_raw.len / @sizeOf(f32)];
    if (scales.len != out_dim) return error.ShapeMismatch;

    const hf = try allocator.alloc(f32, out_dim * in_dim);
    defer allocator.free(hf);
    try qwen_quant.dequantRowQ8(hf, q, scales, out_dim, in_dim);

    if (!transpose) {
        var t = try Tensor.alloc(allocator, .f32, &.{ out_dim, in_dim });
        errdefer t.deinit();
        @memcpy(try t.f32s(), hf);
        return t;
    }
    // Host layout `[in, out]` (same as float projection load).
    var t = try Tensor.alloc(allocator, .f32, &.{ in_dim, out_dim });
    errdefer t.deinit();
    transpose2d(try t.f32s(), hf, out_dim, in_dim);
    return t;
}

fn loadTensorF32(
    allocator: std.mem.Allocator,
    entry: *const artifact.TensorEntry,
    raw: []const u8,
    transpose: bool,
) Error!Tensor {
    const dt = try entry.dtypeTag();
    const rank = entry.rank;
    var shape: [4]usize = undefined;
    var i: u8 = 0;
    while (i < rank) : (i += 1) shape[i] = entry.shape[i];
    const shape_slice = shape[0..rank];

    if (!transpose) {
        var t = try Tensor.alloc(allocator, .f32, shape_slice);
        errdefer t.deinit();
        const dst = try t.f32s();
        try decodeWeights(dst, dt, raw);
        return t;
    }

    if (rank != 2) return error.InvalidShape;
    const rows = shape[0];
    const cols = shape[1];
    var t = try Tensor.alloc(allocator, .f32, &.{ cols, rows });
    errdefer t.deinit();
    var tmp = try Tensor.alloc(allocator, .f32, shape_slice);
    defer tmp.deinit();
    try decodeWeights(try tmp.f32s(), dt, raw);
    transpose2d(try t.f32s(), try tmp.f32s(), rows, cols);
    return t;
}

fn decodeWeights(dst: []f32, dt: DType, raw: []const u8) Error!void {
    switch (dt) {
        .f32 => {
            if (raw.len != dst.len * 4) return error.ShapeMismatch;
            @memcpy(dst, std.mem.bytesAsSlice(f32, raw));
        },
        .bf16 => {
            if (raw.len != dst.len * 2) return error.ShapeMismatch;
            bf16.decodeIntoF32(dst, raw);
        },
        .f16 => {
            if (raw.len != dst.len * 2) return error.ShapeMismatch;
            float16.decodeIntoF32(dst, raw);
        },
        .i8 => return error.InvalidDtype,
    }
}

fn numelFromEntry(entry: *const artifact.TensorEntry) Error!usize {
    var n: usize = 1;
    var i: u8 = 0;
    while (i < entry.rank) : (i += 1) {
        n = std.math.mul(usize, n, entry.shape[i]) catch return error.Overflow;
    }
    return n;
}

fn transpose2dBf16Bytes(dst: []u8, src: []const u8, rows: usize, cols: usize) void {
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        var c: usize = 0;
        while (c < cols) : (c += 1) {
            const src_i = (r * cols + c) * 2;
            const dst_i = (c * rows + r) * 2;
            @memcpy(dst[dst_i..][0..2], src[src_i..][0..2]);
        }
    }
}

/// Copy artifact payload into a GPU-sized bf16 buffer.
/// Returns `true` when the artifact stored f16/bf16 (native half, no f32 host round-trip).
/// Returns `false` for f32 payloads so the caller can narrow from host f32 tensors.
pub fn copyArtifactToBf16(
    dst: []u8,
    art: *const artifact.Artifact,
    name: []const u8,
    transpose: bool,
) Error!bool {
    const entry = try art.findByName(name);
    const raw = try art.tensorBytesByName(name);
    const dt = try entry.dtypeTag();
    const n = try numelFromEntry(entry);
    if (dst.len != n * 2) return error.ShapeMismatch;

    switch (dt) {
        .f32 => return false,
        .bf16 => {
            if (raw.len != n * 2) return error.ShapeMismatch;
            if (!transpose) {
                @memcpy(dst, raw);
                return true;
            }
            if (entry.rank != 2) return error.InvalidShape;
            transpose2dBf16Bytes(dst, raw, entry.shape[0], entry.shape[1]);
            return true;
        },
        .f16 => {
            if (raw.len != n * 2) return error.ShapeMismatch;
            if (!transpose) {
                var i: usize = 0;
                while (i < n) : (i += 1) {
                    const w = std.mem.readInt(u16, raw[i * 2 ..][0..2], .little);
                    std.mem.writeInt(u16, dst[i * 2 ..][0..2], bf16.fromF32(float16.toF32(w)), .little);
                }
                return true;
            }
            if (entry.rank != 2) return error.InvalidShape;
            const rows = entry.shape[0];
            const cols = entry.shape[1];
            var r: usize = 0;
            while (r < rows) : (r += 1) {
                var c: usize = 0;
                while (c < cols) : (c += 1) {
                    const src_i = (r * cols + c) * 2;
                    const w = std.mem.readInt(u16, raw[src_i..][0..2], .little);
                    const dst_i = (c * rows + r) * 2;
                    std.mem.writeInt(u16, dst[dst_i..][0..2], bf16.fromF32(float16.toF32(w)), .little);
                }
            }
            return true;
        },
        .i8 => return false,
    }
}

fn transpose2d(dst: []f32, src: []const f32, rows: usize, cols: usize) void {
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        var c: usize = 0;
        while (c < cols) : (c += 1) {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

fn expectShape1(t: Tensor, n: u32) Error!void {
    if (t.rank != 1 or t.shape[0] != n) return error.ShapeMismatch;
}

fn expectShape2(t: Tensor, a: u32, b: u32) Error!void {
    if (t.rank != 2 or t.shape[0] != a or t.shape[1] != b) return error.ShapeMismatch;
}

/// Test helper: load one tensor by HF name (same path as forward).
pub fn loadTensorNamedForTest(
    allocator: std.mem.Allocator,
    art: *const artifact.Artifact,
    name: []const u8,
    transpose: bool,
) Error!Tensor {
    return loadNamed(allocator, art, name, transpose);
}

test "copyArtifactToBf16 preserves bf16 payload from real checkpoint" {
    const path = "models/qwen3-0.6b.zynfer";
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var art = try artifact.Artifact.loadFile(gpa, io, path);
    defer art.deinit();

    const raw = try art.tensorBytesByName(qwen3.embed_tokens_name);
    var gpu_buf = try gpa.alloc(u8, raw.len);
    defer gpa.free(gpu_buf);
    try std.testing.expect(try copyArtifactToBf16(gpu_buf, &art, qwen3.embed_tokens_name, false));
    try std.testing.expectEqualSlices(u8, raw[0..4096], gpu_buf[0..4096]);
}

test "f16 artifact decode via load path" {
    const gpa = std.testing.allocator;
    const meta = artifact.Meta.fromArch(qwen3.stage11_mini);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const data_f32 = [_]f32{ 1.0, -2.0, 3.5 };
    var f16_bytes: [6]u8 = undefined;
    float16.encodeFromF32(&f16_bytes, &data_f32);
    const shape = try a.dupe(u32, &.{3});
    const owned = try a.dupe(u8, "test.vec");
    var specs: [1]artifact.TensorSpec = .{.{
        .name = owned,
        .tensor_id = 1,
        .dtype = .f16,
        .shape = shape,
        .bytes = &f16_bytes,
    }};
    const bytes = try artifact.build(gpa, meta, &specs);
    defer gpa.free(bytes);
    var art = try artifact.Artifact.loadOwned(gpa, try gpa.dupe(u8, bytes));
    defer art.deinit();
    var t = try loadNamed(gpa, &art, "test.vec", false);
    defer t.deinit();
    const s = try t.f32s();
    try std.testing.expect(@abs(s[0] - 1.0) < 1e-3);
    try std.testing.expect(@abs(s[1] - (-2.0)) < 1e-2);
}

test "real qwen q_proj load matches safetensors transpose layout" {
    const path = "models/qwen3-0.6b.zynfer";
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var art = try artifact.Artifact.loadFile(gpa, io, path);
    defer art.deinit();
    var w = try loadNamed(gpa, &art, "model.layers.0.self_attn.q_proj.weight", true);
    defer w.deinit();
    const s = try w.f32s();
    // Values from models/Qwen3-0.6B/model.safetensors row-major [2048,1024] after transpose [1024,2048].
    try std.testing.expect(@abs(s[0] - 0.0034027099609375) < 1e-6);
    try std.testing.expect(@abs(s[1] - (-0.0244140625)) < 1e-5);
    try std.testing.expect(@abs(s[2048] - (-0.0034637451171875)) < 1e-6);
}

test "real qwen embed row 151643 matches safetensors" {
    const path = "models/qwen3-0.6b.zynfer";
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var art = try artifact.Artifact.loadFile(gpa, io, path);
    defer art.deinit();
    var embed = try loadNamed(gpa, &art, qwen3.embed_tokens_name, false);
    defer embed.deinit();
    const row: usize = 151643;
    const hidden: usize = 1024;
    const s = try embed.f32s();
    // First four f32 values of embed[151643] from HF BF16 checkpoint.
    const base = row * hidden;
    try std.testing.expect(@abs(s[base + 0] - (-0.00274658203125)) < 1e-5);
    try std.testing.expect(@abs(s[base + 1] - 0.035400390625) < 1e-5);
    try std.testing.expect(@abs(s[base + 2] - (-0.00179290771484375)) < 1e-5);
    try std.testing.expect(@abs(s[base + 3] - (-0.015625)) < 1e-5);
}

test "real qwen embed matches lm_head when tied" {
    const path = "models/qwen3-0.6b.zynfer";
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var art = try artifact.Artifact.loadFile(gpa, io, path);
    defer art.deinit();
    const emb = try art.tensorBytesByName("model.embed_tokens.weight");
    const head = try art.tensorBytesByName("lm_head.weight");
    try std.testing.expectEqual(emb.len, head.len);
    try std.testing.expectEqualSlices(u8, emb[0..256], head[0..256]);
}
