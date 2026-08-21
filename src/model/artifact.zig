//! Native `.zynfer` model artifact: format, validate, load, and compile.
//!
//! Stage 10 exit: a deterministic conversion produces bytes the Zig runtime
//! validates and loads. The hot path does not parse Safetensors; that stays
//! in `tools/checkpoint/` (Python) for development-time conversion.

const std = @import("std");
const dtype_mod = @import("../runtime/dtype.zig");
const qwen3 = @import("qwen3.zig");

pub const magic = "ZYNF";
pub const format_version: u16 = 1;
pub const endian_little: u8 = 1;
pub const payload_align: u64 = 64;
pub const max_name_len: usize = 64;
pub const max_rank: usize = 8;
pub const header_size: u32 = @sizeOf(Header);
pub const meta_size: u32 = @sizeOf(Meta);
pub const entry_size: u32 = @sizeOf(TensorEntry);

pub const Error = error{
    InvalidMagic,
    UnsupportedVersion,
    BadEndian,
    Truncated,
    BadHeader,
    BadAlignment,
    ChecksumMismatch,
    DuplicateTensor,
    TensorNotFound,
    ShapeMismatch,
    Overflow,
    InvalidName,
    InvalidDtype,
    InvalidRank,
    OutOfMemory,
};

pub const Header = extern struct {
    magic: [4]u8,
    version: u16,
    endian: u8,
    flags: u8,
    header_bytes: u32,
    meta_offset: u32,
    meta_bytes: u32,
    dir_offset: u32,
    dir_bytes: u32,
    tensor_count: u32,
    reserved0: u32,
    payload_offset: u64,
    payload_bytes: u64,
    /// SHA-256 over the whole file with this field zeroed.
    sha256: [32]u8,
};

comptime {
    // Fixed layout for format_version 1 — bump version if these change.
    if (@sizeOf(Header) != 88) @compileError("Header layout drift");
    if (@sizeOf(Meta) != 116) @compileError("Meta layout drift");
    if (@sizeOf(TensorEntry) != 120) @compileError("TensorEntry layout drift");
}

pub const Meta = extern struct {
    model_id: [64]u8,
    vocab_size: u32,
    hidden_size: u32,
    intermediate_size: u32,
    num_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    max_position_embeddings: u32,
    bos_token_id: u32,
    eos_token_id: u32,
    rope_theta: f32,
    rms_norm_eps: f32,
    tie_word_embeddings: u8,
    _pad: [3]u8 = .{ 0, 0, 0 },

    pub fn modelIdSlice(self: *const Meta) []const u8 {
        return std.mem.sliceTo(&self.model_id, 0);
    }

    pub fn fromArch(arch: qwen3.Arch) Meta {
        var m: Meta = std.mem.zeroes(Meta);
        const id = arch.model_id.name();
        @memcpy(m.model_id[0..id.len], id);
        m.vocab_size = arch.vocab_size;
        m.hidden_size = arch.hidden_size;
        m.intermediate_size = arch.intermediate_size;
        m.num_layers = arch.num_layers;
        m.num_attention_heads = arch.num_attention_heads;
        m.num_key_value_heads = arch.num_key_value_heads;
        m.head_dim = arch.head_dim;
        m.max_position_embeddings = arch.max_position_embeddings;
        m.bos_token_id = arch.bos_token_id;
        m.eos_token_id = arch.eos_token_id;
        m.rope_theta = arch.rope_theta;
        m.rms_norm_eps = arch.rms_norm_eps;
        m.tie_word_embeddings = if (arch.tie_word_embeddings) 1 else 0;
        return m;
    }

    pub fn toArch(self: Meta) !qwen3.Arch {
        const id = self.modelIdSlice();
        const model_id: qwen3.ModelId = if (std.mem.eql(u8, id, "qwen3-0.6b"))
            .qwen3_0_6b
        else
            return error.UnsupportedVersion;
        return .{
            .model_id = model_id,
            .vocab_size = self.vocab_size,
            .hidden_size = self.hidden_size,
            .intermediate_size = self.intermediate_size,
            .num_layers = self.num_layers,
            .num_attention_heads = self.num_attention_heads,
            .num_key_value_heads = self.num_key_value_heads,
            .head_dim = self.head_dim,
            .max_position_embeddings = self.max_position_embeddings,
            .bos_token_id = self.bos_token_id,
            .eos_token_id = self.eos_token_id,
            .rope_theta = self.rope_theta,
            .rms_norm_eps = self.rms_norm_eps,
            .tie_word_embeddings = self.tie_word_embeddings != 0,
        };
    }
};

pub const TensorEntry = extern struct {
    name: [64]u8,
    /// Stable numeric id for hot path (0 = unset / name-only).
    tensor_id: u32,
    dtype: u8,
    rank: u8,
    _pad: [2]u8 = .{ 0, 0 },
    shape: [8]u32,
    /// Byte offset relative to `Header.payload_offset`.
    offset: u64,
    nbytes: u64,

    pub fn nameSlice(self: *const TensorEntry) []const u8 {
        return std.mem.sliceTo(&self.name, 0);
    }

    pub fn dtypeTag(self: TensorEntry) Error!dtype_mod.DType {
        return switch (self.dtype) {
            0 => .f32,
            1 => .f16,
            2 => .bf16,
            else => error.InvalidDtype,
        };
    }
};

pub fn dtypeToTag(dt: dtype_mod.DType) u8 {
    return switch (dt) {
        .f32 => 0,
        .f16 => 1,
        .bf16 => 2,
    };
}

fn alignUp(v: u64, a: u64) u64 {
    return (v + a - 1) / a * a;
}

fn writeName(dest: *[64]u8, name: []const u8) Error!void {
    if (name.len == 0 or name.len >= max_name_len) return error.InvalidName;
    @memset(dest, 0);
    @memcpy(dest[0..name.len], name);
}

fn sha256FileImage(bytes: []const u8) [32]u8 {
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    if (bytes.len < header_size) {
        hasher.update(bytes);
        var out: [32]u8 = undefined;
        hasher.final(&out);
        return out;
    }
    const sha_off = @offsetOf(Header, "sha256");
    hasher.update(bytes[0..sha_off]);
    const zeros = [_]u8{0} ** 32;
    hasher.update(&zeros);
    if (bytes.len > sha_off + 32) {
        hasher.update(bytes[sha_off + 32 ..]);
    }
    var out: [32]u8 = undefined;
    hasher.final(&out);
    return out;
}

pub const TensorSpec = struct {
    name: []const u8,
    tensor_id: u32 = 0,
    dtype: dtype_mod.DType,
    shape: []const u32,
    bytes: []const u8,
};

pub fn build(allocator: std.mem.Allocator, meta: Meta, tensors: []const TensorSpec) Error![]u8 {
    if (tensors.len > std.math.maxInt(u32)) return error.Overflow;

    const dir_bytes: u64 = try std.math.mul(u64, tensors.len, entry_size);
    const meta_offset: u32 = header_size;
    const dir_offset: u32 = meta_offset + meta_size;
    const after_dir: u64 = @as(u64, dir_offset) + dir_bytes;
    const payload_offset = alignUp(after_dir, payload_align);

    var payload_cursor: u64 = 0;
    var entries = try allocator.alloc(TensorEntry, tensors.len);
    defer allocator.free(entries);

    var total_payload: u64 = 0;
    for (tensors, 0..) |t, i| {
        if (t.shape.len == 0 or t.shape.len > max_rank) return error.InvalidRank;
        const rank = t.shape.len;
        var nelem: u64 = 1;
        var shape_buf = [_]u32{0} ** max_rank;
        for (t.shape, 0..) |d, di| {
            if (d == 0) return error.ShapeMismatch;
            nelem = try std.math.mul(u64, nelem, d);
            shape_buf[di] = d;
        }
        const expect = try std.math.mul(u64, nelem, t.dtype.sizeOf());
        if (t.bytes.len != expect) return error.ShapeMismatch;

        const aligned_off = alignUp(payload_cursor, payload_align);
        var ent = std.mem.zeroes(TensorEntry);
        try writeName(&ent.name, t.name);
        ent.tensor_id = t.tensor_id;
        ent.dtype = dtypeToTag(t.dtype);
        ent.rank = @intCast(rank);
        ent.shape = shape_buf;
        ent.offset = aligned_off;
        ent.nbytes = expect;
        entries[i] = ent;
        payload_cursor = aligned_off + expect;
        total_payload = payload_cursor;
    }

    // Duplicate name check.
    var i: usize = 0;
    while (i < entries.len) : (i += 1) {
        var j: usize = i + 1;
        while (j < entries.len) : (j += 1) {
            if (std.mem.eql(u8, entries[i].nameSlice(), entries[j].nameSlice())) {
                return error.DuplicateTensor;
            }
        }
    }

    const file_len = payload_offset + total_payload;
    var out = try allocator.alloc(u8, file_len);
    errdefer allocator.free(out);
    @memset(out, 0);

    var header = Header{
        .magic = magic.*,
        .version = format_version,
        .endian = endian_little,
        .flags = 0,
        .header_bytes = header_size,
        .meta_offset = meta_offset,
        .meta_bytes = meta_size,
        .dir_offset = dir_offset,
        .dir_bytes = @intCast(dir_bytes),
        .tensor_count = @intCast(tensors.len),
        .reserved0 = 0,
        .payload_offset = payload_offset,
        .payload_bytes = total_payload,
        .sha256 = [_]u8{0} ** 32,
    };
    @memcpy(out[0..header_size], std.mem.asBytes(&header));
    @memcpy(out[meta_offset..][0..meta_size], std.mem.asBytes(&meta));
    if (dir_bytes > 0) {
        @memcpy(out[dir_offset..][0..dir_bytes], std.mem.sliceAsBytes(entries));
    }
    for (entries, tensors) |ent, t| {
        const abs = payload_offset + ent.offset;
        @memcpy(out[abs..][0..t.bytes.len], t.bytes);
    }

    const digest = sha256FileImage(out);
    @memcpy(out[@offsetOf(Header, "sha256")..][0..32], &digest);
    return out;
}

/// Loaded artifact owning the file bytes.
pub const Artifact = struct {
    bytes: []u8,
    allocator: std.mem.Allocator,
    header: Header,
    meta: Meta,
    entries: []TensorEntry,

    pub fn deinit(self: *Artifact) void {
        self.allocator.free(self.bytes);
        self.allocator.free(self.entries);
        self.* = undefined;
    }

    pub fn loadOwned(allocator: std.mem.Allocator, bytes: []u8) Error!Artifact {
        const validated = try validate(bytes);
        const entries = try allocator.alloc(TensorEntry, validated.header.tensor_count);
        errdefer allocator.free(entries);
        if (validated.header.tensor_count > 0) {
            const dir = bytes[validated.header.dir_offset..][0..validated.header.dir_bytes];
            const src = std.mem.bytesAsSlice(TensorEntry, dir);
            @memcpy(entries, src);
        }
        return .{
            .bytes = bytes,
            .allocator = allocator,
            .header = validated.header,
            .meta = validated.meta,
            .entries = entries,
        };
    }

    pub fn loadFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !Artifact {
        var file = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io);
        const st = try file.stat(io);
        const size: usize = std.math.cast(usize, st.size) orelse return error.Overflow;
        // Full Qwen3-0.6B BF16 artifact is ~1.5 GiB; Stage 11 may mmap later.
        if (size > 4 * 1024 * 1024 * 1024) return error.Overflow;

        const bytes = try allocator.alloc(u8, size);
        errdefer allocator.free(bytes);

        var off: u64 = 0;
        while (off < size) {
            const chunk = try file.readPositionalAll(io, bytes[@intCast(off)..], off);
            if (chunk == 0) break;
            off += chunk;
        }
        if (off != size) {
            allocator.free(bytes);
            return error.Truncated;
        }

        return loadOwned(allocator, bytes) catch |err| {
            allocator.free(bytes);
            return err;
        };
    }

    pub fn tensorBytes(self: *const Artifact, name: []const u8) Error![]const u8 {
        const ent = try self.find(name);
        const abs = self.header.payload_offset + ent.offset;
        if (abs + ent.nbytes > self.bytes.len) return error.Truncated;
        return self.bytes[abs..][0..ent.nbytes];
    }

    pub fn find(self: *const Artifact, name: []const u8) Error!*const TensorEntry {
        for (self.entries) |*e| {
            if (std.mem.eql(u8, e.nameSlice(), name)) return e;
        }
        return error.TensorNotFound;
    }
};

pub const Validated = struct {
    header: Header,
    meta: Meta,
};

pub fn validate(bytes: []const u8) Error!Validated {
    if (bytes.len < header_size) return error.Truncated;
    var header: Header = undefined;
    @memcpy(std.mem.asBytes(&header), bytes[0..header_size]);

    if (!std.mem.eql(u8, &header.magic, magic)) return error.InvalidMagic;
    if (header.version != format_version) return error.UnsupportedVersion;
    if (header.endian != endian_little) return error.BadEndian;
    if (header.header_bytes != header_size) return error.BadHeader;
    if (header.meta_bytes != meta_size) return error.BadHeader;
    if (header.meta_offset != header_size) return error.BadHeader;
    if (header.dir_offset != header.meta_offset + header.meta_bytes) return error.BadHeader;
    if (header.dir_bytes != header.tensor_count * entry_size) return error.BadHeader;
    if (header.payload_offset % payload_align != 0) return error.BadAlignment;
    if (header.payload_offset < header.dir_offset + header.dir_bytes) return error.BadHeader;

    const need = header.payload_offset + header.payload_bytes;
    if (bytes.len < need) return error.Truncated;

    const digest = sha256FileImage(bytes);
    if (!std.mem.eql(u8, &digest, &header.sha256)) return error.ChecksumMismatch;

    var meta: Meta = undefined;
    @memcpy(std.mem.asBytes(&meta), bytes[header.meta_offset..][0..meta_size]);

    if (header.tensor_count > 0) {
        const dir = bytes[header.dir_offset..][0..header.dir_bytes];
        const entries = std.mem.bytesAsSlice(TensorEntry, dir);
        for (entries) |e| {
            if (e.rank == 0 or e.rank > max_rank) return error.InvalidRank;
            _ = try e.dtypeTag();
            if (e.nameSlice().len == 0) return error.InvalidName;
            if (e.offset + e.nbytes > header.payload_bytes) return error.Truncated;
            if (e.offset % payload_align != 0) return error.BadAlignment;
        }
    }

    return .{ .header = header, .meta = meta };
}

/// Tiny deterministic fixture used by Stage 10 tests and `artifact-compile`.
pub fn buildStage10Fixture(allocator: std.mem.Allocator) Error![]u8 {
    const meta = Meta.fromArch(qwen3.qwen3_0_6b);
    // Two small tensors: not a real Qwen weight set — proves round-trip.
    var w_emb = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };
    var w_lm = [_]f32{ 1.0, 0.0, -1.0, 0.5 };
    const tensors = [_]TensorSpec{
        .{
            .name = "fixture.embed",
            .tensor_id = 1,
            .dtype = .f32,
            .shape = &.{ 2, 4 },
            .bytes = std.mem.sliceAsBytes(w_emb[0..]),
        },
        .{
            .name = "fixture.lm_head",
            .tensor_id = 2,
            .dtype = .f32,
            .shape = &.{ 2, 2 },
            .bytes = std.mem.sliceAsBytes(w_lm[0..]),
        },
    };
    return build(allocator, meta, &tensors);
}

pub fn formatSha256(digest: *const [32]u8, buf: *[64]u8) []const u8 {
    const hex = "0123456789abcdef";
    for (digest.*, 0..) |b, i| {
        buf[i * 2] = hex[b >> 4];
        buf[i * 2 + 1] = hex[b & 0xf];
    }
    return buf[0..64];
}

test "artifact round-trip validates and loads tensors" {
    const gpa = std.testing.allocator;
    const bytes = try buildStage10Fixture(gpa);
    defer gpa.free(bytes);

    const v = try validate(bytes);
    try std.testing.expectEqualStrings("qwen3-0.6b", v.meta.modelIdSlice());
    try std.testing.expectEqual(@as(u32, 28), v.meta.num_layers);
    try std.testing.expectEqual(@as(u32, 2), v.header.tensor_count);

    const owned = try gpa.dupe(u8, bytes);
    var art = try Artifact.loadOwned(gpa, owned);
    defer art.deinit();

    const emb = try art.tensorBytes("fixture.embed");
    try std.testing.expectEqual(@as(usize, 32), emb.len);
    const as_f32 = std.mem.bytesAsSlice(f32, emb);
    try std.testing.expectEqual(@as(f32, 0.1), as_f32[0]);
    try std.testing.expectEqual(@as(f32, 0.8), as_f32[7]);

    const head = try art.find("fixture.lm_head");
    try std.testing.expectEqual(@as(u32, 2), head.tensor_id);
    try std.testing.expectError(error.TensorNotFound, art.find("missing"));
}

test "corrupt checksum is rejected" {
    const gpa = std.testing.allocator;
    const bytes = try buildStage10Fixture(gpa);
    defer gpa.free(bytes);
    var mut = try gpa.dupe(u8, bytes);
    defer gpa.free(mut);
    mut[mut.len - 1] ^= 0xff;
    try std.testing.expectError(error.ChecksumMismatch, validate(mut));
}

test "bad magic is rejected" {
    const gpa = std.testing.allocator;
    const bytes = try buildStage10Fixture(gpa);
    defer gpa.free(bytes);
    var mut = try gpa.dupe(u8, bytes);
    defer gpa.free(mut);
    mut[0] = 'X';
    // Fix checksum so we hit magic first... magic is checked before checksum.
    try std.testing.expectError(error.InvalidMagic, validate(mut));
}
