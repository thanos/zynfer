//! Explicit model registry (Stage M8 — NInfer-style registration).
//!
//! Each entry pins ID, HF source, local paths, architecture dimensions, and
//! an optional artifact SHA-256 (hex). Dims are authoritative; checksums are
//! filled after the first official convert on the lab machine.

const std = @import("std");
const qwen3 = @import("qwen3.zig");

pub const Entry = struct {
    id: qwen3.ModelId,
    /// Human-readable label.
    label: []const u8,
    hf_repo: []const u8,
    /// Default HF snapshot directory under the repo root.
    hf_dir: []const u8,
    /// Default bf16/fp `.zynfer` path.
    artifact_path: []const u8,
    /// Default int8 `.zynfer` path (M5 / M8 quantized capstone).
    int8_artifact_path: []const u8,
    arch: qwen3.Arch,
    /// Artifact format version this entry targets.
    artifact_version: u16 = 1,
    /// Optional lowercase hex SHA-256 of the official int8 artifact body.
    /// Null / empty = dims-only registration (checksum recorded in ledger later).
    int8_sha256_hex: ?[]const u8 = null,
};

pub const entries = [_]Entry{
    .{
        .id = .qwen3_0_6b,
        .label = "Qwen3-0.6B",
        .hf_repo = "Qwen/Qwen3-0.6B",
        .hf_dir = "models/Qwen3-0.6B",
        .artifact_path = "models/qwen3-0.6b.zynfer",
        .int8_artifact_path = "models/qwen3-0.6b-int8.zynfer",
        .arch = qwen3.qwen3_0_6b,
    },
    .{
        .id = .qwen3_4b,
        .label = "Qwen3-4B",
        .hf_repo = "Qwen/Qwen3-4B",
        .hf_dir = "models/Qwen3-4B",
        .artifact_path = "models/qwen3-4b.zynfer",
        .int8_artifact_path = "models/qwen3-4b-int8.zynfer",
        .arch = qwen3.qwen3_4b,
        .int8_sha256_hex = "c7ef40e3312623f1958f5e4473313eb1bc06fbca1ec89721d4f15d2b26d0b541",
    },
};

pub fn byId(id: qwen3.ModelId) Entry {
    for (entries) |e| {
        if (e.id == id) return e;
    }
    unreachable;
}

pub fn byName(name: []const u8) ?Entry {
    const id = qwen3.ModelId.parse(name) orelse return null;
    return byId(id);
}

pub const DimMismatch = error{RegistryDimMismatch};

/// Ensure artifact meta dims match the registered architecture for `id`.
pub fn validateArch(arch: qwen3.Arch) DimMismatch!void {
    const expected = qwen3.archForId(arch.model_id);
    if (arch.vocab_size != expected.vocab_size or
        arch.hidden_size != expected.hidden_size or
        arch.intermediate_size != expected.intermediate_size or
        arch.num_layers != expected.num_layers or
        arch.num_attention_heads != expected.num_attention_heads or
        arch.num_key_value_heads != expected.num_key_value_heads or
        arch.head_dim != expected.head_dim or
        arch.max_position_embeddings != expected.max_position_embeddings or
        arch.bos_token_id != expected.bos_token_id or
        arch.eos_token_id != expected.eos_token_id or
        arch.tie_word_embeddings != expected.tie_word_embeddings)
    {
        return error.RegistryDimMismatch;
    }
}

/// f32 KV bytes for `max_seq` (layers × n_kv × seq × head_dim × 2 × 4).
pub fn estimateKvBytesF32(arch: qwen3.Arch, max_seq: usize) u64 {
    return @as(u64, arch.num_layers) * @as(u64, arch.num_key_value_heads) *
        @as(u64, @intCast(max_seq)) * @as(u64, arch.head_dim) * 2 * @sizeOf(f32);
}

test "registry lists 0.6B and 4B" {
    try std.testing.expectEqual(@as(usize, 2), entries.len);
    try std.testing.expectEqualStrings("qwen3-4b", byId(.qwen3_4b).arch.model_id.name());
    try validateArch(qwen3.qwen3_4b);
    try validateArch(qwen3.qwen3_0_6b);
}

test "registry rejects wrong dims" {
    var bad = qwen3.qwen3_4b;
    bad.hidden_size = 1024;
    try std.testing.expectError(error.RegistryDimMismatch, validateArch(bad));
}
