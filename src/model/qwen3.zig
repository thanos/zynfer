//! Qwen3 architecture constants used by artifacts and (later) forward passes.
//!
//! Values match Hugging Face `Qwen/Qwen3-0.6B` `config.json`. Stage 10 stores
//! them in `.zynfer` metadata; Stage 11 runs the forward.

const std = @import("std");

pub const ModelId = enum {
    qwen3_0_6b,

    pub fn name(self: ModelId) []const u8 {
        return switch (self) {
            .qwen3_0_6b => "qwen3-0.6b",
        };
    }
};

/// Architecture for `Qwen/Qwen3-0.6B` (Instruct/Base share these dims).
pub const qwen3_0_6b = Arch{
    .model_id = .qwen3_0_6b,
    .vocab_size = 151936,
    .hidden_size = 1024,
    .intermediate_size = 3072,
    .num_layers = 28,
    .num_attention_heads = 16,
    .num_key_value_heads = 8,
    .head_dim = 128,
    .max_position_embeddings = 40960,
    .bos_token_id = 151643,
    .eos_token_id = 151645,
    .rope_theta = 1_000_000.0,
    .rms_norm_eps = 1e-6,
    .tie_word_embeddings = true,
};

pub const Arch = struct {
    model_id: ModelId,
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
    tie_word_embeddings: bool,

    pub fn qDim(self: Arch) u32 {
        return self.num_attention_heads * self.head_dim;
    }

    pub fn kvDim(self: Arch) u32 {
        return self.num_key_value_heads * self.head_dim;
    }
};

test "Qwen3-0.6B dims are consistent" {
    const a = qwen3_0_6b;
    try std.testing.expectEqual(@as(u32, 2048), a.qDim());
    try std.testing.expectEqual(@as(u32, 1024), a.kvDim());
    try std.testing.expectEqualStrings("qwen3-0.6b", a.model_id.name());
}

/// Tiny architecture for Stage 11 CI tests (1 layer, f32 fixture artifact).
pub const stage11_mini = Arch{
    .model_id = .qwen3_0_6b,
    .vocab_size = 8,
    .hidden_size = 16,
    .intermediate_size = 32,
    .num_layers = 1,
    .num_attention_heads = 2,
    .num_key_value_heads = 1,
    .head_dim = 8,
    .max_position_embeddings = 32,
    .bos_token_id = 0,
    .eos_token_id = 1,
    .rope_theta = 10_000.0,
    .rms_norm_eps = 1e-6,
    .tie_word_embeddings = true,
};

pub const embed_tokens_name = "model.embed_tokens.weight";
pub const final_norm_name = "model.norm.weight";
pub const lm_head_name = "lm_head.weight";

pub fn layerInputNormName(layer: u32, buf: []u8) []const u8 {
    return std.fmt.bufPrint(buf, "model.layers.{d}.input_layernorm.weight", .{layer}) catch unreachable;
}

pub fn layerPostAttnNormName(layer: u32, buf: []u8) []const u8 {
    return std.fmt.bufPrint(buf, "model.layers.{d}.post_attention_layernorm.weight", .{layer}) catch unreachable;
}

pub fn layerQNormName(layer: u32, buf: []u8) []const u8 {
    return std.fmt.bufPrint(buf, "model.layers.{d}.self_attn.q_norm.weight", .{layer}) catch unreachable;
}

pub fn layerKNormName(layer: u32, buf: []u8) []const u8 {
    return std.fmt.bufPrint(buf, "model.layers.{d}.self_attn.k_norm.weight", .{layer}) catch unreachable;
}

pub fn layerQProjName(layer: u32, buf: []u8) []const u8 {
    return std.fmt.bufPrint(buf, "model.layers.{d}.self_attn.q_proj.weight", .{layer}) catch unreachable;
}

pub fn layerKProjName(layer: u32, buf: []u8) []const u8 {
    return std.fmt.bufPrint(buf, "model.layers.{d}.self_attn.k_proj.weight", .{layer}) catch unreachable;
}

pub fn layerVProjName(layer: u32, buf: []u8) []const u8 {
    return std.fmt.bufPrint(buf, "model.layers.{d}.self_attn.v_proj.weight", .{layer}) catch unreachable;
}

pub fn layerOProjName(layer: u32, buf: []u8) []const u8 {
    return std.fmt.bufPrint(buf, "model.layers.{d}.self_attn.o_proj.weight", .{layer}) catch unreachable;
}

pub fn layerGateProjName(layer: u32, buf: []u8) []const u8 {
    return std.fmt.bufPrint(buf, "model.layers.{d}.mlp.gate_proj.weight", .{layer}) catch unreachable;
}

pub fn layerUpProjName(layer: u32, buf: []u8) []const u8 {
    return std.fmt.bufPrint(buf, "model.layers.{d}.mlp.up_proj.weight", .{layer}) catch unreachable;
}

pub fn layerDownProjName(layer: u32, buf: []u8) []const u8 {
    return std.fmt.bufPrint(buf, "model.layers.{d}.mlp.down_proj.weight", .{layer}) catch unreachable;
}
