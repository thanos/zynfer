//! Per-op Metal adapter for Qwen blocks (Stage M0 baseline path).
//!
//! Routes `qwen_block.forward` through Apple ops the same way the tiny-block
//! baseline `block.Adapter` does. Stage M3 will add a batched Qwen schedule.

const Tensor = @import("../../runtime/tensor.zig").Tensor;
const apple_ops = @import("ops.zig");
const Gpu = @import("gpu.zig").Gpu;

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
