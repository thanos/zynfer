//! Apple Metal schedule for the tiny transformer block fixture.
//!
//! Host tensors and the KV cache stay in `model/tiny_block.zig`. This module
//! runs the same ops on a persistent `Gpu`. Each kernel still uses
//! `encode_and_wait`; that is the measured baseline, not hidden overlap.
//! Decode must not allocate host tensors. GPU shared buffers are still
//! created per op by `apple.ops` — that remaining cost is documented, not
//! claimed away.

const std = @import("std");
const Tensor = @import("../../runtime/tensor.zig").Tensor;
const compare = @import("../../runtime/compare.zig");
const tiny = @import("../../model/tiny_block.zig");
const gpu_mod = @import("gpu.zig");
const apple_ops = @import("ops.zig");

const Gpu = gpu_mod.Gpu;

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

    pub fn init(allocator: std.mem.Allocator, gpu: *Gpu, spec: tiny.Spec) !Session {
        return .{
            .inner = try tiny.Session.init(allocator, spec),
            .gpu = gpu,
        };
    }

    pub fn deinit(self: *Session) void {
        self.inner.deinit();
        self.* = undefined;
    }

    pub fn reset(self: *Session) void {
        self.inner.reset();
    }

    pub fn prefill(self: *Session, x: Tensor, out: Tensor) !void {
        try tiny.forward(Adapter{ .gpu = self.gpu }, &self.inner, x, self.inner.cache.used, out);
    }

    pub fn decode(self: *Session, x: Tensor, out: Tensor) !void {
        if (x.rank != 2 or x.shape[0] != 1) return error.InvalidShape;
        try tiny.forward(Adapter{ .gpu = self.gpu }, &self.inner, x, self.inner.cache.used, out);
    }
};

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
