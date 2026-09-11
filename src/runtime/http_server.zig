//! Stage S4 — small Zig HTTP server (protocol layer only).
//!
//! Listens with `std.Io.net`, frames with `std.http.Server`. Inference stays in
//! `Session.generate`; this module only translates HTTP ↔ generate.
//!
//! Endpoints: GET /health, GET /metrics, POST /v1/completions,
//! POST /v1/chat/completions. Streaming via chunked SSE (`text/event-stream`).

const std = @import("std");
const artifact = @import("../model/artifact.zig");
const qwen3 = @import("../model/qwen3.zig");
const qwen_forward = @import("../model/qwen_forward.zig");
const tokenizer_mod = @import("../model/tokenizer.zig");
const backend_mod = @import("backend.zig");
const sample_mod = @import("sample.zig");

pub const Error = qwen_forward.Error || backend_mod.SelectionError || tokenizer_mod.Error || error{
    BadRequest,
    NotFound,
    MethodNotAllowed,
    MissingPrompt,
    SequenceTooLong,
    TokenizerRequired,
};

pub const Metrics = struct {
    requests_total: u64 = 0,
    completions_total: u64 = 0,
    tokens_generated_total: u64 = 0,
    stream_responses_total: u64 = 0,
};

pub const ServeConfig = struct {
    host: []const u8 = "127.0.0.1",
    port: u16 = 8080,
    /// Exit after this many accepted connections (0 = forever).
    max_connections: u64 = 0,
    default_max_tokens: u32 = 64,
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    kind: backend_mod.BackendKind,
    mini: bool,
    artifact_path: []const u8,
    tokenizer_dir: ?[]const u8,
    art: artifact.Artifact,
    arch: qwen3.Arch,
    tok: ?tokenizer_mod.Tokenizer,
    metrics: Metrics = .{},
    default_max_tokens: u32,

    pub fn initMini(
        allocator: std.mem.Allocator,
        io: std.Io,
        kind: backend_mod.BackendKind,
        default_max_tokens: u32,
    ) !Engine {
        try backend_mod.requireBackend(kind);
        const bytes = try qwen_forward.buildMiniArtifact(allocator);
        const art = try artifact.Artifact.loadOwned(allocator, bytes);
        return .{
            .allocator = allocator,
            .io = io,
            .kind = kind,
            .mini = true,
            .artifact_path = "stage11-mini",
            .tokenizer_dir = null,
            .art = art,
            .arch = qwen3.stage11_mini,
            .tok = null,
            .default_max_tokens = default_max_tokens,
        };
    }

    pub fn initFile(
        allocator: std.mem.Allocator,
        io: std.Io,
        kind: backend_mod.BackendKind,
        artifact_path: []const u8,
        tokenizer_dir: ?[]const u8,
        default_max_tokens: u32,
    ) !Engine {
        try backend_mod.requireBackend(kind);
        var art = try artifact.Artifact.loadFile(allocator, io, artifact_path);
        errdefer art.deinit();
        const arch = try art.meta.toArch();
        var tok: ?tokenizer_mod.Tokenizer = null;
        if (tokenizer_dir) |dir| {
            tok = try tokenizer_mod.Tokenizer.loadHfDir(allocator, io, dir);
        }
        return .{
            .allocator = allocator,
            .io = io,
            .kind = kind,
            .mini = false,
            .artifact_path = artifact_path,
            .tokenizer_dir = tokenizer_dir,
            .art = art,
            .arch = arch,
            .tok = tok,
            .default_max_tokens = default_max_tokens,
        };
    }

    pub fn deinit(self: *Engine) void {
        if (self.tok) |*t| t.deinit();
        self.art.deinit();
        self.* = undefined;
    }
};

const CompletionsJson = struct {
    prompt: ?[]const u8 = null,
    prompt_tokens: ?[]u32 = null,
    max_tokens: ?u32 = null,
    temperature: ?f32 = null,
    top_k: ?u32 = null,
    top_p: ?f32 = null,
    seed: ?u64 = null,
    stream: ?bool = null,
    raw: ?bool = null,
};

const ChatMessageJson = struct {
    role: []const u8 = "",
    content: []const u8 = "",
};

const ChatCompletionsJson = struct {
    messages: []ChatMessageJson = &.{},
    max_tokens: ?u32 = null,
    temperature: ?f32 = null,
    top_k: ?u32 = null,
    top_p: ?f32 = null,
    seed: ?u64 = null,
    stream: ?bool = null,
};

const StreamHttpCtx = struct {
    engine: *Engine,
    body: *std.http.BodyWriter,
    stop_ids: []const u32,
    use_tokenizer: bool,

    fn onToken(ctx: ?*anyopaque, token_id: u32) void {
        const self: *StreamHttpCtx = @ptrCast(@alignCast(ctx.?));
        for (self.stop_ids) |s| if (s == token_id) return;
        if (self.use_tokenizer) {
            const tok = self.engine.tok orelse return;
            const piece = tok.decode(self.engine.allocator, &.{token_id}) catch return;
            defer self.engine.allocator.free(piece);
            self.writeSseText(piece) catch {};
        } else {
            var buf: [32]u8 = undefined;
            const msg = std.fmt.bufPrint(&buf, "{{\"id\":{d}}}", .{token_id}) catch return;
            self.writeSseData(msg) catch {};
        }
    }

    fn writeSseData(self: *StreamHttpCtx, data: []const u8) !void {
        try self.body.writer.print("data: {s}\n\n", .{data});
        try self.body.writer.flush();
        try self.body.flush();
    }

    fn writeSseText(self: *StreamHttpCtx, text: []const u8) !void {
        // Escape minimal JSON string.
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(self.engine.allocator);
        try out.appendSlice(self.engine.allocator, "{\"text\":\"");
        for (text) |c| {
            switch (c) {
                '"' => try out.appendSlice(self.engine.allocator, "\\\""),
                '\\' => try out.appendSlice(self.engine.allocator, "\\\\"),
                '\n' => try out.appendSlice(self.engine.allocator, "\\n"),
                '\r' => try out.appendSlice(self.engine.allocator, "\\r"),
                '\t' => try out.appendSlice(self.engine.allocator, "\\t"),
                else => try out.append(self.engine.allocator, c),
            }
        }
        try out.appendSlice(self.engine.allocator, "\"}");
        try self.writeSseData(out.items);
    }

    fn finish(self: *StreamHttpCtx) !void {
        try self.writeSseData("[DONE]");
        try self.body.end();
    }
};

fn pathOnly(target: []const u8) []const u8 {
    if (std.mem.indexOfScalar(u8, target, '?')) |i| return target[0..i];
    return target;
}

fn readBody(allocator: std.mem.Allocator, request: *std.http.Server.Request) ![]u8 {
    var buf: [4096]u8 = undefined;
    const reader = try request.readerExpectContinue(&buf);
    return try reader.allocRemaining(allocator, .limited(1 << 20));
}

fn respondJson(request: *std.http.Server.Request, status: std.http.Status, body: []const u8) !void {
    try request.respond(body, .{
        .status = status,
        .extra_headers = &.{
            .{ .name = "Content-Type", .value = "application/json; charset=utf-8" },
            .{ .name = "Connection", .value = "close" },
        },
        .keep_alive = false,
    });
}

fn respondText(request: *std.http.Server.Request, status: std.http.Status, body: []const u8, content_type: []const u8) !void {
    try request.respond(body, .{
        .status = status,
        .extra_headers = &.{
            .{ .name = "Content-Type", .value = content_type },
            .{ .name = "Connection", .value = "close" },
        },
        .keep_alive = false,
    });
}

fn handleHealth(engine: *Engine, request: *std.http.Server.Request) !void {
    var buf: [512]u8 = undefined;
    const body = try std.fmt.bufPrint(&buf, "{{\"ok\":true,\"mini\":{},\"model\":\"{s}\",\"backend\":\"{s}\"}}", .{
        engine.mini,
        engine.artifact_path,
        engine.kind.name(),
    });
    try respondJson(request, .ok, body);
}

fn handleMetrics(engine: *Engine, request: *std.http.Server.Request) !void {
    var buf: [512]u8 = undefined;
    const body = try std.fmt.bufPrint(
        &buf,
        \\# TYPE zynfer_requests_total counter
        \\zynfer_requests_total {d}
        \\# TYPE zynfer_completions_total counter
        \\zynfer_completions_total {d}
        \\# TYPE zynfer_tokens_generated_total counter
        \\zynfer_tokens_generated_total {d}
        \\# TYPE zynfer_stream_responses_total counter
        \\zynfer_stream_responses_total {d}
        \\
    ,
        .{
            engine.metrics.requests_total,
            engine.metrics.completions_total,
            engine.metrics.tokens_generated_total,
            engine.metrics.stream_responses_total,
        },
    );
    try respondText(request, .ok, body, "text/plain; version=0.0.4; charset=utf-8");
}

fn resolvePromptIds(
    engine: *Engine,
    allocator: std.mem.Allocator,
    prompt: ?[]const u8,
    prompt_tokens: ?[]const u32,
    raw: bool,
) Error![]u32 {
    if (prompt_tokens) |ids| {
        if (ids.len == 0) return error.MissingPrompt;
        return try allocator.dupe(u32, ids);
    }
    const text = prompt orelse return error.MissingPrompt;
    if (text.len == 0) return error.MissingPrompt;
    const tok = engine.tok orelse return error.TokenizerRequired;
    const wrapped = if (raw)
        try allocator.dupe(u8, text)
    else
        try tok.applyChatTemplate(allocator, text);
    defer allocator.free(wrapped);
    return tok.encode(allocator, wrapped) catch return error.BadRequest;
}

fn generateAndRespond(
    engine: *Engine,
    request: *std.http.Server.Request,
    prompt_ids: []const u32,
    max_tokens: u32,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    seed: u64,
    stream: bool,
) !void {
    const max_seq = prompt_ids.len + max_tokens;
    if (max_seq == 0 or max_seq > engine.arch.max_position_embeddings) return error.SequenceTooLong;

    var sess = try qwen_forward.Session.initWithBackend(engine.allocator, &engine.art, engine.arch, max_seq, engine.kind);
    defer sess.deinit();

    var stop_buf: [3]u32 = .{ 1, 1, 1 };
    var stop_len: usize = 0;
    if (engine.tok) |*tok| {
        stop_buf[0] = tok.eos_token_id;
        stop_buf[1] = tok.endoftext_id;
        stop_buf[2] = tok.im_end_id;
        stop_len = 3;
    } else {
        stop_buf[0] = engine.arch.eos_token_id;
        stop_len = 1;
    }
    const stop_ids = stop_buf[0..stop_len];

    var out_ids: std.ArrayList(u32) = .empty;
    defer out_ids.deinit(engine.allocator);
    try out_ids.ensureTotalCapacity(engine.allocator, max_tokens);

    var rng = std.Random.DefaultPrng.init(seed);
    const sample: sample_mod.Config = .{
        .temperature = temperature,
        .top_k = top_k,
        .top_p = top_p,
        .seed = seed,
    };

    if (stream) {
        var send_buf: [4096]u8 = undefined;
        var body = try request.respondStreaming(&send_buf, .{
            .respond_options = .{
                .keep_alive = false,
                .extra_headers = &.{
                    .{ .name = "Content-Type", .value = "text/event-stream; charset=utf-8" },
                    .{ .name = "Cache-Control", .value = "no-cache" },
                },
            },
        });
        var stream_ctx: StreamHttpCtx = .{
            .engine = engine,
            .body = &body,
            .stop_ids = stop_ids,
            .use_tokenizer = engine.tok != null,
        };

        const stats = try sess.generate(engine.io, prompt_ids, &out_ids, .{
            .max_new_tokens = max_tokens,
            .sample = sample,
            .stop_ids = stop_ids,
            .on_token = StreamHttpCtx.onToken,
            .on_token_ctx = @ptrCast(&stream_ctx),
            .use_kv_cache = true,
        }, &rng);
        engine.metrics.tokens_generated_total += stats.generated_tokens;
        engine.metrics.stream_responses_total += 1;
        try stream_ctx.finish();
    } else {
        const stats = try sess.generate(engine.io, prompt_ids, &out_ids, .{
            .max_new_tokens = max_tokens,
            .sample = sample,
            .stop_ids = stop_ids,
            .use_kv_cache = true,
        }, &rng);
        engine.metrics.tokens_generated_total += stats.generated_tokens;

        var text: []u8 = &.{};
        var free_text = false;
        defer if (free_text) engine.allocator.free(text);
        if (engine.tok) |*tok| {
            text = try tok.decode(engine.allocator, out_ids.items);
            free_text = true;
        } else {
            var list: std.ArrayList(u8) = .empty;
            defer list.deinit(engine.allocator);
            for (out_ids.items, 0..) |id, i| {
                if (i != 0) try list.append(engine.allocator, ' ');
                var ibuf: [16]u8 = undefined;
                const s = try std.fmt.bufPrint(&ibuf, "{d}", .{id});
                try list.appendSlice(engine.allocator, s);
            }
            text = try engine.allocator.dupe(u8, list.items);
            free_text = true;
        }

        // Minimal JSON (escape text).
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(engine.allocator);
        try out.appendSlice(engine.allocator, "{\"id\":\"cmpl-zynfer\",\"object\":\"text_completion\",\"choices\":[{\"text\":\"");
        for (text) |c| {
            switch (c) {
                '"' => try out.appendSlice(engine.allocator, "\\\""),
                '\\' => try out.appendSlice(engine.allocator, "\\\\"),
                '\n' => try out.appendSlice(engine.allocator, "\\n"),
                '\r' => try out.appendSlice(engine.allocator, "\\r"),
                else => try out.append(engine.allocator, c),
            }
        }
        try out.appendSlice(engine.allocator, "\",\"index\":0}],\"usage\":{\"prompt_tokens\":");
        {
            var nbuf: [32]u8 = undefined;
            try out.appendSlice(engine.allocator, try std.fmt.bufPrint(&nbuf, "{d}", .{stats.prompt_tokens}));
        }
        try out.appendSlice(engine.allocator, ",\"completion_tokens\":");
        {
            var nbuf: [32]u8 = undefined;
            try out.appendSlice(engine.allocator, try std.fmt.bufPrint(&nbuf, "{d}", .{stats.generated_tokens}));
        }
        try out.appendSlice(engine.allocator, "}}");
        try respondJson(request, .ok, out.items);
    }
}

fn handleCompletions(engine: *Engine, request: *std.http.Server.Request) !void {
    const body = try readBody(engine.allocator, request);
    defer engine.allocator.free(body);

    var parsed = std.json.parseFromSlice(CompletionsJson, engine.allocator, body, .{
        .ignore_unknown_fields = true,
        .allocate = .alloc_always,
    }) catch return error.BadRequest;
    defer parsed.deinit();
    const req = parsed.value;

    const prompt_ids = try resolvePromptIds(
        engine,
        engine.allocator,
        req.prompt,
        if (req.prompt_tokens) |p| p else null,
        req.raw orelse engine.mini,
    );
    defer engine.allocator.free(prompt_ids);

    const max_tokens = req.max_tokens orelse engine.default_max_tokens;
    const stream = req.stream orelse true;
    engine.metrics.completions_total += 1;
    try generateAndRespond(
        engine,
        request,
        prompt_ids,
        max_tokens,
        req.temperature orelse 0,
        req.top_k orelse 0,
        req.top_p orelse 1.0,
        req.seed orelse 0,
        stream,
    );
}

fn handleChatCompletions(engine: *Engine, request: *std.http.Server.Request) !void {
    if (engine.tok == null) return error.TokenizerRequired;
    const body = try readBody(engine.allocator, request);
    defer engine.allocator.free(body);

    var parsed = std.json.parseFromSlice(ChatCompletionsJson, engine.allocator, body, .{
        .ignore_unknown_fields = true,
        .allocate = .alloc_always,
    }) catch return error.BadRequest;
    defer parsed.deinit();
    const req = parsed.value;
    if (req.messages.len == 0) return error.MissingPrompt;

    // Use last user message as the chat prompt (matches CLI chat wrapping).
    var user_text: ?[]const u8 = null;
    for (req.messages) |m| {
        if (std.mem.eql(u8, m.role, "user")) user_text = m.content;
    }
    const text = user_text orelse req.messages[req.messages.len - 1].content;

    const prompt_ids = try resolvePromptIds(engine, engine.allocator, text, null, false);
    defer engine.allocator.free(prompt_ids);

    engine.metrics.completions_total += 1;
    try generateAndRespond(
        engine,
        request,
        prompt_ids,
        req.max_tokens orelse engine.default_max_tokens,
        req.temperature orelse 0,
        req.top_k orelse 0,
        req.top_p orelse 1.0,
        req.seed orelse 0,
        req.stream orelse true,
    );
}

pub fn handleRequest(engine: *Engine, request: *std.http.Server.Request) !void {
    engine.metrics.requests_total += 1;
    dispatch(engine, request) catch |err| {
        const status: std.http.Status = switch (err) {
            error.NotFound => .not_found,
            error.MethodNotAllowed => .method_not_allowed,
            error.BadRequest, error.MissingPrompt, error.SequenceTooLong, error.TokenizerRequired => .bad_request,
            else => .internal_server_error,
        };
        var ebuf: [128]u8 = undefined;
        const msg = try std.fmt.bufPrint(&ebuf, "{{\"error\":\"{s}\"}}", .{@errorName(err)});
        try respondJson(request, status, msg);
    };
}

fn dispatch(engine: *Engine, request: *std.http.Server.Request) !void {
    const path = pathOnly(request.head.target);
    const method = request.head.method;

    if (std.mem.eql(u8, path, "/health") or std.mem.eql(u8, path, "/v1/health")) {
        if (method != .GET) return error.MethodNotAllowed;
        try handleHealth(engine, request);
        return;
    }
    if (std.mem.eql(u8, path, "/metrics")) {
        if (method != .GET) return error.MethodNotAllowed;
        try handleMetrics(engine, request);
        return;
    }
    if (std.mem.eql(u8, path, "/v1/completions")) {
        if (method != .POST) return error.MethodNotAllowed;
        try handleCompletions(engine, request);
        return;
    }
    if (std.mem.eql(u8, path, "/v1/chat/completions")) {
        if (method != .POST) return error.MethodNotAllowed;
        try handleChatCompletions(engine, request);
        return;
    }
    return error.NotFound;
}

fn serveConnection(engine: *Engine, stream: std.Io.net.Stream) void {
    const io = engine.io;
    defer {
        var copy = stream;
        copy.close(io);
    }
    var send_buffer: [8192]u8 = undefined;
    var recv_buffer: [8192]u8 = undefined;
    var connection_reader = stream.reader(io, &recv_buffer);
    var connection_writer = stream.writer(io, &send_buffer);
    var server: std.http.Server = .init(&connection_reader.interface, &connection_writer.interface);

    while (true) {
        var request = server.receiveHead() catch |err| switch (err) {
            error.HttpConnectionClosing => return,
            else => return,
        };
        handleRequest(engine, &request) catch return;
        // One request per connection for simplicity (Connection: close).
        return;
    }
}

/// Blocking accept loop. Returns after `max_connections` (if non-zero).
pub fn serveLoop(engine: *Engine, cfg: ServeConfig) !void {
    const addr = try std.Io.net.IpAddress.parse(cfg.host, cfg.port);
    var tcp = try addr.listen(engine.io, .{ .reuse_address = true });
    defer tcp.deinit(engine.io);

    const bound = tcp.socket.address;
    std.log.info("zynfer serve listening at http://{f}/", .{bound});

    var accepted: u64 = 0;
    while (true) {
        const stream = try tcp.accept(engine.io);
        serveConnection(engine, stream);
        accepted += 1;
        if (cfg.max_connections != 0 and accepted >= cfg.max_connections) break;
    }
}

const SmokeShared = struct {
    engine: *Engine,
    tcp: *std.Io.net.Server,
    io: std.Io,
    err: ?anyerror = null,
};

fn smokeAcceptOne(shared: *SmokeShared) void {
    const stream = shared.tcp.accept(shared.io) catch |err| {
        shared.err = err;
        return;
    };
    serveConnection(shared.engine, stream);
}

/// In-process smoke: listen on ephemeral port, one connection for /health and
/// one for streaming completions (two accepts), then return.
pub fn runSmoke(engine: *Engine) !void {
    const addr = try std.Io.net.IpAddress.parseIp4("127.0.0.1", 0);
    var tcp = try addr.listen(engine.io, .{ .reuse_address = true });
    defer tcp.deinit(engine.io);
    const port = tcp.socket.address.getPort();

    var url_buf: [64]u8 = undefined;
    const health_url = try std.fmt.bufPrint(&url_buf, "http://127.0.0.1:{d}/health", .{port});

    // Health
    {
        var shared: SmokeShared = .{ .engine = engine, .tcp = &tcp, .io = engine.io };
        var fut = try engine.io.concurrent(smokeAcceptOne, .{&shared});
        var client: std.http.Client = .{ .allocator = engine.allocator, .io = engine.io };
        defer client.deinit();
        var response_body: std.Io.Writer.Allocating = .init(engine.allocator);
        defer response_body.deinit();
        const result = try client.fetch(.{
            .location = .{ .url = health_url },
            .response_writer = &response_body.writer,
        });
        _ = fut.await(engine.io);
        if (shared.err) |e| return e;
        if (result.status != .ok) return error.BadRequest;
        if (std.mem.indexOf(u8, response_body.written(), "\"ok\":true") == null) return error.BadRequest;
    }

    // Streaming completions (mini prompt_tokens)
    {
        var shared: SmokeShared = .{ .engine = engine, .tcp = &tcp, .io = engine.io };
        var fut = try engine.io.concurrent(smokeAcceptOne, .{&shared});
        var client: std.http.Client = .{ .allocator = engine.allocator, .io = engine.io };
        defer client.deinit();
        var response_body: std.Io.Writer.Allocating = .init(engine.allocator);
        defer response_body.deinit();
        var curl_buf: [80]u8 = undefined;
        const completions_url = try std.fmt.bufPrint(&curl_buf, "http://127.0.0.1:{d}/v1/completions", .{port});
        const payload =
            \\{"prompt_tokens":[2,3,2,3],"max_tokens":4,"stream":true,"temperature":0}
        ;
        const result = try client.fetch(.{
            .location = .{ .url = completions_url },
            .method = .POST,
            .payload = payload,
            .extra_headers = &.{
                .{ .name = "Content-Type", .value = "application/json" },
            },
            .response_writer = &response_body.writer,
        });
        _ = fut.await(engine.io);
        if (shared.err) |e| return e;
        if (result.status != .ok) return error.BadRequest;
        const body = response_body.written();
        if (std.mem.indexOf(u8, body, "data:") == null) return error.BadRequest;
        if (std.mem.indexOf(u8, body, "[DONE]") == null) return error.BadRequest;
    }
}

test "http smoke health + streamed completions on mini" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var engine = try Engine.initMini(gpa, io, .cpu, 4);
    defer engine.deinit();
    try runSmoke(&engine);
    try std.testing.expect(engine.metrics.completions_total >= 1);
    try std.testing.expect(engine.metrics.stream_responses_total >= 1);
}
