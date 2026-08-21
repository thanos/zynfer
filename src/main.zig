const std = @import("std");
const zynfer = @import("zynfer");

const usage =
    \\zynfer — Zig LLM inference (CPU oracle + Apple Metal + AMD HIP probe)
    \\
    \\Usage:
    \\  zynfer              Environment report + GPU probe
    \\  zynfer env          Development-environment report
    \\  zynfer gpu          HIP device enumeration (AMD host)
    \\  zynfer caps         Backend/device capabilities and fallbacks
    \\  zynfer stage7       SME / Core ML Stage 7 probe + retain/reject ledger
    \\  zynfer stage8       Hardening leftovers + retain/reject ledger
    \\  zynfer stage10      Checkpoint / .zynfer artifact Stage 10 ledger
    \\  zynfer inspect PATH Validate and print a .zynfer artifact
    \\  zynfer artifact-compile [--out PATH]  Write Stage 10 fixture .zynfer
    \\  zynfer backends     List selectable backends
    \\  zynfer ops-bench    CPU vs Apple op microbenchmarks
    \\  zynfer block-bench  Tiny-block prefill/decode timings
    \\  zynfer bench        HIP query timing (AMD host)
    \\  zynfer help
    \\
    \\Force a backend (invalid choices fail; they do not fall back):
    \\  zynfer caps --backend cpu
    \\  zynfer caps --backend apple
    \\  ZYNFER_BACKEND=cpu zynfer caps
    \\
    \\Stage 7 experimental forces (must fail loud; paths are not retained):
    \\  ZYNFER_FORCE_SME=1 zynfer stage7
    \\  ZYNFER_FORCE_COREML=1 zynfer stage7
    \\
;

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const host = zynfer.util.Host{
        .gpa = allocator,
        .io = io,
        .environ = init.environ_map,
    };

    var args_it = std.process.Args.Iterator.init(init.minimal.args);
    _ = args_it.skip();
    var command: []const u8 = "all";
    var have_command = false;
    var forced_backend: ?[]const u8 = null;
    var out_path: ?[]const u8 = null;
    var positionals: [8][]const u8 = undefined;
    var n_pos: usize = 0;
    while (args_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--backend")) {
            forced_backend = args_it.next() orelse {
                std.debug.print("missing value for --backend\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.startsWith(u8, arg, "--backend=")) {
            forced_backend = arg["--backend=".len..];
        } else if (std.mem.eql(u8, arg, "--out")) {
            out_path = args_it.next() orelse {
                std.debug.print("missing value for --out\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.startsWith(u8, arg, "--out=")) {
            out_path = arg["--out=".len..];
        } else if (!have_command and !std.mem.startsWith(u8, arg, "-")) {
            command = arg;
            have_command = true;
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            if (n_pos >= positionals.len) {
                std.debug.print("too many arguments\n", .{});
                std.process.exit(2);
            }
            positionals[n_pos] = arg;
            n_pos += 1;
        } else {
            std.debug.print("unknown flag: {s}\n", .{arg});
            std.process.exit(2);
        }
    }
    if (forced_backend == null) {
        if (host.environ) |env_map| {
            forced_backend = env_map.get("ZYNFER_BACKEND");
        }
    }

    var buf: [8192]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(io, &buf);
    const writer = &stdout_writer.interface;

    if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "-h") or std.mem.eql(u8, command, "--help")) {
        try writer.writeAll(usage);
        try writer.flush();
        return;
    }

    // Stage 7: forced experimental paths must fail clearly, never silently.
    rejectForcedExperimentalPaths();

    if (std.mem.eql(u8, command, "env")) {
        try printEnv(host, writer);
    } else if (std.mem.eql(u8, command, "gpu")) {
        try printGpu(writer);
    } else if (std.mem.eql(u8, command, "caps")) {
        try printCaps(writer, forced_backend);
    } else if (std.mem.eql(u8, command, "stage7")) {
        try printStage7(writer);
    } else if (std.mem.eql(u8, command, "stage8")) {
        try printStage8(writer);
    } else if (std.mem.eql(u8, command, "stage10")) {
        try printStage10(writer);
    } else if (std.mem.eql(u8, command, "inspect")) {
        if (n_pos < 1) {
            std.debug.print("usage: zynfer inspect PATH.zynfer\n", .{});
            std.process.exit(2);
        }
        try runInspect(allocator, io, writer, positionals[0]);
    } else if (std.mem.eql(u8, command, "artifact-compile")) {
        const path = out_path orelse (if (n_pos >= 1) positionals[0] else "stage10-fixture.zynfer");
        try runArtifactCompile(allocator, io, writer, path);
    } else if (std.mem.eql(u8, command, "backends")) {
        try printBackends(writer);
    } else if (std.mem.eql(u8, command, "ops-bench")) {
        try runOpsBench(allocator, io, writer, forced_backend);
    } else if (std.mem.eql(u8, command, "block-bench")) {
        try runBlockBench(allocator, io, writer, forced_backend);
    } else if (std.mem.eql(u8, command, "bench")) {
        try printEnv(host, writer);
        try writer.writeAll("\n");
        try printGpu(writer);
        try writer.writeAll("\n");
        try runHipBench(io, writer);
    } else if (std.mem.eql(u8, command, "all")) {
        try printEnv(host, writer);
        try writer.writeAll("\n");
        try printCaps(writer, forced_backend);
        try writer.writeAll("\n");
        try printGpu(writer);
    } else {
        try writer.print("unknown command: {s}\n\n", .{command});
        try writer.writeAll(usage);
        try writer.flush();
        std.process.exit(2);
    }

    try writer.flush();
}

fn printEnv(host: zynfer.util.Host, writer: *std.Io.Writer) !void {
    var report = try zynfer.env.collect(host);
    defer report.deinit();
    try zynfer.env.print(writer, report);
}

fn printGpu(writer: *std.Io.Writer) !void {
    try writer.print("zynfer HIP GPU report\n", .{});
    try writer.print("=====================\n\n", .{});
    try zynfer.hip.printDevices(writer);
}

fn printBackends(writer: *std.Io.Writer) !void {
    try writer.print("selectable backends\n", .{});
    try writer.print("-------------------\n", .{});
    for ([3]zynfer.BackendKind{ .cpu, .apple, .amd_hip }) |kind| {
        const status: []const u8 = if (zynfer.backend.isBackendBuildable(kind)) "buildable" else "not compiled";
        try writer.print("  {s: <10} {s}\n", .{ kind.name(), status });
    }
    try writer.print("\nDefault kind on this host: {s}\n", .{zynfer.backend.defaultKind().name()});
}

fn resolveKind(forced: ?[]const u8) !zynfer.BackendKind {
    if (forced) |name| {
        const kind = zynfer.backend.parseBackendKind(name) catch {
            std.debug.print("unknown backend '{s}'. use cpu, apple, or amd-hip.\n", .{name});
            std.process.exit(2);
        };
        zynfer.backend.requireBackend(kind) catch {
            std.debug.print("backend '{s}' is not available in this build.\n", .{kind.name()});
            std.process.exit(2);
        };
        return kind;
    }
    return zynfer.backend.defaultKind();
}

fn rejectForcedExperimentalPaths() void {
    if (zynfer.cpu.sme.forceRequested()) {
        std.debug.print(
            "ZYNFER_FORCE_SME requested but SME kernels are not retained (Stage 7: no Zig/Clang SME path).\n",
            .{},
        );
        std.process.exit(2);
    }
    if (zynfer.apple.coreml.forceRequested()) {
        std.debug.print(
            "ZYNFER_FORCE_COREML requested but Core ML/ANE inference is not retained (Stage 7: no measured subgraph).\n",
            .{},
        );
        std.process.exit(2);
    }
}

fn printStage7(writer: *std.Io.Writer) !void {
    try writer.print("zynfer Stage 7 — SME / Core ML experiments\n", .{});
    try writer.print("==========================================\n\n", .{});

    const sme_p = zynfer.cpu.sme.probe();
    try writer.print("SME / SME2\n", .{});
    try writer.print("  FEAT_SME (sysctl):  {s}\n", .{yn(sme_p.feat_sme)});
    try writer.print("  FEAT_SME2 (sysctl): {s}\n", .{yn(sme_p.feat_sme2)});
    try writer.print("  Zig target sme:    {s}\n", .{yn(sme_p.target_sme)});
    try writer.print("  path retained:     {s}\n", .{yn(sme_p.path_retained)});
    try writer.print("  detail: {s}\n\n", .{sme_p.detail});

    const cm_p = zynfer.apple.coreml.probe();
    try writer.print("Core ML / ANE\n", .{});
    try writer.print("  framework linked:           {s}\n", .{yn(cm_p.framework_linked)});
    try writer.print("  MLModelConfiguration ok:    {s}\n", .{yn(cm_p.configuration_ok)});
    try writer.print("  computeUnits All ok:        {s}\n", .{yn(cm_p.compute_units_all_ok)});
    try writer.print("  computeUnits CPU+ANE ok:    {s}\n", .{yn(cm_p.compute_units_cpu_and_ane_ok)});
    try writer.print("  ANE execution verified:     {s}\n", .{yn(cm_p.ane_execution_verified)});
    try writer.print("  path retained:              {s}\n", .{yn(cm_p.path_retained)});
    try writer.print("  detail: {s}\n\n", .{cm_p.detail});

    try writer.print("Accelerate (public CPU matrix path; AMX may be internal only)\n", .{});
    try writer.print("  have_accelerate: {s}\n", .{yn(zynfer.cpu.accelerate.have_accelerate)});
    try writer.print("  matmul gate M*N*K>={d}; matvec M*K>={d}\n\n", .{
        zynfer.cpu.accelerate.matmul_min_flops,
        zynfer.cpu.accelerate.matvec_min_flops,
    });

    try writer.print("Stage 7 decisions\n", .{});
    try writer.print("  SME kernels:     REJECT — detection only; no brittle assembly\n", .{});
    try writer.print("  Core ML/ANE ops: REJECT — framework probe only; no end-to-end subgraph\n", .{});
    try writer.print("  Accelerate:      RETAIN — measured Stage 5 size-gated vDSP path\n", .{});
    try writer.print("  Metal Stage 6:   RETAIN — default tiny-block schedule\n", .{});
    try writer.print("\nSee bench/results/apple-stage7-dev-laptop.md\n", .{});
}

fn printStage8(writer: *std.Io.Writer) !void {
    try writer.print("zynfer Stage 8 — hardening + Stage 6 leftovers\n", .{});
    try writer.print("==============================================\n\n", .{});

    try writer.print("Done in Stage 8\n", .{});
    try writer.print("  attention kv_len cap:     {d} (was 64; thread-local scores)\n", .{zynfer.apple.ops.max_attention_kv});
    try writer.print("  fused vs baseline A/B:    retained (Stage 6 test)\n", .{});
    try writer.print("  signposts:                ZYNFER_SIGNPOSTS=1 (prefill/decode/weights_upload + encode/batch)\n", .{});
    try writer.print("  peak_rss_bytes:           block-bench JSON + docs/benchmarks.md matrix Peak memory\n", .{});
    try writer.print("  energy_per_token:         null (not measured)\n", .{});
    try writer.print("  stress tests:             Session init×3 + full max_seq; batch abort; dual-Gpu concurrency\n", .{});
    try writer.print("  fp16/bf16 Metal:          Unsupported stubs (matmulF16/matvecF16)\n\n", .{});

    try writer.print("Rejected / deferred with reasons\n", .{});
    try writer.print("  ICB / encode-once replay: REJECT — KV/q_len change every decode step;\n", .{});
    try writer.print("                            Stage 6 already collapsed waits; re-encode is cheap\n", .{});
    try writer.print("  Extra MSL fusions:        REJECT for tiny-block — add_rmsnorm did not beat\n", .{});
    try writer.print("                            unfused Stage 6 batching in ns; revisit at Qwen scale\n", .{});
    try writer.print("                            (master Stage 16)\n", .{});
    try writer.print("  Int8 tiny-block Session:  REJECT — ops Q8DeviceWeights retained; Session stays\n", .{});
    try writer.print("                            f32 until realistic shapes (Stages 11/16)\n", .{});
    try writer.print("  TTFT / tok/s:             N/A until Stages 10–12\n\n", .{});

    try writer.print("Retained paths\n", .{});
    try writer.print("  Metal Stage 6 path={s}\n", .{zynfer.apple.block.path_staged});
    try writer.print("  Baseline A/B path={s}\n", .{zynfer.apple.block.path_baseline});
    try writer.print("  Accelerate size-gated vDSP (Stage 5)\n", .{});
    try writer.print("  SME/Core ML inference: rejected (Stage 7)\n", .{});
    try writer.print("\nSee bench/results/apple-stage8-dev-laptop.md\n", .{});
}

fn printStage10(writer: *std.Io.Writer) !void {
    try writer.print("zynfer Stage 10 — checkpoint inspection + .zynfer artifact\n", .{});
    try writer.print("=========================================================\n\n", .{});
    try writer.print("Done\n", .{});
    try writer.print("  format:           magic ZYNF v{d}, little-endian, 64-byte payload align\n", .{zynfer.artifact.format_version});
    try writer.print("  meta:             Qwen3-0.6B dims (HF config) in binary Meta\n", .{});
    try writer.print("  integrity:        SHA-256 over file with checksum field zeroed\n", .{});
    try writer.print("  load:             mmap (posix) with heap fallback; hot path findById\n", .{});
    try writer.print("  Zig API:          artifact.build / validate / Artifact.load*\n", .{});
    try writer.print("  CLI:              inspect PATH; artifact-compile --out PATH\n", .{});
    try writer.print("  converter:        tools/checkpoint/safetensors_to_zynfer.py (single or shards)\n\n", .{});
    try writer.print("Not in Stage 10\n", .{});
    try writer.print("  full Qwen weight conversion in CI (needs HF download)\n", .{});
    try writer.print("  forward pass / logits — Stage 11\n", .{});
    try writer.print("  tokenizer / sampling / TTFT — Stage 12\n\n", .{});
    try writer.print("See docs/artifact-format.md and bench/results/stage10-dev-laptop.md\n", .{});
}

fn runInspect(allocator: std.mem.Allocator, io: std.Io, writer: *std.Io.Writer, path: []const u8) !void {
    var art = zynfer.artifact.Artifact.loadFile(allocator, io, path) catch |err| {
        std.debug.print("inspect failed ({s}): {s}\n", .{ path, @errorName(err) });
        std.process.exit(2);
    };
    defer art.deinit();

    var hex_buf: [64]u8 = undefined;
    const hex = zynfer.artifact.formatSha256(&art.header.sha256, &hex_buf);

    try writer.print("zynfer artifact\n", .{});
    try writer.print("===============\n\n", .{});
    try writer.print("path:            {s}\n", .{path});
    try writer.print("format_version:  {d}\n", .{art.header.version});
    try writer.print("sha256:          {s}\n", .{hex});
    try writer.print("bytes:           {d}\n", .{art.bytes.len});
    try writer.print("payload_bytes:   {d}\n", .{art.header.payload_bytes});
    try writer.print("storage:         {s}\n", .{if (art.mapped != null) "mmap" else "heap"});
    try writer.print("\nmodel_id:        {s}\n", .{art.meta.modelIdSlice()});
    try writer.print("vocab_size:      {d}\n", .{art.meta.vocab_size});
    try writer.print("hidden_size:     {d}\n", .{art.meta.hidden_size});
    try writer.print("intermediate:    {d}\n", .{art.meta.intermediate_size});
    try writer.print("layers:          {d}\n", .{art.meta.num_layers});
    try writer.print("heads / kv:      {d} / {d}\n", .{ art.meta.num_attention_heads, art.meta.num_key_value_heads });
    try writer.print("head_dim:        {d}\n", .{art.meta.head_dim});
    try writer.print("max_position:    {d}\n", .{art.meta.max_position_embeddings});
    try writer.print("rope_theta:      {d}\n", .{art.meta.rope_theta});
    try writer.print("rms_norm_eps:    {e}\n", .{art.meta.rms_norm_eps});
    try writer.print("tie_embeddings:  {d}\n", .{art.meta.tie_word_embeddings});
    try writer.print("\ntensors ({d}):\n", .{art.entries.len});
    for (art.entries) |e| {
        try writer.print("  - {s}  id={d}  dtype={s}  rank={d}  shape=[", .{
            e.nameSlice(),
            e.tensor_id,
            (try e.dtypeTag()).name(),
            e.rank,
        });
        var i: u8 = 0;
        while (i < e.rank) : (i += 1) {
            if (i != 0) try writer.writeAll(",");
            try writer.print("{d}", .{e.shape[i]});
        }
        try writer.print("]  nbytes={d}\n", .{e.nbytes});
    }
}

fn runArtifactCompile(allocator: std.mem.Allocator, io: std.Io, writer: *std.Io.Writer, path: []const u8) !void {
    const bytes = try zynfer.artifact.buildStage10Fixture(allocator);
    defer allocator.free(bytes);

    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes });

    var hex_buf: [64]u8 = undefined;
    const v = try zynfer.artifact.validate(bytes);
    const hex = zynfer.artifact.formatSha256(&v.header.sha256, &hex_buf);
    try writer.print("wrote {s} ({d} bytes, sha256={s}, tensors={d})\n", .{
        path,
        bytes.len,
        hex,
        v.header.tensor_count,
    });
}

fn printCaps(writer: *std.Io.Writer, forced: ?[]const u8) !void {
    const kind = try resolveKind(forced);
    try writer.print("zynfer capabilities\n", .{});
    try writer.print("===================\n\n", .{});
    try writer.print("requested backend: {s}\n", .{kind.name()});

    const caps = switch (kind) {
        .cpu => zynfer.backend.cpuCapabilities(),
        .apple => zynfer.apple.gpu.capabilities(),
        .amd_hip => blk: {
            var c = zynfer.backend.cpuCapabilities();
            c.backend = .amd_hip;
            c.hip = zynfer.hip.have_hip;
            c.addDisabled("HIP backend is device enumeration only; transformer ops are not implemented here yet");
            break :blk c;
        },
    };

    try writer.print("device architecture: {s}\n", .{caps.arch.name()});
    try writer.print("unified memory: {s}\n", .{if (caps.unified_memory) "yes" else "no"});
    try writer.print("fp32: {s}  fp16: {s}  bf16: {s}\n", .{
        yn(caps.fp32),
        yn(caps.fp16),
        yn(caps.bf16),
    });
    try writer.print("simdgroup_matrix hardware: {s}\n", .{yn(caps.simdgroup_matrix)});
    try writer.print("Accelerate path: {s}\n", .{yn(caps.accelerate)});
    try writer.print("SME inference path: {s}\n", .{yn(caps.sme)});
    try writer.print("Core ML inference path: {s}\n", .{yn(caps.core_ml)});
    try writer.print("HIP linked: {s}\n", .{yn(caps.hip or zynfer.hip.have_hip)});

    const sme_p = zynfer.cpu.sme.probe();
    const cm_p = zynfer.apple.coreml.probe();
    try writer.print("\nStage 7 probes (hardware/framework ≠ retained path)\n", .{});
    try writer.print("  SME hardware FEAT_SME/SME2: {s}/{s}\n", .{ yn(sme_p.feat_sme), yn(sme_p.feat_sme2) });
    try writer.print("  Core ML framework linked:   {s}\n", .{yn(cm_p.framework_linked)});
    try writer.print("  ANE execution verified:     {s}\n", .{yn(cm_p.ane_execution_verified)});

    switch (caps.arch) {
        .apple_m => |feat| {
            try writer.print("\nApple Metal device (label only; not used for kernel correctness)\n", .{});
            try writer.print("  name: {s}\n", .{feat.nameSlice()});
            try writer.print("  recommended working set: {d} bytes\n", .{feat.recommended_working_set_bytes});
            try writer.print("  max buffer: {d} bytes\n", .{feat.max_buffer_bytes});
            try writer.print("  max threads/threadgroup: {d}\n", .{feat.max_threads_per_threadgroup});
            try writer.print("  GPU family Apple7/8/9: {s}/{s}/{s}\n", .{
                yn(feat.gpu_family_apple7),
                yn(feat.gpu_family_apple8),
                yn(feat.gpu_family_apple9),
            });
            try writer.print("  chosen kernels: naive f32 Metal + gated matmul_f32_simdgroup (M*N*K>={d}) + forceable matmul_f32_simdgroup_x4 + matvec/matmul_q8_f32; attention kv_len<={d}\n", .{
                zynfer.apple.ops.simdgroup_min_flops,
                zynfer.apple.ops.max_attention_kv,
            });
            const auto64: []const u8 = if (feat.simdgroup_matrix_available) "matmul_f32_simdgroup" else "matmul_f32";
            const auto256: []const u8 = if (feat.simdgroup_matrix_available) "matmul_f32_simdgroup" else "matmul_f32";
            try writer.print("  matmul auto-path (64^3 / 256^3): {s} / {s}  (x4 measured slower at 256^3; force with ZYNFER_MATMUL_PATH=simdgroup_x4)\n", .{ auto64, auto256 });
            try writer.print("  packed q8 GEMM/GEMV: explicit API; persistent via Q8DeviceWeights (fair benches; not auto over f32)\n", .{});
            try writer.print("  Accelerate CPU: vDSP matmul M*N*K>={d}; matvec M*K>={d} (do not claim AMX)\n", .{
                zynfer.cpu.accelerate.matmul_min_flops,
                zynfer.cpu.accelerate.matvec_min_flops,
            });
            try writer.print("  Stage 6 tiny-block path={s}: one CB/wait + resident KV + add_rmsnorm\n", .{zynfer.apple.block.path_staged});
            try writer.print("  A/B: ZYNFER_APPLE_BLOCK=baseline → path={s} (per-op waits)\n", .{zynfer.apple.block.path_baseline});
            try writer.print("  Stage 7: SME/Core ML inference paths rejected; see `zynfer stage7`\n", .{});
            try writer.print("  Stage 8: kv_len<={d}; signposts via ZYNFER_SIGNPOSTS=1; see `zynfer stage8`\n", .{zynfer.apple.ops.max_attention_kv});
        },
        else => {},
    }

    try writer.print("\ndisabled paths\n", .{});
    var i: usize = 0;
    while (i < caps.disabled_len) : (i += 1) {
        try writer.print("  - {s}\n", .{caps.disabled[i]});
    }
}

fn yn(v: bool) []const u8 {
    return if (v) "yes" else "no";
}

fn runOpsBench(gpa: std.mem.Allocator, io: std.Io, writer: *std.Io.Writer, forced: ?[]const u8) !void {
    const kind = try resolveKind(forced);
    try writer.print("zynfer ops-bench\n", .{});
    try writer.print("================\n", .{});
    try writer.print("backend={s}  (cpu always runs as oracle)\n", .{kind.name()});
    try writer.print("note: Apple times include per-op shared-buffer fill + encode_and_wait.\n", .{});
    try writer.print("      That is the current baseline, not a fused production decode path.\n", .{});
    try writer.print("      Fair q8 rows pack once outside the timed loop (path field in JSON).\n\n", .{});

    const warmup = 2;
    const iters = 8;
    var metal_init_ns: ?u64 = null;
    var gpu_storage: zynfer.apple.gpu.Gpu = undefined;
    var gpu_ptr: ?*zynfer.apple.gpu.Gpu = null;
    if (kind == .apple and zynfer.apple.gpu.have_apple) {
        const t0 = std.Io.Clock.awake.now(io);
        gpu_storage = zynfer.apple.gpu.Gpu.init() catch |err| {
            try writer.print("Apple Metal init failed: {s}\n", .{@errorName(err)});
            try writer.print("If shaders fail to compile, install the Metal Toolchain:\n", .{});
            try writer.print("  xcodebuild -downloadComponent MetalToolchain\n", .{});
            return;
        };
        gpu_ptr = &gpu_storage;
        const t1 = std.Io.Clock.awake.now(io);
        metal_init_ns = @intCast(@max(@as(i96, 0), t1.nanoseconds - t0.nanoseconds));
        try writer.print("metal_device_create_plus_shader_compile_ns={d}\n\n", .{metal_init_ns.?});
    }
    defer if (gpu_ptr) |g| g.deinit();

    var rows: [18]BenchRow = undefined;
    rows[0] = try benchNamed(gpa, io, writer, gpu_ptr, "add_f32_4096", benchAdd, warmup, iters, "add_f32");
    rows[1] = try benchNamed(gpa, io, writer, gpu_ptr, "silu_mul_f32_4096", benchSiluMul, warmup, iters, "silu_mul_f32");
    rows[2] = try benchNamed(gpa, io, writer, gpu_ptr, "matvec_f32_256x256", benchMatvec, warmup, iters, "matvec_f32");
    rows[3] = try benchNamed(gpa, io, writer, gpu_ptr, "matmul_f32_32x64x64", benchMatmul, warmup, iters, "matmul_auto");
    rows[4] = try benchNamed(gpa, io, writer, gpu_ptr, "matmul_f32_naive_64x64x64", benchMatmulNaive64, warmup, iters, "matmul_f32");
    rows[5] = try benchNamed(gpa, io, writer, gpu_ptr, "matmul_f32_simdgroup_64x64x64", benchMatmulSimd64, warmup, iters, "matmul_f32_simdgroup");
    rows[6] = try benchNamed(gpa, io, writer, gpu_ptr, "matmul_f32_naive_256x256x256", benchMatmulNaive256, warmup, iters, "matmul_f32");
    rows[7] = try benchNamed(gpa, io, writer, gpu_ptr, "matmul_f32_simdgroup_256x256x256", benchMatmulSimd256, warmup, iters, "matmul_f32_simdgroup");
    rows[8] = try benchNamed(gpa, io, writer, gpu_ptr, "matmul_f32_simdgroup_x4_256x256x256", benchMatmulSimdX4_256, warmup, iters, "matmul_f32_simdgroup_x4");
    rows[9] = try benchNamed(gpa, io, writer, gpu_ptr, "matmul_f32_auto_256x256x256", benchMatmulAuto256, warmup, iters, "matmul_auto");
    rows[10] = try benchNamed(gpa, io, writer, null, "matmul_accelerate_64x64x64", benchMatmulAccelerate64, warmup, iters, "accelerate_vDSP_mmul");
    rows[11] = try benchNamed(gpa, io, writer, null, "matvec_accelerate_256x256", benchMatvecAccelerate256, warmup, iters, "accelerate_vDSP_matvec");
    rows[12] = try benchFairQ8Matvec(gpa, io, writer, gpu_ptr, warmup, iters);
    rows[13] = try benchPersistentQ8Matvec(gpa, io, writer, gpu_ptr, warmup, iters);
    rows[14] = try benchFairQ8Matmul(gpa, io, writer, gpu_ptr, warmup, iters);
    rows[15] = try benchPersistentQ8Matmul(gpa, io, writer, gpu_ptr, warmup, iters);
    rows[16] = try benchNamed(gpa, io, writer, gpu_ptr, "matvec_f32_256x256_ref", benchMatvec, warmup, iters, "matvec_f32");
    rows[17] = try benchNamed(gpa, io, writer, gpu_ptr, "matmul_f32_128x128x128_ref", benchMatmul128, warmup, iters, "matmul_auto");

    try writer.print("\njson\n", .{});
    try writer.print("{{\"backend\":\"{s}\",\"zig\":\"{s}\",\"warmup\":{d},\"iters\":{d}", .{
        kind.name(),
        @import("builtin").zig_version_string,
        warmup,
        iters,
    });
    if (metal_init_ns) |ns| {
        try writer.print(",\"metal_init_ns\":{d}", .{ns});
    } else {
        try writer.print(",\"metal_init_ns\":null", .{});
    }
    try writer.print(",\"ops\":[", .{});
    for (rows, 0..) |row, i| {
        if (i != 0) try writer.print(",", .{});
        try writer.print("{{\"name\":\"{s}\",\"path\":\"{s}\",\"cpu_ns\":{d},", .{ row.name, row.path, row.cpu_ns });
        if (row.apple_ns) |ns| {
            try writer.print("\"apple_metal_ns\":{d}}}", .{ns});
        } else {
            try writer.print("\"apple_metal_ns\":null}}", .{});
        }
    }
    try writer.print("]}}\n", .{});
}

const BenchRow = struct {
    name: []const u8,
    path: []const u8,
    cpu_ns: u64,
    apple_ns: ?u64,
};

const BenchFn = *const fn (gpa: std.mem.Allocator, gpu: ?*zynfer.apple.gpu.Gpu) anyerror!void;

fn benchNamed(
    gpa: std.mem.Allocator,
    io: std.Io,
    writer: *std.Io.Writer,
    gpu: ?*zynfer.apple.gpu.Gpu,
    name: []const u8,
    func: BenchFn,
    warmup: usize,
    iters: usize,
    path: []const u8,
) !BenchRow {
    var i: usize = 0;
    while (i < warmup) : (i += 1) try func(gpa, null);
    const cpu_ns = try timeIters(io, iters, func, gpa, null);
    try writer.print("{s} path={s} cpu_ns={d} iters={d}\n", .{ name, path, cpu_ns / iters, iters });

    var apple_ns: ?u64 = null;
    if (gpu) |g| {
        i = 0;
        while (i < warmup) : (i += 1) try func(gpa, g);
        const total = try timeIters(io, iters, func, gpa, g);
        apple_ns = total / iters;
        try writer.print("{s} path={s} apple_metal_ns={d} iters={d}\n", .{ name, path, apple_ns.?, iters });
    } else {
        try writer.print("{s} path={s} apple_metal_ns=N/A\n", .{ name, path });
    }
    return .{ .name = name, .path = path, .cpu_ns = cpu_ns / iters, .apple_ns = apple_ns };
}

fn timeIters(io: std.Io, iters: usize, func: BenchFn, gpa: std.mem.Allocator, gpu: ?*zynfer.apple.gpu.Gpu) !u64 {
    const start = std.Io.Clock.awake.now(io);
    var i: usize = 0;
    while (i < iters) : (i += 1) try func(gpa, gpu);
    const end = std.Io.Clock.awake.now(io);
    return @intCast(@max(@as(i96, 0), end.nanoseconds - start.nanoseconds));
}

fn benchAdd(gpa: std.mem.Allocator, gpu: ?*zynfer.apple.gpu.Gpu) !void {
    var a = try zynfer.Tensor.alloc(gpa, .f32, &.{4096});
    defer a.deinit();
    var b = try zynfer.Tensor.alloc(gpa, .f32, &.{4096});
    defer b.deinit();
    var o = try zynfer.Tensor.alloc(gpa, .f32, &.{4096});
    defer o.deinit();
    try a.fillF32(1);
    try b.fillF32(2);
    if (gpu) |g| {
        try zynfer.apple.ops.add(g, o, a, b);
    } else {
        try zynfer.cpu.ops.add(o, a, b);
    }
}

fn benchSiluMul(gpa: std.mem.Allocator, gpu: ?*zynfer.apple.gpu.Gpu) !void {
    var a = try zynfer.Tensor.alloc(gpa, .f32, &.{4096});
    defer a.deinit();
    var b = try zynfer.Tensor.alloc(gpa, .f32, &.{4096});
    defer b.deinit();
    var o = try zynfer.Tensor.alloc(gpa, .f32, &.{4096});
    defer o.deinit();
    try a.fillF32(0.5);
    try b.fillF32(1.5);
    if (gpu) |g| {
        try zynfer.apple.ops.siluMul(g, o, a, b);
    } else {
        try zynfer.cpu.ops.siluMul(o, a, b);
    }
}

fn benchMatvec(gpa: std.mem.Allocator, gpu: ?*zynfer.apple.gpu.Gpu) !void {
    var a = try zynfer.Tensor.alloc(gpa, .f32, &.{ 256, 256 });
    defer a.deinit();
    var x = try zynfer.Tensor.alloc(gpa, .f32, &.{256});
    defer x.deinit();
    var y = try zynfer.Tensor.alloc(gpa, .f32, &.{256});
    defer y.deinit();
    try a.fillF32(0.01);
    try x.fillF32(0.02);
    if (gpu) |g| {
        try zynfer.apple.ops.matvec(g, y, a, x);
    } else {
        try zynfer.cpu.ops.matvec(y, a, x);
    }
}

fn benchMatmul(gpa: std.mem.Allocator, gpu: ?*zynfer.apple.gpu.Gpu) !void {
    var a = try zynfer.Tensor.alloc(gpa, .f32, &.{ 32, 64 });
    defer a.deinit();
    var b = try zynfer.Tensor.alloc(gpa, .f32, &.{ 64, 64 });
    defer b.deinit();
    var c = try zynfer.Tensor.alloc(gpa, .f32, &.{ 32, 64 });
    defer c.deinit();
    try a.fillF32(0.01);
    try b.fillF32(0.02);
    if (gpu) |g| {
        try zynfer.apple.ops.matmul(g, c, a, b);
    } else {
        try zynfer.cpu.ops.matmul(c, a, b);
    }
}

fn benchMatmulNaive64(gpa: std.mem.Allocator, gpu: ?*zynfer.apple.gpu.Gpu) !void {
    try benchMatmulPathSized(gpa, gpu, .naive, 64);
}

fn benchMatmulSimd64(gpa: std.mem.Allocator, gpu: ?*zynfer.apple.gpu.Gpu) !void {
    try benchMatmulPathSized(gpa, gpu, .simdgroup, 64);
}

fn benchMatmulNaive256(gpa: std.mem.Allocator, gpu: ?*zynfer.apple.gpu.Gpu) !void {
    try benchMatmulPathSized(gpa, gpu, .naive, 256);
}

fn benchMatmulSimd256(gpa: std.mem.Allocator, gpu: ?*zynfer.apple.gpu.Gpu) !void {
    try benchMatmulPathSized(gpa, gpu, .simdgroup, 256);
}

fn benchMatmulSimdX4_256(gpa: std.mem.Allocator, gpu: ?*zynfer.apple.gpu.Gpu) !void {
    try benchMatmulPathSized(gpa, gpu, .simdgroup_x4, 256);
}

fn benchMatmulAuto256(gpa: std.mem.Allocator, gpu: ?*zynfer.apple.gpu.Gpu) !void {
    var a = try zynfer.Tensor.alloc(gpa, .f32, &.{ 256, 256 });
    defer a.deinit();
    var b = try zynfer.Tensor.alloc(gpa, .f32, &.{ 256, 256 });
    defer b.deinit();
    var c = try zynfer.Tensor.alloc(gpa, .f32, &.{ 256, 256 });
    defer c.deinit();
    try a.fillF32(0.01);
    try b.fillF32(0.02);
    if (gpu) |g| {
        try zynfer.apple.ops.matmul(g, c, a, b);
    } else {
        try zynfer.cpu.ops.matmul(c, a, b);
    }
}

fn benchMatmul128(gpa: std.mem.Allocator, gpu: ?*zynfer.apple.gpu.Gpu) !void {
    var a = try zynfer.Tensor.alloc(gpa, .f32, &.{ 128, 128 });
    defer a.deinit();
    var b = try zynfer.Tensor.alloc(gpa, .f32, &.{ 128, 128 });
    defer b.deinit();
    var c = try zynfer.Tensor.alloc(gpa, .f32, &.{ 128, 128 });
    defer c.deinit();
    try a.fillF32(0.01);
    try b.fillF32(0.02);
    if (gpu) |g| {
        try zynfer.apple.ops.matmul(g, c, a, b);
    } else {
        try zynfer.cpu.ops.matmul(c, a, b);
    }
}

fn benchMatmulPathSized(gpa: std.mem.Allocator, gpu: ?*zynfer.apple.gpu.Gpu, path: zynfer.apple.ops.MatmulPath, dim: usize) !void {
    var a = try zynfer.Tensor.alloc(gpa, .f32, &.{ dim, dim });
    defer a.deinit();
    var b = try zynfer.Tensor.alloc(gpa, .f32, &.{ dim, dim });
    defer b.deinit();
    var c = try zynfer.Tensor.alloc(gpa, .f32, &.{ dim, dim });
    defer c.deinit();
    try a.fillF32(0.01);
    try b.fillF32(0.02);
    if (gpu) |g| {
        if ((path == .simdgroup or path == .simdgroup_x4) and !g.features.simdgroup_matrix_available) {
            try zynfer.apple.ops.matmulPath(g, c, a, b, .naive);
            return;
        }
        try zynfer.apple.ops.matmulPath(g, c, a, b, path);
    } else {
        try zynfer.cpu.ops.matmul(c, a, b);
    }
}

fn benchMatmulAccelerate64(gpa: std.mem.Allocator, gpu: ?*zynfer.apple.gpu.Gpu) !void {
    _ = gpu;
    var a = try zynfer.Tensor.alloc(gpa, .f32, &.{ 64, 64 });
    defer a.deinit();
    var b = try zynfer.Tensor.alloc(gpa, .f32, &.{ 64, 64 });
    defer b.deinit();
    var c = try zynfer.Tensor.alloc(gpa, .f32, &.{ 64, 64 });
    defer c.deinit();
    try a.fillF32(0.01);
    try b.fillF32(0.02);
    if (zynfer.cpu.accelerate.have_accelerate) {
        try zynfer.cpu.accelerate.matmul(c, a, b);
    } else {
        try zynfer.cpu.ops.matmul(c, a, b);
    }
}

fn benchMatvecAccelerate256(gpa: std.mem.Allocator, gpu: ?*zynfer.apple.gpu.Gpu) !void {
    _ = gpu;
    var a = try zynfer.Tensor.alloc(gpa, .f32, &.{ 256, 256 });
    defer a.deinit();
    var x = try zynfer.Tensor.alloc(gpa, .f32, &.{256});
    defer x.deinit();
    var y = try zynfer.Tensor.alloc(gpa, .f32, &.{256});
    defer y.deinit();
    try a.fillF32(0.01);
    try x.fillF32(0.02);
    if (zynfer.cpu.accelerate.have_accelerate) {
        try zynfer.cpu.accelerate.matvec(y, a, x);
    } else {
        try zynfer.cpu.ops.matvec(y, a, x);
    }
}

/// Fair int8 GEMV: pack once outside the timed loop; only matvec is measured.
fn benchFairQ8Matvec(
    gpa: std.mem.Allocator,
    io: std.Io,
    writer: *std.Io.Writer,
    gpu: ?*zynfer.apple.gpu.Gpu,
    warmup: usize,
    iters: usize,
) !BenchRow {
    const name = "matvec_q8_f32_256x256_prepacked";
    const path = "matvec_q8_f32_per_row";
    const m: usize = 256;
    const k: usize = 256;
    var w = try zynfer.Tensor.alloc(gpa, .f32, &.{ m, k });
    defer w.deinit();
    var x = try zynfer.Tensor.alloc(gpa, .f32, &.{k});
    defer x.deinit();
    var y = try zynfer.Tensor.alloc(gpa, .f32, &.{m});
    defer y.deinit();
    try w.fillF32(0.01);
    try x.fillF32(0.02);
    const q = try gpa.alloc(i8, m * k);
    defer gpa.free(q);
    const scale = try gpa.alloc(f32, m);
    defer gpa.free(scale);
    try zynfer.cpu.ops.packRowQ8(try w.f32s(), m, k, q, scale);

    var i: usize = 0;
    while (i < warmup) : (i += 1) {
        try zynfer.cpu.ops.matvecQ8(try y.f32s(), q, scale, try x.f32s(), m, k, .per_row);
    }
    const t0 = std.Io.Clock.awake.now(io);
    i = 0;
    while (i < iters) : (i += 1) {
        try zynfer.cpu.ops.matvecQ8(try y.f32s(), q, scale, try x.f32s(), m, k, .per_row);
    }
    const t1 = std.Io.Clock.awake.now(io);
    const cpu_ns: u64 = @intCast(@max(@as(i96, 0), t1.nanoseconds - t0.nanoseconds));
    try writer.print("{s} path={s} cpu_ns={d} iters={d} (pack excluded)\n", .{ name, path, cpu_ns / iters, iters });

    var apple_ns: ?u64 = null;
    if (gpu) |g| {
        i = 0;
        while (i < warmup) : (i += 1) {
            try zynfer.apple.ops.matvecQ8(g, y, q, scale, x, .per_row);
        }
        const a0 = std.Io.Clock.awake.now(io);
        i = 0;
        while (i < iters) : (i += 1) {
            try zynfer.apple.ops.matvecQ8(g, y, q, scale, x, .per_row);
        }
        const a1 = std.Io.Clock.awake.now(io);
        apple_ns = @as(u64, @intCast(@max(@as(i96, 0), a1.nanoseconds - a0.nanoseconds))) / iters;
        try writer.print("{s} path={s} apple_metal_ns={d} iters={d} (pack excluded)\n", .{ name, path, apple_ns.?, iters });
    } else {
        try writer.print("{s} path={s} apple_metal_ns=N/A\n", .{ name, path });
    }
    return .{ .name = name, .path = path, .cpu_ns = cpu_ns / iters, .apple_ns = apple_ns };
}

/// Fair int8 GEMM: pack once; compare to f32 ref row at 128³.
fn benchFairQ8Matmul(
    gpa: std.mem.Allocator,
    io: std.Io,
    writer: *std.Io.Writer,
    gpu: ?*zynfer.apple.gpu.Gpu,
    warmup: usize,
    iters: usize,
) !BenchRow {
    const name = "matmul_q8_f32_128x128x128_prepacked";
    const path = "matmul_q8_f32_per_row";
    const m: usize = 128;
    const k: usize = 128;
    const n: usize = 128;
    var w = try zynfer.Tensor.alloc(gpa, .f32, &.{ m, k });
    defer w.deinit();
    var b = try zynfer.Tensor.alloc(gpa, .f32, &.{ k, n });
    defer b.deinit();
    var c = try zynfer.Tensor.alloc(gpa, .f32, &.{ m, n });
    defer c.deinit();
    try w.fillF32(0.01);
    try b.fillF32(0.02);
    const q = try gpa.alloc(i8, m * k);
    defer gpa.free(q);
    const scale = try gpa.alloc(f32, m);
    defer gpa.free(scale);
    try zynfer.cpu.ops.packRowQ8(try w.f32s(), m, k, q, scale);

    var i: usize = 0;
    while (i < warmup) : (i += 1) {
        try zynfer.cpu.ops.matmulQ8(try c.f32s(), q, scale, try b.f32s(), m, n, k, .per_row);
    }
    const t0 = std.Io.Clock.awake.now(io);
    i = 0;
    while (i < iters) : (i += 1) {
        try zynfer.cpu.ops.matmulQ8(try c.f32s(), q, scale, try b.f32s(), m, n, k, .per_row);
    }
    const t1 = std.Io.Clock.awake.now(io);
    const cpu_ns: u64 = @intCast(@max(@as(i96, 0), t1.nanoseconds - t0.nanoseconds));
    try writer.print("{s} path={s} cpu_ns={d} iters={d} (pack excluded)\n", .{ name, path, cpu_ns / iters, iters });

    var apple_ns: ?u64 = null;
    if (gpu) |g| {
        i = 0;
        while (i < warmup) : (i += 1) {
            try zynfer.apple.ops.matmulQ8(g, c, q, scale, b, .per_row);
        }
        const a0 = std.Io.Clock.awake.now(io);
        i = 0;
        while (i < iters) : (i += 1) {
            try zynfer.apple.ops.matmulQ8(g, c, q, scale, b, .per_row);
        }
        const a1 = std.Io.Clock.awake.now(io);
        apple_ns = @as(u64, @intCast(@max(@as(i96, 0), a1.nanoseconds - a0.nanoseconds))) / iters;
        try writer.print("{s} path={s} apple_metal_ns={d} iters={d} (pack excluded)\n", .{ name, path, apple_ns.?, iters });
    } else {
        try writer.print("{s} path={s} apple_metal_ns=N/A\n", .{ name, path });
    }
    return .{ .name = name, .path = path, .cpu_ns = cpu_ns / iters, .apple_ns = apple_ns };
}

/// Persistent Metal int8 weights: upload once, time only activation traffic + kernel.
fn benchPersistentQ8Matvec(
    gpa: std.mem.Allocator,
    io: std.Io,
    writer: *std.Io.Writer,
    gpu: ?*zynfer.apple.gpu.Gpu,
    warmup: usize,
    iters: usize,
) !BenchRow {
    const name = "matvec_q8_f32_256x256_persistent";
    const path = "matvec_q8_f32_persistent_per_row";
    const m: usize = 256;
    const k: usize = 256;
    var w = try zynfer.Tensor.alloc(gpa, .f32, &.{ m, k });
    defer w.deinit();
    var x = try zynfer.Tensor.alloc(gpa, .f32, &.{k});
    defer x.deinit();
    var y = try zynfer.Tensor.alloc(gpa, .f32, &.{m});
    defer y.deinit();
    try w.fillF32(0.01);
    try x.fillF32(0.02);
    const q = try gpa.alloc(i8, m * k);
    defer gpa.free(q);
    const scale = try gpa.alloc(f32, m);
    defer gpa.free(scale);
    try zynfer.cpu.ops.packRowQ8(try w.f32s(), m, k, q, scale);

    var i: usize = 0;
    while (i < warmup) : (i += 1) {
        try zynfer.cpu.ops.matvecQ8(try y.f32s(), q, scale, try x.f32s(), m, k, .per_row);
    }
    const t0 = std.Io.Clock.awake.now(io);
    i = 0;
    while (i < iters) : (i += 1) {
        try zynfer.cpu.ops.matvecQ8(try y.f32s(), q, scale, try x.f32s(), m, k, .per_row);
    }
    const t1 = std.Io.Clock.awake.now(io);
    const cpu_ns: u64 = @intCast(@max(@as(i96, 0), t1.nanoseconds - t0.nanoseconds));
    try writer.print("{s} path={s} cpu_ns={d} iters={d} (pack excluded)\n", .{ name, path, cpu_ns / iters, iters });

    var apple_ns: ?u64 = null;
    if (gpu) |g| {
        var persisted = try zynfer.apple.ops.Q8DeviceWeights.upload(g, q, scale, m, k, .per_row);
        defer persisted.deinit();
        i = 0;
        while (i < warmup) : (i += 1) {
            try zynfer.apple.ops.matvecQ8Persistent(g, y, persisted, x);
        }
        const a0 = std.Io.Clock.awake.now(io);
        i = 0;
        while (i < iters) : (i += 1) {
            try zynfer.apple.ops.matvecQ8Persistent(g, y, persisted, x);
        }
        const a1 = std.Io.Clock.awake.now(io);
        apple_ns = @as(u64, @intCast(@max(@as(i96, 0), a1.nanoseconds - a0.nanoseconds))) / iters;
        try writer.print("{s} path={s} apple_metal_ns={d} iters={d} (weights resident)\n", .{ name, path, apple_ns.?, iters });
    } else {
        try writer.print("{s} path={s} apple_metal_ns=N/A\n", .{ name, path });
    }
    return .{ .name = name, .path = path, .cpu_ns = cpu_ns / iters, .apple_ns = apple_ns };
}

fn benchPersistentQ8Matmul(
    gpa: std.mem.Allocator,
    io: std.Io,
    writer: *std.Io.Writer,
    gpu: ?*zynfer.apple.gpu.Gpu,
    warmup: usize,
    iters: usize,
) !BenchRow {
    const name = "matmul_q8_f32_128x128x128_persistent";
    const path = "matmul_q8_f32_persistent_per_row";
    const m: usize = 128;
    const k: usize = 128;
    const n: usize = 128;
    var w = try zynfer.Tensor.alloc(gpa, .f32, &.{ m, k });
    defer w.deinit();
    var b = try zynfer.Tensor.alloc(gpa, .f32, &.{ k, n });
    defer b.deinit();
    var c = try zynfer.Tensor.alloc(gpa, .f32, &.{ m, n });
    defer c.deinit();
    try w.fillF32(0.01);
    try b.fillF32(0.02);
    const q = try gpa.alloc(i8, m * k);
    defer gpa.free(q);
    const scale = try gpa.alloc(f32, m);
    defer gpa.free(scale);
    try zynfer.cpu.ops.packRowQ8(try w.f32s(), m, k, q, scale);

    var i: usize = 0;
    while (i < warmup) : (i += 1) {
        try zynfer.cpu.ops.matmulQ8(try c.f32s(), q, scale, try b.f32s(), m, n, k, .per_row);
    }
    const t0 = std.Io.Clock.awake.now(io);
    i = 0;
    while (i < iters) : (i += 1) {
        try zynfer.cpu.ops.matmulQ8(try c.f32s(), q, scale, try b.f32s(), m, n, k, .per_row);
    }
    const t1 = std.Io.Clock.awake.now(io);
    const cpu_ns: u64 = @intCast(@max(@as(i96, 0), t1.nanoseconds - t0.nanoseconds));
    try writer.print("{s} path={s} cpu_ns={d} iters={d} (pack excluded)\n", .{ name, path, cpu_ns / iters, iters });

    var apple_ns: ?u64 = null;
    if (gpu) |g| {
        var persisted = try zynfer.apple.ops.Q8DeviceWeights.upload(g, q, scale, m, k, .per_row);
        defer persisted.deinit();
        i = 0;
        while (i < warmup) : (i += 1) {
            try zynfer.apple.ops.matmulQ8Persistent(g, c, persisted, b);
        }
        const a0 = std.Io.Clock.awake.now(io);
        i = 0;
        while (i < iters) : (i += 1) {
            try zynfer.apple.ops.matmulQ8Persistent(g, c, persisted, b);
        }
        const a1 = std.Io.Clock.awake.now(io);
        apple_ns = @as(u64, @intCast(@max(@as(i96, 0), a1.nanoseconds - a0.nanoseconds))) / iters;
        try writer.print("{s} path={s} apple_metal_ns={d} iters={d} (weights resident)\n", .{ name, path, apple_ns.?, iters });
    } else {
        try writer.print("{s} path={s} apple_metal_ns=N/A\n", .{ name, path });
    }
    return .{ .name = name, .path = path, .cpu_ns = cpu_ns / iters, .apple_ns = apple_ns };
}

fn runBlockBench(gpa: std.mem.Allocator, io: std.Io, writer: *std.Io.Writer, forced: ?[]const u8) !void {
    const kind = try resolveKind(forced);
    const spec = zynfer.tiny_block.fixture_spec;
    const prefill_tokens: usize = 8;
    const decode_steps: usize = 8;
    const warmup: usize = 1;
    const iters: usize = 4;

    try writer.print("zynfer block-bench\n", .{});
    try writer.print("==================\n", .{});
    try writer.print("backend={s}  fixture=tiny-block hidden={d} n_q={d} n_kv={d} head_dim={d}\n", .{
        kind.name(),
        spec.hidden,
        spec.n_q,
        spec.n_kv,
        spec.head_dim,
    });
    try writer.print("prefill_tokens={d} decode_steps={d} max_seq={d}\n", .{ prefill_tokens, decode_steps, spec.max_seq });
    try writer.print("note: Apple Stage 6 default path={s} (one CB/wait + resident KV + add_rmsnorm).\n", .{zynfer.apple.block.path_staged});
    try writer.print("      ZYNFER_APPLE_BLOCK=baseline → path={s} (per-op waits) for A/B.\n", .{zynfer.apple.block.path_baseline});
    try writer.print("      JSON fields: apple_block_path / apple_block_waits / apple_block_encodes / peak_rss_bytes.\n", .{});
    try writer.print("      Optional: ZYNFER_SIGNPOSTS=1 for Instruments (prefill/decode/weights_upload + encode/batch).\n", .{});
    try writer.print("      This is not Qwen3 and not a production decode path.\n\n", .{});

    var metal_init_ns: ?u64 = null;
    var gpu_storage: zynfer.apple.gpu.Gpu = undefined;
    var gpu_ptr: ?*zynfer.apple.gpu.Gpu = null;
    if (kind == .apple and zynfer.apple.gpu.have_apple) {
        const t0 = std.Io.Clock.awake.now(io);
        gpu_storage = zynfer.apple.gpu.Gpu.init() catch |err| {
            try writer.print("Apple Metal init failed: {s}\n", .{@errorName(err)});
            try writer.print("If shaders fail to compile, install the Metal Toolchain:\n", .{});
            try writer.print("  xcodebuild -downloadComponent MetalToolchain\n", .{});
            return;
        };
        gpu_ptr = &gpu_storage;
        const t1 = std.Io.Clock.awake.now(io);
        metal_init_ns = nsDelta(t0, t1);
        try writer.print("metal_device_create_plus_shader_compile_ns={d}\n\n", .{metal_init_ns.?});
    }
    defer if (gpu_ptr) |g| g.deinit();

    const cpu_times = try timeBlock(gpa, io, null, spec, prefill_tokens, decode_steps, warmup, iters);
    try writer.print("cpu prefill_ns={d} decode_ns_per_token={d} iters={d}\n", .{
        cpu_times.prefill_ns,
        cpu_times.decode_ns_per_token,
        iters,
    });

    var apple_times: ?BlockTimes = null;
    if (gpu_ptr) |g| {
        apple_times = try timeBlock(gpa, io, g, spec, prefill_tokens, decode_steps, warmup, iters);
        try writer.print("apple_metal prefill_ns={d} decode_ns_per_token={d} iters={d}\n", .{
            apple_times.?.prefill_ns,
            apple_times.?.decode_ns_per_token,
            iters,
        });
    } else {
        try writer.print("apple_metal prefill_ns=N/A decode_ns_per_token=N/A\n", .{});
    }

    try writer.print("\njson\n", .{});
    try writer.print("{{\"backend\":\"{s}\",\"fixture\":\"tiny-block\",\"hidden\":{d},\"prefill_tokens\":{d},\"decode_steps\":{d},\"warmup\":{d},\"iters\":{d}", .{
        kind.name(),
        spec.hidden,
        prefill_tokens,
        decode_steps,
        warmup,
        iters,
    });
    if (metal_init_ns) |ns| {
        try writer.print(",\"metal_init_ns\":{d}", .{ns});
    } else {
        try writer.print(",\"metal_init_ns\":null", .{});
    }
    if (zynfer.util.peakRssBytes()) |rss| {
        try writer.print(",\"peak_rss_bytes\":{d}", .{rss});
    } else {
        try writer.print(",\"peak_rss_bytes\":null", .{});
    }
    try writer.print(",\"energy_per_token\":null", .{});
    try writer.print(",\"cpu_prefill_ns\":{d},\"cpu_decode_ns_per_token\":{d}", .{
        cpu_times.prefill_ns,
        cpu_times.decode_ns_per_token,
    });
    if (apple_times) |t| {
        try writer.print(",\"apple_prefill_ns\":{d},\"apple_decode_ns_per_token\":{d},\"apple_block_path\":\"{s}\",\"apple_block_waits\":{d},\"apple_block_encodes\":{d}}}\n", .{
            t.prefill_ns,
            t.decode_ns_per_token,
            zynfer.apple.block.last_block_path,
            zynfer.apple.block.last_block_waits,
            zynfer.apple.block.last_block_encodes,
        });
    } else {
        try writer.print(",\"apple_prefill_ns\":null,\"apple_decode_ns_per_token\":null,\"apple_block_path\":null}}\n", .{});
    }
}

const BlockTimes = struct {
    prefill_ns: u64,
    decode_ns_per_token: u64,
};

fn nsDelta(start: std.Io.Timestamp, end: std.Io.Timestamp) u64 {
    return @intCast(@max(@as(i96, 0), end.nanoseconds - start.nanoseconds));
}

fn timeBlock(
    gpa: std.mem.Allocator,
    io: std.Io,
    gpu: ?*zynfer.apple.gpu.Gpu,
    spec: zynfer.tiny_block.Spec,
    prefill_tokens: usize,
    decode_steps: usize,
    warmup: usize,
    iters: usize,
) !BlockTimes {
    var i: usize = 0;
    while (i < warmup) : (i += 1) {
        _ = try runBlockOnce(gpa, io, gpu, spec, prefill_tokens, decode_steps);
    }

    var prefill_total: u64 = 0;
    var decode_total: u64 = 0;
    i = 0;
    while (i < iters) : (i += 1) {
        const sample = try runBlockOnce(gpa, io, gpu, spec, prefill_tokens, decode_steps);
        prefill_total += sample.prefill_ns;
        decode_total += sample.decode_ns;
    }
    return .{
        .prefill_ns = prefill_total / iters,
        .decode_ns_per_token = decode_total / (iters * decode_steps),
    };
}

fn runBlockOnce(
    gpa: std.mem.Allocator,
    io: std.Io,
    gpu: ?*zynfer.apple.gpu.Gpu,
    spec: zynfer.tiny_block.Spec,
    prefill_tokens: usize,
    decode_steps: usize,
) !struct { prefill_ns: u64, decode_ns: u64 } {
    var x_prefill = try zynfer.Tensor.alloc(gpa, .f32, &.{ prefill_tokens, spec.hidden });
    defer x_prefill.deinit();
    var y_prefill = try zynfer.Tensor.alloc(gpa, .f32, &.{ prefill_tokens, spec.hidden });
    defer y_prefill.deinit();
    var x_step = try zynfer.Tensor.alloc(gpa, .f32, &.{ 1, spec.hidden });
    defer x_step.deinit();
    var y_step = try zynfer.Tensor.alloc(gpa, .f32, &.{ 1, spec.hidden });
    defer y_step.deinit();
    try zynfer.tiny_block.iotaFill(x_prefill, 0.1, 0.01);
    try zynfer.tiny_block.iotaFill(x_step, 0.2, 0.01);

    if (gpu) |g| {
        var sess = try zynfer.apple.block.Session.init(gpa, g, spec);
        defer sess.deinit();
        try sess.inner.weights.fillFixture();
        sess.markWeightsDirty();
        const t0 = std.Io.Clock.awake.now(io);
        try sess.prefill(x_prefill, y_prefill);
        const t1 = std.Io.Clock.awake.now(io);
        var s: usize = 0;
        while (s < decode_steps) : (s += 1) try sess.decode(x_step, y_step);
        const t2 = std.Io.Clock.awake.now(io);
        return .{ .prefill_ns = nsDelta(t0, t1), .decode_ns = nsDelta(t1, t2) };
    }

    var sess = try zynfer.tiny_block.Session.init(gpa, spec);
    defer sess.deinit();
    try sess.weights.fillFixture();
    const t0 = std.Io.Clock.awake.now(io);
    try sess.prefill(x_prefill, y_prefill);
    const t1 = std.Io.Clock.awake.now(io);
    var s: usize = 0;
    while (s < decode_steps) : (s += 1) try sess.decode(x_step, y_step);
    const t2 = std.Io.Clock.awake.now(io);
    return .{ .prefill_ns = nsDelta(t0, t1), .decode_ns = nsDelta(t1, t2) };
}

fn runHipBench(io: std.Io, writer: *std.Io.Writer) !void {
    try writer.print("zynfer HIP probe benchmark\n", .{});
    try writer.print("==========================\n\n", .{});

    if (!zynfer.hip.have_hip) {
        try writer.print("HIP is not linked. Enumeration latency cannot be measured on this host.\n", .{});
        return;
    }

    const warmup = 8;
    const iters = 32;
    var i: usize = 0;
    while (i < warmup) : (i += 1) {
        _ = zynfer.hip.deviceCount() catch |err| {
            try writer.print("warmup failed: {s}\n", .{@errorName(err)});
            return;
        };
    }
    const start = std.Io.Clock.awake.now(io);
    i = 0;
    var last_count: u32 = 0;
    while (i < iters) : (i += 1) {
        last_count = try zynfer.hip.deviceCount();
        if (last_count > 0) _ = try zynfer.hip.describeDevice(0);
    }
    const elapsed_ns: u64 = @intCast(@max(@as(i96, 0), std.Io.Clock.awake.now(io).nanoseconds - start.nanoseconds));
    try writer.print("avg hip query: {d} ns  devices={d}\n", .{ elapsed_ns / iters, last_count });
}
