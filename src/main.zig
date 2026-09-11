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
    \\  zynfer stage11      Qwen forward + golden logits Stage 11 ledger
    \\  zynfer stage12      Tokenizer + sampling Stage 12 ledger
    \\  zynfer stage13      KV cache Stage 13 ledger
    \\  zynfer stageM0      Metal Qwen forward Stage M0 ledger
    \\  zynfer stageM1      Prefill/decode split Stage M1 ledger
    \\  zynfer stageM2      Profile one decode token Stage M2 ledger
    \\  zynfer stageM3      Qwen-scale schedule + fusion Stage M3 ledger
    \\  zynfer stageM4      bf16/fp16 Metal weights+KV Stage M4 ledger
    \\  zynfer stageM5      int8 weight quantization Stage M5 ledger
    \\  zynfer stageM6      static decode plan Stage M6 ledger
    \\  zynfer stageM7      ANE / Core ML Qwen-scale Stage M7 ledger
    \\  zynfer stageM8      Apple capstone: Qwen3-4B + benchmark matrix
    \\  zynfer stageS1      Batching / scheduling Stage S1 ledger
    \\  zynfer stageS2      Prefix reuse / cache Stage S2 ledger
    \\  zynfer stageS3      Speculative decoding Stage S3 ledger
    \\  zynfer stageS4      HTTP server Stage S4 ledger
    \\  zynfer coreml-smoke [PATH]  Load toy Core ML .mlpackage + one predict (M7 polish)
    \\  zynfer mem-report   weights / KV / scratch / peak RSS (Stage M6)
    \\  zynfer inspect PATH Validate and print a .zynfer artifact
    \\  zynfer artifact-compile [--out PATH] [--mini]  Write fixture .zynfer
    \\  zynfer forward-golden ARTIFACT [--tokens IDS] [--golden PATH] [--dump DIR] [--backend cpu|apple]
    \\  zynfer run ARTIFACT --prompt TEXT [--tokenizer DIR] [sampling flags]
    \\  zynfer chat [ARTIFACT] "PROMPT"   Interactive-style generate (streams tokens)
    \\  zynfer kv-bench [ARTIFACT] [--mini] [--layout] [--prompt TEXT] [--max-tokens N]
    \\  zynfer qwen-bench [ARTIFACT] [--mini] [--prompt TEXT] [--max-tokens N]
    \\  zynfer qwen-profile [ARTIFACT] [--mini] [--prompt TEXT] [--backend apple|cpu]
    \\  zynfer batch-bench [ARTIFACT] [--mini] [--batch-size N] [--max-inflight N] [--max-tokens N]
    \\  zynfer prefix-bench [ARTIFACT] [--mini] [--prefix-len N] [--suffix-len N] [--trials N]
    \\  zynfer spec-bench [ARTIFACT] [--mini] [--proposal-depth K] [--ngram-order N] [--max-tokens N]
    \\  zynfer serve [ARTIFACT] [--mini] [--host ADDR] [--port N] [--smoke] [--tokenizer DIR]
    \\  zynfer setup [--model 0.6b|4b] [--quantize] [--skip-golden] [--skip-pip]
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
    \\Stage 7 / M7 experimental forces (must fail loud; paths are not retained):
    \\  ZYNFER_FORCE_SME=1 zynfer stage7
    \\  ZYNFER_FORCE_COREML=1 zynfer stageM7
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
    var tokens_arg: ?[]const u8 = null;
    var golden_path: ?[]const u8 = null;
    var dump_dir: ?[]const u8 = null;
    var prompt_arg: ?[]const u8 = null;
    var tokenizer_dir: ?[]const u8 = null;
    var max_tokens: u32 = 64;
    var batch_size: u32 = 2;
    var max_inflight: u32 = 2;
    var prefix_len: u32 = 0;
    var suffix_len: u32 = 0;
    var trials: u32 = 4;
    var proposal_depth: u32 = 4;
    var ngram_order: u32 = 2;
    var serve_host: []const u8 = "127.0.0.1";
    var serve_port: u16 = 8080;
    var serve_smoke = false;
    var temperature: f32 = 0;
    var top_k: u32 = 0;
    var top_p: f32 = 1.0;
    var seed: u64 = 0;
    var raw_prompt = false;
    var no_stream = false;
    var no_kv_cache = false;
    var skip_golden = false;
    var skip_pip = false;
    var skip_download = false;
    var artifact_mini = false;
    var kv_layout_bench = false;
    var setup_quantize = false;
    var setup_model: []const u8 = "0.6b";
    var positionals: [16][]const u8 = undefined;
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
        } else if (std.mem.eql(u8, arg, "--mini")) {
            artifact_mini = true;
        } else if (std.mem.eql(u8, arg, "--layout")) {
            kv_layout_bench = true;
        } else if (std.mem.eql(u8, arg, "--tokens")) {
            tokens_arg = args_it.next() orelse {
                std.debug.print("missing value for --tokens\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.startsWith(u8, arg, "--tokens=")) {
            tokens_arg = arg["--tokens=".len..];
        } else if (std.mem.eql(u8, arg, "--golden")) {
            golden_path = args_it.next() orelse {
                std.debug.print("missing value for --golden\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.startsWith(u8, arg, "--golden=")) {
            golden_path = arg["--golden=".len..];
        } else if (std.mem.eql(u8, arg, "--dump")) {
            dump_dir = args_it.next() orelse {
                std.debug.print("missing value for --dump\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.startsWith(u8, arg, "--dump=")) {
            dump_dir = arg["--dump=".len..];
        } else if (std.mem.eql(u8, arg, "--prompt")) {
            prompt_arg = args_it.next() orelse {
                std.debug.print("missing value for --prompt\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.startsWith(u8, arg, "--prompt=")) {
            prompt_arg = arg["--prompt=".len..];
        } else if (std.mem.eql(u8, arg, "--tokenizer")) {
            tokenizer_dir = args_it.next() orelse {
                std.debug.print("missing value for --tokenizer\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.startsWith(u8, arg, "--tokenizer=")) {
            tokenizer_dir = arg["--tokenizer=".len..];
        } else if (std.mem.eql(u8, arg, "--max-tokens")) {
            const v = args_it.next() orelse {
                std.debug.print("missing value for --max-tokens\n", .{});
                std.process.exit(2);
            };
            max_tokens = std.fmt.parseInt(u32, v, 10) catch {
                std.debug.print("invalid --max-tokens\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.startsWith(u8, arg, "--max-tokens=")) {
            max_tokens = std.fmt.parseInt(u32, arg["--max-tokens=".len..], 10) catch {
                std.debug.print("invalid --max-tokens\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.eql(u8, arg, "--batch-size")) {
            const v = args_it.next() orelse {
                std.debug.print("missing value for --batch-size\n", .{});
                std.process.exit(2);
            };
            batch_size = std.fmt.parseInt(u32, v, 10) catch {
                std.debug.print("invalid --batch-size\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.startsWith(u8, arg, "--batch-size=")) {
            batch_size = std.fmt.parseInt(u32, arg["--batch-size=".len..], 10) catch {
                std.debug.print("invalid --batch-size\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.eql(u8, arg, "--max-inflight")) {
            const v = args_it.next() orelse {
                std.debug.print("missing value for --max-inflight\n", .{});
                std.process.exit(2);
            };
            max_inflight = std.fmt.parseInt(u32, v, 10) catch {
                std.debug.print("invalid --max-inflight\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.startsWith(u8, arg, "--max-inflight=")) {
            max_inflight = std.fmt.parseInt(u32, arg["--max-inflight=".len..], 10) catch {
                std.debug.print("invalid --max-inflight\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.eql(u8, arg, "--prefix-len")) {
            const v = args_it.next() orelse {
                std.debug.print("missing value for --prefix-len\n", .{});
                std.process.exit(2);
            };
            prefix_len = std.fmt.parseInt(u32, v, 10) catch {
                std.debug.print("invalid --prefix-len\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.startsWith(u8, arg, "--prefix-len=")) {
            prefix_len = std.fmt.parseInt(u32, arg["--prefix-len=".len..], 10) catch {
                std.debug.print("invalid --prefix-len\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.eql(u8, arg, "--suffix-len")) {
            const v = args_it.next() orelse {
                std.debug.print("missing value for --suffix-len\n", .{});
                std.process.exit(2);
            };
            suffix_len = std.fmt.parseInt(u32, v, 10) catch {
                std.debug.print("invalid --suffix-len\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.startsWith(u8, arg, "--suffix-len=")) {
            suffix_len = std.fmt.parseInt(u32, arg["--suffix-len=".len..], 10) catch {
                std.debug.print("invalid --suffix-len\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.eql(u8, arg, "--trials")) {
            const v = args_it.next() orelse {
                std.debug.print("missing value for --trials\n", .{});
                std.process.exit(2);
            };
            trials = std.fmt.parseInt(u32, v, 10) catch {
                std.debug.print("invalid --trials\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.startsWith(u8, arg, "--trials=")) {
            trials = std.fmt.parseInt(u32, arg["--trials=".len..], 10) catch {
                std.debug.print("invalid --trials\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.eql(u8, arg, "--proposal-depth")) {
            const v = args_it.next() orelse {
                std.debug.print("missing value for --proposal-depth\n", .{});
                std.process.exit(2);
            };
            proposal_depth = std.fmt.parseInt(u32, v, 10) catch {
                std.debug.print("invalid --proposal-depth\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.startsWith(u8, arg, "--proposal-depth=")) {
            proposal_depth = std.fmt.parseInt(u32, arg["--proposal-depth=".len..], 10) catch {
                std.debug.print("invalid --proposal-depth\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.eql(u8, arg, "--ngram-order")) {
            const v = args_it.next() orelse {
                std.debug.print("missing value for --ngram-order\n", .{});
                std.process.exit(2);
            };
            ngram_order = std.fmt.parseInt(u32, v, 10) catch {
                std.debug.print("invalid --ngram-order\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.startsWith(u8, arg, "--ngram-order=")) {
            ngram_order = std.fmt.parseInt(u32, arg["--ngram-order=".len..], 10) catch {
                std.debug.print("invalid --ngram-order\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.eql(u8, arg, "--host")) {
            serve_host = args_it.next() orelse {
                std.debug.print("missing value for --host\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.startsWith(u8, arg, "--host=")) {
            serve_host = arg["--host=".len..];
        } else if (std.mem.eql(u8, arg, "--port")) {
            const v = args_it.next() orelse {
                std.debug.print("missing value for --port\n", .{});
                std.process.exit(2);
            };
            serve_port = std.fmt.parseInt(u16, v, 10) catch {
                std.debug.print("invalid --port\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.startsWith(u8, arg, "--port=")) {
            serve_port = std.fmt.parseInt(u16, arg["--port=".len..], 10) catch {
                std.debug.print("invalid --port\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.eql(u8, arg, "--smoke")) {
            serve_smoke = true;
        } else if (std.mem.eql(u8, arg, "--temperature") or std.mem.eql(u8, arg, "--temp")) {
            const v = args_it.next() orelse {
                std.debug.print("missing value for --temperature\n", .{});
                std.process.exit(2);
            };
            temperature = std.fmt.parseFloat(f32, v) catch {
                std.debug.print("invalid --temperature\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.startsWith(u8, arg, "--temperature=")) {
            temperature = std.fmt.parseFloat(f32, arg["--temperature=".len..]) catch {
                std.debug.print("invalid --temperature\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.startsWith(u8, arg, "--temp=")) {
            temperature = std.fmt.parseFloat(f32, arg["--temp=".len..]) catch {
                std.debug.print("invalid --temp\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.eql(u8, arg, "--top-k")) {
            const v = args_it.next() orelse {
                std.debug.print("missing value for --top-k\n", .{});
                std.process.exit(2);
            };
            top_k = std.fmt.parseInt(u32, v, 10) catch {
                std.debug.print("invalid --top-k\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.startsWith(u8, arg, "--top-k=")) {
            top_k = std.fmt.parseInt(u32, arg["--top-k=".len..], 10) catch {
                std.debug.print("invalid --top-k\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.eql(u8, arg, "--top-p")) {
            const v = args_it.next() orelse {
                std.debug.print("missing value for --top-p\n", .{});
                std.process.exit(2);
            };
            top_p = std.fmt.parseFloat(f32, v) catch {
                std.debug.print("invalid --top-p\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.startsWith(u8, arg, "--top-p=")) {
            top_p = std.fmt.parseFloat(f32, arg["--top-p=".len..]) catch {
                std.debug.print("invalid --top-p\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.eql(u8, arg, "--seed")) {
            const v = args_it.next() orelse {
                std.debug.print("missing value for --seed\n", .{});
                std.process.exit(2);
            };
            seed = std.fmt.parseInt(u64, v, 10) catch {
                std.debug.print("invalid --seed\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.startsWith(u8, arg, "--seed=")) {
            seed = std.fmt.parseInt(u64, arg["--seed=".len..], 10) catch {
                std.debug.print("invalid --seed\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.eql(u8, arg, "--raw")) {
            raw_prompt = true;
        } else if (std.mem.eql(u8, arg, "--no-stream")) {
            no_stream = true;
        } else if (std.mem.eql(u8, arg, "--no-kv-cache")) {
            no_kv_cache = true;
        } else if (std.mem.eql(u8, arg, "--skip-golden")) {
            skip_golden = true;
        } else if (std.mem.eql(u8, arg, "--skip-pip")) {
            skip_pip = true;
        } else if (std.mem.eql(u8, arg, "--skip-download")) {
            skip_download = true;
        } else if (std.mem.eql(u8, arg, "--quantize")) {
            setup_quantize = true;
        } else if (std.mem.eql(u8, arg, "--model")) {
            setup_model = args_it.next() orelse {
                std.debug.print("missing value for --model\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.startsWith(u8, arg, "--model=")) {
            setup_model = arg["--model=".len..];
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
    } else if (std.mem.eql(u8, command, "stage11")) {
        try printStage11(writer);
    } else if (std.mem.eql(u8, command, "stage12")) {
        try printStage12(writer);
    } else if (std.mem.eql(u8, command, "stage13")) {
        try printStage13(writer);
    } else if (std.mem.eql(u8, command, "stageM0") or std.mem.eql(u8, command, "stagem0")) {
        try printStageM0(writer);
    } else if (std.mem.eql(u8, command, "stageM1") or std.mem.eql(u8, command, "stagem1")) {
        try printStageM1(writer);
    } else if (std.mem.eql(u8, command, "stageM2") or std.mem.eql(u8, command, "stagem2")) {
        try printStageM2(writer);
    } else if (std.mem.eql(u8, command, "stageM3") or std.mem.eql(u8, command, "stagem3")) {
        try printStageM3(writer);
    } else if (std.mem.eql(u8, command, "stageM4") or std.mem.eql(u8, command, "stagem4")) {
        try printStageM4(writer);
    } else if (std.mem.eql(u8, command, "stageM5") or std.mem.eql(u8, command, "stagem5")) {
        try printStageM5(writer);
    } else if (std.mem.eql(u8, command, "stageM6") or std.mem.eql(u8, command, "stagem6")) {
        try printStageM6(writer);
    } else if (std.mem.eql(u8, command, "stageM7") or std.mem.eql(u8, command, "stagem7")) {
        try printStageM7(writer);
    } else if (std.mem.eql(u8, command, "stageM8") or std.mem.eql(u8, command, "stagem8")) {
        try printStageM8(writer);
    } else if (std.mem.eql(u8, command, "stageS1") or std.mem.eql(u8, command, "stages1")) {
        try printStageS1(writer);
    } else if (std.mem.eql(u8, command, "stageS2") or std.mem.eql(u8, command, "stages2")) {
        try printStageS2(writer);
    } else if (std.mem.eql(u8, command, "stageS3") or std.mem.eql(u8, command, "stages3")) {
        try printStageS3(writer);
    } else if (std.mem.eql(u8, command, "stageS4") or std.mem.eql(u8, command, "stages4")) {
        try printStageS4(writer);
    } else if (std.mem.eql(u8, command, "coreml-smoke") or std.mem.eql(u8, command, "coremlsmoke")) {
        const path = if (n_pos >= 1) positionals[0] else "tools/fixtures/coreml_toy.mlpackage";
        try cmdCoreMlSmoke(writer, path);
    } else if (std.mem.eql(u8, command, "mem-report") or std.mem.eql(u8, command, "memreport")) {
        const backend = try resolveKind(forced_backend);
        const path = if (n_pos >= 1) positionals[0] else null;
        try cmdMemReport(host, writer, path, artifact_mini, max_tokens, backend);
    } else if (std.mem.eql(u8, command, "inspect")) {
        if (n_pos < 1) {
            std.debug.print("usage: zynfer inspect PATH.zynfer\n", .{});
            std.process.exit(2);
        }
        try runInspect(allocator, io, writer, positionals[0]);
    } else if (std.mem.eql(u8, command, "artifact-compile")) {
        const path = out_path orelse (if (n_pos >= 1) positionals[0] else if (artifact_mini) "stage11-mini.zynfer" else "stage10-fixture.zynfer");
        try runArtifactCompile(allocator, io, writer, path, artifact_mini);
    } else if (std.mem.eql(u8, command, "forward-golden")) {
        if (n_pos < 1) {
            std.debug.print("usage: zynfer forward-golden ARTIFACT.zynfer [--tokens 1,2,3] [--golden PATH] [--dump DIR]\n", .{});
            std.process.exit(2);
        }
        try runForwardGolden(allocator, io, writer, positionals[0], tokens_arg, golden_path, dump_dir, try resolveKind(forced_backend));
    } else if (std.mem.eql(u8, command, "run")) {
        if (n_pos < 1 or prompt_arg == null) {
            std.debug.print(
                "usage: zynfer run ARTIFACT.zynfer --prompt TEXT [--tokenizer DIR] [--max-tokens N] [--temperature T] [--top-k K] [--top-p P] [--seed S] [--raw] [--no-stream] [--no-kv-cache] [--backend cpu|apple]\n",
                .{},
            );
            std.process.exit(2);
        }
        try runGenerate(
            allocator,
            io,
            writer,
            positionals[0],
            prompt_arg.?,
            tokenizer_dir,
            max_tokens,
            temperature,
            top_k,
            top_p,
            seed,
            raw_prompt,
            !no_stream,
            !no_kv_cache,
            try resolveKind(forced_backend),
        );
    } else if (std.mem.eql(u8, command, "kv-bench")) {
        try runKvBench(
            allocator,
            io,
            writer,
            if (n_pos >= 1) positionals[0] else null,
            prompt_arg,
            tokenizer_dir,
            max_tokens,
            seed,
            raw_prompt,
            artifact_mini,
            tokens_arg,
            kv_layout_bench,
        );
    } else if (std.mem.eql(u8, command, "qwen-bench")) {
        try runQwenBench(
            allocator,
            io,
            writer,
            if (n_pos >= 1) positionals[0] else null,
            prompt_arg,
            tokenizer_dir,
            max_tokens,
            seed,
            raw_prompt,
            artifact_mini,
            tokens_arg,
        );
    } else if (std.mem.eql(u8, command, "qwen-profile")) {
        try runQwenProfile(
            allocator,
            io,
            writer,
            if (n_pos >= 1) positionals[0] else null,
            prompt_arg,
            tokenizer_dir,
            seed,
            raw_prompt,
            artifact_mini,
            tokens_arg,
            forced_backend,
        );
    } else if (std.mem.eql(u8, command, "batch-bench")) {
        try runBatchBench(
            allocator,
            io,
            writer,
            if (n_pos >= 1) positionals[0] else null,
            artifact_mini,
            batch_size,
            max_inflight,
            max_tokens,
            seed,
            try resolveKind(forced_backend),
        );
    } else if (std.mem.eql(u8, command, "prefix-bench")) {
        try runPrefixBench(
            allocator,
            io,
            writer,
            if (n_pos >= 1) positionals[0] else null,
            artifact_mini,
            prefix_len,
            suffix_len,
            trials,
            try resolveKind(forced_backend),
        );
    } else if (std.mem.eql(u8, command, "spec-bench")) {
        try runSpecBench(
            allocator,
            io,
            writer,
            if (n_pos >= 1) positionals[0] else null,
            artifact_mini,
            proposal_depth,
            ngram_order,
            max_tokens,
            try resolveKind(forced_backend),
        );
    } else if (std.mem.eql(u8, command, "serve")) {
        try runServe(
            allocator,
            io,
            writer,
            if (n_pos >= 1) positionals[0] else null,
            artifact_mini,
            serve_host,
            serve_port,
            serve_smoke,
            tokenizer_dir,
            max_tokens,
            try resolveKind(forced_backend),
        );
    } else if (std.mem.eql(u8, command, "chat")) {
        // zynfer chat "prompt"  OR  zynfer chat ARTIFACT "prompt"  OR  --prompt=
        var artifact_path: []const u8 = defaultRegisteredArtifact(io);
        var chat_prompt: ?[]const u8 = prompt_arg;
        if (n_pos >= 1 and std.mem.endsWith(u8, positionals[0], ".zynfer")) {
            artifact_path = positionals[0];
            if (n_pos >= 2) chat_prompt = positionals[1];
        } else if (n_pos >= 1) {
            chat_prompt = positionals[0];
        }
        if (chat_prompt == null or chat_prompt.?.len == 0) {
            std.debug.print(
                "usage: zynfer chat [ARTIFACT.zynfer] \"PROMPT\" [--tokenizer DIR] [--max-tokens N] …\n",
                .{},
            );
            std.process.exit(2);
        }
        if (!zynfer.util.fileExists(io, artifact_path)) {
            std.debug.print(
                "chat: artifact not found ({s})\n  run: ./zig-out/bin/zynfer setup\n",
                .{artifact_path},
            );
            std.process.exit(2);
        }
        try runGenerate(
            allocator,
            io,
            writer,
            artifact_path,
            chat_prompt.?,
            tokenizer_dir,
            max_tokens,
            temperature,
            top_k,
            top_p,
            seed,
            raw_prompt,
            !no_stream,
            !no_kv_cache,
            try resolveKind(forced_backend),
        );
    } else if (std.mem.eql(u8, command, "setup")) {
        try runSetup(host, writer, skip_pip, skip_download, skip_golden, setup_model, setup_quantize);
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
            "ZYNFER_FORCE_COREML requested but Core ML/ANE inference is not retained (Stage M7: Qwen-scale REJECT — no Instruments ANE proof / no e2e win over Metal).\n",
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
    try writer.print("  MLState class available:    {s}\n", .{yn(cm_p.ml_state_available)});
    try writer.print("  Darwin major:               {d}\n", .{cm_p.macos_major});
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
    try writer.print("  Core ML/ANE ops: REJECT — framework probe only; Qwen-scale closed in M7\n", .{});
    try writer.print("  Accelerate:      RETAIN — measured Stage 5 size-gated vDSP path\n", .{});
    try writer.print("  Metal Stage 6:   RETAIN — default tiny-block schedule\n", .{});
    try writer.print("\nSee bench/results/apple-stage7-dev-laptop.md and stageM7\n", .{});
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
    try writer.print("  fp16/bf16 Metal:          RETAINED (M4) — bf16 weights+KV, f32 activations/softmax\n\n", .{});

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
    try writer.print("  forward pass / logits — Stage 11 (see docs/stages/11-qwen-forward.md)\n", .{});
    try writer.print("  tokenizer / sampling / TTFT — Stage 12\n", .{});
    try writer.print("  HF download in CI — never; local convert only\n\n", .{});
    try writer.print("See docs/artifact-format.md and bench/results/stage10-dev-laptop.md\n", .{});
}

fn printStage11(writer: *std.Io.Writer) !void {
    try writer.print("zynfer Stage 11 — Qwen3 forward + golden logits (CPU)\n", .{});
    try writer.print("====================================================\n\n", .{});
    try writer.print("Done\n", .{});
    try writer.print("  forward:          embed → {d} blocks → norm → lm_head\n", .{zynfer.qwen3.qwen3_0_6b.num_layers});
    try writer.print("  Qwen3 extras:     QK-norm, GQA, SwiGLU, RoPE (theta=1e6)\n", .{});
    try writer.print("  weights:          BF16→F32 at load; HF linear transpose\n", .{});
    try writer.print("  CLI:              forward-golden ARTIFACT [--tokens IDS] [--golden PATH] [--dump DIR]\n", .{});
    try writer.print("  golden (local):   python3 tools/fixtures/gen_golden_logits.py → --golden ref_logits.f32\n", .{});
    try writer.print("  debug:            --dump DIR + tools/fixtures/ref_forward_numpy.py\n", .{});
    try writer.print("  CI fixture:       artifact-compile --mini → stage11-mini.zynfer\n", .{});
    try writer.print("  tests:            mini artifact forward + determinism\n\n", .{});
    try writer.print("Not in Stage 11\n", .{});
    try writer.print("  tokenizer / sampling / TTFT — Stage 12\n", .{});
    try writer.print("  Metal Qwen path — after CPU golden matches\n", .{});
    try writer.print("  HF download in CI — never\n\n", .{});
    try writer.print("See docs/stages/11-qwen-forward.md\n", .{});
}

fn printStage12(writer: *std.Io.Writer) !void {
    try writer.print("zynfer Stage 12 — tokenizer + sampling (CPU)\n", .{});
    try writer.print("=============================================\n\n", .{});
    try writer.print("Done\n", .{});
    try writer.print("  tokenizer:        Qwen2 byte-level BPE (vocab.json + merges.txt)\n", .{});
    try writer.print("  sampling:         greedy / temperature / top-k / top-p / seeded RNG\n", .{});
    try writer.print("  generate:         prefill + KV-cached decode loop (streaming)\n", .{});
    try writer.print("  CLI:              run / chat / setup\n", .{});
    try writer.print("  metrics:          TTFT, prefill_ms, decode_tok_s, ITL p50/p95/p99\n", .{});
    try writer.print("  chat wrap:        Qwen3 non-thinking template (disable with --raw)\n\n", .{});
    try writer.print("Not in Stage 12\n", .{});
    try writer.print("  Metal Qwen path — later\n", .{});
    try writer.print("  HF download in CI — never (use local `zynfer setup`)\n\n", .{});
    try writer.print("See docs/stages/12-tokenizer-sampling.md\n", .{});
}

fn printStage13(writer: *std.Io.Writer) !void {
    try writer.print("zynfer Stage 13 — KV cache (CPU Qwen)\n", .{});
    try writer.print("=====================================\n\n", .{});
    try writer.print("Done\n", .{});
    try writer.print("  layout:           [n_kv, max_seq, head_dim] retained (see kv-bench --layout)\n", .{});
    try writer.print("  cached path:      one prefill, then decodeToken appends one position\n", .{});
    try writer.print("  uncached path:    each step resets and recomputes the full prefix\n", .{});
    try writer.print("  parity:           greedy tokens match (unit test + kv-bench)\n", .{});
    try writer.print("  bake-off:         heads-outer beats seq-outer on decode attn (~3×)\n", .{});
    try writer.print("  CLI:              kv-bench [--mini] [--layout] | run/chat --no-kv-cache\n\n", .{});

    try writer.print("Three memories (do not confuse)\n", .{});
    try writer.print("  weights:     fixed parameters loaded once from the .zynfer artifact\n", .{});
    try writer.print("  activations: scratch tensors for the current forward (hidden, Q/K/V, …)\n", .{});
    try writer.print("  KV cache:    growing per-layer K/V history across decode steps\n\n", .{});

    const a = zynfer.qwen3.qwen3_0_6b;
    const per_tok = zynfer.kv_cache.estimateModelBytes(
        a.num_layers,
        a.num_key_value_heads,
        1,
        a.head_dim,
    );
    const at_4k = zynfer.kv_cache.estimateModelBytes(
        a.num_layers,
        a.num_key_value_heads,
        4096,
        a.head_dim,
    );
    try writer.print("KV memory (f32, Qwen3-0.6B)\n", .{});
    try writer.print("  formula:  layers × n_kv × seq × head_dim × 2 × 4\n", .{});
    try writer.print("  per token: {d} bytes (~{d:.2} MiB)\n", .{
        per_tok,
        @as(f64, @floatFromInt(per_tok)) / (1024.0 * 1024.0),
    });
    try writer.print("  at seq=4096: {d} bytes (~{d:.1} MiB)\n\n", .{
        at_4k,
        @as(f64, @floatFromInt(at_4k)) / (1024.0 * 1024.0),
    });

    try writer.print("Not in Stage 13\n", .{});
    try writer.print("  Metal-resident Qwen KV — later\n", .{});
    try writer.print("  paged / quantized KV — Stages 18 / 22\n\n", .{});
    try writer.print("See docs/stages/13-kv-cache.md\n", .{});
}

fn printStageM0(writer: *std.Io.Writer) !void {
    try writer.print("zynfer Stage M0 — Metal Qwen forward + generate (f32)\n", .{});
    try writer.print("===================================================\n\n", .{});
    try writer.print("Status: IN PROGRESS (baseline landed; gate not closed)\n", .{});
    try writer.print("Plan:   baoulo/prompts/fable-5-prompt.md  Phase M\n\n", .{});
    try writer.print("Done (baseline)\n", .{});
    try writer.print("  routing:          qwen_block → apple qwen_adapter (per-op Metal)\n", .{});
    try writer.print("  backend:          --backend cpu|apple on run / forward-golden / chat\n", .{});
    try writer.print("  attention cap:    kv_len ≤ {d} ({d} thread-local fast path)\n", .{
        zynfer.apple.ops.max_attention_kv,
        zynfer.apple.ops.max_attention_kv_threadlocal,
    });
    try writer.print("  oracle:           CPU reference; embed/norm/lm_head on CPU (M0)\n", .{});
    try writer.print("  CI test:          mini artifact Metal logits vs CPU\n\n", .{});
    try writer.print("Open (must close before M0 = done)\n", .{});
    try writer.print("  [ ] full-model Metal greedy tokens == CPU golden\n", .{});
    try writer.print("  [ ] Metal TTFT / decode tok/s in stageM0 bench ledger\n", .{});
    try writer.print("  [ ] per-layer dump ladder on Metal\n", .{});
    try writer.print("  [ ] LM-head GEMV path A/B (naive / simdgroup / Accelerate)\n", .{});
    try writer.print("  [ ] attention parity at kv_len > 256 (device scores)\n", .{});
    try writer.print("  (Stage 6 resident-KV / one-CB → done in M3)\n\n", .{});
    try writer.print("See docs/stages/M0-metal-qwen-forward.md\n", .{});
}

fn printStageM1(writer: *std.Io.Writer) !void {
    try writer.print("zynfer Stage M1 — Prefill vs decode on Qwen\n", .{});
    try writer.print("===========================================\n\n", .{});
    try writer.print("Done\n", .{});
    try writer.print("  CLI:              qwen-bench [--mini] → CPU + Apple split report\n", .{});
    try writer.print("  prefill:          latency_ms, prompt_tok_s, GEMM shapes\n", .{});
    try writer.print("  decode:           tok_s, ms/token, bytes/token est., measured enc/wait per tok\n", .{});
    try writer.print("  output:           human table + json line\n", .{});
    try writer.print("  also:             run/chat footer prints prefill_tok_s separately\n\n", .{});
    try writer.print("Not in Stage M1\n", .{});
    try writer.print("  per-op decode profile / roofline — M2\n", .{});
    try writer.print("  batched Metal schedule — M3\n\n", .{});
    try writer.print("See docs/stages/M1-prefill-vs-decode-qwen.md\n", .{});
}

fn printStageM2(writer: *std.Io.Writer) !void {
    try writer.print("zynfer Stage M2 — Profile one decode token\n", .{});
    try writer.print("==========================================\n\n", .{});
    try writer.print("Done\n", .{});
    try writer.print("  CLI:              qwen-profile [--mini] → per-family table + top3 + json\n", .{});
    try writer.print("  families:         RMSNorm, QKV, RoPE, attention, O-proj, MLP,\n", .{});
    try writer.print("                    host layout, embed, LM head, sampling\n", .{});
    try writer.print("  Metal:            measured enc/wait + empty-launch overhead estimate\n", .{});
    try writer.print("  roofline:         STREAM triad bandwidth / B/tok_est → ideal tok/s\n", .{});
    try writer.print("  signposts:        ZYNFER_SIGNPOSTS=1 → qwen.* family intervals\n\n", .{});
    try writer.print("Not in Stage M2\n", .{});
    try writer.print("  batched one-CB schedule / fusion ledger — M3\n\n", .{});
    try writer.print("See docs/stages/M2-profile-one-decode-token.md\n", .{});
}

fn printStageM3(writer: *std.Io.Writer) !void {
    try writer.print("zynfer Stage M3 — Qwen-scale schedule + fusion\n", .{});
    try writer.print("==============================================\n\n", .{});
    try writer.print("Done\n", .{});
    try writer.print("  path:             batched_resident_kv_fused (default on Apple)\n", .{});
    try writer.print("  A/B:              ZYNFER_QWEN_METAL=baseline → M0 per-op path\n", .{});
    try writer.print("  schedule:         one CB for all layers + one CB for final norm/LM head\n", .{});
    try writer.print("                    (≈2 waits/forward; resident weights + Metal KV)\n", .{});
    try writer.print("  retained:         silu_mul, add_rmsnorm_f32, Metal LM-head matvec\n", .{});
    try writer.print("  rejected:         Q/K+RoPE fuse, attention tiling, dequant-GEMV (→M5),\n", .{});
    try writer.print("                    ICB/encode-once (KV mutates every decode)\n\n", .{});
    try writer.print("Not in Stage M3\n", .{});
    try writer.print("  fp16/bf16 weights+KV — M4\n", .{});
    try writer.print("  int8 session weights — M5\n\n", .{});
    try writer.print("See docs/stages/M3-qwen-schedule-fusion.md\n", .{});
}

fn printStageM4(writer: *std.Io.Writer) !void {
    try writer.print("zynfer Stage M4 — bf16/fp16 Metal weights + KV\n", .{});
    try writer.print("============================================\n\n", .{});
    try writer.print("Done\n", .{});
    try writer.print("  path:             batched_resident_kv_bf16 (ZYNFER_QWEN_METAL=bf16|half|fp16)\n", .{});
    try writer.print("  storage:          bf16 resident weights + bf16 KV on GPU\n", .{});
    try writer.print("  compute:          f32 activations; f32 accum in GEMM/attention/softmax\n", .{});
    try writer.print("  artifact:         .zynfer dtype tags 1=f16 2=bf16 (converter preserves bytes)\n", .{});
    try writer.print("  load:             CPU oracle f32; GPU half path copies artifact f16/bf16 bytes\n", .{});
    try writer.print("  tolerance:        5e-3 atol vs CPU logits (dtype-justified)\n", .{});
    try writer.print("  bytes/token:      estimateDecodeBytesPerTokenHalf ≈ ½ f32 estimate\n\n", .{});
    try writer.print("Not in Stage M4\n", .{});
    try writer.print("  int8 session weights — M5 (done; see stageM5)\n", .{});
    try writer.print("  zero-allocation static decode — M6 (done; see stageM6)\n\n", .{});
    try writer.print("See docs/stages/M4-half-precision-metal.md\n", .{});
}

fn printStageM5(writer: *std.Io.Writer) !void {
    try writer.print("zynfer Stage M5 — int8 weight quantization (Apple)\n", .{});
    try writer.print("================================================\n\n", .{});
    try writer.print("Done\n", .{});
    try writer.print("  path:             batched_resident_kv_q8 (ZYNFER_QWEN_METAL=int8|q8)\n", .{});
    try writer.print("  scheme:           per-row symmetric int8 (scale = max_abs/127, no ZP)\n", .{});
    try writer.print("  layout:           HF [out,in]; prefer artifact i8+qscale → Metal\n", .{});
    try writer.print("  fallback pack:    host f32 [in,out] when artifact is float\n", .{});
    try writer.print("  kernels:          matmul_aq8_f32 (fused dequant) + matvec_q8_f32 / bf16 tied lm_head\n", .{});
    try writer.print("  not quantized:    norms (f32), embed table (bf16), activations (f32)\n", .{});
    try writer.print("  KV:               bf16 on int8 Metal path\n", .{});
    try writer.print("  artifact:         dtype tag 3=i8; tools/checkpoint/quantize_zynfer_int8.py\n", .{});
    try writer.print("  decoder:          qwen_quant.dequantRowQ8 (CPU round-trip tests)\n", .{});
    try writer.print("  tolerance:        5e-2 atol vs CPU logits (packing-justified)\n", .{});
    try writer.print("  bytes/token:      estimateDecodeBytesPerTokenQ8 (i8 weights + f32 scales + bf16 KV)\n\n", .{});
    try writer.print("Not in Stage M5\n", .{});
    try writer.print("  4-bit weights — only if int8 decode wins and quality allows\n", .{});
    try writer.print("  zero-allocation static decode — M6 (done; see stageM6)\n\n", .{});
    try writer.print("See docs/stages/M5-weight-quantization-apple.md\n", .{});
}

fn printStageM6(writer: *std.Io.Writer) !void {
    try writer.print("zynfer Stage M6 — static decode plan (Apple)\n", .{});
    try writer.print("===========================================\n\n", .{});
    try writer.print("Done\n", .{});
    try writer.print("  after warm-up:    weights resident, KV to max_seq, fixed GPU scratch\n", .{});
    try writer.print("  host mirror:      batched Metal skips host per-layer KV/scratch twin\n", .{});
    try writer.print("  sample scratch:   Session-owned probs/idx; reuse Session.logits\n", .{});
    try writer.print("  out_ids:          ensureTotalCapacity + appendAssumeCapacity\n", .{});
    try writer.print("  asserted:         FailingAllocator flat over decode / generateCached\n", .{});
    try writer.print("  encode/submit:    2 waits/forward (M3); ICB/replay — REJECT (finalize)\n", .{});
    try writer.print("  memory report:    zynfer mem-report [--mini | ARTIFACT] [--max-tokens N]\n", .{});
    try writer.print("  ITL variance:     run / qwen-bench print itl_ms p50/p95/p99\n\n", .{});
    try writer.print("ICB decision (final)\n", .{});
    try writer.print("  REJECT — KV/q_len change each decode; cite Stage 8 + M3 ledgers.\n", .{});
    try writer.print("  Encode count still high; waits already minimal (2). No ICB path.\n\n", .{});
    try writer.print("Next\n", .{});
    try writer.print("  ANE / Core ML — M7 done (REJECT); see stageM7\n\n", .{});
    try writer.print("See docs/stages/M6-static-decode-plan.md\n", .{});
}

fn printStageM7(writer: *std.Io.Writer) !void {
    try writer.print("zynfer Stage M7 — ANE / Core ML at Qwen scale\n", .{});
    try writer.print("=============================================\n\n", .{});

    const cm_p = zynfer.apple.coreml.probe();
    try writer.print("Probe (does not load a model; does not claim ANE)\n", .{});
    try writer.print("  framework linked:           {s}\n", .{yn(cm_p.framework_linked)});
    try writer.print("  MLModelConfiguration ok:    {s}\n", .{yn(cm_p.configuration_ok)});
    try writer.print("  computeUnits All ok:        {s}\n", .{yn(cm_p.compute_units_all_ok)});
    try writer.print("  computeUnits CPU+ANE ok:    {s}\n", .{yn(cm_p.compute_units_cpu_and_ane_ok)});
    try writer.print("  MLState class available:    {s}\n", .{yn(cm_p.ml_state_available)});
    try writer.print("  Darwin major:               {d}\n", .{cm_p.macos_major});
    try writer.print("  .mlmodel in tree:           {s}\n", .{yn(cm_p.model_artifact_present)});
    try writer.print("  ANE execution verified:     {s}\n", .{yn(cm_p.ane_execution_verified)});
    try writer.print("  path retained:              {s}\n", .{yn(cm_p.path_retained)});
    try writer.print("  detail: {s}\n\n", .{cm_p.detail});

    try writer.print("Candidates considered\n", .{});
    try writer.print("  (a) fixed-length prefill subgraph — SKIP: no Qwen .mlmodel / converter;\n", .{});
    try writer.print("      fair race is most of the stage. Handoff/repack vs resident Metal\n", .{});
    try writer.print("      taxes TTFT (unified memory ≠ free Metal↔Core ML splice)\n", .{});
    try writer.print("  (b) stateful decode graph (MLState) — SKIP: API present on Darwin 24+\n", .{});
    try writer.print("      but no Qwen KV graph exported; mutable KV + dynamic shapes remain\n", .{});
    try writer.print("      the known hazard; no Instruments ANE placement proof\n", .{});
    try writer.print("  (c) nothing — CHOSEN (did not start (a)/(b); retain criteria unmet)\n\n", .{});

    try writer.print("Retain criteria (all required)\n", .{});
    try writer.print("  [ ] Instruments confirms ANE placement (not CPU/GPU fallback)\n", .{});
    try writer.print("  [ ] e2e TTFT or tok/s beats M6 Metal at equal quality\n", .{});
    try writer.print("  [ ] I/O / layout handoffs do not erase the win\n", .{});
    try writer.print("  [ ] graph stable enough to amortize compilation\n\n", .{});

    try writer.print("Optional polish (does not clear retain)\n", .{});
    try writer.print("  toy model:   tools/fixtures/coreml_toy.mlpackage (make_coreml_toy.py)\n", .{});
    try writer.print("  load smoke:  zynfer coreml-smoke [PATH]\n", .{});
    try writer.print("  xctrace:     see apple-ane-qwen-dev-laptop.md (Core ML template once)\n\n", .{});

    try writer.print("Stage M7 decision\n", .{});
    try writer.print("  Core ML / ANE at Qwen scale:  REJECT (final for this project)\n", .{});
    try writer.print("  Retained Apple engine:        Metal (M0–M6) + Accelerate (Stage 5)\n", .{});
    try writer.print("  ZYNFER_FORCE_COREML:          exit 2 (loud; no zombie path)\n", .{});
    try writer.print("  Next:                         M8 Apple capstone (Qwen3-4B + matrix)\n\n", .{});
    try writer.print("See bench/results/apple-ane-qwen-dev-laptop.md\n", .{});
    try writer.print("See docs/stages/M7-ane-coreml-qwen.md\n", .{});
    try writer.print("See docs/tutorials/21-the-neural-engine-question.md\n", .{});
}

fn printStageM8(writer: *std.Io.Writer) !void {
    try writer.print("zynfer Stage M8 — Apple capstone (Qwen3-4B)\n", .{});
    try writer.print("==========================================\n\n", .{});
    try writer.print("Milestone: Apple-complete (Backend 1)\n\n", .{});

    try writer.print("Registered models\n", .{});
    for (zynfer.registry.entries) |e| {
        try writer.print("  {s}\n", .{e.id.name()});
        try writer.print("    label:     {s}\n", .{e.label});
        try writer.print("    hf:        {s}\n", .{e.hf_repo});
        try writer.print("    dims:      h={d} inter={d} layers={d} heads={d}/{d} head_dim={d}\n", .{
            e.arch.hidden_size,
            e.arch.intermediate_size,
            e.arch.num_layers,
            e.arch.num_attention_heads,
            e.arch.num_key_value_heads,
            e.arch.head_dim,
        });
        try writer.print("    artifact:  {s}\n", .{e.artifact_path});
        try writer.print("    int8:      {s}\n", .{e.int8_artifact_path});
        if (e.int8_sha256_hex) |sha| {
            try writer.print("    int8_sha:  {s}\n", .{sha});
        }
    }
    try writer.writeAll("\n");

    const e4 = zynfer.registry.byId(.qwen3_4b);
    try writer.print("KV budget (Qwen3-4B, bf16 K/V — Metal int8 path)\n", .{});
    inline for ([_]usize{ 256, 512, 1024, 2048 }) |seq| {
        const kv = zynfer.registry.estimateKvBytesF32(e4.arch, seq) / 2;
        const wt = e4.arch.estimateDecodeBytesPerTokenQ8(seq);
        try writer.print("  max_seq={d: <5}  KV≈{d} MiB   decode-bytes/tok (int8 w + bf16 KV)≈{d} MiB\n", .{
            seq,
            kv / (1024 * 1024),
            wt / (1024 * 1024),
        });
    }
    try writer.print("  Metal attention kv_len cap: 2048 (Unsupported above)\n\n", .{});

    try writer.print("Final Apple matrix (see bench/results/apple-capstone-dev-laptop.md)\n", .{});
    try writer.print("  paths: CPU / Accelerate / Metal f32 / Metal bf16 / Metal int8 /\n", .{});
    try writer.print("         Core ML-ANE (REJECT — Stage M7)\n", .{});
    try writer.print("  axes:  short+long prefill × batch-1 decode × cold/warm\n", .{});
    try writer.print("  external: llama.cpp-Metal / MLX permitted (same machine; document gaps)\n\n", .{});

    try writer.print("Setup\n", .{});
    try writer.print("  python3 tools/setup_qwen.py --model 4b --quantize --skip-golden\n", .{});
    try writer.print("  ZYNFER_QWEN_METAL=int8 ./zig-out/bin/zynfer chat models/qwen3-4b-int8.zynfer \"…\"\n\n", .{});

    try writer.print("Stage M8 decision\n", .{});
    try writer.print("  Backend 1 (Apple):  Metal M0–M6 + Accelerate; Core ML REJECT (M7)\n", .{});
    try writer.print("  Capstone model:    registered Qwen3-4B (int8 preferred)\n", .{});
    try writer.print("  Next phases:       R (AMD when hardware), S (serving)\n\n", .{});
    try writer.print("See docs/stages/M8-apple-capstone.md\n", .{});
    try writer.print("See docs/tutorials/22-what-makes-apple-inference-fast.md\n", .{});
    try writer.print("See bench/results/apple-capstone-dev-laptop.md\n", .{});
}

fn printStageS1(writer: *std.Io.Writer) !void {
    try writer.print("zynfer Stage S1 — batching and scheduling\n", .{});
    try writer.print("=========================================\n\n", .{});
    try writer.print("Done (request-level)\n", .{});
    try writer.print("  queue:            FIFO admit (one job/iteration) into max_inflight slots\n", .{});
    try writer.print("  decode:           round-robin one token across active slots\n", .{});
    try writer.print("  ownership:        each request owns its Session / KV (no shared cache)\n", .{});
    try writer.print("  baseline:         serial generate A/B vs scheduled\n", .{});
    try writer.print("  metrics:          concurrency, aggregate tok/s, per-req tok/s, TTFT, mean ITL\n", .{});
    try writer.print("  CLI:              batch-bench [--mini] [--batch-size N] [--max-inflight N]\n\n", .{});
    try writer.print("Not Stage S1 (explicit non-goals)\n", .{});
    try writer.print("  packed n_seq Metal forward / continuous batch in one CB\n", .{});
    try writer.print("  prefix / paged KV reuse (S2)\n", .{});
    try writer.print("  speculative / MTP (S3)\n", .{});
    try writer.print("  HTTP server (S4)\n", .{});
    try writer.print("  Metal CB packing — that is Stage M3, not request scheduling\n\n", .{});
    try writer.print("Gate\n", .{});
    try writer.print("  >=2 requests; greedy token parity serial vs scheduled (mini)\n", .{});
    try writer.print("  batch-bench prints TTFT / aggregate tok/s + JSON\n", .{});
    try writer.print("  tutorial distinguishes request scheduling vs Metal CB batching\n\n", .{});
    try writer.print("See docs/stages/S1-batching-and-scheduling.md\n", .{});
    try writer.print("See docs/tutorials/23-batching-and-scheduling.md\n", .{});
    try writer.print("See bench/results/stageS1-dev-laptop.md\n", .{});
}

fn printStageS2(writer: *std.Io.Writer) !void {
    try writer.print("zynfer Stage S2 — prefix reuse / cache management\n", .{});
    try writer.print("================================================\n\n", .{});
    try writer.print("Done (dense contiguous)\n", .{});
    try writer.print("  identity:         exact token-id prefix registry (PrefixCache)\n", .{});
    try writer.print("  lookup:           longest registered exact prefix of a prompt\n", .{});
    try writer.print("  eviction:         LRU when max_entries exceeded\n", .{});
    try writer.print("  KV policy:        truncateTo(n) + prefillContinue(suffix)\n", .{});
    try writer.print("  ownership:        same Session retains dense KV; no paged blocks yet\n", .{});
    try writer.print("  CLI:              prefix-bench [--mini] [--prefix-len N] [--suffix-len N]\n\n", .{});
    try writer.print("Not Stage S2 (explicit non-goals)\n", .{});
    try writer.print("  paged / block-allocator KV (teach contiguous first)\n", .{});
    try writer.print("  packed n_seq Metal continuous batching\n", .{});
    try writer.print("  speculative / MTP (S3)\n", .{});
    try writer.print("  HTTP server (S4)\n\n", .{});
    try writer.print("Gate\n", .{});
    try writer.print("  warm suffix prefill logits match cold full prefill (mini)\n", .{});
    try writer.print("  prefix-bench: warm prefill_ns << cold; savings_ratio > 0 + JSON\n", .{});
    try writer.print("  tutorial covers identity / truncate / fragmentation vs paged\n\n", .{});
    try writer.print("See docs/stages/S2-prefix-reuse.md\n", .{});
    try writer.print("See docs/tutorials/24-prefix-reuse.md\n", .{});
    try writer.print("See bench/results/stageS2-dev-laptop.md\n", .{});
}

fn printStageS3(writer: *std.Io.Writer) !void {
    try writer.print("zynfer Stage S3 — speculative decoding\n", .{});
    try writer.print("======================================\n\n", .{});
    try writer.print("Done (n-gram draft + target verify)\n", .{});
    try writer.print("  draft:            history n-gram proposal (no second model / no MTP heads)\n", .{});
    try writer.print("  verify:           greedy argmax vs draft before each decodeToken\n", .{});
    try writer.print("  reject:           commit target truth; end round (verify precedes decode)\n", .{});
    try writer.print("  metrics:          proposal depth, acceptance rate, tokens/round, committed tok/s\n", .{});
    try writer.print("  CLI:              spec-bench [--mini] [--proposal-depth K] [--ngram-order N]\n\n", .{});
    try writer.print("MTP note\n", .{});
    try writer.print("  Registered Qwen3-0.6B/4B are CausalLM only — no MTP tensors in artifacts.\n", .{});
    try writer.print("  MTP / Medusa / EAGLE / draft-LM / layer-skip → future proposal (not S3).\n\n", .{});
    try writer.print("Not Stage S3 (explicit non-goals)\n", .{});
    try writer.print("  separate draft model / EAGLE / Medusa / layer-skip (proposed S3b)\n", .{});
    try writer.print("  tree attention / multi-position parallel verify\n", .{});
    try writer.print("  HTTP server (S4)\n", .{});
    try writer.print("  reporting proposed-token throughput as useful tok/s\n\n", .{});
    try writer.print("Gate\n", .{});
    try writer.print("  greedy token parity vs non-speculative baseline (mini)\n", .{});
    try writer.print("  spec-bench JSON: depth, acceptance, tokens/round, committed tok/s\n", .{});
    try writer.print("  tutorial teaches draft/verify/accept and when speculation loses\n\n", .{});
    try writer.print("See docs/stages/S3-speculative-decoding.md\n", .{});
    try writer.print("See docs/tutorials/25-speculative-decoding.md\n", .{});
    try writer.print("See docs/proposals/speculative-draft-followons.md\n", .{});
    try writer.print("See bench/results/stageS3-dev-laptop.md\n", .{});
}

fn printStageS4(writer: *std.Io.Writer) !void {
    try writer.print("zynfer Stage S4 — HTTP server\n", .{});
    try writer.print("=============================\n\n", .{});
    try writer.print("Done (protocol layer)\n", .{});
    try writer.print("  listen:           std.Io.net + std.http.Server (one request/connection)\n", .{});
    try writer.print("  endpoints:        GET /health, GET /metrics\n", .{});
    try writer.print("                    POST /v1/completions, POST /v1/chat/completions\n", .{});
    try writer.print("  streaming:        SSE text/event-stream via Session.generate on_token\n", .{});
    try writer.print("  separation:       http_server.zig translates HTTP ↔ generate only\n", .{});
    try writer.print("  CLI:              serve [--mini] [--host] [--port] [--smoke]\n\n", .{});
    try writer.print("Not Stage S4 (explicit non-goals)\n", .{});
    try writer.print("  TLS / auth / multi-model routing\n", .{});
    try writer.print("  full OpenAI API parity\n", .{});
    try writer.print("  S1 scheduler over HTTP / concurrent Metal requests\n", .{});
    try writer.print("  S3b draft-model speculation\n\n", .{});
    try writer.print("Gate\n", .{});
    try writer.print("  local client streams tokens over HTTP (serve --smoke / unit)\n", .{});
    try writer.print("  /health + /metrics + streamed /v1/completions\n", .{});
    try writer.print("  tutorial teaches protocol vs engine boundary\n\n", .{});
    try writer.print("See docs/stages/S4-http-server.md\n", .{});
    try writer.print("See docs/tutorials/26-http-server.md\n", .{});
    try writer.print("See bench/results/stageS4-dev-laptop.md\n", .{});
}

fn cmdCoreMlSmoke(writer: *std.Io.Writer, path: []const u8) !void {
    try writer.print("zynfer coreml-smoke — Stage M7 toy load\n", .{});
    try writer.print("======================================\n\n", .{});
    try writer.print("path: {s}\n", .{path});

    const s = zynfer.apple.coreml.smoke(path);
    try writer.print("status:         {d}\n", .{s.status});
    try writer.print("load_ok:        {s}\n", .{yn(s.load_ok)});
    try writer.print("predict_ok:     {s}\n", .{yn(s.predict_ok)});
    try writer.print("compute_units:  {s}\n", .{s.compute_units});
    try writer.print("y:              [{d:.6}, {d:.6}, {d:.6}, {d:.6}]\n", .{ s.y[0], s.y[1], s.y[2], s.y[3] });
    try writer.print("detail:         {s}\n\n", .{s.detail});
    try writer.print("Note: load/predict success does NOT verify ANE placement.\n", .{});
    try writer.print("      ane_execution_verified stays false without Instruments.\n", .{});

    if (comptime !zynfer.apple.coreml.have_coreml) {
        try writer.print("Core ML smoke is unsupported on this platform; skipping.\n", .{});
        try writer.flush();
        return;
    }
    try writer.flush();
    if (s.status != 0 or !s.load_ok or !s.predict_ok) {
        std.process.exit(1);
    }
}

fn cmdMemReport(
    host: zynfer.util.Host,
    writer: *std.Io.Writer,
    artifact_path: ?[]const u8,
    force_mini: bool,
    max_seq_in: u32,
    backend: zynfer.BackendKind,
) !void {
    const allocator = host.gpa;
    const io = host.io;

    try writer.print("zynfer mem-report — Stage M6 static plan\n", .{});
    try writer.print("=======================================\n\n", .{});

    const use_mini = force_mini or artifact_path == null;
    if (use_mini) {
        const bytes = try zynfer.qwen_forward.buildMiniArtifact(allocator);
        defer allocator.free(bytes);
        var art = try zynfer.artifact.Artifact.loadOwned(allocator, try allocator.dupe(u8, bytes));
        defer art.deinit();
        const arch = zynfer.qwen3.stage11_mini;
        const max_seq: usize = if (max_seq_in == 64)
            @intCast(arch.max_position_embeddings)
        else
            @min(@as(usize, max_seq_in), @as(usize, @intCast(arch.max_position_embeddings)));
        var sess = try zynfer.qwen_forward.Session.initWithBackend(allocator, &art, arch, max_seq, backend);
        defer sess.deinit();
        if (backend == .apple and sess.metal_stack != null) {
            const logits = try sess.logits.f32s();
            const prompt = [_]u32{ 2, 3 };
            try sess.prefillLastLogits(&prompt, logits);
        }
        try printMemoryReport(writer, "stage11-mini", sess.memoryReport());
        return;
    }

    const path = artifact_path.?;
    var art = try zynfer.artifact.Artifact.loadFile(allocator, io, path);
    defer art.deinit();
    const arch = try art.meta.toArch();
    const max_seq: usize = if (max_seq_in == 64)
        @min(@as(usize, @intCast(arch.max_position_embeddings)), 2048)
    else
        max_seq_in;
    var sess = try zynfer.qwen_forward.Session.initWithBackend(allocator, &art, arch, max_seq, backend);
    defer sess.deinit();
    try printMemoryReport(writer, path, sess.memoryReport());
}

fn printMemoryReport(writer: *std.Io.Writer, label: []const u8, r: zynfer.qwen_forward.MemoryReport) !void {
    try writer.print("fixture/model:  {s}\n", .{label});
    try writer.print("backend:        {s}\n", .{r.backend});
    try writer.print("max_seq:        {d}\n\n", .{r.max_seq});
    try writer.print("  host_weights      {d}\n", .{r.host_weights_bytes});
    try writer.print("  host_kv_cap       {d}\n", .{r.host_kv_cap_bytes});
    try writer.print("  host_scratch      {d}\n", .{r.host_scratch_bytes});
    try writer.print("  metal_weights     {d}\n", .{r.metal_weights_bytes});
    try writer.print("  metal_kv_cap      {d}\n", .{r.metal_kv_cap_bytes});
    try writer.print("  metal_scratch     {d}\n", .{r.metal_scratch_bytes});
    try writer.print("  accounted_total   {d}\n", .{r.totalAccounted()});
    if (r.peak_rss_bytes) |rss| {
        try writer.print("  peak_rss          {d}\n", .{rss});
    } else {
        try writer.print("  peak_rss          (unavailable)\n", .{});
    }
    try writer.print("\njson\n", .{});
    try writer.print(
        "{{\"cmd\":\"mem-report\",\"backend\":\"{s}\",\"max_seq\":{d},\"host_weights\":{d},\"host_kv_cap\":{d},\"host_scratch\":{d},\"metal_weights\":{d},\"metal_kv_cap\":{d},\"metal_scratch\":{d},\"accounted\":{d}",
        .{
            r.backend,
            r.max_seq,
            r.host_weights_bytes,
            r.host_kv_cap_bytes,
            r.host_scratch_bytes,
            r.metal_weights_bytes,
            r.metal_kv_cap_bytes,
            r.metal_scratch_bytes,
            r.totalAccounted(),
        },
    );
    if (r.peak_rss_bytes) |rss| {
        try writer.print(",\"peak_rss\":{d}}}\n", .{rss});
    } else {
        try writer.print(",\"peak_rss\":null}}\n", .{});
    }
}

const StreamCtx = struct {
    tok: *const zynfer.tokenizer.Tokenizer,
    allocator: std.mem.Allocator,
    io: std.Io,
    writer: *std.Io.Writer,
    pending: std.ArrayList(u8),
    stop_ids: []const u32,

    fn onToken(ctx: ?*anyopaque, token_id: u32) void {
        const self: *StreamCtx = @ptrCast(@alignCast(ctx.?));
        for (self.stop_ids) |s| if (s == token_id) return;
        const piece = self.tok.decode(self.allocator, &.{token_id}) catch return;
        defer self.allocator.free(piece);
        self.pending.appendSlice(self.allocator, piece) catch return;
        self.flushUtf8() catch {};
    }

    fn flushUtf8(self: *StreamCtx) !void {
        var i: usize = 0;
        while (i < self.pending.items.len) {
            const n = std.unicode.utf8ByteSequenceLength(self.pending.items[i]) catch break;
            if (i + n > self.pending.items.len) break;
            _ = std.unicode.utf8Decode(self.pending.items[i..][0..n]) catch break;
            try self.writer.writeAll(self.pending.items[i .. i + n]);
            i += n;
        }
        if (i > 0) {
            const rest = self.pending.items[i..];
            std.mem.copyForwards(u8, self.pending.items[0..rest.len], rest);
            self.pending.shrinkRetainingCapacity(rest.len);
            try self.writer.flush();
        }
    }

    fn finish(self: *StreamCtx) !void {
        if (self.pending.items.len != 0) {
            try self.writer.writeAll(self.pending.items);
            self.pending.clearRetainingCapacity();
            try self.writer.flush();
        }
        try self.writer.writeAll("\n");
        try self.writer.flush();
    }
};

fn runSetup(
    host: zynfer.util.Host,
    writer: *std.Io.Writer,
    skip_pip: bool,
    skip_download: bool,
    skip_golden: bool,
    model: []const u8,
    quantize: bool,
) !void {
    try writer.print("zynfer setup — download registered Qwen3, convert .zynfer\n", .{});
    try writer.print("(live output from python3 tools/setup_qwen.py --model {s})\n\n", .{model});
    try writer.flush();

    var argv_list: std.ArrayList([]const u8) = .empty;
    defer argv_list.deinit(host.gpa);
    try argv_list.append(host.gpa, "python3");
    try argv_list.append(host.gpa, "tools/setup_qwen.py");
    try argv_list.append(host.gpa, "--model");
    try argv_list.append(host.gpa, model);
    if (skip_pip) try argv_list.append(host.gpa, "--skip-pip");
    if (skip_download) try argv_list.append(host.gpa, "--skip-download");
    if (skip_golden) try argv_list.append(host.gpa, "--skip-golden");
    if (quantize) try argv_list.append(host.gpa, "--quantize");

    var child = std.process.spawn(host.io, .{
        .argv = argv_list.items,
        .stdin = .ignore,
        .stdout = .inherit,
        .stderr = .inherit,
    }) catch {
        std.debug.print("setup: failed to spawn python3 tools/setup_qwen.py\n", .{});
        std.process.exit(2);
    };
    const term = child.wait(host.io) catch {
        std.debug.print("setup: wait failed\n", .{});
        std.process.exit(2);
    };
    switch (term) {
        .exited => |code| if (code != 0) {
            std.debug.print("setup: exited with code {d}\n", .{code});
            std.process.exit(2);
        },
        else => {
            std.debug.print("setup: process terminated abnormally\n", .{});
            std.process.exit(2);
        },
    }
}

/// Prefer registered int8 4B, then bf16 4B, then 0.6B int8/bf16 (Stage M8).
fn defaultRegisteredArtifact(io: std.Io) []const u8 {
    const prefer = [_][]const u8{
        "models/qwen3-4b-int8.zynfer",
        "models/qwen3-4b.zynfer",
        "models/qwen3-0.6b-int8.zynfer",
        "models/qwen3-0.6b.zynfer",
    };
    for (prefer) |p| {
        if (zynfer.util.fileExists(io, p)) return p;
    }
    return "models/qwen3-0.6b.zynfer";
}

fn tokenizerDirHasFiles(io: std.Io, dir: []const u8) bool {
    var vbuf: [512]u8 = undefined;
    const vocab = std.fmt.bufPrint(&vbuf, "{s}/vocab.json", .{dir}) catch return false;
    var mbuf: [512]u8 = undefined;
    const merges = std.fmt.bufPrint(&mbuf, "{s}/merges.txt", .{dir}) catch return false;
    return zynfer.util.fileExists(io, vocab) and zynfer.util.fileExists(io, merges);
}

fn resolveTokenizerDir(
    allocator: std.mem.Allocator,
    io: std.Io,
    artifact_path: []const u8,
    explicit: ?[]const u8,
) ![]const u8 {
    if (explicit) |d| {
        if (tokenizerDirHasFiles(io, d)) return try allocator.dupe(u8, d);
        std.debug.print(
            "tokenizer: --tokenizer={s} missing vocab.json and/or merges.txt\n",
            .{d},
        );
        std.process.exit(2);
    }

    // 1) registry HF dirs next to artifact and at conventional paths
    for (zynfer.registry.entries) |entry| {
        if (std.fs.path.dirname(artifact_path)) |parent| {
            var cand_buf: [512]u8 = undefined;
            const base = std.fs.path.basename(entry.hf_dir);
            if (std.fmt.bufPrint(&cand_buf, "{s}/{s}", .{ parent, base })) |cand| {
                if (tokenizerDirHasFiles(io, cand)) return try allocator.dupe(u8, cand);
            } else |_| {}
        }
        if (tokenizerDirHasFiles(io, entry.hf_dir)) {
            return try allocator.dupe(u8, entry.hf_dir);
        }
    }

    std.debug.print(
        \\tokenizer: could not find vocab.json + merges.txt
        \\  looked next to artifact and at registered models/*/ HF dirs
        \\  fix with:  ./zig-out/bin/zynfer setup --model 4b
        \\       or:  --tokenizer models/Qwen3-4B
        \\
    , .{});
    std.process.exit(2);
}

fn percentileNs(sorted: []u64, pct: f64) u64 {
    if (sorted.len == 0) return 0;
    if (sorted.len == 1) return sorted[0];
    const rank = pct / 100.0 * @as(f64, @floatFromInt(sorted.len - 1));
    const lo: usize = @intFromFloat(@floor(rank));
    const hi = @min(lo + 1, sorted.len - 1);
    const frac = rank - @as(f64, @floatFromInt(lo));
    const a: f64 = @floatFromInt(sorted[lo]);
    const b: f64 = @floatFromInt(sorted[hi]);
    return @intFromFloat(a + (b - a) * frac);
}

const ItlStats = struct {
    n: usize = 0,
    p50_ns: u64 = 0,
    p95_ns: u64 = 0,
    p99_ns: u64 = 0,

    fn fromSlice(buf: []u64, count: usize) ItlStats {
        if (count == 0) return .{};
        const scratch = buf[0..count];
        std.mem.sort(u64, scratch, {}, std.sort.asc(u64));
        return .{
            .n = count,
            .p50_ns = percentileNs(scratch, 50),
            .p95_ns = percentileNs(scratch, 95),
            .p99_ns = percentileNs(scratch, 99),
        };
    }
};

fn runGenerate(
    allocator: std.mem.Allocator,
    io: std.Io,
    writer: *std.Io.Writer,
    artifact_path: []const u8,
    prompt: []const u8,
    tokenizer_dir_opt: ?[]const u8,
    max_new_tokens: u32,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    seed: u64,
    raw_prompt: bool,
    stream: bool,
    use_kv_cache: bool,
    backend: zynfer.BackendKind,
) !void {
    const tok_dir = try resolveTokenizerDir(allocator, io, artifact_path, tokenizer_dir_opt);
    defer allocator.free(tok_dir);

    var tok = zynfer.tokenizer.Tokenizer.loadHfDir(allocator, io, tok_dir) catch |err| {
        std.debug.print(
            "run: tokenizer load failed ({s}): {s}\n  expected vocab.json + merges.txt (zynfer setup)\n",
            .{ tok_dir, @errorName(err) },
        );
        std.process.exit(2);
    };
    defer tok.deinit();

    const wrapped = if (raw_prompt)
        try allocator.dupe(u8, prompt)
    else
        try tok.applyChatTemplate(allocator, prompt);
    defer allocator.free(wrapped);

    const prompt_ids = tok.encode(allocator, wrapped) catch |err| {
        std.debug.print("run: encode failed: {s}\n", .{@errorName(err)});
        std.process.exit(2);
    };
    defer allocator.free(prompt_ids);

    var art = zynfer.artifact.Artifact.loadFile(allocator, io, artifact_path) catch |err| {
        std.debug.print(
            "run: load failed ({s}): {s}\n  run: ./zig-out/bin/zynfer setup\n",
            .{ artifact_path, @errorName(err) },
        );
        std.process.exit(2);
    };
    defer art.deinit();

    const arch = try art.meta.toArch();
    const max_seq = prompt_ids.len + max_new_tokens;
    if (max_seq == 0 or max_seq > arch.max_position_embeddings) {
        std.debug.print("run: sequence too long (prompt={d} + max_tokens={d})\n", .{ prompt_ids.len, max_new_tokens });
        std.process.exit(2);
    }

    var sess = try zynfer.qwen_forward.Session.initWithBackend(allocator, &art, arch, max_seq, backend);
    defer sess.deinit();

    const stop_ids = [_]u32{ tok.eos_token_id, tok.endoftext_id, tok.im_end_id };
    var out_ids: std.ArrayList(u32) = .empty;
    defer out_ids.deinit(allocator);
    try out_ids.ensureTotalCapacity(allocator, max_new_tokens);

    const itl_buf = try allocator.alloc(u64, max_new_tokens);
    defer allocator.free(itl_buf);

    var stream_ctx: StreamCtx = .{
        .tok = &tok,
        .allocator = allocator,
        .io = io,
        .writer = writer,
        .pending = .empty,
        .stop_ids = &stop_ids,
    };
    defer stream_ctx.pending.deinit(allocator);
    if (stream) {
        // UTF-8 streaming: reserve ~8 bytes per token to avoid per-token realloc.
        try stream_ctx.pending.ensureTotalCapacity(allocator, @as(usize, max_new_tokens) * 8);
    }

    var rng = std.Random.DefaultPrng.init(seed);
    const stats = sess.generate(io, prompt_ids, &out_ids, .{
        .max_new_tokens = max_new_tokens,
        .sample = .{
            .temperature = temperature,
            .top_k = top_k,
            .top_p = top_p,
            .seed = seed,
        },
        .stop_ids = &stop_ids,
        .itl_ns_out = itl_buf,
        .on_token = if (stream) StreamCtx.onToken else null,
        .on_token_ctx = if (stream) @ptrCast(&stream_ctx) else null,
        .use_kv_cache = use_kv_cache,
    }, &rng) catch |err| {
        std.debug.print("run: generate failed: {s}\n", .{@errorName(err)});
        std.process.exit(2);
    };

    if (stream) {
        try stream_ctx.finish();
    } else {
        const text = tok.decode(allocator, out_ids.items) catch |err| {
            std.debug.print("run: decode failed: {s}\n", .{@errorName(err)});
            std.process.exit(2);
        };
        defer allocator.free(text);
        try writer.print("{s}\n", .{text});
    }

    try writer.print("\n---\n", .{});
    try writer.print("prompt_tokens={d} generated_tokens={d} kv_cache={s} backend={s}\n", .{
        stats.prompt_tokens,
        stats.generated_tokens,
        if (stats.use_kv_cache) "on" else "off",
        sess.backendName(),
    });
    if (use_kv_cache) {
        try writer.print("kv_bytes_used={d} kv_bytes_cap={d}\n", .{ sess.kvBytesUsed(), sess.kvBytesCapacity() });
    }
    // Stage M1: always report prefill and decode as separate regimes.
    try writer.print("prefill_ms={d:.3}", .{@as(f64, @floatFromInt(stats.prefill_ns)) / 1e6});
    if (stats.prompt_tokens > 0 and stats.prefill_ns > 0) {
        const prefill_tok_s = @as(f64, @floatFromInt(stats.prompt_tokens)) /
            (@as(f64, @floatFromInt(stats.prefill_ns)) / 1e9);
        try writer.print(" prefill_tok_s={d:.3}", .{prefill_tok_s});
    }
    try writer.print(" ttft_ms={d:.3}", .{@as(f64, @floatFromInt(stats.ttft_ns)) / 1e6});
    if (stats.generated_tokens > 1 and stats.decode_ns > 0) {
        const decode_tokens = stats.generated_tokens - 1;
        const tok_s = @as(f64, @floatFromInt(decode_tokens)) / (@as(f64, @floatFromInt(stats.decode_ns)) / 1e9);
        const ms_tok = (@as(f64, @floatFromInt(stats.decode_ns)) / 1e6) / @as(f64, @floatFromInt(decode_tokens));
        try writer.print(" decode_tok_s={d:.3} decode_ms_per_tok={d:.3}", .{ tok_s, ms_tok });
    } else if (stats.generated_tokens == 1) {
        try writer.print(" decode_tok_s=n/a (single token; decode interval after first)", .{});
    }
    if (stats.itl_count > 0) {
        const itl = ItlStats.fromSlice(itl_buf, stats.itl_count);
        try writer.print(
            "\nitl_ms p50={d:.3} p95={d:.3} p99={d:.3} (n={d})",
            .{
                @as(f64, @floatFromInt(itl.p50_ns)) / 1e6,
                @as(f64, @floatFromInt(itl.p95_ns)) / 1e6,
                @as(f64, @floatFromInt(itl.p99_ns)) / 1e6,
                itl.n,
            },
        );
    }
    try writer.print("\n", .{});
}

fn runKvBench(
    allocator: std.mem.Allocator,
    io: std.Io,
    writer: *std.Io.Writer,
    artifact_path_opt: ?[]const u8,
    prompt_opt: ?[]const u8,
    tokenizer_dir_opt: ?[]const u8,
    max_new_tokens_in: u32,
    seed: u64,
    raw_prompt: bool,
    force_mini: bool,
    tokens_arg: ?[]const u8,
    layout_only: bool,
) !void {
    if (layout_only) {
        try runKvLayoutBench(allocator, io, writer);
        return;
    }

    const default_path = "models/qwen3-0.6b.zynfer";
    const use_mini = force_mini or (artifact_path_opt == null and !zynfer.util.fileExists(io, default_path));
    const max_new: u32 = if (max_new_tokens_in == 64 and use_mini) 4 else max_new_tokens_in;

    try writer.print("zynfer kv-bench — cached vs uncached decode\n", .{});
    try writer.print("==========================================\n\n", .{});

    var art: zynfer.artifact.Artifact = undefined;
    var prompt_ids: []u32 = undefined;
    var free_prompt = false;
    defer if (free_prompt) allocator.free(prompt_ids);

    var arch: zynfer.qwen3.Arch = undefined;

    if (use_mini) {
        try writer.print("fixture: stage11-mini (in-memory)\n", .{});
        const bytes = try zynfer.qwen_forward.buildMiniArtifact(allocator);
        art = try zynfer.artifact.Artifact.loadOwned(allocator, bytes);
        arch = zynfer.qwen3.stage11_mini;
        if (tokens_arg) |s| {
            prompt_ids = try parseCsvTokenIds(allocator, s);
            free_prompt = true;
        } else {
            prompt_ids = try allocator.dupe(u32, &.{ 2, 3 });
            free_prompt = true;
        }
    } else {
        const path = artifact_path_opt orelse default_path;
        try writer.print("artifact: {s}\n", .{path});
        art = zynfer.artifact.Artifact.loadFile(allocator, io, path) catch |err| {
            std.debug.print("kv-bench: load failed ({s}): {s}\n", .{ path, @errorName(err) });
            std.process.exit(2);
        };
        arch = try art.meta.toArch();

        if (tokens_arg) |s| {
            prompt_ids = try parseCsvTokenIds(allocator, s);
            free_prompt = true;
        } else {
            const prompt = prompt_opt orelse "Explain gravity simply.";
            const tok_dir = try resolveTokenizerDir(allocator, io, path, tokenizer_dir_opt);
            defer allocator.free(tok_dir);
            var tok = zynfer.tokenizer.Tokenizer.loadHfDir(allocator, io, tok_dir) catch |err| {
                std.debug.print("kv-bench: tokenizer load failed ({s}): {s}\n", .{ tok_dir, @errorName(err) });
                std.process.exit(2);
            };
            defer tok.deinit();
            const wrapped = if (raw_prompt)
                try allocator.dupe(u8, prompt)
            else
                try tok.applyChatTemplate(allocator, prompt);
            defer allocator.free(wrapped);
            prompt_ids = tok.encode(allocator, wrapped) catch |err| {
                std.debug.print("kv-bench: encode failed: {s}\n", .{@errorName(err)});
                std.process.exit(2);
            };
            free_prompt = true;
        }
    }
    defer art.deinit();

    const max_seq = prompt_ids.len + max_new;
    if (max_seq == 0 or max_seq > arch.max_position_embeddings) {
        std.debug.print("kv-bench: sequence too long\n", .{});
        std.process.exit(2);
    }

    const kv_cap = zynfer.kv_cache.estimateModelBytes(
        arch.num_layers,
        arch.num_key_value_heads,
        max_seq,
        arch.head_dim,
    );
    try writer.print("prompt_tokens={d} max_new={d} kv_cap_bytes={d}\n\n", .{ prompt_ids.len, max_new, kv_cap });

    var cached_ids: std.ArrayList(u32) = .empty;
    defer cached_ids.deinit(allocator);
    var uncached_ids: std.ArrayList(u32) = .empty;
    defer uncached_ids.deinit(allocator);

    var sess_c = try zynfer.qwen_forward.Session.init(allocator, &art, arch, max_seq);
    defer sess_c.deinit();
    var rng_c = std.Random.DefaultPrng.init(seed);
    const stats_c = try sess_c.generate(io, prompt_ids, &cached_ids, .{
        .max_new_tokens = max_new,
        .sample = .{ .temperature = 0, .seed = seed },
        .use_kv_cache = true,
    }, &rng_c);

    var sess_u = try zynfer.qwen_forward.Session.init(allocator, &art, arch, max_seq);
    defer sess_u.deinit();
    var rng_u = std.Random.DefaultPrng.init(seed);
    const stats_u = try sess_u.generate(io, prompt_ids, &uncached_ids, .{
        .max_new_tokens = max_new,
        .sample = .{ .temperature = 0, .seed = seed },
        .use_kv_cache = false,
    }, &rng_u);

    const match = std.mem.eql(u32, cached_ids.items, uncached_ids.items);
    try writer.print("token_parity: {s}\n", .{if (match) "PASS" else "FAIL"});
    try writer.print("cached_ids:   ", .{});
    for (cached_ids.items, 0..) |id, i| {
        if (i != 0) try writer.writeAll(",");
        try writer.print("{d}", .{id});
    }
    try writer.print("\nuncached_ids: ", .{});
    for (uncached_ids.items, 0..) |id, i| {
        if (i != 0) try writer.writeAll(",");
        try writer.print("{d}", .{id});
    }
    try writer.print("\n\n", .{});

    try printKvBenchRow(writer, "cached", stats_c, sess_c.kvBytesUsed(), sess_c.kvBytesCapacity());
    try printKvBenchRow(writer, "uncached", stats_u, 0, 0);

    if (stats_c.decode_ns > 0 and stats_u.decode_ns > 0 and stats_c.generated_tokens > 1) {
        const speedup = @as(f64, @floatFromInt(stats_u.decode_ns)) / @as(f64, @floatFromInt(stats_c.decode_ns));
        try writer.print("\ndecode_speedup (uncached/cached wall): {d:.2}×\n", .{speedup});
    }

    if (!match) {
        try writer.flush();
        std.process.exit(1);
    }
}

fn printKvBenchRow(
    writer: *std.Io.Writer,
    label: []const u8,
    stats: zynfer.qwen_forward.Session.GenerateStats,
    kv_used: u64,
    kv_cap: u64,
) !void {
    try writer.print("{s}:\n", .{label});
    try writer.print("  generated={d} prefill_ms={d:.3} ttft_ms={d:.3}", .{
        stats.generated_tokens,
        @as(f64, @floatFromInt(stats.prefill_ns)) / 1e6,
        @as(f64, @floatFromInt(stats.ttft_ns)) / 1e6,
    });
    if (stats.generated_tokens > 1 and stats.decode_ns > 0) {
        const n = stats.generated_tokens - 1;
        const tok_s = @as(f64, @floatFromInt(n)) / (@as(f64, @floatFromInt(stats.decode_ns)) / 1e9);
        try writer.print(" decode_tok_s={d:.3}", .{tok_s});
    }
    if (kv_cap > 0) {
        try writer.print("\n  kv_bytes_used={d} kv_bytes_cap={d}", .{ kv_used, kv_cap });
    }
    try writer.print("\n", .{});
}

fn runKvLayoutBench(allocator: std.mem.Allocator, io: std.Io, writer: *std.Io.Writer) !void {
    try writer.print("zynfer kv-bench --layout — host KV physical layout bake-off\n", .{});
    try writer.print("==========================================================\n\n", .{});
    try writer.print("Shapes mimic Qwen3-0.6B GQA (n_q=16, n_kv=8, head_dim=128).\n", .{});
    try writer.print("attn = decode attention K/V scan; append = write one token.\n\n", .{});

    const report = try zynfer.kv_cache.benchLayouts(allocator, io, .{});
    try writer.print(
        "cfg: n_kv={d} n_q={d} max_seq={d} kv_len={d} head_dim={d} warmup={d} iters={d}\n\n",
        .{
            report.cfg.n_kv,
            report.cfg.n_q,
            report.cfg.max_seq,
            report.cfg.kv_len,
            report.cfg.head_dim,
            report.cfg.warmup,
            report.cfg.iters,
        },
    );

    for (report.rows) |row| {
        const tag: []const u8 = if (row.layout.retained()) "RETAIN" else "reject";
        try writer.print("{s}  {s}\n", .{ tag, row.layout.name() });
        try writer.print("  attn_ns={d}  append_ns={d}\n", .{ row.attn_ns, row.append_ns });
    }

    const attn_ratio = @as(f64, @floatFromInt(report.rows[1].attn_ns)) /
        @as(f64, @floatFromInt(@max(report.rows[0].attn_ns, 1)));
    const append_ratio = @as(f64, @floatFromInt(report.rows[0].append_ns)) /
        @as(f64, @floatFromInt(@max(report.rows[1].append_ns, 1)));
    try writer.print("\nattn speedup (seq-outer / heads-outer): {d:.2}×\n", .{attn_ratio});
    try writer.print("append cost ratio (heads-outer / seq-outer): {d:.2}×\n", .{append_ratio});
    try writer.print(
        "\nDecision: retain [n_kv, max_seq, head_dim] — decode attention is the hot path;\n",
        .{},
    );
    try writer.print("contiguous per-head prefixes beat strided seq-outer gathers.\n", .{});
}

/// Stage M1: prefill vs decode split report on CPU and (when available) Apple Metal.
fn runQwenBench(
    allocator: std.mem.Allocator,
    io: std.Io,
    writer: *std.Io.Writer,
    artifact_path_opt: ?[]const u8,
    prompt_opt: ?[]const u8,
    tokenizer_dir_opt: ?[]const u8,
    max_new_tokens_in: u32,
    seed: u64,
    raw_prompt: bool,
    force_mini: bool,
    tokens_arg: ?[]const u8,
) !void {
    const default_path = "models/qwen3-0.6b.zynfer";
    const use_mini = force_mini or (artifact_path_opt == null and !zynfer.util.fileExists(io, default_path));
    const max_new: u32 = if (max_new_tokens_in == 64 and use_mini) 4 else if (max_new_tokens_in == 64) 16 else max_new_tokens_in;

    try writer.print("zynfer qwen-bench — prefill vs decode (Stage M1)\n", .{});
    try writer.print("================================================\n\n", .{});

    var art: zynfer.artifact.Artifact = undefined;
    var prompt_ids: []u32 = undefined;
    var free_prompt = false;
    defer if (free_prompt) allocator.free(prompt_ids);
    var arch: zynfer.qwen3.Arch = undefined;

    if (use_mini) {
        try writer.print("fixture: stage11-mini (in-memory)\n", .{});
        const bytes = try zynfer.qwen_forward.buildMiniArtifact(allocator);
        art = try zynfer.artifact.Artifact.loadOwned(allocator, bytes);
        arch = zynfer.qwen3.stage11_mini;
        if (tokens_arg) |s| {
            prompt_ids = try parseCsvTokenIds(allocator, s);
            free_prompt = true;
        } else {
            prompt_ids = try allocator.dupe(u32, &.{ 2, 3 });
            free_prompt = true;
        }
    } else {
        const path = artifact_path_opt orelse default_path;
        try writer.print("artifact: {s}\n", .{path});
        art = zynfer.artifact.Artifact.loadFile(allocator, io, path) catch |err| {
            std.debug.print("qwen-bench: load failed ({s}): {s}\n", .{ path, @errorName(err) });
            std.process.exit(2);
        };
        arch = try art.meta.toArch();
        if (tokens_arg) |s| {
            prompt_ids = try parseCsvTokenIds(allocator, s);
            free_prompt = true;
        } else {
            const prompt = prompt_opt orelse "Explain gravity simply.";
            const tok_dir = try resolveTokenizerDir(allocator, io, path, tokenizer_dir_opt);
            defer allocator.free(tok_dir);
            var tok = zynfer.tokenizer.Tokenizer.loadHfDir(allocator, io, tok_dir) catch |err| {
                std.debug.print("qwen-bench: tokenizer load failed ({s}): {s}\n", .{ tok_dir, @errorName(err) });
                std.process.exit(2);
            };
            defer tok.deinit();
            const wrapped = if (raw_prompt)
                try allocator.dupe(u8, prompt)
            else
                try tok.applyChatTemplate(allocator, prompt);
            defer allocator.free(wrapped);
            prompt_ids = tok.encode(allocator, wrapped) catch |err| {
                std.debug.print("qwen-bench: encode failed: {s}\n", .{@errorName(err)});
                std.process.exit(2);
            };
            free_prompt = true;
        }
    }
    defer art.deinit();

    const max_seq = prompt_ids.len + max_new;
    if (max_seq == 0 or max_seq > arch.max_position_embeddings) {
        std.debug.print("qwen-bench: sequence too long\n", .{});
        std.process.exit(2);
    }

    var gemm_buf: [512]u8 = undefined;
    const gemm_prefill = arch.describePrefillGemms(prompt_ids.len, &gemm_buf);
    var gemm_dec_buf: [512]u8 = undefined;
    const gemm_decode = arch.describeDecodeGemms(&gemm_dec_buf);

    try writer.print("prompt_tokens={d} max_new={d} layers={d} hidden={d}\n", .{
        prompt_ids.len,
        max_new,
        arch.num_layers,
        arch.hidden_size,
    });
    try writer.print("prefill_gemms: {s}\n", .{gemm_prefill});
    try writer.print("decode_gemms:  {s}\n\n", .{gemm_decode});

    try writer.print("{s:<8} {s:>12} {s:>12} {s:>12} {s:>12} {s:>14} {s:>12} {s:>10} {s:>10}\n", .{
        "backend",
        "prefill_ms",
        "prefill_t/s",
        "ttft_ms",
        "decode_t/s",
        "decode_ms/tok",
        "B/tok_est",
        "enc/tok",
        "wait/tok",
    });
    try writer.print("{s:-<8} {s:->12} {s:->12} {s:->12} {s:->12} {s:->14} {s:->12} {s:->10} {s:->10}\n", .{
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    });

    const Row = struct {
        backend: []const u8,
        prefill_ns: u64,
        ttft_ns: u64,
        decode_ns: u64,
        prompt_tokens: usize,
        generated_tokens: usize,
        bytes_per_tok: u64,
        encodes_per_tok: f64,
        waits_per_tok: f64,
        metal_encodes_prefill: u64,
        metal_waits_prefill: u64,
        metal_encodes_decode: u64,
        metal_waits_decode: u64,
        decode_steps: usize,
        itl_n: usize,
        itl_p50_ns: u64,
        itl_p95_ns: u64,
        itl_p99_ns: u64,
    };

    var rows: [2]Row = undefined;
    var n_rows: usize = 0;

    const kinds = [_]zynfer.BackendKind{ .cpu, .apple };
    for (kinds) |kind| {
        if (kind == .apple) {
            if (!zynfer.backend.isBackendBuildable(.apple)) continue;
            zynfer.backend.requireBackend(.apple) catch continue;
        }

        var out_ids: std.ArrayList(u32) = .empty;
        defer out_ids.deinit(allocator);
        try out_ids.ensureTotalCapacity(allocator, max_new);

        const itl_buf = try allocator.alloc(u64, max_new);
        defer allocator.free(itl_buf);

        var sess = zynfer.qwen_forward.Session.initWithBackend(allocator, &art, arch, max_seq, kind) catch |err| {
            try writer.print("{s:<8}  SKIP ({s})\n", .{ kind.name(), @errorName(err) });
            continue;
        };
        defer sess.deinit();

        var rng = std.Random.DefaultPrng.init(seed);
        const stats = sess.generate(io, prompt_ids, &out_ids, .{
            .max_new_tokens = max_new,
            .sample = .{ .temperature = 0, .seed = seed },
            .use_kv_cache = true,
            .itl_ns_out = itl_buf,
        }, &rng) catch |err| {
            try writer.print("{s:<8}  FAIL ({s})\n", .{ kind.name(), @errorName(err) });
            continue;
        };

        const kv_len = prompt_ids.len + @max(stats.generated_tokens, 1);
        const bytes_tok = if (kind == .apple and zynfer.apple.qwen_schedule.useQ8Path())
            arch.estimateDecodeBytesPerTokenQ8(kv_len)
        else if (kind == .apple and zynfer.apple.qwen_schedule.useHalfPath())
            arch.estimateDecodeBytesPerTokenHalf(kv_len)
        else
            arch.estimateDecodeBytesPerToken(kv_len);
        const denom: f64 = @floatFromInt(@max(stats.decode_steps, 1));
        const enc_tok: f64 = if (stats.decode_steps > 0)
            @as(f64, @floatFromInt(stats.metal_encodes_decode)) / denom
        else
            0;
        const wait_tok: f64 = if (stats.decode_steps > 0)
            @as(f64, @floatFromInt(stats.metal_waits_decode)) / denom
        else
            0;

        const prefill_ms = @as(f64, @floatFromInt(stats.prefill_ns)) / 1e6;
        const ttft_ms = @as(f64, @floatFromInt(stats.ttft_ns)) / 1e6;
        const prefill_tok_s: f64 = if (stats.prefill_ns > 0)
            @as(f64, @floatFromInt(stats.prompt_tokens)) / (@as(f64, @floatFromInt(stats.prefill_ns)) / 1e9)
        else
            0;
        var decode_tok_s: f64 = 0;
        var decode_ms_tok: f64 = 0;
        if (stats.generated_tokens > 1 and stats.decode_ns > 0) {
            const n = stats.generated_tokens - 1;
            decode_tok_s = @as(f64, @floatFromInt(n)) / (@as(f64, @floatFromInt(stats.decode_ns)) / 1e9);
            decode_ms_tok = (@as(f64, @floatFromInt(stats.decode_ns)) / 1e6) / @as(f64, @floatFromInt(n));
        }

        const itl = ItlStats.fromSlice(itl_buf, stats.itl_count);

        try writer.print("{s:<8} {d:>12.3} {d:>12.3} {d:>12.3} {d:>12.3} {d:>14.3} {d:>12} {d:>10.1} {d:>10.1}\n", .{
            kind.name(),
            prefill_ms,
            prefill_tok_s,
            ttft_ms,
            decode_tok_s,
            decode_ms_tok,
            bytes_tok,
            enc_tok,
            wait_tok,
        });
        if (itl.n > 0) {
            try writer.print(
                "         itl_ms p50={d:.3} p95={d:.3} p99={d:.3} (n={d})\n",
                .{
                    @as(f64, @floatFromInt(itl.p50_ns)) / 1e6,
                    @as(f64, @floatFromInt(itl.p95_ns)) / 1e6,
                    @as(f64, @floatFromInt(itl.p99_ns)) / 1e6,
                    itl.n,
                },
            );
        }

        rows[n_rows] = .{
            .backend = kind.name(),
            .prefill_ns = stats.prefill_ns,
            .ttft_ns = stats.ttft_ns,
            .decode_ns = stats.decode_ns,
            .prompt_tokens = stats.prompt_tokens,
            .generated_tokens = stats.generated_tokens,
            .bytes_per_tok = bytes_tok,
            .encodes_per_tok = enc_tok,
            .waits_per_tok = wait_tok,
            .metal_encodes_prefill = stats.metal_encodes_prefill,
            .metal_waits_prefill = stats.metal_waits_prefill,
            .metal_encodes_decode = stats.metal_encodes_decode,
            .metal_waits_decode = stats.metal_waits_decode,
            .decode_steps = stats.decode_steps,
            .itl_n = itl.n,
            .itl_p50_ns = itl.p50_ns,
            .itl_p95_ns = itl.p95_ns,
            .itl_p99_ns = itl.p99_ns,
        };
        n_rows += 1;
    }

    try writer.print("\nnotes:\n", .{});
    try writer.print("  enc/tok + wait/tok = measured Metal launches / waits per decodeToken.\n", .{});
    try writer.print("  M3 default (batched): waits ≈ 2/forward; M0 baseline: waits ≈ encodes.\n", .{});
    try writer.print("  Force baseline with ZYNFER_QWEN_METAL=baseline. CPU rows show 0.\n", .{});
    try writer.print("  B/tok_est = f32 weight reads + KV read at end-of-run kv_len (approx).\n", .{});
    try writer.print("  decode_t/s uses generated_tokens-1 (intervals after first token).\n", .{});
    try writer.print("  ITL p50/p95/p99 (Stage M6): inter-token latency after first token; apple row when n≥2.\n", .{});
    var ri: usize = 0;
    while (ri < n_rows) : (ri += 1) {
        const r = rows[ri];
        if (r.metal_encodes_prefill > 0 or r.metal_waits_prefill > 0) {
            try writer.print(
                "  {s} prefill measured: encodes={d} waits={d}; decode totals: encodes={d} waits={d} steps={d}\n",
                .{
                    r.backend,
                    r.metal_encodes_prefill,
                    r.metal_waits_prefill,
                    r.metal_encodes_decode,
                    r.metal_waits_decode,
                    r.decode_steps,
                },
            );
        }
    }
    try writer.print("\n", .{});

    try writer.print("json\n", .{});
    try writer.print("{{\"cmd\":\"qwen-bench\",\"prompt_tokens\":{d},\"max_new\":{d},\"layers\":{d},\"hidden\":{d},\"mini\":{},\"rows\":[", .{
        prompt_ids.len,
        max_new,
        arch.num_layers,
        arch.hidden_size,
        use_mini,
    });
    var i: usize = 0;
    while (i < n_rows) : (i += 1) {
        if (i != 0) try writer.writeAll(",");
        const r = rows[i];
        try writer.print(
            "{{\"backend\":\"{s}\",\"prefill_ns\":{d},\"ttft_ns\":{d},\"decode_ns\":{d},\"generated_tokens\":{d},\"bytes_per_tok_est\":{d},\"metal_encodes_prefill\":{d},\"metal_waits_prefill\":{d},\"metal_encodes_decode\":{d},\"metal_waits_decode\":{d},\"decode_steps\":{d},\"encodes_per_tok\":{d:.3},\"waits_per_tok\":{d:.3},\"itl_n\":{d},\"itl_p50_ns\":{d},\"itl_p95_ns\":{d},\"itl_p99_ns\":{d}}}",
            .{
                r.backend,
                r.prefill_ns,
                r.ttft_ns,
                r.decode_ns,
                r.generated_tokens,
                r.bytes_per_tok,
                r.metal_encodes_prefill,
                r.metal_waits_prefill,
                r.metal_encodes_decode,
                r.metal_waits_decode,
                r.decode_steps,
                r.encodes_per_tok,
                r.waits_per_tok,
                r.itl_n,
                r.itl_p50_ns,
                r.itl_p95_ns,
                r.itl_p99_ns,
            },
        );
    }
    try writer.print("]}}\n", .{});

    if (n_rows == 0) {
        try writer.flush();
        std.process.exit(1);
    }
}

/// Stage S1: serial vs scheduled multi-request A/B (independent Sessions).
fn runBatchBench(
    allocator: std.mem.Allocator,
    io: std.Io,
    writer: *std.Io.Writer,
    artifact_path_opt: ?[]const u8,
    force_mini: bool,
    batch_size_in: u32,
    max_inflight_in: u32,
    max_new_tokens_in: u32,
    seed: u64,
    kind: zynfer.BackendKind,
) !void {
    const default_path = "models/qwen3-0.6b.zynfer";
    const use_mini = force_mini or (artifact_path_opt == null and !zynfer.util.fileExists(io, default_path));
    const batch_size: usize = if (batch_size_in == 0) 2 else batch_size_in;
    const inflight: usize = if (max_inflight_in == 0) 2 else max_inflight_in;
    const max_new: u32 = if (max_new_tokens_in == 64 and use_mini) 4 else if (max_new_tokens_in == 64) 8 else max_new_tokens_in;

    try writer.print("zynfer batch-bench — request scheduling (Stage S1)\n", .{});
    try writer.print("=================================================\n\n", .{});
    try writer.print("note: request-level FIFO+RR over independent Sessions;\n", .{});
    try writer.print("      not Metal CB packing (M3) and not packed n_seq forward.\n\n", .{});

    var art: zynfer.artifact.Artifact = undefined;
    var arch: zynfer.qwen3.Arch = undefined;
    var prompt_pool: [4][]const u32 = undefined;
    var owned_prompts: [4]?[]u32 = .{ null, null, null, null };
    defer for (owned_prompts) |p| if (p) |ids| allocator.free(ids);

    if (use_mini) {
        try writer.print("fixture: stage11-mini (in-memory)\n", .{});
        const bytes = try zynfer.qwen_forward.buildMiniArtifact(allocator);
        art = try zynfer.artifact.Artifact.loadOwned(allocator, bytes);
        arch = zynfer.qwen3.stage11_mini;
        const a = try allocator.dupe(u32, &.{ 2, 3 });
        const b = try allocator.dupe(u32, &.{ 2, 4 });
        const c = try allocator.dupe(u32, &.{ 2, 5 });
        const d = try allocator.dupe(u32, &.{ 2, 6 });
        owned_prompts = .{ a, b, c, d };
        prompt_pool = .{ a, b, c, d };
    } else {
        const path = artifact_path_opt orelse default_path;
        try writer.print("artifact: {s}\n", .{path});
        art = zynfer.artifact.Artifact.loadFile(allocator, io, path) catch |err| {
            std.debug.print("batch-bench: load failed ({s}): {s}\n", .{ path, @errorName(err) });
            std.process.exit(2);
        };
        arch = try art.meta.toArch();
        // Distinct token prompts without requiring a tokenizer (CSV-style ids).
        const a = try allocator.dupe(u32, &.{ 151643, 8948 });
        const b = try allocator.dupe(u32, &.{ 151643, 872 });
        const c = try allocator.dupe(u32, &.{ 151643, 2610 });
        const d = try allocator.dupe(u32, &.{ 151643, 320 });
        owned_prompts = .{ a, b, c, d };
        prompt_pool = .{ a, b, c, d };
    }
    defer art.deinit();

    if (batch_size == 0) {
        std.debug.print("batch-bench: --batch-size must be >= 1\n", .{});
        std.process.exit(2);
    }

    const jobs = try allocator.alloc(zynfer.scheduler.Job, batch_size);
    defer allocator.free(jobs);
    var max_prompt: usize = 0;
    for (jobs, 0..) |*job, i| {
        const ids = prompt_pool[i % prompt_pool.len];
        job.* = .{
            .prompt_ids = ids,
            .max_new_tokens = max_new,
            .sample = .{ .temperature = 0 },
            .seed = seed +% i,
        };
        max_prompt = @max(max_prompt, ids.len);
    }
    const max_seq = max_prompt + max_new;
    if (max_seq == 0 or max_seq > arch.max_position_embeddings) {
        std.debug.print("batch-bench: sequence too long\n", .{});
        std.process.exit(2);
    }

    try writer.print("backend={s} batch_size={d} max_inflight={d} max_new={d} max_seq={d}\n\n", .{
        kind.name(),
        batch_size,
        inflight,
        max_new,
        max_seq,
    });

    var serial = zynfer.scheduler.runSerial(allocator, io, &art, arch, kind, max_seq, jobs) catch |err| {
        std.debug.print("batch-bench: serial failed: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
    defer serial.deinit();

    var scheduled = zynfer.scheduler.runScheduled(allocator, io, &art, arch, kind, max_seq, jobs, inflight) catch |err| {
        std.debug.print("batch-bench: scheduled failed: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
    defer scheduled.deinit();

    var parity_ok = true;
    if (serial.n_requests != scheduled.n_requests) parity_ok = false;
    for (serial.requests, scheduled.requests) |a, b| {
        if (!std.mem.eql(u32, a.token_ids, b.token_ids)) {
            parity_ok = false;
            break;
        }
    }

    try writer.print("{s:<10} {s:>10} {s:>12} {s:>14} {s:>10}\n", .{
        "mode",
        "inflight",
        "wall_ms",
        "agg_tok/s",
        "gen_tok",
    });
    try writer.print("{s:-<10} {s:->10} {s:->12} {s:->14} {s:->10}\n", .{ "", "", "", "", "" });
    inline for (.{ &serial, &scheduled }) |rep| {
        try writer.print("{s:<10} {d:>10} {d:>12.3} {d:>14.3} {d:>10}\n", .{
            rep.mode,
            rep.max_inflight,
            @as(f64, @floatFromInt(rep.wall_ns)) / 1e6,
            rep.aggregate_tok_s,
            rep.total_generated,
        });
    }
    try writer.writeAll("\n");

    try writer.print("{s:<6} {s:<10} {s:>8} {s:>10} {s:>10} {s:>10} {s:>10} {s:>12}\n", .{
        "id",
        "mode",
        "gen",
        "ttft_ms",
        "e2e_ms",
        "itl_ms",
        "tok/s",
        "queue_ms",
    });
    try writer.print("{s:-<6} {s:-<10} {s:->8} {s:->10} {s:->10} {s:->10} {s:->10} {s:->12}\n", .{
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    });

    const printReq = struct {
        fn go(w: *std.Io.Writer, mode: []const u8, r: zynfer.scheduler.RequestStats) !void {
            const e2e_s = @as(f64, @floatFromInt(r.e2e_ns)) * 1e-9;
            const tok_s = if (e2e_s == 0) 0 else @as(f64, @floatFromInt(r.generated_tokens)) / e2e_s;
            try w.print("{d:<6} {s:<10} {d:>8} {d:>10.3} {d:>10.3} {d:>10.3} {d:>10.3} {d:>12.3}\n", .{
                r.id,
                mode,
                r.generated_tokens,
                @as(f64, @floatFromInt(r.ttft_ns)) / 1e6,
                @as(f64, @floatFromInt(r.e2e_ns)) / 1e6,
                @as(f64, @floatFromInt(r.mean_itl_ns)) / 1e6,
                tok_s,
                @as(f64, @floatFromInt(r.queue_wait_ns)) / 1e6,
            });
        }
    }.go;

    for (serial.requests) |r| try printReq(writer, "serial", r);
    for (scheduled.requests) |r| try printReq(writer, "scheduled", r);

    try writer.print("\ntoken_parity: {s}\n\n", .{if (parity_ok) "PASS" else "FAIL"});

    try writer.print("json\n", .{});
    try writer.print("{{\"cmd\":\"batch-bench\",\"backend\":\"{s}\",\"mini\":{},\"batch_size\":{d},\"max_inflight\":{d},\"max_new\":{d},\"token_parity\":{},\"serial\":{{\"wall_ns\":{d},\"aggregate_tok_s\":{d:.6},\"total_generated\":{d}}},\"scheduled\":{{\"wall_ns\":{d},\"aggregate_tok_s\":{d:.6},\"total_generated\":{d},\"max_inflight\":{d}}},\"requests\":[", .{
        kind.name(),
        use_mini,
        batch_size,
        inflight,
        max_new,
        parity_ok,
        serial.wall_ns,
        serial.aggregate_tok_s,
        serial.total_generated,
        scheduled.wall_ns,
        scheduled.aggregate_tok_s,
        scheduled.total_generated,
        scheduled.max_inflight,
    });
    for (scheduled.requests, 0..) |r, i| {
        if (i != 0) try writer.writeAll(",");
        const e2e_s = @as(f64, @floatFromInt(r.e2e_ns)) * 1e-9;
        const tok_s = if (e2e_s == 0) 0 else @as(f64, @floatFromInt(r.generated_tokens)) / e2e_s;
        try writer.print("{{\"id\":{d},\"prompt_tokens\":{d},\"generated\":{d},\"ttft_ns\":{d},\"e2e_ns\":{d},\"mean_itl_ns\":{d},\"queue_wait_ns\":{d},\"tok_s\":{d:.6}}}", .{
            r.id,
            r.prompt_tokens,
            r.generated_tokens,
            r.ttft_ns,
            r.e2e_ns,
            r.mean_itl_ns,
            r.queue_wait_ns,
            tok_s,
        });
    }
    try writer.print("]}}\n", .{});

    if (!parity_ok) {
        try writer.flush();
        std.process.exit(1);
    }
}

/// Stage S2: cold full prefill vs warm truncate+continue on a shared prefix.
fn runPrefixBench(
    allocator: std.mem.Allocator,
    io: std.Io,
    writer: *std.Io.Writer,
    artifact_path_opt: ?[]const u8,
    force_mini: bool,
    prefix_len_in: u32,
    suffix_len_in: u32,
    trials_in: u32,
    kind: zynfer.BackendKind,
) !void {
    const default_path = "models/qwen3-0.6b.zynfer";
    const use_mini = force_mini or (artifact_path_opt == null and !zynfer.util.fileExists(io, default_path));
    const n_trials: usize = if (trials_in == 0) 4 else trials_in;
    const pref_len: usize = if (prefix_len_in != 0) prefix_len_in else if (use_mini) 8 else 64;
    const suf_len: usize = if (suffix_len_in != 0) suffix_len_in else if (use_mini) 2 else 8;

    try writer.print("zynfer prefix-bench — prefix reuse (Stage S2)\n", .{});
    try writer.print("=============================================\n\n", .{});
    try writer.print("note: dense truncateTo + prefillContinue; not paged KV / packed n_seq.\n\n", .{});

    var art: zynfer.artifact.Artifact = undefined;
    var arch: zynfer.qwen3.Arch = undefined;
    if (use_mini) {
        try writer.print("fixture: stage11-mini (in-memory)\n", .{});
        const bytes = try zynfer.qwen_forward.buildMiniArtifact(allocator);
        art = try zynfer.artifact.Artifact.loadOwned(allocator, bytes);
        arch = zynfer.qwen3.stage11_mini;
    } else {
        const path = artifact_path_opt orelse default_path;
        try writer.print("artifact: {s}\n", .{path});
        art = zynfer.artifact.Artifact.loadFile(allocator, io, path) catch |err| {
            std.debug.print("prefix-bench: load failed ({s}): {s}\n", .{ path, @errorName(err) });
            std.process.exit(2);
        };
        arch = try art.meta.toArch();
    }
    defer art.deinit();

    if (pref_len == 0 or suf_len == 0 or n_trials == 0) {
        std.debug.print("prefix-bench: --prefix-len/--suffix-len/--trials must be >= 1\n", .{});
        std.process.exit(2);
    }

    const max_seq = pref_len + suf_len;
    if (max_seq > arch.max_position_embeddings) {
        std.debug.print("prefix-bench: sequence too long for arch\n", .{});
        std.process.exit(2);
    }

    const prefix = try allocator.alloc(u32, pref_len);
    defer allocator.free(prefix);
    // Stable ids in vocab range (mini vocab is tiny — stay in 2..vocab-1).
    const vocab_u: u32 = @intCast(arch.vocab_size);
    var pi: usize = 0;
    while (pi < pref_len) : (pi += 1) {
        prefix[pi] = 2 + @as(u32, @intCast(pi % @max(@as(usize, 1), @as(usize, vocab_u) -| 3)));
    }

    var suffix_store = try allocator.alloc(u32, n_trials * suf_len);
    defer allocator.free(suffix_store);
    const suffixes = try allocator.alloc([]const u32, n_trials);
    defer allocator.free(suffixes);
    var ti: usize = 0;
    while (ti < n_trials) : (ti += 1) {
        const slice = suffix_store[ti * suf_len ..][0..suf_len];
        var si: usize = 0;
        while (si < suf_len) : (si += 1) {
            slice[si] = 2 + @as(u32, @intCast((ti * 3 + si + 1) % @max(@as(usize, 1), @as(usize, vocab_u) -| 3)));
        }
        suffixes[ti] = slice;
    }

    // Register identity in PrefixCache (lookup smoke for the ledger narrative).
    var cache = zynfer.prefix_cache.PrefixCache.init(allocator, 4);
    defer cache.deinit();
    try cache.insert(prefix);
    const looked = cache.lookupLongest(prefix) orelse {
        std.debug.print("prefix-bench: PrefixCache lookup failed\n", .{});
        std.process.exit(1);
    };

    const atol: f32 = if (use_mini) 1e-4 else 3e-3;
    var report = zynfer.prefix_cache.runColdWarmPrefill(
        allocator,
        io,
        &art,
        arch,
        kind,
        max_seq,
        prefix,
        suffixes,
        atol,
    ) catch |err| {
        std.debug.print("prefix-bench: failed: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
    defer report.deinit();

    try writer.print("backend={s} prefix_len={d} suffix_len={d} trials={d} cache_hit_len={d}\n\n", .{
        kind.name(),
        pref_len,
        suf_len,
        n_trials,
        looked.prefix_len,
    });

    try writer.print("{s:<8} {s:>14} {s:>14} {s:>12}\n", .{ "mode", "prefill_ms", "tokens", "ms/tok" });
    try writer.print("{s:-<8} {s:->14} {s:->14} {s:->12}\n", .{ "", "", "", "" });

    const cold_tok = pref_len + suf_len;
    const cold_ms = @as(f64, @floatFromInt(report.cold_prefill_ns)) / 1e6;
    const warm_ms = @as(f64, @floatFromInt(report.warm_prefill_ns)) / 1e6;
    const prefix_ms = @as(f64, @floatFromInt(report.warm_prefix_ns)) / 1e6;
    const cold_ms_tok = cold_ms / @as(f64, @floatFromInt(cold_tok * n_trials));
    const warm_ms_tok = warm_ms / @as(f64, @floatFromInt(suf_len * n_trials));

    try writer.print("{s:<8} {d:>14.3} {d:>14} {d:>12.4}\n", .{ "cold", cold_ms, cold_tok * n_trials, cold_ms_tok });
    try writer.print("{s:<8} {d:>14.3} {d:>14} {d:>12.4}\n", .{ "warm", warm_ms, suf_len * n_trials, warm_ms_tok });
    try writer.print("{s:<8} {d:>14.3} {d:>14} {s:>12}\n", .{ "prefix*", prefix_ms, pref_len, "(once)" });
    try writer.print("\nsavings_ratio={d:.3}  (1 - warm/cold trial prefill; excludes one-time prefix)\n", .{report.savings_ratio});
    try writer.print("logits_match: {s}\n\n", .{if (report.all_logits_match) "PASS" else "FAIL"});

    try writer.print("json\n", .{});
    try writer.print("{{\"cmd\":\"prefix-bench\",\"backend\":\"{s}\",\"mini\":{},\"prefix_len\":{d},\"suffix_len\":{d},\"trials\":{d},\"cold_prefill_ns\":{d},\"warm_prefill_ns\":{d},\"warm_prefix_ns\":{d},\"savings_ratio\":{d:.6},\"logits_match\":{}}}\n", .{
        kind.name(),
        use_mini,
        pref_len,
        suf_len,
        n_trials,
        report.cold_prefill_ns,
        report.warm_prefill_ns,
        report.warm_prefix_ns,
        report.savings_ratio,
        report.all_logits_match,
    });

    if (!report.all_logits_match or report.savings_ratio <= 0) {
        try writer.flush();
        std.process.exit(1);
    }
}

/// Stage S3: n-gram speculative vs greedy baseline A/B.
fn runSpecBench(
    allocator: std.mem.Allocator,
    io: std.Io,
    writer: *std.Io.Writer,
    artifact_path_opt: ?[]const u8,
    force_mini: bool,
    proposal_depth_in: u32,
    ngram_order_in: u32,
    max_new_tokens_in: u32,
    kind: zynfer.BackendKind,
) !void {
    const default_path = "models/qwen3-0.6b.zynfer";
    const use_mini = force_mini or (artifact_path_opt == null and !zynfer.util.fileExists(io, default_path));
    const depth: u32 = if (proposal_depth_in == 0) 4 else proposal_depth_in;
    const order: u32 = if (ngram_order_in < 2) 2 else ngram_order_in;
    const max_new: u32 = if (max_new_tokens_in == 64 and use_mini) 8 else if (max_new_tokens_in == 64) 16 else max_new_tokens_in;

    try writer.print("zynfer spec-bench — speculative decoding (Stage S3)\n", .{});
    try writer.print("==================================================\n\n", .{});
    try writer.print("note: n-gram draft + greedy verify; committed tok/s only (not proposed).\n", .{});
    try writer.print("      Qwen3-0.6B/4B artifacts have no MTP heads.\n\n", .{});

    var art: zynfer.artifact.Artifact = undefined;
    var arch: zynfer.qwen3.Arch = undefined;
    var prompt_ids: []u32 = undefined;
    var free_prompt = false;
    defer if (free_prompt) allocator.free(prompt_ids);

    if (use_mini) {
        try writer.print("fixture: stage11-mini (in-memory)\n", .{});
        const bytes = try zynfer.qwen_forward.buildMiniArtifact(allocator);
        art = try zynfer.artifact.Artifact.loadOwned(allocator, bytes);
        arch = zynfer.qwen3.stage11_mini;
        // Repeated bigrams so n-gram draft has something to propose.
        prompt_ids = try allocator.dupe(u32, &.{ 2, 3, 2, 3, 4, 2, 3 });
        free_prompt = true;
    } else {
        const path = artifact_path_opt orelse default_path;
        try writer.print("artifact: {s}\n", .{path});
        art = zynfer.artifact.Artifact.loadFile(allocator, io, path) catch |err| {
            std.debug.print("spec-bench: load failed ({s}): {s}\n", .{ path, @errorName(err) });
            std.process.exit(2);
        };
        arch = try art.meta.toArch();
        prompt_ids = try allocator.dupe(u32, &.{ 151643, 8948, 151643, 8948, 872, 151643, 8948 });
        free_prompt = true;
    }
    defer art.deinit();

    const max_seq = prompt_ids.len + max_new;
    if (max_seq == 0 or max_seq > arch.max_position_embeddings) {
        std.debug.print("spec-bench: sequence too long\n", .{});
        std.process.exit(2);
    }

    try writer.print("backend={s} proposal_depth={d} ngram_order={d} max_new={d} prompt_tokens={d}\n\n", .{
        kind.name(),
        depth,
        order,
        max_new,
        prompt_ids.len,
    });

    var report = zynfer.speculative.runSpecBench(
        allocator,
        io,
        &art,
        arch,
        kind,
        max_seq,
        prompt_ids,
        max_new,
        depth,
        order,
    ) catch |err| {
        std.debug.print("spec-bench: failed: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
    defer report.deinit();

    const s = report.speculative;
    try writer.print("{s:<12} {s:>12} {s:>12} {s:>14} {s:>10}\n", .{
        "mode",
        "wall_ms",
        "gen_tok",
        "committed_t/s",
        "rounds",
    });
    try writer.print("{s:-<12} {s:->12} {s:->12} {s:->14} {s:->10}\n", .{ "", "", "", "", "" });
    try writer.print("{s:<12} {d:>12.3} {d:>12} {d:>14.3} {s:>10}\n", .{
        "baseline",
        @as(f64, @floatFromInt(report.baseline_wall_ns)) / 1e6,
        report.baseline_generated,
        report.baseline_tok_s,
        "—",
    });
    try writer.print("{s:<12} {d:>12.3} {d:>12} {d:>14.3} {d:>10}\n", .{
        "speculative",
        @as(f64, @floatFromInt(s.wall_ns)) / 1e6,
        s.generated_tokens,
        s.committed_tok_s,
        s.rounds,
    });

    try writer.print("\nproposal_depth={d} proposed={d} accepted_draft={d} acceptance_rate={d:.3} tokens/round={d:.3}\n", .{
        s.proposal_depth,
        s.proposed_tokens,
        s.accepted_draft_tokens,
        s.acceptance_rate,
        s.tokens_per_round,
    });
    try writer.print("token_parity: {s}\n\n", .{if (report.token_parity) "PASS" else "FAIL"});

    try writer.print("json\n", .{});
    try writer.print("{{\"cmd\":\"spec-bench\",\"backend\":\"{s}\",\"mini\":{},\"proposal_depth\":{d},\"ngram_order\":{d},\"max_new\":{d},\"token_parity\":{},\"baseline\":{{\"wall_ns\":{d},\"generated\":{d},\"tok_s\":{d:.6}}},\"speculative\":{{\"wall_ns\":{d},\"generated\":{d},\"committed_tok_s\":{d:.6},\"proposed\":{d},\"accepted_draft\":{d},\"acceptance_rate\":{d:.6},\"tokens_per_round\":{d:.6},\"rounds\":{d}}}}}\n", .{
        kind.name(),
        use_mini,
        depth,
        order,
        max_new,
        report.token_parity,
        report.baseline_wall_ns,
        report.baseline_generated,
        report.baseline_tok_s,
        s.wall_ns,
        s.generated_tokens,
        s.committed_tok_s,
        s.proposed_tokens,
        s.accepted_draft_tokens,
        s.acceptance_rate,
        s.tokens_per_round,
        s.rounds,
    });

    if (!report.token_parity) {
        try writer.flush();
        std.process.exit(1);
    }
}

fn runServe(
    allocator: std.mem.Allocator,
    io: std.Io,
    writer: *std.Io.Writer,
    artifact_path_opt: ?[]const u8,
    force_mini: bool,
    host: []const u8,
    port: u16,
    smoke: bool,
    tokenizer_dir_opt: ?[]const u8,
    default_max_tokens: u32,
    kind: zynfer.BackendKind,
) !void {
    const use_mini = force_mini or (artifact_path_opt == null and smoke);

    try writer.print("zynfer serve — HTTP server (Stage S4)\n", .{});
    try writer.print("=====================================\n\n", .{});

    var engine: zynfer.http_server.Engine = undefined;
    var owned_tok_dir: ?[]const u8 = null;
    defer if (owned_tok_dir) |d| allocator.free(d);

    if (use_mini) {
        try writer.print("fixture: stage11-mini (in-memory)\n", .{});
        engine = try zynfer.http_server.Engine.initMini(allocator, io, kind, default_max_tokens);
    } else {
        const path = artifact_path_opt orelse defaultRegisteredArtifact(io);
        if (!zynfer.util.fileExists(io, path)) {
            std.debug.print(
                "serve: artifact not found ({s})\n  run: ./zig-out/bin/zynfer setup\n  or:  --mini\n",
                .{path},
            );
            std.process.exit(2);
        }
        owned_tok_dir = try resolveTokenizerDir(allocator, io, path, tokenizer_dir_opt);
        try writer.print("artifact: {s}\n", .{path});
        try writer.print("tokenizer: {s}\n", .{owned_tok_dir.?});
        engine = try zynfer.http_server.Engine.initFile(
            allocator,
            io,
            kind,
            path,
            owned_tok_dir,
            default_max_tokens,
        );
    }
    defer engine.deinit();

    try writer.print("backend={s} default_max_tokens={d}\n", .{ kind.name(), default_max_tokens });
    try writer.flush();

    if (smoke) {
        try writer.print("mode: smoke (health + streamed completions)\n", .{});
        try writer.flush();
        zynfer.http_server.runSmoke(&engine) catch |err| {
            std.debug.print("serve --smoke failed: {s}\n", .{@errorName(err)});
            std.process.exit(1);
        };
        try writer.print("smoke: PASS\n", .{});
        try writer.print("metrics: requests={d} completions={d} tokens={d} streams={d}\n", .{
            engine.metrics.requests_total,
            engine.metrics.completions_total,
            engine.metrics.tokens_generated_total,
            engine.metrics.stream_responses_total,
        });
        try writer.print("json\n", .{});
        try writer.print("{{\"cmd\":\"serve-smoke\",\"backend\":\"{s}\",\"mini\":{},\"ok\":true,\"requests\":{d},\"completions\":{d},\"tokens\":{d},\"streams\":{d}}}\n", .{
            kind.name(),
            use_mini,
            engine.metrics.requests_total,
            engine.metrics.completions_total,
            engine.metrics.tokens_generated_total,
            engine.metrics.stream_responses_total,
        });
        return;
    }

    try writer.print("listening http://{s}:{d}/  (Ctrl-C to stop)\n", .{ host, port });
    try writer.print("  GET  /health\n", .{});
    try writer.print("  GET  /metrics\n", .{});
    try writer.print("  POST /v1/completions\n", .{});
    try writer.print("  POST /v1/chat/completions\n", .{});
    try writer.flush();

    try zynfer.http_server.serveLoop(&engine, .{
        .host = host,
        .port = port,
        .default_max_tokens = default_max_tokens,
    });
}

/// Stage M2: per-family wall profile for one Metal (or CPU) decode token + roofline.
fn runQwenProfile(
    allocator: std.mem.Allocator,
    io: std.Io,
    writer: *std.Io.Writer,
    artifact_path_opt: ?[]const u8,
    prompt_opt: ?[]const u8,
    tokenizer_dir_opt: ?[]const u8,
    seed: u64,
    raw_prompt: bool,
    force_mini: bool,
    tokens_arg: ?[]const u8,
    forced_backend: ?[]const u8,
) !void {
    const default_path = "models/qwen3-0.6b.zynfer";
    const use_mini = force_mini or (artifact_path_opt == null and !zynfer.util.fileExists(io, default_path));

    try writer.print("zynfer qwen-profile — one decode token (Stage M2)\n", .{});
    try writer.print("=================================================\n\n", .{});

    var art: zynfer.artifact.Artifact = undefined;
    var prompt_ids: []u32 = undefined;
    var free_prompt = false;
    defer if (free_prompt) allocator.free(prompt_ids);
    var arch: zynfer.qwen3.Arch = undefined;

    if (use_mini) {
        try writer.print("fixture: stage11-mini (in-memory)\n", .{});
        const bytes = try zynfer.qwen_forward.buildMiniArtifact(allocator);
        art = try zynfer.artifact.Artifact.loadOwned(allocator, bytes);
        arch = zynfer.qwen3.stage11_mini;
        if (tokens_arg) |s| {
            prompt_ids = try parseCsvTokenIds(allocator, s);
            free_prompt = true;
        } else {
            prompt_ids = try allocator.dupe(u32, &.{ 2, 3 });
            free_prompt = true;
        }
    } else {
        const path = artifact_path_opt orelse default_path;
        try writer.print("artifact: {s}\n", .{path});
        art = zynfer.artifact.Artifact.loadFile(allocator, io, path) catch |err| {
            std.debug.print("qwen-profile: load failed ({s}): {s}\n", .{ path, @errorName(err) });
            std.process.exit(2);
        };
        arch = try art.meta.toArch();
        if (tokens_arg) |s| {
            prompt_ids = try parseCsvTokenIds(allocator, s);
            free_prompt = true;
        } else {
            const prompt = prompt_opt orelse "Explain gravity simply.";
            const tok_dir = try resolveTokenizerDir(allocator, io, path, tokenizer_dir_opt);
            defer allocator.free(tok_dir);
            var tok = zynfer.tokenizer.Tokenizer.loadHfDir(allocator, io, tok_dir) catch |err| {
                std.debug.print("qwen-profile: tokenizer load failed ({s}): {s}\n", .{ tok_dir, @errorName(err) });
                std.process.exit(2);
            };
            defer tok.deinit();
            const wrapped = if (raw_prompt)
                try allocator.dupe(u8, prompt)
            else
                try tok.applyChatTemplate(allocator, prompt);
            defer allocator.free(wrapped);
            prompt_ids = tok.encode(allocator, wrapped) catch |err| {
                std.debug.print("qwen-profile: encode failed: {s}\n", .{@errorName(err)});
                std.process.exit(2);
            };
            free_prompt = true;
        }
    }
    defer art.deinit();

    const max_seq = prompt_ids.len + 2;
    if (max_seq == 0 or max_seq > arch.max_position_embeddings) {
        std.debug.print("qwen-profile: sequence too long\n", .{});
        std.process.exit(2);
    }

    var kind: zynfer.BackendKind = .apple;
    if (forced_backend) |name| {
        kind = zynfer.backend.parseBackendKind(name) catch {
            std.debug.print("qwen-profile: unknown --backend {s}\n", .{name});
            std.process.exit(2);
        };
    } else if (use_mini) {
        // Prefer Apple when available for the M2 gate; fall back to CPU.
        zynfer.backend.requireBackend(.apple) catch {
            kind = .cpu;
        };
    }
    zynfer.backend.requireBackend(kind) catch |err| {
        std.debug.print("qwen-profile: backend {s} unavailable: {s}\n", .{ kind.name(), @errorName(err) });
        std.process.exit(2);
    };

    try writer.print("backend={s} prompt_tokens={d} layers={d} hidden={d}\n", .{
        kind.name(),
        prompt_ids.len,
        arch.num_layers,
        arch.hidden_size,
    });
    try writer.print("signposts: set ZYNFER_SIGNPOSTS=1 for Instruments (qwen.* + encode/batch)\n\n", .{});

    var sess = zynfer.qwen_forward.Session.initWithBackend(allocator, &art, arch, max_seq, kind) catch |err| {
        std.debug.print("qwen-profile: session init failed: {s}\n", .{@errorName(err)});
        std.process.exit(2);
    };
    defer sess.deinit();

    const logits = try allocator.alloc(f32, arch.vocab_size);
    defer allocator.free(logits);
    try sess.prefillLastLogits(prompt_ids, logits);

    // Warm one unprofiled decode so the profiled step is not cold-start dominated.
    if (prompt_ids.len + 1 < max_seq) {
        const warm_id = try zynfer.sample.argmax(logits);
        try sess.decodeToken(warm_id, logits);
    }

    var buckets = zynfer.decode_profile.Accumulators{};
    var rng = std.Random.DefaultPrng.init(if (seed != 0) seed else 1);
    const profile_token = try zynfer.sample.argmax(logits);
    _ = try sess.profileDecodeToken(io, profile_token, logits, &buckets, true, true, .{ .temperature = 0 }, &rng);

    const sum_ns = buckets.sumFamilies();
    const wall_ms = @as(f64, @floatFromInt(buckets.wall_ns)) / 1e6;
    try writer.print("{s:<28} {s:>10} {s:>8}\n", .{ "operation", "ms", "%" });
    try writer.print("{s:-<28} {s:->10} {s:->8}\n", .{ "", "", "" });

    const Family = zynfer.decode_profile.Family;
    var fi: usize = 0;
    while (fi < zynfer.decode_profile.family_count) : (fi += 1) {
        const fam: Family = @enumFromInt(fi);
        const ns = buckets.ns[fi];
        const ms = @as(f64, @floatFromInt(ns)) / 1e6;
        const pct = if (sum_ns == 0) 0 else 100.0 * @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(sum_ns));
        try writer.print("{s:<28} {d:>10.3} {d:>7.1}\n", .{ fam.label(), ms, pct });
    }
    try writer.print("{s:-<28} {s:->10} {s:->8}\n", .{ "", "", "" });
    try writer.print("{s:<28} {d:>10.3}\n", .{ "sum(families)", @as(f64, @floatFromInt(sum_ns)) / 1e6 });
    try writer.print("{s:<28} {d:>10.3}\n", .{ "wall decode token", wall_ms });
    const unaccounted = if (buckets.wall_ns > sum_ns) buckets.wall_ns - sum_ns else 0;
    try writer.print("{s:<28} {d:>10.3}\n\n", .{ "unaccounted", @as(f64, @floatFromInt(unaccounted)) / 1e6 });

    const top = buckets.top3();
    try writer.print("top3:\n", .{});
    try writer.print("  1. {s} ({d:.1}%)\n", .{
        top[0].family.label(),
        if (sum_ns == 0) 0 else 100.0 * @as(f64, @floatFromInt(top[0].ns)) / @as(f64, @floatFromInt(sum_ns)),
    });
    try writer.print("  2. {s} ({d:.1}%)\n", .{
        top[1].family.label(),
        if (sum_ns == 0) 0 else 100.0 * @as(f64, @floatFromInt(top[1].ns)) / @as(f64, @floatFromInt(sum_ns)),
    });
    try writer.print("  3. {s} ({d:.1}%)\n\n", .{
        top[2].family.label(),
        if (sum_ns == 0) 0 else 100.0 * @as(f64, @floatFromInt(top[2].ns)) / @as(f64, @floatFromInt(sum_ns)),
    });

    try writer.print("metal_encodes={d} metal_waits={d} kv_len={d}\n", .{
        buckets.metal_encodes,
        buckets.metal_waits,
        buckets.kv_len,
    });

    var empty_launch_ns: u64 = 0;
    var bw_gbps: f64 = 0;
    if (kind == .apple) {
        if (sess.gpu) |g| {
            empty_launch_ns = zynfer.apple.ops.measureEmptyLaunchNs(g, io, 8, 64) catch 0;
            const est = zynfer.decode_profile.emptyLaunchOverheadNs(buckets.metal_encodes, empty_launch_ns);
            try writer.print(
                "empty_encode_wait_ns={d} → est_launch_overhead_ms={d:.3} (embedded in Metal op families on M0 path)\n",
                .{ empty_launch_ns, @as(f64, @floatFromInt(est)) / 1e6 },
            );
            // ~64 MiB elements: large enough for STREAM, small enough for laptop CI.
            const elems: u32 = if (use_mini) (1 << 20) else (16 << 20);
            if (zynfer.apple.ops.measureSustainableBandwidth(g, io, elems, 2, 4)) |bw| {
                bw_gbps = bw.gbps();
                try writer.print(
                    "bandwidth_stream_triad: elems={d} bytes_moved={d} elapsed_ms={d:.3} → {d:.1} GB/s (measured)\n",
                    .{
                        elems,
                        bw.bytes_moved,
                        @as(f64, @floatFromInt(bw.elapsed_ns)) / 1e6,
                        bw_gbps,
                    },
                );
            } else |_| {
                try writer.print("bandwidth_stream_triad: measurement failed\n", .{});
            }
        }
    } else {
        try writer.print("bandwidth_stream_triad: n/a (CPU backend)\n", .{});
    }

    const bytes_tok = if (kind == .apple and zynfer.apple.qwen_schedule.useQ8Path())
        arch.estimateDecodeBytesPerTokenQ8(buckets.kv_len)
    else if (kind == .apple and zynfer.apple.qwen_schedule.useHalfPath())
        arch.estimateDecodeBytesPerTokenHalf(buckets.kv_len)
    else
        arch.estimateDecodeBytesPerToken(buckets.kv_len);
    const roof = zynfer.decode_profile.roofline(bytes_tok, bw_gbps, buckets.wall_ns);
    try writer.print("bytes_per_tok_est={d}\n", .{bytes_tok});
    if (bw_gbps > 0) {
        try writer.print(
            "roofline: ideal_tok_s={d:.3} measured_tok_s={d:.3} fraction={d:.4}\n",
            .{ roof.ideal_tok_s, roof.measured_tok_s, roof.fraction },
        );
    } else {
        try writer.print(
            "roofline: ideal_tok_s=n/a measured_tok_s={d:.3} (no bandwidth)\n",
            .{roof.measured_tok_s},
        );
    }

    try writer.print("\njson\n", .{});
    try writer.print(
        "{{\"cmd\":\"qwen-profile\",\"backend\":\"{s}\",\"mini\":{},\"prompt_tokens\":{d},\"kv_len\":{d},\"wall_ns\":{d},\"sum_families_ns\":{d},\"metal_encodes\":{d},\"metal_waits\":{d},\"empty_launch_ns\":{d},\"bandwidth_gbps\":{d:.3},\"bytes_per_tok_est\":{d},\"ideal_tok_s\":{d:.6},\"measured_tok_s\":{d:.6},\"roofline_fraction\":{d:.6},\"top3\":[\"{s}\",\"{s}\",\"{s}\"],\"families\":{{",
        .{
            kind.name(),
            use_mini,
            prompt_ids.len,
            buckets.kv_len,
            buckets.wall_ns,
            sum_ns,
            buckets.metal_encodes,
            buckets.metal_waits,
            empty_launch_ns,
            bw_gbps,
            bytes_tok,
            roof.ideal_tok_s,
            roof.measured_tok_s,
            roof.fraction,
            top[0].family.label(),
            top[1].family.label(),
            top[2].family.label(),
        },
    );
    fi = 0;
    while (fi < zynfer.decode_profile.family_count) : (fi += 1) {
        if (fi != 0) try writer.writeAll(",");
        const fam: Family = @enumFromInt(fi);
        try writer.print("\"{s}\":{d}", .{ fam.label(), buckets.ns[fi] });
    }
    try writer.print("}}}}\n", .{});
}

fn parseCsvTokenIds(allocator: std.mem.Allocator, csv: []const u8) ![]u32 {
    var count: usize = 0;
    var it = std.mem.splitScalar(u8, csv, ',');
    while (it.next()) |p| {
        if (p.len == 0) continue;
        count += 1;
    }
    const out = try allocator.alloc(u32, count);
    errdefer allocator.free(out);
    var i: usize = 0;
    it = std.mem.splitScalar(u8, csv, ',');
    while (it.next()) |p| {
        if (p.len == 0) continue;
        out[i] = std.fmt.parseInt(u32, p, 10) catch {
            std.debug.print("invalid token id in --tokens\n", .{});
            std.process.exit(2);
        };
        i += 1;
    }
    return out;
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

fn runArtifactCompile(allocator: std.mem.Allocator, io: std.Io, writer: *std.Io.Writer, path: []const u8, mini: bool) !void {
    const bytes = if (mini)
        try zynfer.qwen_forward.buildMiniArtifact(allocator)
    else
        try zynfer.artifact.buildStage10Fixture(allocator);
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

fn runForwardGolden(
    allocator: std.mem.Allocator,
    io: std.Io,
    writer: *std.Io.Writer,
    artifact_path: []const u8,
    tokens_arg: ?[]const u8,
    golden_path: ?[]const u8,
    dump_dir: ?[]const u8,
    backend: zynfer.BackendKind,
) !void {
    var art = zynfer.artifact.Artifact.loadFile(allocator, io, artifact_path) catch |err| {
        std.debug.print("forward-golden: load failed ({s}): {s}\n", .{ artifact_path, @errorName(err) });
        std.process.exit(2);
    };
    defer art.deinit();

    const arch = try art.meta.toArch();
    var token_list: std.ArrayList(u32) = .empty;
    defer token_list.deinit(allocator);

    if (tokens_arg) |ts| {
        try parseTokenIds(allocator, &token_list, ts);
    } else {
        try token_list.append(allocator, arch.bos_token_id);
        try token_list.append(allocator, 2);
        try token_list.append(allocator, 3);
    }

    var sess = try zynfer.qwen_forward.Session.initWithBackend(allocator, &art, arch, token_list.items.len, backend);
    defer sess.deinit();

    const logits = try allocator.alloc(f32, arch.vocab_size);
    defer allocator.free(logits);

    const DumpCtx = struct {
        dir: []const u8,
        io: std.Io,
        allocator: std.mem.Allocator,
    };

    var dump_ctx: DumpCtx = undefined;
    const dump_hook: ?zynfer.qwen_forward.DumpHook = if (dump_dir) |dir| blk: {
        dump_ctx = .{ .dir = dir, .io = io, .allocator = allocator };
        break :blk dumpWriteF32;
    } else null;

    try sess.prefillLastLogitsDump(
        token_list.items,
        logits,
        dump_hook,
        if (dump_hook != null) @ptrCast(&dump_ctx) else null,
    );

    if (golden_path) |gpath| {
        const golden_bytes = std.Io.Dir.cwd().readFileAlloc(io, gpath, allocator, .limited(512 * 1024 * 1024)) catch |err| {
            if (err == error.FileNotFound) {
                std.debug.print(
                    "forward-golden: golden file not found ({s})\n",
                    .{gpath},
                );
                std.debug.print(
                    "  generate with: python3 tools/fixtures/gen_golden_logits.py --tokens=151643,2,3 --out {s}\n",
                    .{gpath},
                );
                std.debug.print(
                    "  (use the same --tokens as zynfer forward-golden)\n",
                    .{},
                );
            } else {
                std.debug.print("forward-golden: golden read failed ({s}): {s}\n", .{ gpath, @errorName(err) });
            }
            std.process.exit(2);
        };
        defer allocator.free(golden_bytes);
        if (golden_bytes.len != logits.len * 4) {
            std.debug.print("forward-golden: golden size mismatch (expected {d} bytes)\n", .{logits.len * 4});
            std.process.exit(2);
        }
        const expected = @as([*]align(4) const f32, @ptrCast(@alignCast(golden_bytes.ptr)))[0..logits.len];
        try zynfer.compare.expectClose(expected, logits, 1e-3, 1e-2);
        try writer.print("golden OK ({s}, vocab={d})\n", .{ gpath, arch.vocab_size });
    }

    var top: [8]zynfer.qwen_forward.TopK = undefined;
    zynfer.qwen_forward.topK(logits, 8, &top);

    try writer.print("forward-golden: {s} backend={s}\n", .{ artifact_path, sess.backendName() });
    try writer.print("tokens: {d}\n", .{token_list.items.len});
    try writer.print("top logits (last token):\n", .{});
    for (top) |entry| {
        if (entry.logit == -std.math.inf(f32)) break;
        try writer.print("  id={d} logit={d:.6}\n", .{ entry.id, entry.logit });
    }
    try writer.print("\n", .{});
}

fn dumpWriteF32(ctx: ?*anyopaque, name: []const u8, data: []const f32) void {
    const c: *struct {
        dir: []const u8,
        io: std.Io,
        allocator: std.mem.Allocator,
    } = @ptrCast(@alignCast(ctx.?));
    var path_buf: [512]u8 = undefined;
    const rel = std.fmt.bufPrint(&path_buf, "{s}/{s}.f32", .{ c.dir, name }) catch return;
    const bytes = std.mem.sliceAsBytes(data);
    std.Io.Dir.cwd().writeFile(c.io, .{ .sub_path = rel, .data = bytes }) catch {
        std.debug.print("forward-golden: dump write failed ({s})\n", .{rel});
        return;
    };
    std.debug.print("dump: {s} ({d} f32)\n", .{ rel, data.len });
}

fn parseTokenIds(allocator: std.mem.Allocator, out: *std.ArrayList(u32), text: []const u8) !void {
    var parts = std.mem.splitScalar(u8, text, ',');
    while (parts.next()) |part| {
        const trimmed = std.mem.trim(u8, part, &.{ ' ', '\t' });
        if (trimmed.len == 0) continue;
        const id = try std.fmt.parseInt(u32, trimmed, 10);
        try out.append(allocator, id);
    }
    if (out.items.len == 0) return error.InvalidShape;
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
    try writer.print("\nStage 7 / M7 probes (hardware/framework ≠ retained path)\n", .{});
    try writer.print("  SME hardware FEAT_SME/SME2: {s}/{s}\n", .{ yn(sme_p.feat_sme), yn(sme_p.feat_sme2) });
    try writer.print("  Core ML framework linked:   {s}\n", .{yn(cm_p.framework_linked)});
    try writer.print("  MLState class available:    {s}\n", .{yn(cm_p.ml_state_available)});
    try writer.print("  ANE execution verified:     {s}\n", .{yn(cm_p.ane_execution_verified)});
    try writer.print("  Core ML path retained:      {s} (M7 final REJECT)\n", .{yn(cm_p.path_retained)});

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
            try writer.print("  Stage 7: SME rejected; Core ML framework probe only — see `zynfer stage7`\n", .{});
            try writer.print("  Stage M7: Core ML/ANE at Qwen scale REJECT (final) — see `zynfer stageM7`\n", .{});
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
