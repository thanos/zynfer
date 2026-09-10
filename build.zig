const std = @import("std");

const HipMode = enum { auto, on, off };

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const hip_mode = b.option(HipMode, "hip", "HIP linking: auto (default), on, or off") orelse .auto;
    const hip_path_opt = b.option([]const u8, "hip-path", "ROCm/HIP prefix (default: HIP_PATH, ROCM_PATH, or /opt/rocm)");

    const detected_hip = detectHipPrefix(b, hip_path_opt);
    const have_hip = switch (hip_mode) {
        .off => false,
        .on => true,
        .auto => detected_hip != null,
    };

    if (hip_mode == .on and detected_hip == null) {
        std.log.err("HIP was requested (-Dhip=on) but ROCm/HIP was not found.", .{});
        std.log.err("Pass -Dhip-path=/opt/rocm or set HIP_PATH / ROCM_PATH.", .{});
        std.process.fatal("missing HIP installation", .{});
    }

    const hip_path = detected_hip orelse "";
    const have_apple = target.result.os.tag == .macos;

    const options = b.addOptions();
    options.addOption(bool, "have_hip", have_hip);
    options.addOption([]const u8, "hip_path", hip_path);
    options.addOption(bool, "have_apple", have_apple);

    const zynfer_mod = b.addModule("zynfer", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    zynfer_mod.link_libc = true;
    zynfer_mod.addOptions("build_options", options);
    configureHip(b, zynfer_mod, have_hip, hip_path);
    configureApple(b, zynfer_mod, have_apple);

    const exe = b.addExecutable(.{
        .name = "zynfer",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zynfer", .module = zynfer_mod },
            },
        }),
    });
    exe.root_module.link_libc = true;
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run zynfer (default: env + GPU report)");
    run_step.dependOn(&run_cmd.step);

    const env_cmd = b.addRunArtifact(exe);
    env_cmd.step.dependOn(b.getInstallStep());
    env_cmd.addArg("env");
    const env_step = b.step("env", "Print the development-environment report");
    env_step.dependOn(&env_cmd.step);

    const gpu_cmd = b.addRunArtifact(exe);
    gpu_cmd.step.dependOn(b.getInstallStep());
    gpu_cmd.addArg("gpu");
    const gpu_step = b.step("gpu", "Enumerate HIP devices");
    gpu_step.dependOn(&gpu_cmd.step);

    const caps_cmd = b.addRunArtifact(exe);
    caps_cmd.step.dependOn(b.getInstallStep());
    caps_cmd.addArg("caps");
    const caps_step = b.step("caps", "Print backend capabilities and fallbacks");
    caps_step.dependOn(&caps_cmd.step);

    const stage7_cmd = b.addRunArtifact(exe);
    stage7_cmd.step.dependOn(b.getInstallStep());
    stage7_cmd.addArg("stage7");
    stage7_cmd.expectStdOutMatch("Stage 7 decisions");
    stage7_cmd.expectExitCode(0);
    const stage7_step = b.step("stage7", "SME / Core ML Stage 7 probe and retain/reject ledger");
    stage7_step.dependOn(&stage7_cmd.step);

    const stage8_cmd = b.addRunArtifact(exe);
    stage8_cmd.step.dependOn(b.getInstallStep());
    stage8_cmd.addArg("stage8");
    stage8_cmd.expectStdOutMatch("Stage 8");
    stage8_cmd.expectExitCode(0);
    const stage8_step = b.step("stage8", "Apple Stage 8 hardening ledger");
    stage8_step.dependOn(&stage8_cmd.step);

    const stage10_cmd = b.addRunArtifact(exe);
    stage10_cmd.step.dependOn(b.getInstallStep());
    stage10_cmd.addArg("stage10");
    stage10_cmd.expectStdOutMatch("Stage 10");
    stage10_cmd.expectExitCode(0);
    const stage10_step = b.step("stage10", "Checkpoint / .zynfer artifact Stage 10 ledger");
    stage10_step.dependOn(&stage10_cmd.step);

    const stage11_cmd = b.addRunArtifact(exe);
    stage11_cmd.step.dependOn(b.getInstallStep());
    stage11_cmd.addArg("stage11");
    stage11_cmd.expectStdOutMatch("Stage 11");
    stage11_cmd.expectExitCode(0);
    const stage11_step = b.step("stage11", "Qwen forward + golden logits Stage 11 ledger");
    stage11_step.dependOn(&stage11_cmd.step);

    const stage12_cmd = b.addRunArtifact(exe);
    stage12_cmd.step.dependOn(b.getInstallStep());
    stage12_cmd.addArg("stage12");
    stage12_cmd.expectStdOutMatch("Stage 12");
    stage12_cmd.expectExitCode(0);
    const stage12_step = b.step("stage12", "Tokenizer + sampling Stage 12 ledger");
    stage12_step.dependOn(&stage12_cmd.step);

    const stage13_cmd = b.addRunArtifact(exe);
    stage13_cmd.step.dependOn(b.getInstallStep());
    stage13_cmd.addArg("stage13");
    stage13_cmd.expectStdOutMatch("Stage 13");
    stage13_cmd.expectExitCode(0);
    const stage13_step = b.step("stage13", "KV cache Stage 13 ledger");
    stage13_step.dependOn(&stage13_cmd.step);

    const stageM0_cmd = b.addRunArtifact(exe);
    stageM0_cmd.step.dependOn(b.getInstallStep());
    stageM0_cmd.addArg("stageM0");
    stageM0_cmd.expectStdOutMatch("Stage M0");
    stageM0_cmd.expectExitCode(0);
    const stageM0_step = b.step("stageM0", "Metal Qwen forward Stage M0 ledger");
    stageM0_step.dependOn(&stageM0_cmd.step);

    const stageM1_cmd = b.addRunArtifact(exe);
    stageM1_cmd.step.dependOn(b.getInstallStep());
    stageM1_cmd.addArg("stageM1");
    stageM1_cmd.expectStdOutMatch("Stage M1");
    stageM1_cmd.expectExitCode(0);
    const stageM1_step = b.step("stageM1", "Prefill/decode split Stage M1 ledger");
    stageM1_step.dependOn(&stageM1_cmd.step);

    const stageM2_cmd = b.addRunArtifact(exe);
    stageM2_cmd.step.dependOn(b.getInstallStep());
    stageM2_cmd.addArg("stageM2");
    stageM2_cmd.expectStdOutMatch("Stage M2");
    stageM2_cmd.expectExitCode(0);
    const stageM2_step = b.step("stageM2", "Profile one decode token Stage M2 ledger");
    stageM2_step.dependOn(&stageM2_cmd.step);

    const stageM3_cmd = b.addRunArtifact(exe);
    stageM3_cmd.step.dependOn(b.getInstallStep());
    stageM3_cmd.addArg("stageM3");
    stageM3_cmd.expectStdOutMatch("Stage M3");
    stageM3_cmd.expectExitCode(0);
    const stageM3_step = b.step("stageM3", "Qwen schedule/fusion Stage M3 ledger");
    stageM3_step.dependOn(&stageM3_cmd.step);

    const stageM4_cmd = b.addRunArtifact(exe);
    stageM4_cmd.step.dependOn(b.getInstallStep());
    stageM4_cmd.addArg("stageM4");
    stageM4_cmd.expectStdOutMatch("Stage M4");
    stageM4_cmd.expectExitCode(0);
    const stageM4_step = b.step("stageM4", "bf16/fp16 Metal Stage M4 ledger");
    stageM4_step.dependOn(&stageM4_cmd.step);

    const stageM5_cmd = b.addRunArtifact(exe);
    stageM5_cmd.step.dependOn(b.getInstallStep());
    stageM5_cmd.addArg("stageM5");
    stageM5_cmd.expectStdOutMatch("Stage M5");
    stageM5_cmd.expectExitCode(0);
    const stageM5_step = b.step("stageM5", "int8 quantization Stage M5 ledger");
    stageM5_step.dependOn(&stageM5_cmd.step);

    const stageM6_cmd = b.addRunArtifact(exe);
    stageM6_cmd.step.dependOn(b.getInstallStep());
    stageM6_cmd.addArg("stageM6");
    stageM6_cmd.expectStdOutMatch("Stage M6");
    stageM6_cmd.expectExitCode(0);
    const stageM6_step = b.step("stageM6", "static decode plan Stage M6 ledger");
    stageM6_step.dependOn(&stageM6_cmd.step);

    const stageM7_cmd = b.addRunArtifact(exe);
    stageM7_cmd.step.dependOn(b.getInstallStep());
    stageM7_cmd.addArg("stageM7");
    stageM7_cmd.expectStdOutMatch("Stage M7");
    stageM7_cmd.expectStdOutMatch("REJECT");
    stageM7_cmd.expectExitCode(0);
    const stageM7_step = b.step("stageM7", "ANE/Core ML Qwen-scale Stage M7 ledger");
    stageM7_step.dependOn(&stageM7_cmd.step);

    const stageM8_cmd = b.addRunArtifact(exe);
    stageM8_cmd.step.dependOn(b.getInstallStep());
    stageM8_cmd.addArg("stageM8");
    stageM8_cmd.expectStdOutMatch("Stage M8");
    stageM8_cmd.expectStdOutMatch("Apple-complete");
    stageM8_cmd.expectStdOutMatch("qwen3-4b");
    stageM8_cmd.expectExitCode(0);
    const stageM8_step = b.step("stageM8", "Apple capstone Qwen3-4B Stage M8 ledger");
    stageM8_step.dependOn(&stageM8_cmd.step);

    const stageS1_cmd = b.addRunArtifact(exe);
    stageS1_cmd.step.dependOn(b.getInstallStep());
    stageS1_cmd.addArg("stageS1");
    stageS1_cmd.expectStdOutMatch("Stage S1");
    stageS1_cmd.expectExitCode(0);
    const stageS1_step = b.step("stageS1", "batching/scheduling Stage S1 ledger");
    stageS1_step.dependOn(&stageS1_cmd.step);

    const batch_bench_cmd = b.addRunArtifact(exe);
    batch_bench_cmd.step.dependOn(b.getInstallStep());
    batch_bench_cmd.addArg("batch-bench");
    batch_bench_cmd.addArg("--mini");
    batch_bench_cmd.addArg("--batch-size");
    batch_bench_cmd.addArg("2");
    batch_bench_cmd.addArg("--max-tokens");
    batch_bench_cmd.addArg("4");
    batch_bench_cmd.expectStdOutMatch("request scheduling");
    batch_bench_cmd.expectStdOutMatch("token_parity: PASS");
    batch_bench_cmd.expectStdOutMatch("json");
    batch_bench_cmd.expectExitCode(0);
    const batch_bench_step = b.step("batch-bench", "multi-request scheduling A/B (Stage S1)");
    batch_bench_step.dependOn(&batch_bench_cmd.step);

    const qwen_bench_cmd = b.addRunArtifact(exe);
    qwen_bench_cmd.step.dependOn(b.getInstallStep());
    qwen_bench_cmd.addArg("qwen-bench");
    qwen_bench_cmd.addArg("--mini");
    qwen_bench_cmd.addArg("--max-tokens");
    qwen_bench_cmd.addArg("4");
    qwen_bench_cmd.expectStdOutMatch("prefill vs decode");
    qwen_bench_cmd.expectStdOutMatch("json");
    qwen_bench_cmd.expectExitCode(0);
    const qwen_bench_step = b.step("qwen-bench", "Qwen prefill/decode split (Stage M1)");
    qwen_bench_step.dependOn(&qwen_bench_cmd.step);

    const qwen_profile_cmd = b.addRunArtifact(exe);
    qwen_profile_cmd.step.dependOn(b.getInstallStep());
    qwen_profile_cmd.addArg("qwen-profile");
    qwen_profile_cmd.addArg("--mini");
    qwen_profile_cmd.expectStdOutMatch("one decode token");
    qwen_profile_cmd.expectStdOutMatch("top3");
    qwen_profile_cmd.expectStdOutMatch("json");
    qwen_profile_cmd.expectExitCode(0);
    const qwen_profile_step = b.step("qwen-profile", "Qwen one-token profile (Stage M2)");
    qwen_profile_step.dependOn(&qwen_profile_cmd.step);

    const kv_bench_cmd = b.addRunArtifact(exe);
    kv_bench_cmd.step.dependOn(b.getInstallStep());
    kv_bench_cmd.addArg("kv-bench");
    kv_bench_cmd.addArg("--mini");
    kv_bench_cmd.addArg("--max-tokens");
    kv_bench_cmd.addArg("4");
    kv_bench_cmd.expectStdOutMatch("token_parity: PASS");
    kv_bench_cmd.expectExitCode(0);
    const kv_bench_step = b.step("kv-bench", "Cached vs uncached decode (Stage 13)");
    kv_bench_step.dependOn(&kv_bench_cmd.step);

    const kv_layout_cmd = b.addRunArtifact(exe);
    kv_layout_cmd.step.dependOn(b.getInstallStep());
    kv_layout_cmd.addArg("kv-bench");
    kv_layout_cmd.addArg("--layout");
    kv_layout_cmd.expectStdOutMatch("RETAIN");
    kv_layout_cmd.expectStdOutMatch("[n_kv, max_seq, head_dim]");
    kv_layout_cmd.expectExitCode(0);
    const kv_layout_step = b.step("kv-layout", "Host KV layout bake-off (Stage 13)");
    kv_layout_step.dependOn(&kv_layout_cmd.step);

    const ops_bench_cmd = b.addRunArtifact(exe);
    ops_bench_cmd.step.dependOn(b.getInstallStep());
    ops_bench_cmd.addArg("ops-bench");
    const ops_bench_step = b.step("ops-bench", "CPU vs Apple operation microbenchmarks");
    ops_bench_step.dependOn(&ops_bench_cmd.step);

    const block_bench_cmd = b.addRunArtifact(exe);
    block_bench_cmd.step.dependOn(b.getInstallStep());
    block_bench_cmd.addArg("block-bench");
    const block_bench_step = b.step("block-bench", "Tiny-block prefill/decode timings");
    block_bench_step.dependOn(&block_bench_cmd.step);

    const bench_cmd = b.addRunArtifact(exe);
    bench_cmd.step.dependOn(b.getInstallStep());
    bench_cmd.addArg("bench");
    const bench_step = b.step("bench", "Time HIP device enumeration");
    bench_step.dependOn(&bench_cmd.step);

    const install_tests = b.option(bool, "install-tests", "Install test binaries (for kcov)") orelse false;

    const unit_tests = b.addTest(.{
        .name = "test-unit",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    unit_tests.root_module.link_libc = true;
    unit_tests.root_module.addOptions("build_options", options);
    configureHip(b, unit_tests.root_module, have_hip, hip_path);
    configureApple(b, unit_tests.root_module, have_apple);

    const numerical_tests = b.addTest(.{
        .name = "test-numerical",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/numerical/ops.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zynfer", .module = zynfer_mod },
            },
        }),
    });

    const smoke_tests = b.addTest(.{
        .name = "test-smoke",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/unit/smoke.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zynfer", .module = zynfer_mod },
            },
        }),
    });

    const integration_tests = b.addTest(.{
        .name = "test-integration",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/integration/cli.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zynfer", .module = zynfer_mod },
            },
        }),
    });

    if (install_tests) {
        b.installArtifact(unit_tests);
        b.installArtifact(numerical_tests);
        b.installArtifact(smoke_tests);
        b.installArtifact(integration_tests);
    }

    const run_unit_tests = b.addRunArtifact(unit_tests);
    const run_smoke_tests = b.addRunArtifact(smoke_tests);
    const run_numerical_tests = b.addRunArtifact(numerical_tests);
    const test_step = b.step("test", "Run unit, smoke, and numerical regression tests");
    test_step.dependOn(&run_unit_tests.step);
    test_step.dependOn(&run_smoke_tests.step);
    test_step.dependOn(&run_numerical_tests.step);

    const run_integration = b.addRunArtifact(integration_tests);
    run_integration.step.dependOn(b.getInstallStep());
    run_integration.setEnvironmentVariable("ZYNFER_BIN", b.getInstallPath(.bin, "zynfer"));

    const help_run = b.addRunArtifact(exe);
    help_run.addArg("help");
    help_run.expectStdOutMatch("Usage:");
    help_run.expectExitCode(0);

    const env_ok = b.addRunArtifact(exe);
    env_ok.addArg("env");
    env_ok.expectStdOutMatch("Zig version:");
    env_ok.expectExitCode(0);

    const caps_cpu = b.addRunArtifact(exe);
    caps_cpu.addArg("caps");
    caps_cpu.addArg("--backend");
    caps_cpu.addArg("cpu");
    caps_cpu.expectStdOutMatch("requested backend: cpu");
    caps_cpu.expectExitCode(0);

    const bad_backend = b.addRunArtifact(exe);
    bad_backend.addArg("caps");
    bad_backend.addArg("--backend");
    bad_backend.addArg("cuda");
    // Zig 0.16 treats unmatched stderr as a diagnostic warning ("w" / "failed
    // command") unless a stderr check is present. Assert the message so this
    // intentional rejection is a real check, not a spurious build warning.
    bad_backend.expectStdErrMatch("unknown backend");
    bad_backend.expectExitCode(2);

    const block_cpu = b.addRunArtifact(exe);
    block_cpu.addArg("block-bench");
    block_cpu.addArg("--backend");
    block_cpu.addArg("cpu");
    block_cpu.expectStdOutMatch("tiny-block");
    block_cpu.expectExitCode(0);

    const stage7_ok = b.addRunArtifact(exe);
    stage7_ok.addArg("stage7");
    stage7_ok.expectStdOutMatch("Stage 7 decisions");
    stage7_ok.expectStdOutMatch("REJECT");
    stage7_ok.expectExitCode(0);

    const force_sme = b.addRunArtifact(exe);
    force_sme.setEnvironmentVariable("ZYNFER_FORCE_SME", "1");
    force_sme.addArg("stage7");
    force_sme.expectStdErrMatch("ZYNFER_FORCE_SME");
    force_sme.expectExitCode(2);

    const force_coreml = b.addRunArtifact(exe);
    force_coreml.setEnvironmentVariable("ZYNFER_FORCE_COREML", "1");
    force_coreml.addArg("caps");
    force_coreml.expectStdErrMatch("ZYNFER_FORCE_COREML");
    force_coreml.expectExitCode(2);

    const integration_step = b.step("integration", "Run CLI integration tests");
    integration_step.dependOn(&run_integration.step);
    integration_step.dependOn(&help_run.step);
    integration_step.dependOn(&env_ok.step);
    integration_step.dependOn(&caps_cpu.step);
    integration_step.dependOn(&bad_backend.step);
    integration_step.dependOn(&block_cpu.step);
    integration_step.dependOn(&stage7_ok.step);
    integration_step.dependOn(&force_sme.step);
    integration_step.dependOn(&force_coreml.step);

    const stage8_ok = b.addRunArtifact(exe);
    stage8_ok.addArg("stage8");
    stage8_ok.expectStdOutMatch("Stage 8");
    stage8_ok.expectStdOutMatch("REJECT");
    stage8_ok.expectExitCode(0);
    integration_step.dependOn(&stage8_ok.step);

    const stage11_ok = b.addRunArtifact(exe);
    stage11_ok.addArg("stage11");
    stage11_ok.expectStdOutMatch("Stage 11");
    stage11_ok.expectExitCode(0);
    integration_step.dependOn(&stage11_ok.step);

    const stage12_ok = b.addRunArtifact(exe);
    stage12_ok.addArg("stage12");
    stage12_ok.expectStdOutMatch("Stage 12");
    stage12_ok.expectExitCode(0);
    integration_step.dependOn(&stage12_ok.step);

    const stage13_ok = b.addRunArtifact(exe);
    stage13_ok.addArg("stage13");
    stage13_ok.expectStdOutMatch("Stage 13");
    stage13_ok.expectExitCode(0);
    integration_step.dependOn(&stage13_ok.step);

    const stageM0_ok = b.addRunArtifact(exe);
    stageM0_ok.addArg("stageM0");
    stageM0_ok.expectStdOutMatch("Stage M0");
    stageM0_ok.expectExitCode(0);
    integration_step.dependOn(&stageM0_ok.step);

    const stageM1_ok = b.addRunArtifact(exe);
    stageM1_ok.addArg("stageM1");
    stageM1_ok.expectStdOutMatch("Stage M1");
    stageM1_ok.expectExitCode(0);
    integration_step.dependOn(&stageM1_ok.step);

    const stageM2_ok = b.addRunArtifact(exe);
    stageM2_ok.addArg("stageM2");
    stageM2_ok.expectStdOutMatch("Stage M2");
    stageM2_ok.expectExitCode(0);
    integration_step.dependOn(&stageM2_ok.step);

    const stageM3_ok = b.addRunArtifact(exe);
    stageM3_ok.addArg("stageM3");
    stageM3_ok.expectStdOutMatch("Stage M3");
    stageM3_ok.expectExitCode(0);
    integration_step.dependOn(&stageM3_ok.step);

    const stageM4_ok = b.addRunArtifact(exe);
    stageM4_ok.addArg("stageM4");
    stageM4_ok.expectStdOutMatch("Stage M4");
    stageM4_ok.expectExitCode(0);
    integration_step.dependOn(&stageM4_ok.step);

    const stageM5_ok = b.addRunArtifact(exe);
    stageM5_ok.addArg("stageM5");
    stageM5_ok.expectStdOutMatch("Stage M5");
    stageM5_ok.expectExitCode(0);
    integration_step.dependOn(&stageM5_ok.step);

    const stageM6_ok = b.addRunArtifact(exe);
    stageM6_ok.addArg("stageM6");
    stageM6_ok.expectStdOutMatch("Stage M6");
    stageM6_ok.expectExitCode(0);
    integration_step.dependOn(&stageM6_ok.step);

    const stageM7_ok = b.addRunArtifact(exe);
    stageM7_ok.addArg("stageM7");
    stageM7_ok.expectStdOutMatch("Stage M7");
    stageM7_ok.expectStdOutMatch("REJECT");
    stageM7_ok.expectExitCode(0);
    integration_step.dependOn(&stageM7_ok.step);

    const stageM8_ok = b.addRunArtifact(exe);
    stageM8_ok.addArg("stageM8");
    stageM8_ok.expectStdOutMatch("Stage M8");
    stageM8_ok.expectStdOutMatch("qwen3-4b");
    stageM8_ok.expectExitCode(0);
    integration_step.dependOn(&stageM8_ok.step);

    const stageS1_ok = b.addRunArtifact(exe);
    stageS1_ok.addArg("stageS1");
    stageS1_ok.expectStdOutMatch("Stage S1");
    stageS1_ok.expectExitCode(0);
    integration_step.dependOn(&stageS1_ok.step);

    const batch_bench_ok = b.addRunArtifact(exe);
    batch_bench_ok.addArg("batch-bench");
    batch_bench_ok.addArg("--mini");
    batch_bench_ok.addArg("--batch-size");
    batch_bench_ok.addArg("2");
    batch_bench_ok.addArg("--max-tokens");
    batch_bench_ok.addArg("4");
    batch_bench_ok.expectStdOutMatch("token_parity: PASS");
    batch_bench_ok.expectStdOutMatch("json");
    batch_bench_ok.expectExitCode(0);
    integration_step.dependOn(&batch_bench_ok.step);

    const coreml_smoke_ok = b.addRunArtifact(exe);
    coreml_smoke_ok.addArg("coreml-smoke");
    coreml_smoke_ok.addArg("tools/fixtures/coreml_toy.mlpackage");
    if (have_apple) {
        // Real Core ML load/predict only on macOS.
        coreml_smoke_ok.expectStdOutMatch("predict_ok");
        coreml_smoke_ok.expectStdOutMatch("does NOT verify ANE");
        coreml_smoke_ok.expectExitCode(0);
    } else {
        // Linux CI: command must report unsupported and exit 0 (not fail the suite).
        coreml_smoke_ok.expectStdOutMatch("unsupported");
        coreml_smoke_ok.expectStdOutMatch("does NOT verify ANE");
        coreml_smoke_ok.expectExitCode(0);
    }
    integration_step.dependOn(&coreml_smoke_ok.step);

    const mem_report_ok = b.addRunArtifact(exe);
    mem_report_ok.addArg("mem-report");
    mem_report_ok.addArg("--mini");
    mem_report_ok.expectStdOutMatch("mem-report");
    mem_report_ok.expectStdOutMatch("json");
    mem_report_ok.expectExitCode(0);
    integration_step.dependOn(&mem_report_ok.step);

    const qwen_bench_ok = b.addRunArtifact(exe);
    qwen_bench_ok.addArg("qwen-bench");
    qwen_bench_ok.addArg("--mini");
    qwen_bench_ok.addArg("--max-tokens");
    qwen_bench_ok.addArg("4");
    qwen_bench_ok.expectStdOutMatch("prefill vs decode");
    qwen_bench_ok.expectStdOutMatch("json");
    qwen_bench_ok.expectExitCode(0);
    integration_step.dependOn(&qwen_bench_ok.step);

    const qwen_profile_ok = b.addRunArtifact(exe);
    qwen_profile_ok.addArg("qwen-profile");
    qwen_profile_ok.addArg("--mini");
    qwen_profile_ok.expectStdOutMatch("one decode token");
    qwen_profile_ok.expectStdOutMatch("top3");
    qwen_profile_ok.expectStdOutMatch("json");
    qwen_profile_ok.expectExitCode(0);
    integration_step.dependOn(&qwen_profile_ok.step);

    const kv_bench_ok = b.addRunArtifact(exe);
    kv_bench_ok.addArg("kv-bench");
    kv_bench_ok.addArg("--mini");
    kv_bench_ok.addArg("--max-tokens");
    kv_bench_ok.addArg("4");
    kv_bench_ok.expectStdOutMatch("token_parity: PASS");
    kv_bench_ok.expectExitCode(0);
    integration_step.dependOn(&kv_bench_ok.step);

    const kv_layout_ok = b.addRunArtifact(exe);
    kv_layout_ok.addArg("kv-bench");
    kv_layout_ok.addArg("--layout");
    kv_layout_ok.expectStdOutMatch("RETAIN");
    kv_layout_ok.expectExitCode(0);
    integration_step.dependOn(&kv_layout_ok.step);

    const forward_mini_compile = b.addRunArtifact(exe);
    forward_mini_compile.step.dependOn(b.getInstallStep());
    forward_mini_compile.addArg("artifact-compile");
    forward_mini_compile.addArg("--mini");
    forward_mini_compile.addArg("--out");
    forward_mini_compile.addArg("zig-out/stage11-mini.zynfer");
    forward_mini_compile.expectExitCode(0);

    const forward_mini = b.addRunArtifact(exe);
    forward_mini.step.dependOn(b.getInstallStep());
    forward_mini.addArg("forward-golden");
    forward_mini.addArg("zig-out/stage11-mini.zynfer");
    forward_mini.addArg("--tokens");
    forward_mini.addArg("2,3");
    forward_mini.expectStdOutMatch("forward-golden");
    forward_mini.expectExitCode(0);
    forward_mini.step.dependOn(&forward_mini_compile.step);
    integration_step.dependOn(&forward_mini.step);

    // Optional local full-model smoke: only when models/qwen3-0.6b.zynfer exists.
    const run_local_if = b.addSystemCommand(&.{
        "sh",
        "-c",
        \\if [ -f models/qwen3-0.6b.zynfer ] && [ -f models/Qwen3-0.6B/vocab.json ]; then
        \\  ./zig-out/bin/zynfer chat "Say hi in one short sentence." --max-tokens 8 --no-stream;
        \\else
        \\  echo "skip local zynfer run (no models/qwen3-0.6b.zynfer)";
        \\fi
    });
    run_local_if.step.dependOn(b.getInstallStep());
    run_local_if.expectExitCode(0);
    integration_step.dependOn(&run_local_if.step);

    const docs_lib = b.addLibrary(.{
        .name = "zynfer",
        .root_module = zynfer_mod,
        .linkage = .static,
    });
    const install_docs = b.addInstallDirectory(.{
        .source_dir = docs_lib.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs/api",
    });
    const docs_step = b.step("docs", "Generate Zig autodoc into zig-out/docs/api");
    docs_step.dependOn(&install_docs.step);

    const fmt_step = b.step("fmt", "Check Zig formatting");
    const fmt_cmd = b.addSystemCommand(&.{
        b.graph.zig_exe,
        "fmt",
        "--check",
        "build.zig",
        "src",
        "tests",
    });
    fmt_step.dependOn(&fmt_cmd.step);

    const ci_step = b.step("ci", "Local stand-in for CI: fmt, tests, integration, docs");
    ci_step.dependOn(fmt_step);
    ci_step.dependOn(test_step);
    ci_step.dependOn(integration_step);
    ci_step.dependOn(docs_step);
}

fn configureApple(b: *std.Build, mod: *std.Build.Module, have_apple: bool) void {
    if (!have_apple) return;
    mod.link_libc = true;
    mod.addIncludePath(b.path("src/backends/apple"));
    mod.linkFramework("Foundation", .{});
    mod.linkFramework("Metal", .{});
    mod.linkFramework("Accelerate", .{});
    mod.linkFramework("CoreML", .{});
    mod.addCSourceFile(.{
        .file = b.path("src/backends/apple/bridge.m"),
        .flags = &.{"-fobjc-arc"},
        .language = .objective_c,
    });
    mod.addCSourceFile(.{
        .file = b.path("src/backends/apple/coreml_bridge.m"),
        .flags = &.{"-fobjc-arc"},
        .language = .objective_c,
    });
}

fn configureHip(b: *std.Build, mod: *std.Build.Module, have_hip: bool, hip_path: []const u8) void {
    if (!have_hip) return;

    mod.link_libc = true;
    mod.addIncludePath(b.path("src"));
    mod.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include", .{hip_path}) });
    mod.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib", .{hip_path}) });
    mod.addRPath(.{ .cwd_relative = b.fmt("{s}/lib", .{hip_path}) });
    mod.linkSystemLibrary("amdhip64", .{});
    mod.addCSourceFile(.{
        .file = b.path("src/hip_probe.c"),
        .flags = &.{
            "-std=c11",
            "-D__HIP_PLATFORM_AMD__",
        },
    });
}

fn detectHipPrefix(b: *std.Build, explicit: ?[]const u8) ?[]const u8 {
    if (explicit) |path| {
        if (hipLooksValid(b, path)) return path;
        std.log.warn("HIP path '{s}' does not contain include/hip/hip_runtime_api.h", .{path});
        return null;
    }
    if (b.graph.environ_map.get("HIP_PATH")) |path| {
        if (hipLooksValid(b, path)) return path;
    }
    if (b.graph.environ_map.get("ROCM_PATH")) |path| {
        if (hipLooksValid(b, path)) return path;
    }
    if (hipLooksValid(b, "/opt/rocm")) return "/opt/rocm";
    if (hipLooksValid(b, "/usr")) return "/usr";
    return null;
}

fn hipLooksValid(b: *std.Build, prefix: []const u8) bool {
    var buf: [512]u8 = undefined;
    const header = std.fmt.bufPrint(&buf, "{s}/include/hip/hip_runtime_api.h", .{prefix}) catch return false;
    if (std.fs.path.isAbsolute(header)) {
        std.Io.Dir.accessAbsolute(b.graph.io, header, .{}) catch return false;
        return true;
    }
    std.Io.Dir.cwd().access(b.graph.io, header, .{}) catch return false;
    return true;
}
