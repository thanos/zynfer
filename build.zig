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
