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

    const options = b.addOptions();
    options.addOption(bool, "have_hip", have_hip);
    options.addOption([]const u8, "hip_path", hip_path);

    const zynfer_mod = b.addModule("zynfer", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    zynfer_mod.addOptions("build_options", options);
    configureHip(b, zynfer_mod, have_hip, hip_path);

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

    const bench_cmd = b.addRunArtifact(exe);
    bench_cmd.step.dependOn(b.getInstallStep());
    bench_cmd.addArg("bench");
    const bench_step = b.step("bench", "Time HIP device enumeration");
    bench_step.dependOn(&bench_cmd.step);

    const unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    unit_tests.root_module.addOptions("build_options", options);
    configureHip(b, unit_tests.root_module, have_hip, hip_path);

    const smoke_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/unit/smoke.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zynfer", .module = zynfer_mod },
            },
        }),
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);
    const run_smoke_tests = b.addRunArtifact(smoke_tests);
    const test_step = b.step("test", "Run Stage 0 tests");
    test_step.dependOn(&run_unit_tests.step);
    test_step.dependOn(&run_smoke_tests.step);
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
