const std = @import("std");
const vkgen = @import("vulkan_zig");
const ShaderCompileStep = vkgen.ShaderCompileStep;

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "zig-vulkan-learn",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(exe);

    const glfw_dep = b.dependency("mach_glfw", .{});
    exe.root_module.addImport("glfw", glfw_dep.module("mach-glfw"));

    const vk_registry = b.dependency("vulkan_headers", .{}).path("registry/vk.xml");

    const vk_gen = b.dependency("vulkan_zig", .{}).artifact("vulkan-zig-generator");
    const vk_gen_cmd = b.addRunArtifact(vk_gen);
    vk_gen_cmd.addFileArg(vk_registry);

    exe.root_module.addAnonymousImport("vk", .{
        .root_source_file = vk_gen_cmd.addOutputFileArg("vk.zig"),
    });

    const shaders = ShaderCompileStep.create(
        b,
        &[_][]const u8{ "glslc", "--target-env=vulkan1.3" },
        "-o",
    );
    shaders.add("main_vert", "src/shaders/main.vert", .{});
    shaders.add("main_frag", "src/shaders/main.frag", .{});
    exe.root_module.addImport("shaders", shaders.getModule());

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}