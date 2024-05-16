const std = @import("std");
const vkgen = @import("vulkan_zig");
const ShaderCompileStep = vkgen.ShaderCompileStep;

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const vk_registry = b.dependency("vulkan_headers", .{}).path("registry/vk.xml");

    const vk_gen = b.dependency("vulkan_zig", .{}).artifact("vulkan-zig-generator");
    const vk_gen_cmd = b.addRunArtifact(vk_gen);
    vk_gen_cmd.addFileArg(vk_registry);

    const vk = b.addModule("vk", .{
        .root_source_file = vk_gen_cmd.addOutputFileArg("vk.zig"),
    });
    const image = b.addModule("image", .{
        .root_source_file = b.path("src/image.zig"),
    });
    image.addImport("vk", vk);

    const shaders = ShaderCompileStep.create(
        b,
        &[_][]const u8{ "glslc", "--target-env=vulkan1.3" },
        "-o",
    );
    shaders.add("main_vert", "shaders/main.vert", .{});
    shaders.add("main_frag", "shaders/main.frag", .{});

    const exe = b.addExecutable(.{
        .name = "zig-vulkan-learn",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(exe);

    exe.root_module.addImport("vk", vk);
    exe.root_module.addImport("image", image);  
    exe.root_module.addImport("shaders", shaders.getModule());
    exe.root_module.addImport("glfw", b.dependency("mach_glfw", .{}).module("mach-glfw"));

    const assembler = b.addExecutable(.{
        .name = "assembler",
        .root_source_file = b.path("src/assembler.zig"),
        .target = target,
        .optimize = optimize,
    });

    assembler.root_module.addImport("vk", vk);
    assembler.root_module.addImport("image", image);  
    
    const run_exe = b.addRunArtifact(exe);
    const run_assembler = b.addRunArtifact(assembler);
    run_exe.step.dependOn(&run_assembler.step);

    const run_exe_step = b.step("run", "Run main executable");
    const run_assemble_step = b.step("assemble", "Assemble assets into single asset file");

    run_exe_step.dependOn(&run_exe.step);
    run_assemble_step.dependOn(&run_assembler.step);    
}