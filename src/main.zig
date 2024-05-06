const std = @import("std");
const glfw = @import("glfw");
const vk = @import("vk");
const shaders = @import("shaders");
const print = std.debug.print;

const required_deviceice_extensions = [_][*:0]const u8{
    "VK_KHR_swapchain",
};
const required_instance_extensions = [_][*:0]const u8{
    "VK_KHR_surface",
    "VK_KHR_win32_surface",
    "VK_EXT_debug_utils",
};
const required_validation_layers = [_][*:0]const u8{
    "VK_LAYER_KHRONOS_validation",
};

const present_modes_cap = 4;
const surface_formats_cap = 4;
const extensions_cap = 128;
const pdevices_cap = 4;
const validation_layers_cap = 16;
const propps_cap = 4;
const swapchain_images_cap = 8;

fn debugCallback(
    message_severity: vk.DebugUtilsMessageSeverityFlagsEXT, 
    message_type: vk.DebugUtilsMessageTypeFlagsEXT, 
    p_callback_data: ?*const vk.DebugUtilsMessengerCallbackDataEXT, 
    _: ?*anyopaque,
) callconv(.C) vk.Bool32 {
    const data = p_callback_data.?;

    print(
        \\validation layer:
        \\  message_severity: {} 
        \\  message_type: {}
        \\  .flags = {},
        \\  .p_message_id_name = {s},
        \\  .p_message =
        \\      {s},
        \\
        , 
        .{
            message_severity,
            message_type,
            data.flags,
            data.p_message_id_name.?,
            data.p_message.?,
        },
    );

    return vk.FALSE;
}

const apis: []const vk.ApiInfo = &.{
    vk.features.version_1_0,
    vk.features.version_1_3,

    vk.extensions.khr_surface,
    vk.extensions.khr_swapchain,
    
    vk.extensions.ext_debug_utils,
};

/// Next, pass the `apis` to the wrappers to create dispatch tables.
const BaseDispatch = vk.BaseWrapper(apis);
const InstanceDispatch = vk.InstanceWrapper(apis);
const DeviceDispatch = vk.DeviceWrapper(apis);

pub fn main() !void {
    if (!glfw.init(.{})) {
        return error.GlfwInitFailed;
    }
    if (!glfw.vulkanSupported()) {
        return error.VulkanNotSupported;
    }
    defer glfw.terminate();

    print("\n", .{});

    const app_name = "Down with chemistry!";

    // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    // defer _ = gpa.deinit();
    // const allocator = gpa.allocator();

    var extent: vk.Extent2D = .{
        .width = 1200,
        .height = 800,
    };
    var window = glfw.Window.create(
        extent.width, 
        extent.height, 
        app_name, 
        null, 
        null, 
        .{
            .client_api = .no_api,
        }
    ) orelse return error.WindowCreateFailed;
    defer window.destroy();

    const vkb: BaseDispatch = try BaseDispatch.load(@as(vk.PfnGetInstanceProcAddr, @ptrCast(&glfw.getInstanceProcAddress)));

    var validation_layers_len: u32 = undefined;
    _ = try vkb.enumerateInstanceLayerProperties(&validation_layers_len, null);

    var validation_layers: [validation_layers_cap]vk.LayerProperties = undefined;
    _ = try vkb.enumerateInstanceLayerProperties(&validation_layers_len, &validation_layers);

    outer: for (required_validation_layers) |required_layer| {
        for (0..validation_layers_len) |validation_layer_i| {
            const validation_layer = validation_layers[validation_layer_i];
            if (std.mem.eql(u8, std.mem.span(required_layer), std.mem.sliceTo(&validation_layer.layer_name, 0))) {
                continue :outer;
            }
        }
        return error.ValidationLayerNotFound;
    }
        
    const instance = try vkb.createInstance(
        &.{
            .p_application_info = &.{
                .p_application_name = app_name,
                .application_version = vk.makeApiVersion(0, 0, 0, 0),
                .p_engine_name = app_name,
                .engine_version = vk.makeApiVersion(0, 0, 0, 0),
                .api_version = vk.API_VERSION_1_3,
            },
            .enabled_layer_count = required_validation_layers.len,
            .pp_enabled_layer_names = &required_validation_layers,
            .enabled_extension_count = required_instance_extensions.len,
            .pp_enabled_extension_names = &required_instance_extensions,
        }, 
        null,
    );

    const vki = try InstanceDispatch.load(instance, vkb.dispatch.vkGetInstanceProcAddr);
    defer vki.destroyInstance(instance, null);

    const debug_messenger = try vki.createDebugUtilsMessengerEXT(
        instance,
        &.{
            .message_severity = .{ 
                .verbose_bit_ext = true, 
                .warning_bit_ext = true, 
                .error_bit_ext = true,
            },
            .message_type = .{ 
                .general_bit_ext = true, 
                .validation_bit_ext = true, 
                .performance_bit_ext = true, 
            },
            .pfn_user_callback = &debugCallback,
        }, 
        null
    );
    defer vki.destroyDebugUtilsMessengerEXT(instance, debug_messenger, null);

    var surface: vk.SurfaceKHR = undefined;
    if (@as(vk.Result, @enumFromInt(glfw.createWindowSurface(instance, window, null, &surface))) != .success) {
        return error.SurfaceInitFailed;
    }
    defer vki.destroySurfaceKHR(instance, surface, null);

    var pdevices_len: u32 = undefined;
    _ = try vki.enumeratePhysicalDevices(instance, &pdevices_len, null);

    var pdevices: [pdevices_cap]vk.PhysicalDevice = undefined;
    _ = try vki.enumeratePhysicalDevices(instance, &pdevices_len, &pdevices);
    
    var pdevice: vk.PhysicalDevice = undefined;
    var pdevice_i: usize = 0;

    var _graphics_family: ?u32 = undefined;
    var _present_family: ?u32 = undefined;

    var pdevice_features: vk.PhysicalDeviceFeatures = undefined;
    var pdevice_props: vk.PhysicalDeviceProperties = undefined;
    var surface_format: vk.SurfaceFormatKHR = undefined;
    var present_mode: vk.PresentModeKHR = undefined;

    outer: while (true) {
        pdevice = pdevices[pdevice_i];
        pdevice_props = vki.getPhysicalDeviceProperties(pdevice);
        pdevice_features = vki.getPhysicalDeviceFeatures(pdevice);

        var propss_len: u32 = undefined;
        vki.getPhysicalDeviceQueueFamilyProperties(pdevice, &propss_len, null);

        var propss: [propps_cap]vk.QueueFamilyProperties = undefined;
        vki.getPhysicalDeviceQueueFamilyProperties(pdevice, &propss_len, &propss);

        _graphics_family = null;
        _present_family = null;
        var queue_family: u32 = 0;

        pdevice: while (queue_family != propss_len) {
            const props = propss[queue_family];
            if (props.queue_flags.graphics_bit) {
                _graphics_family = queue_family;
            }

            const present = try vki.getPhysicalDeviceSurfaceSupportKHR(pdevice, queue_family, surface) == vk.TRUE;
            if (present and (_graphics_family == null or _graphics_family.? != queue_family)) {
                _present_family = queue_family;
            }

            if (_graphics_family != null and _present_family != null) {
                var surface_formats_len: u32 = undefined;
                _ = try vki.getPhysicalDeviceSurfaceFormatsKHR(pdevice, surface, &surface_formats_len, null);
                if (surface_formats_len == 0) {
                    break :pdevice;
                }

                var surface_formats: [surface_formats_cap]vk.SurfaceFormatKHR = undefined;
                _ = try vki.getPhysicalDeviceSurfaceFormatsKHR(pdevice, surface, &surface_formats_len, &surface_formats);

                var present_modes_len: u32 = undefined;
                _ = try vki.getPhysicalDeviceSurfacePresentModesKHR(pdevice, surface, &present_modes_len, null);
                if (present_modes_len == 0) {
                    break :pdevice;
                }

                var present_modes: [present_modes_cap]vk.PresentModeKHR = undefined;
                _ = try vki.getPhysicalDeviceSurfacePresentModesKHR(pdevice, surface, &present_modes_len, &present_modes);
                
                var extensions_len: u32 = undefined;
                _ = try vki.enumerateDeviceExtensionProperties(pdevice, null, &extensions_len, null);

                var extensions: [extensions_cap]vk.ExtensionProperties = undefined;
                _ = try vki.enumerateDeviceExtensionProperties(pdevice, null, &extensions_len, &extensions);

                required_ext: for (required_deviceice_extensions) |required_ext| {
                    for (0..extensions_len) |ext_i| {
                        const ext = extensions[ext_i];

                        if (std.mem.eql(u8, std.mem.span(required_ext), std.mem.sliceTo(&ext.extension_name, 0))) {
                            continue :required_ext;
                        }
                    }
                    break :pdevice;
                }

                surface_format = chooseSurfaceFormat(&surface_formats, surface_formats_len);
                present_mode = choosePresentMode(&present_modes, present_modes_len);

                break :outer;
            }
            queue_family += 1;
        }

        pdevice_i += 1;
        if (pdevice_i == pdevices_len) {
            return error.SuitablePhysicalDeviceNotFound;
        }
    }
    // print("\nselected physical deviceice: {s}\n", .{pdevice_name});

    const graphics_family = _graphics_family.?;
    const present_family = _present_family.?;
    const device = try vki.createDevice(
        pdevice, 
        &.{
            .p_queue_create_infos = &.{
                    .{
                        .queue_family_index = graphics_family,
                        .queue_count = 1,
                        .p_queue_priorities = &[_]f32 {1.0},
                    },
                    // todo: present_family may == graphics_family
                    .{
                        .queue_family_index = present_family,
                        .queue_count = 1,
                        .p_queue_priorities = &[_]f32 {1.0},
                    },
            },
            .enabled_extension_count = required_deviceice_extensions.len,
            .pp_enabled_extension_names = &required_deviceice_extensions,
            .queue_create_info_count = 2,
            .p_enabled_features = &pdevice_features,
        }, 
        null
    );
    const vkd = try DeviceDispatch.load(device, vki.dispatch.vkGetDeviceProcAddr);
    defer vkd.destroyDevice(device, null);

    const surface_capabilities = try vki.getPhysicalDeviceSurfaceCapabilitiesKHR(pdevice, surface);
    var swapchain_images_len = surface_capabilities.min_image_count + 1;
    if (surface_capabilities.max_image_count != 0 and swapchain_images_len > surface_capabilities.max_image_count) {
        swapchain_images_len = surface_capabilities.max_image_count;
    }
    
    setExtent(&extent, surface_capabilities);

    const swapchain = try vkd.createSwapchainKHR(
        device,
        &.{
            .surface = surface,
            .min_image_count = swapchain_images_len,
            .image_format = surface_format.format,
            .image_color_space = surface_format.color_space,
            .image_extent = extent,
            .image_array_layers = 1,
            .image_usage = .{ .color_attachment_bit = true },
            .image_sharing_mode = vk.SharingMode.concurrent,
            .queue_family_index_count = 2,
            .p_queue_family_indices = &[_]u32 {graphics_family, present_family},
            .composite_alpha = .{ .opaque_bit_khr = true, },
            .present_mode = present_mode,
            .clipped = vk.TRUE,
            .pre_transform = surface_capabilities.current_transform,
            .old_swapchain = .null_handle,
        }, 
        null,
    );
    defer vkd.destroySwapchainKHR(device, swapchain, null);

    _ = try vkd.getSwapchainImagesKHR(device, swapchain, &swapchain_images_len, null);
    var swapchain_images: [swapchain_images_cap]vk.Image = undefined;
    _ = try vkd.getSwapchainImagesKHR(device, swapchain, &swapchain_images_len, &swapchain_images);

    var swapchain_image_views: [swapchain_images_cap]vk.ImageView = undefined;
    for (0..swapchain_images_len) |i| {
        swapchain_image_views[i] = try vkd.createImageView(
            device,
            &.{
                .image = swapchain_images[i],
                .components = .{
                    .r = .identity,
                    .g = .identity,
                    .b = .identity,
                    .a = .identity,
                },
                .subresource_range = .{
                    .aspect_mask = .{ .color_bit = true, },
                    .base_mip_level = 0,
                    .level_count = 1,
                    .base_array_layer = 0,
                    .layer_count = 1,
                },
                .format = surface_format.format,
                .view_type = .@"2d",
            }, 
            null,
        );
    }

    const color_attachment = vk.AttachmentDescription {
        .format = surface_format.format,
        .samples = .{ .@"1_bit" = true },
        .load_op = .clear,
        .store_op = .store,
        .stencil_load_op = .dont_care,
        .stencil_store_op = .dont_care,
        .initial_layout = .undefined,
        .final_layout = .present_src_khr,
    };

    const color_attachment_ref = vk.AttachmentReference {
        .attachment = 0,
        .layout = .color_attachment_optimal,
    };

    const subpass = vk.SubpassDescription {
        .pipeline_bind_point = .graphics,
        .color_attachment_count = 1,
        .p_color_attachments = @ptrCast(&color_attachment_ref),
    };

    const subpass_dependency = vk.SubpassDependency {
        .src_subpass = vk.SUBPASS_EXTERNAL,
        .dst_subpass = 0,
        .src_stage_mask = .{ 
            .color_attachment_output_bit = true, 
        },
        .src_access_mask = .{
        },
        .dst_stage_mask = .{
            .color_attachment_output_bit = true,
        },
        .dst_access_mask = .{
            .color_attachment_write_bit = true,
        },
    };

    const render_pass = try vkd.createRenderPass(
        device,
        &.{
            .attachment_count = 1,
            .p_attachments = @ptrCast(&color_attachment),
            .subpass_count = 1,
            .p_subpasses = @ptrCast(&subpass),
            .dependency_count = 1,
            .p_dependencies = @ptrCast(&subpass_dependency),
        },
        null,
    );
    defer vkd.destroyRenderPass(device, render_pass, null);

    const vert = try vkd.createShaderModule(
        device, 
        &.{
            .code_size = shaders.main_vert.len,
            .p_code = @ptrCast(&shaders.main_vert),
        }, 
        null,
    );
    defer vkd.destroyShaderModule(device, vert, null);

    const frag = try vkd.createShaderModule(
        device,
        &.{
            .code_size = shaders.main_frag.len,
            .p_code = @ptrCast(&shaders.main_frag),
        },
        null,
    );
    defer vkd.destroyShaderModule(device, frag, null);

    const shader_stage_cis = [_]vk.PipelineShaderStageCreateInfo{
        .{
            .stage = .{ .vertex_bit = true },
            .module = vert,
            .p_name = "main",
        },
        .{
            .stage = .{ .fragment_bit = true },
            .module = frag,
            .p_name = "main",
        },
    };

    const vertex_input_state_ci = vk.PipelineVertexInputStateCreateInfo{
        .vertex_binding_description_count = 0,
        .p_vertex_binding_descriptions = null,
        .vertex_attribute_description_count = 0,
        .p_vertex_attribute_descriptions = null,
    };

    const input_assembly_state_ci = vk.PipelineInputAssemblyStateCreateInfo{
        .topology = .triangle_list,
        .primitive_restart_enable = vk.FALSE,
    };

    const viewport_state_ci = vk.PipelineViewportStateCreateInfo{
        .viewport_count = 1,
        .p_viewports = null, // set in createCommandBuffers with cmdSetViewport
        .scissor_count = 1,
        .p_scissors = null, // set in createCommandBuffers with cmdSetScissor
    };

    const rasterization_state_ci = vk.PipelineRasterizationStateCreateInfo{
        .depth_clamp_enable = vk.FALSE,
        .rasterizer_discard_enable = vk.FALSE,
        .polygon_mode = .fill,
        .cull_mode = .{ .back_bit = true },
        .front_face = .clockwise,
        .depth_bias_enable = vk.FALSE,
        .depth_bias_constant_factor = 0,
        .depth_bias_clamp = 0,
        .depth_bias_slope_factor = 0,
        .line_width = 1,
    };

    const multi_sample_state_ci = vk.PipelineMultisampleStateCreateInfo{
        .rasterization_samples = .{ .@"1_bit" = true },
        .sample_shading_enable = vk.FALSE,
        .min_sample_shading = 1,
        .alpha_to_coverage_enable = vk.FALSE,
        .alpha_to_one_enable = vk.FALSE,
    };

    const blend_attachment_states = [_]vk.PipelineColorBlendAttachmentState{
        // present framebuffer
        vk.PipelineColorBlendAttachmentState{
            .blend_enable = vk.FALSE,
            .src_color_blend_factor = .one,
            .dst_color_blend_factor = .zero,
            .color_blend_op = .add,
            .src_alpha_blend_factor = .one,
            .dst_alpha_blend_factor = .zero,
            .alpha_blend_op = .add,
            .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
        },
    };

    const blend_state_ci = vk.PipelineColorBlendStateCreateInfo{
        .logic_op_enable = vk.FALSE,
        .logic_op = .copy,
        .attachment_count = 1,
        .p_attachments = &blend_attachment_states,
        .blend_constants = [_]f32{ 0, 0, 0, 0, },
    };

    const dynamic_states = [_]vk.DynamicState{ 
        .viewport, 
        .scissor, 
    };
    const dynamic_state_ci = vk.PipelineDynamicStateCreateInfo{
        .flags = .{},
        .dynamic_state_count = dynamic_states.len,
        .p_dynamic_states = &dynamic_states,
    };

    const pipeline_layout = try vkd.createPipelineLayout(
        device,
        &.{
            .set_layout_count = 0,
            .p_set_layouts = null,
            .push_constant_range_count = 0,
            .p_push_constant_ranges = null,
        }, 
        null
    );
    defer vkd.destroyPipelineLayout(device, pipeline_layout, null);

    var pipeline: vk.Pipeline = undefined;
    _ = try vkd.createGraphicsPipelines(
        device,
        .null_handle,
        1,
        @ptrCast(&vk.GraphicsPipelineCreateInfo{
            .flags = .{},
            .stage_count = shader_stage_cis.len,
            .p_stages = &shader_stage_cis,
            .p_vertex_input_state = &vertex_input_state_ci,
            .p_input_assembly_state = &input_assembly_state_ci,
            .p_tessellation_state = null,
            .p_viewport_state = &viewport_state_ci,
            .p_rasterization_state = &rasterization_state_ci,
            .p_multisample_state = &multi_sample_state_ci,
            .p_depth_stencil_state = null,
            .p_color_blend_state = &blend_state_ci,
            .p_dynamic_state = &dynamic_state_ci,
            .layout = pipeline_layout,
            .render_pass = render_pass,
            .subpass = 0,
            .base_pipeline_handle = .null_handle,
            .base_pipeline_index = -1,
        }),
        null,
        @ptrCast(&pipeline),
    );
    defer vkd.destroyPipeline(device, pipeline, null);

    var swapchain_framebuffers: [swapchain_images_cap]vk.Framebuffer = undefined;
    for (0..swapchain_images_len) |i| {
        const attachments = [_]vk.ImageView{
            swapchain_image_views[i],
        };

        swapchain_framebuffers[i] = try vkd.createFramebuffer(
            device,
            &.{
                .render_pass = render_pass,
                .attachment_count = attachments.len,
                .p_attachments = &attachments,
                .width = extent.width,
                .height = extent.height,
                .layers = 1,
            }, 
            null,
        );
    }

    const command_pool = try vkd.createCommandPool(
        device,
        &.{
            .flags = .{ 
                .reset_command_buffer_bit = true,
            },
            .queue_family_index = graphics_family,
        }, 
        null
    );
    defer vkd.destroyCommandPool(device, command_pool, null);

    var command_buffer: vk.CommandBuffer = undefined;
    _ = try vkd.allocateCommandBuffers(
        device,
        @ptrCast(&vk.CommandBufferAllocateInfo{
            .command_pool = command_pool,
            .level = .primary,
            .command_buffer_count = 1,
        }), 
        @ptrCast(&command_buffer),
    );

    const graphics_queue = vkd.getDeviceQueue(device, graphics_family, 0);
    const present_queue = vkd.getDeviceQueue(device, graphics_family, 0);
    
    const image_available_semaphore = try vkd.createSemaphore(
        device,
        &.{
            .flags = .{},
        },
        null,
    );
    defer vkd.destroySemaphore(device, image_available_semaphore, null);


    const render_finished_semaphore = try vkd.createSemaphore(
        device,
        &.{
            .flags = .{},
        },
        null,
    );
    defer vkd.destroySemaphore(device, render_finished_semaphore, null);


    const in_flight_fence = try vkd.createFence(
        device, 
        &.{
            .flags = .{
                .signaled_bit = true,
            },
        }, 
        null,
    );
    defer vkd.destroyFence(device, in_flight_fence, null);

    while (true) {
        if (window.getKey(.escape) == .press) {
            break;
        }

        _ = try vkd.waitForFences(
            device, 
            1, 
            @ptrCast(&in_flight_fence), 
            vk.TRUE, 
            std.math.maxInt(u64)
        );
        _ = try vkd.resetFences(
            device, 
            1, 
            @ptrCast(&in_flight_fence)
        );

        const swapchain_image_index = (try vkd.acquireNextImageKHR(
            device,
            swapchain, 
            std.math.maxInt(u64), 
            image_available_semaphore, 
            .null_handle
        )).image_index;

        _ = try vkd.resetCommandBuffer(command_buffer, .{});

        _ = try vkd.beginCommandBuffer(
            command_buffer, 
            @ptrCast(&vk.CommandBufferBeginInfo {
                .flags = .{},
                .p_inheritance_info = null,
            }),
        );

        const clear_colors = [_]vk.ClearValue {
            .{
                .color = .{
                    .float_32 = [4]f32 {0, 0, 0, 0,},
                },
            },
        };
        const render_pass_bi = vk.RenderPassBeginInfo {
            .render_pass = render_pass,
            .framebuffer = swapchain_framebuffers[swapchain_image_index],
            .render_area = vk.Rect2D {
                .extent = extent,
                .offset = .{
                    .x = 0,
                    .y = 0,
                },
            },
            .clear_value_count = clear_colors.len,
            .p_clear_values = &clear_colors,
        };

        vkd.cmdBeginRenderPass(
            command_buffer, 
            &render_pass_bi, 
            .@"inline"
        );

        vkd.cmdBindPipeline(
            command_buffer, 
            .graphics, 
            pipeline
        );

        vkd.cmdSetViewport(
            command_buffer, 
            0, 
            1, 
            @ptrCast(&vk.Viewport{
                .x = 0,
                .y = 0,
                .width = @floatFromInt(extent.width),
                .height = @floatFromInt(extent.height),
                .min_depth = 0.0,
                .max_depth = 1.0,
            }),
        );

        vkd.cmdSetScissor(
            command_buffer, 
            0, 
            1, 
            @ptrCast(&vk.Rect2D {
                .offset = .{
                    .x = 0,
                    .y = 0,
                },
                .extent = extent,
            }),
        );

        vkd.cmdDraw(
            command_buffer, 
            3, 
            1,
            0, 
            0
        );

        vkd.cmdEndRenderPass(command_buffer);

        _ = try vkd.endCommandBuffer(command_buffer);

        _ = try vkd.queueSubmit(
            graphics_queue, 
            1, 
            @ptrCast(&vk.SubmitInfo{
                .command_buffer_count = 1,
                .p_command_buffers = @ptrCast(&command_buffer),
                .wait_semaphore_count = 1,
                .p_wait_semaphores = @ptrCast(&image_available_semaphore),
                .p_wait_dst_stage_mask = @ptrCast(&vk.PipelineStageFlags {
                    .color_attachment_output_bit = true,
                }),
                .signal_semaphore_count = 1,
                .p_signal_semaphores = @ptrCast(&render_finished_semaphore),
            }),
            in_flight_fence,
        );

        _ = try vkd.queuePresentKHR(
            present_queue, 
            &.{
                .swapchain_count = 1,
                .p_swapchains = @ptrCast(&swapchain),
                .p_image_indices = @ptrCast(&swapchain_image_index),
                .wait_semaphore_count = 1,
                .p_wait_semaphores = @ptrCast(&render_finished_semaphore),
            },
        );

        glfw.pollEvents();
    }

    _ = try vkd.deviceWaitIdle(device);

    for (0..swapchain_images_len) |i| {
        vkd.destroyFramebuffer(device, swapchain_framebuffers[i], null);
        vkd.destroyImageView(device, swapchain_image_views[i], null);
    }
}

fn setExtent(extent: *vk.Extent2D, capabilities: vk.SurfaceCapabilitiesKHR) void {
    if (capabilities.current_extent.width != 0xFFFF_FFFF) {
        return;
    }

    if (extent.width < capabilities.min_image_extent.width) {
        extent.width = capabilities.min_image_extent.width;
    } else if (extent.width > capabilities.max_image_extent.width) {
        extent.width = capabilities.max_image_extent.width;
    }

    if (extent.height < capabilities.min_image_extent.height) {
        extent.height = capabilities.min_image_extent.height;
    } else if (extent.height > capabilities.max_image_extent.height) {
        extent.height = capabilities.max_image_extent.height;
    }
}

fn choosePresentMode(present_modes: *const [present_modes_cap]vk.PresentModeKHR, len: u32) vk.PresentModeKHR {
    for (0..len) |i| {
        if (present_modes[i] == .mailbox_khr) {
            return .mailbox_khr;
        }
    }
    return .fifo_khr;
}

fn chooseSurfaceFormat(surface_formats: *const [surface_formats_cap]vk.SurfaceFormatKHR, len: u32) vk.SurfaceFormatKHR {
    const wanted = .{
        .format = .b8g8r8a8_srgb,
        .color_space = .srgb_nonlinear_khr,
    };
    for (0..len) |i| {
        if (std.meta.eql(surface_formats[i], wanted)) {
            return wanted;
        }
    }

    return surface_formats[0];
}