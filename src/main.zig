const std = @import("std");
const glfw = @import("glfw");
const vk = @import("vk");
const shaders = @import("shaders");
const math = @import("math.zig");

const print = std.debug.print;

const required_device_extensions = [_][*:0]const u8{
    "VK_KHR_swapchain",
};
const required_instance_extensions = [_][*:0]const u8{
    "VK_KHR_surface",
    "VK_KHR_win32_surface",
    "VK_EXT_debug_utils",
    // "VK_EXT_layer_settings",
};
const required_validation_layers = [_][*:0]const u8{
    "VK_LAYER_KHRONOS_validation",
};

const present_modes_cap = 4;
const surface_formats_cap = 4;
const extensions_cap = 256;
const pdevices_cap = 4;
const validation_layers_cap = 16;
const propss_cap = 4;
const swapchain_cap = 4;

const Vertex = extern struct {
    x: f32, y: f32, z: f32,
    r: f32, g: f32, b: f32,

    const binding_description = vk.VertexInputBindingDescription {
        .binding = 0,
        .stride = @sizeOf(Vertex),
        .input_rate = .vertex,
    };

    const attribute_descriptions = [_]vk.VertexInputAttributeDescription {
        vk.VertexInputAttributeDescription {
            .binding = 0,
            .location = 0,
            .format = .r32g32b32_sfloat,
            .offset = @offsetOf(Vertex, "x"),
        },
        vk.VertexInputAttributeDescription {
            .binding = 0,
            .location = 1,
            .format = .r32g32b32_sfloat,
            .offset = @offsetOf(Vertex, "r"),
        },
    };
};

const CameraUBO = extern struct {
    view: math.Affine3,
    near_z: f32,    
};

const vertices = [_]Vertex {
    Vertex {
        .x = -0.5, .y = -0.5, .z = 0.0,
        .r = 0.5, .g = 0.5, .b = 0.0,
    },
    Vertex {
        .x = 0.5, .y = -0.5, .z = 0.0,
        .r = 0.0, .g = 1.0, .b = 0.0,
    },
    Vertex {
        .x = 0.5, .y = 0.5, .z = 0.0,
        .r = 1.0, .g = 0.0, .b = 1.0,
    },
    Vertex {
        .x = -0.5, .y = 0.5, .z = 0.0,
        .r = 0.0, .g = 1.0, .b = 1.0,
    },
};

const indices = [_]u16 {
    0, 1, 2, 
    2, 3, 0,
};

fn debugCallback(
    message_severity: vk.DebugUtilsMessageSeverityFlagsEXT, 
    _: vk.DebugUtilsMessageTypeFlagsEXT, 
    p_callback_data: ?*const vk.DebugUtilsMessengerCallbackDataEXT, 
    _: ?*anyopaque,
) callconv(.C) vk.Bool32 {
    const data = p_callback_data.?;

    // check ansi standard
    var color_code: u8 = undefined;
    if (message_severity.verbose_bit_ext) {
        color_code = '0';
    } else if (message_severity.info_bit_ext) {
        color_code = '2';
    } else if (message_severity.warning_bit_ext) {
        color_code = '3';
    } else {
        color_code = '1';
    }

    print(
        "\x1b[9{c};1;4m{s}\x1b[m\x1b[9{c}m\n{s}\n\x1b[m\n",
        .{
            color_code,
            data.p_message_id_name.?,
            color_code,
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

    const debug_create_info = vk.DebugUtilsMessengerCreateInfoEXT{
        .message_severity = .{ 
            .verbose_bit_ext = true,
            .info_bit_ext = true, 
            .warning_bit_ext = true,
            .error_bit_ext = true,
        },
        .message_type = .{ 
            .general_bit_ext = true, 
            .validation_bit_ext = true, 
            .performance_bit_ext = true,
            .device_address_binding_bit_ext = true, 
        },
        .pfn_user_callback = &debugCallback,
    };

    // ensure .FALSE and .TRUE are of type u32
    const report_flags = &[_][]const u8{"info","warn","perf","error","debug"};
    const validate_sync = vk.TRUE;
    const validate_core = vk.TRUE;
    const thread_safety = vk.TRUE;
    const best_practices = vk.TRUE;
    const debug_action = "VK_DBG_LAYER_ACTION_LOG_MSG";
    const enable_message_limit = vk.TRUE;
    const duplicate_message_limit: u32 = 1;

    const layer_settings = [_]vk.LayerSettingEXT {
        vk.LayerSettingEXT {
            .p_layer_name = "VK_LAYER_KHRONOS_validation",
            .p_setting_name = "validate_core",
            .type = .bool32_ext,
            .value_count = 1,
            .p_values = &validate_core,
        },
        vk.LayerSettingEXT {
            .p_layer_name = "VK_LAYER_KHRONOS_validation",
            .p_setting_name = "validate_sync",
            .type = .bool32_ext,
            .value_count = 1,
            .p_values = &validate_sync,
        },
        vk.LayerSettingEXT {
            .p_layer_name = "VK_LAYER_KHRONOS_validation",
            .p_setting_name = "validate_best_practices",
            .type = .bool32_ext,
            .value_count = 1,
            .p_values = &best_practices,
        },
        vk.LayerSettingEXT {
            .p_layer_name = "VK_LAYER_KHRONOS_validation",
            .p_setting_name = "thread_safety",
            .type = .bool32_ext,
            .value_count = 1,
            .p_values = &thread_safety,
        },
        vk.LayerSettingEXT {
            .p_layer_name = "VK_LAYER_KHRONOS_validation",
            .p_setting_name = "debug_action",
            .type = .string_ext,
            .value_count = 1,
            .p_values = @ptrCast(debug_action),
        },
        vk.LayerSettingEXT {
            .p_layer_name = "VK_LAYER_KHRONOS_validation",
            .p_setting_name = "report_flags",
            .type = .string_ext,
            .value_count = report_flags.len,
            .p_values = @ptrCast(&report_flags),
        },
        vk.LayerSettingEXT {
            .p_layer_name = "VK_LAYER_KHRONOS_validation",
            .p_setting_name = "enable_message_limit",
            .type = .bool32_ext,
            .value_count = 1,
            .p_values = &enable_message_limit,
        },
        vk.LayerSettingEXT {
            .p_layer_name = "VK_LAYER_KHRONOS_validation",
            .p_setting_name = "duplicate_message_limit",
            .type = .uint32_ext,
            .value_count = 1,
            .p_values = &duplicate_message_limit,
        },
    };


    const layer_settings_create_info = vk.LayerSettingsCreateInfoEXT {
        .setting_count = layer_settings.len,
        .p_settings = &layer_settings,
        .p_next = &debug_create_info,
    };

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
            .p_next = &layer_settings_create_info,
        }, 
        null,
    );

    const vki = try InstanceDispatch.load(instance, vkb.dispatch.vkGetInstanceProcAddr);
    defer vki.destroyInstance(instance, null);


    const debug_messenger = try vki.createDebugUtilsMessengerEXT(
        instance,
        &debug_create_info, 
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

        var propss: [propss_cap]vk.QueueFamilyProperties = undefined;
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

                required_ext: for (required_device_extensions) |required_ext| {
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
            .enabled_extension_count = required_device_extensions.len,
            .pp_enabled_extension_names = &required_device_extensions,
            .queue_create_info_count = 2,
            .p_enabled_features = &pdevice_features,
        },
        null
    );
    const vkd = try DeviceDispatch.load(device, vki.dispatch.vkGetDeviceProcAddr);
    defer vkd.destroyDevice(device, null);

    var surface_capabilities = try vki.getPhysicalDeviceSurfaceCapabilitiesKHR(pdevice, surface);
    var swapchain_images_len: u32 = surface_capabilities.min_image_count + 1;
    if (surface_capabilities.max_image_count != 0 and swapchain_images_len > surface_capabilities.max_image_count) {
        swapchain_images_len = surface_capabilities.max_image_count;
    }
    
    setExtent(&extent, &window, surface_capabilities);

    var swapchain = try vkd.createSwapchainKHR(
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
    var swapchain_images: [swapchain_cap]vk.Image = undefined;
    _ = try vkd.getSwapchainImagesKHR(device, swapchain, &swapchain_images_len, &swapchain_images);
    
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

    const descriptor_layout_bindings = [_]vk.DescriptorSetLayoutBinding {
        vk.DescriptorSetLayoutBinding{
            .binding = 0,     
            .descriptor_type = .uniform_buffer_dynamic,
            .descriptor_count = 1,
            .stage_flags = .{ .vertex_bit = true },
        },
    };
    const descriptor_set_layout = try vkd.createDescriptorSetLayout(
        device, 
        &.{
            .binding_count = descriptor_layout_bindings.len,
            .p_bindings = &descriptor_layout_bindings,
        }, 
        null,
    );
    defer vkd.destroyDescriptorSetLayout(device, descriptor_set_layout, null);


    const vertex_input_state_ci = vk.PipelineVertexInputStateCreateInfo{
        .vertex_binding_description_count = 1,
        .p_vertex_binding_descriptions = @ptrCast(&Vertex.binding_description),
        .vertex_attribute_description_count = Vertex.attribute_descriptions.len,
        .p_vertex_attribute_descriptions = &Vertex.attribute_descriptions,
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

    var command_buffers: [swapchain_cap]vk.CommandBuffer = undefined;
    _ = try vkd.allocateCommandBuffers(
        device,
        @ptrCast(&vk.CommandBufferAllocateInfo{
            .command_pool = command_pool,
            .level = .primary,
            .command_buffer_count = swapchain_images_len,
        }), 
        &command_buffers,
    );

    const graphics_queue = vkd.getDeviceQueue(device, graphics_family, 0);
    const present_queue = vkd.getDeviceQueue(device, present_family, 0);

    var image_acquired_array: [swapchain_cap]vk.Semaphore = undefined;
    var image_rendered_array: [swapchain_cap]vk.Semaphore = undefined;
    var frame_ready_array: [swapchain_cap]vk.Fence = undefined;

    var swapchain_image_views: [swapchain_cap]vk.ImageView = undefined;
    var swapchain_framebuffers: [swapchain_cap]vk.Framebuffer = undefined;

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

        image_acquired_array[i] = try vkd.createSemaphore(
            device,
            &.{
                .flags = .{},
            },
            null,
        );

        image_rendered_array[i] = try vkd.createSemaphore(
            device,
            &.{
                .flags = .{},
            },
            null,
        );

        frame_ready_array[i] = try vkd.createFence(
            device, 
            &.{
                .flags = .{
                    .signaled_bit = true,
                },
            }, 
            null,
        );
    }

    defer for (0..swapchain_images_len) |i| {
        vkd.destroyFramebuffer(device, swapchain_framebuffers[i], null);
        vkd.destroyImageView(device, swapchain_image_views[i], null);
        vkd.destroyFence(device, frame_ready_array[i], null);
        vkd.destroySemaphore(device, image_acquired_array[i], null);
        vkd.destroySemaphore(device, image_rendered_array[i], null);
    };

    const pdevice_mem_props = vki.getPhysicalDeviceMemoryProperties(pdevice);

    const vertex_buffer_info = vk.BufferCreateInfo {
        .size = @sizeOf(Vertex) * vertices.len,
        .usage = .{ 
            .vertex_buffer_bit = true,
            .transfer_dst_bit = true,
        },
        .sharing_mode = .exclusive
    };
    const vertex_buffer = try vkd.createBuffer(
        device, 
        &vertex_buffer_info,
        null,
    );

    const vertex_mem_requirements = vkd.getBufferMemoryRequirements(device, vertex_buffer);

    const vertex_memory = try vkd.allocateMemory(
        device, 
        &.{
            .allocation_size = vertex_mem_requirements.size,
            .memory_type_index = findMemoryTypeIndex(
                vertex_mem_requirements.memory_type_bits,
                .{
                    .device_local_bit = true,
                },
                pdevice_mem_props, 
            ),
        },
        null,
    );
    defer vkd.freeMemory(device, vertex_memory, null);
    _ = try vkd.bindBufferMemory(
        device, 
        vertex_buffer, 
        vertex_memory, 
        0,
    );

    const index_buffer_info = vk.BufferCreateInfo {
        .size = @sizeOf(u16) * indices.len,
        .usage = .{ 
            .index_buffer_bit = true,
            .transfer_dst_bit = true,
        },
        .sharing_mode = .exclusive
    };
    const index_buffer = try vkd.createBuffer(
        device, 
        &index_buffer_info,
        null,
    );

    const index_mem_requirements = vkd.getBufferMemoryRequirements(device, index_buffer);
    const index_memory = try vkd.allocateMemory(
        device, 
        &.{
            .allocation_size = index_mem_requirements.size,
            .memory_type_index = findMemoryTypeIndex(
                index_mem_requirements.memory_type_bits,
                .{
                    .device_local_bit = true,
                },
                pdevice_mem_props, 
            ),
        },
        null,
    );
    defer vkd.freeMemory(device, index_memory, null);
    _ = try vkd.bindBufferMemory(
        device, 
        index_buffer, 
        index_memory, 
        0,
    );

    const staging_buffer_info = vk.BufferCreateInfo {
        .size = vertex_buffer_info.size + index_buffer_info.size,
        .usage = .{ 
            .transfer_src_bit = true,
        },
        .sharing_mode = .exclusive
    };
    const staging_buffer = try vkd.createBuffer(
        device, 
        &staging_buffer_info,
        null,
    );

    const staging_memory_requirements = vkd.getBufferMemoryRequirements(device, staging_buffer);
    const staging_memmory = try vkd.allocateMemory(
        device, 
        &.{
            .allocation_size = staging_memory_requirements.size,
            .memory_type_index = findMemoryTypeIndex(
                staging_memory_requirements.memory_type_bits, 
                .{
                    .host_visible_bit = true,
                    .host_coherent_bit = true,
                }, 
                pdevice_mem_props,
            ),
        }, 
        null,
    );
    defer vkd.freeMemory(device, staging_memmory, null);
    _ = try vkd.bindBufferMemory(
        device, 
        staging_buffer, 
        staging_memmory, 
        0,
    );

    const uniform_buffer_info = vk.BufferCreateInfo {
        .size = @sizeOf(CameraUBO) * swapchain_images_len,
        .usage = .{
            .uniform_buffer_bit = true
        },
        .sharing_mode = .exclusive,
    };
    const uniform_buffer = try vkd.createBuffer(device, &uniform_buffer_info, null);
    const uniform_memory_requirements = vkd.getBufferMemoryRequirements(device, uniform_buffer);

    const uniform_memory = try vkd.allocateMemory(
        device, 
        &.{
            .allocation_size = uniform_memory_requirements.size,
            .memory_type_index = findMemoryTypeIndex(
                uniform_memory_requirements.memory_type_bits, 
                .{
                    .host_visible_bit = true,
                    .host_coherent_bit = true,
                },
                pdevice_mem_props,
            ),
        },
        null,
    );
    defer vkd.freeMemory(device, uniform_memory, null);
    var camera_ubos: []CameraUBO = @as([*]CameraUBO, @ptrCast(@alignCast(
        (try vkd.mapMemory(
            device, 
            uniform_memory, 
            0, 
            uniform_buffer_info.size, 
            .{},
        )).?
    )))[0..swapchain_images_len];
    camera_ubos[0] = CameraUBO{
        .view = math.Affine3.identity,
        .near_z = 1,
    };
    defer vkd.unmapMemory(device, uniform_memory);

    const staging_ptr: []u8 = @as([*]u8, @ptrCast(
        (try vkd.mapMemory(
            device, 
            staging_memmory, 
            0, 
            staging_buffer_info.size, 
            .{},
        )).?
    ))[0..staging_buffer_info.size];
    @memcpy(staging_ptr[0..vertex_buffer_info.size], @as([*]const u8, @ptrCast(&vertices)));
    @memcpy(staging_ptr[vertex_buffer_info.size..staging_buffer_info.size], @as([*]const u8, @ptrCast(&indices)));
    vkd.unmapMemory(device, staging_memmory);

    defer vkd.destroyBuffer(device, vertex_buffer, null);
    defer vkd.destroyBuffer(device, staging_buffer, null);
    defer vkd.destroyBuffer(device, index_buffer, null);
    defer vkd.destroyBuffer(device, uniform_buffer, null);
    // copy memory
    {
        const command_buffer = command_buffers[0];
        _ = try vkd.beginCommandBuffer(
            command_buffer,
            &.{}
        );

        vkd.cmdCopyBuffer(
            command_buffer, 
            staging_buffer, 
            vertex_buffer, 
            1, 
            @ptrCast(&vk.BufferCopy{
                .src_offset = 0,
                .dst_offset = 0,
                .size = vertex_buffer_info.size,
            }),
        );

        vkd.cmdCopyBuffer(
            command_buffer, 
            staging_buffer, 
            index_buffer, 
            1, 
            @ptrCast(&vk.BufferCopy{
                .src_offset = vertex_buffer_info.size,
                .dst_offset = 0,
                .size = index_buffer_info.size,
            }),
        );

        _ = try vkd.endCommandBuffer(command_buffer);

        _ = try vkd.queueSubmit(
            graphics_queue, 
            1, 
            @ptrCast(&vk.SubmitInfo{
                .command_buffer_count = 1,
                .p_command_buffers = @ptrCast(&command_buffer),
            }),
            .null_handle,
        );

        _ = try vkd.queueWaitIdle(graphics_queue);
    }

    var swapchain_image_index: u32 = 0;
    while (true) {
        if (window.shouldClose() or window.getKey(.escape) == .press) {
            break;
        }

        const image_acquired = image_acquired_array[swapchain_image_index];
        const image_rendered = image_rendered_array[swapchain_image_index];
        const frame_ready = frame_ready_array[swapchain_image_index];
        const command_buffer = command_buffers[swapchain_image_index];
        const swapchain_framebuffer = swapchain_framebuffers[swapchain_image_index];

        _ = try vkd.waitForFences(
            device, 
            1, 
            @ptrCast(&frame_ready), 
            vk.TRUE,
            std.math.maxInt(u64)
        );

        const result_image_index = (try vkd.acquireNextImageKHR(
            device,
            swapchain,
            std.math.maxInt(u64),
            image_acquired,
            .null_handle
        )).image_index;
        if (result_image_index != swapchain_image_index) {
            return error.AcquireNextImageMismatch;
        }
        
        _ = try vkd.resetFences(
            device, 
            1,
            @ptrCast(&frame_ready)
        );

        _ = try vkd.resetCommandBuffer(command_buffer, .{});

        _ = try vkd.beginCommandBuffer(
            command_buffer, 
            &.{},
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
            .framebuffer = swapchain_framebuffer,
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

        vkd.cmdBindVertexBuffers(
            command_buffer, 
            0, 
            1, 
            @ptrCast(&vertex_buffer)
            , 
            &.{0},
        );

        vkd.cmdBindIndexBuffer(
            command_buffer, 
            index_buffer, 
            0, 
            .uint16,
        );

        vkd.cmdDrawIndexed(
            command_buffer, 
            indices.len, 
            1, 
            0, 
            0, 
            0,
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
                .p_wait_semaphores = @ptrCast(&image_acquired),
                .p_wait_dst_stage_mask = @ptrCast(&vk.PipelineStageFlags {
                    .color_attachment_output_bit = true,
                }),
                .signal_semaphore_count = 1,
                .p_signal_semaphores = @ptrCast(&image_rendered),
            }),
            frame_ready,
        );

        const result = try vkd.queuePresentKHR(
            present_queue,
            &.{
                .swapchain_count = 1,
                .p_swapchains = @ptrCast(&swapchain),
                .p_image_indices = @ptrCast(&swapchain_image_index),
                .wait_semaphore_count = 1,
                .p_wait_semaphores = @ptrCast(&image_rendered),
            },
        );
        switch (result) {
            .error_out_of_date_khr, .suboptimal_khr => {
                _ = try vkd.deviceWaitIdle(device);

                while (true) {
                    surface_capabilities = try vki.getPhysicalDeviceSurfaceCapabilitiesKHR(pdevice, surface);
                    setExtent(&extent, &window, surface_capabilities);

                    if (extent.width != 0 and extent.height != 0) {
                        break;
                    }
                    glfw.waitEvents();
                }

                const old_swapchain = swapchain;
                swapchain = try vkd.createSwapchainKHR(
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
                        .old_swapchain = old_swapchain,
                    }, 
                    null,
                );
                vkd.destroySwapchainKHR(device, old_swapchain, null);
                _ = try vkd.getSwapchainImagesKHR(device, swapchain, &swapchain_images_len, null);
                _ = try vkd.getSwapchainImagesKHR(device, swapchain, &swapchain_images_len, &swapchain_images);

                for (0..swapchain_images_len) |i| {
                    vkd.destroyFramebuffer(device, swapchain_framebuffers[i], null);
                    vkd.destroyImageView(device, swapchain_image_views[i], null);

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
                swapchain_image_index = 0;
                continue;
            },
            .success => {},
            else => return error.AcquireNextImageFailed,
        }

        if (swapchain_image_index == swapchain_images_len - 1) {
            swapchain_image_index = 0;
        } else {
            swapchain_image_index += 1;
        }

        glfw.pollEvents();
    }

    _ = try vkd.deviceWaitIdle(device);
}

fn setExtent(extent: *vk.Extent2D, window: *glfw.Window, capabilities: vk.SurfaceCapabilitiesKHR) void {
    if (capabilities.current_extent.width != 0xFFFF_FFFF) {
        extent.* = capabilities.current_extent;
        return;
    }

    const size = window.getFramebufferSize();
    extent.* = .{
        .width = size.width,
        .height = size.height, 
    };

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

fn findMemoryTypeIndex(
    type_mask: u32, 
    required_props: vk.MemoryPropertyFlags,
    props: vk.PhysicalDeviceMemoryProperties,
) u32 {
    for (props.memory_types[0..props.memory_type_count], 0..) |mem_type, i| {
        if (type_mask & (@as(u32, 1) << @truncate(i)) != 0 and mem_type.property_flags.contains(required_props)) {
            return @truncate(i);
        }
    }
    unreachable;
}

fn chooseSurfaceFormat(surface_formats: *const [surface_formats_cap]vk.SurfaceFormatKHR, len: u32) vk.SurfaceFormatKHR {
    const wanted: vk.SurfaceFormatKHR = .{
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