const std = @import("std");
const glfw = @import("glfw");
const vk = @import("vk");
const shaders = @import("shaders");
const math = @import("math.zig");
const img = @import("image");

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
const present_frames_cap = 4;
const render_frames_cap = present_frames_cap - 1;

const target_fps = 60;
const target_dt: f32 = 1.0 / @as(f32, @floatFromInt(target_fps));

const Vertex = extern struct {
    x: f32, y: f32, z: f32,
    r: f32, g: f32, b: f32,
    u: f32, v: f32,

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
        vk.VertexInputAttributeDescription {
            .binding = 0,
            .location = 2,
            .format = .r32g32_sfloat,
            .offset = @offsetOf(Vertex, "u"),
        },
    };
};

const Camera = extern struct {
    position: math.Vector3,
    near_z: f32,
    cos_z_to_y: f32,
    sin_z_to_y: f32,
    cos_z_to_x: f32,
    sin_z_to_x: f32,
    x_scale: f32,
    y_scale: f32,
};


const vertices = [_]Vertex {
    Vertex {
        .x = 0.5, .y = 0.5, .z = 0.5,
        .r = 0.5, .g = 0.5, .b = 0.0,
        .u = 1.0, .v = 0.0,
    },
    Vertex {
        .x = 0.5, .y = -0.5, .z = 0.5,
        .r = 0.0, .g = 1.0, .b = 0.0,
        .u = 1.0, .v = 1.0,
    },
    Vertex {
        .x = -0.5, .y = -0.5, .z = 0.5,
        .r = 1.0, .g = 0.0, .b = 1.0,
        .u = 0.0, .v = 1.0,
    },
    Vertex {
        .x = -0.5, .y = 0.5, .z = 0.5,
        .r = 0.0, .g = 1.0, .b = 1.0,
        .u = 0.0, .v = 0.0,
    },

    Vertex {
        .x = 0.5, .y = 0.5, .z = 1.0,
        .r = 0.5, .g = 0.5, .b = 0.0,
        .u = 1.0, .v = 0.0,
    },
    Vertex {
        .x = 0.5, .y = -0.5, .z = 1.0,
        .r = 0.0, .g = 1.0, .b = 0.0,
        .u = 1.0, .v = 1.0,
    },
    Vertex {
        .x = -0.5, .y = -0.5, .z = 1.0,
        .r = 1.0, .g = 0.0, .b = 1.0,
        .u = 0.0, .v = 1.0,
    },
    Vertex {
        .x = -0.5, .y = 0.5, .z = 1.0,
        .r = 0.0, .g = 1.0, .b = 1.0,
        .u = 0.0, .v = 0.0,
    },
};

const indices = [_]u16 {
    0, 1, 2,
    2, 3, 0,

    4, 5, 6,
    6, 7, 4,
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
    if (data.p_message_id_name != null and data.p_message != null) {
    print(
        "\x1b[9{c};1;4m{s}\x1b[m\x1b[9{c}m\n{s}\n\x1b[m\n",
        .{
            color_code,
            data.p_message_id_name.?,
            color_code,
            data.p_message.?,
        },
    );
    }

    
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

    var validation_layers_count: u32 = undefined;
    _ = try vkb.enumerateInstanceLayerProperties(&validation_layers_count, null);

    var validation_layers: [validation_layers_cap]vk.LayerProperties = undefined;
    _ = try vkb.enumerateInstanceLayerProperties(&validation_layers_count, &validation_layers);

    outer: for (required_validation_layers) |required_layer| {
        for (0..validation_layers_count) |validation_layer_i| {
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

    const layer_settings = [_]vk.LayerSettingEXT {
        vk.LayerSettingEXT {
            .p_layer_name = "VK_LAYER_KHRONOS_validation",
            .p_setting_name = "validate_core",
            .type = .bool32_ext,
            .value_count = 1,
            .p_values = &vk.TRUE,
        },
        vk.LayerSettingEXT {
            .p_layer_name = "VK_LAYER_KHRONOS_validation",
            .p_setting_name = "validate_sync",
            .type = .bool32_ext,
            .value_count = 1,
            .p_values = &vk.TRUE,
        },
        vk.LayerSettingEXT {
            .p_layer_name = "VK_LAYER_KHRONOS_validation",
            .p_setting_name = "validate_best_practices",
            .type = .bool32_ext,
            .value_count = 1,
            .p_values = &vk.TRUE,
        },
        vk.LayerSettingEXT {
            .p_layer_name = "VK_LAYER_KHRONOS_validation",
            .p_setting_name = "thread_safety",
            .type = .bool32_ext,
            .value_count = 1,
            .p_values = &vk.TRUE,
        },
        vk.LayerSettingEXT {
            .p_layer_name = "VK_LAYER_KHRONOS_validation",
            .p_setting_name = "debug_action",
            .type = .string_ext,
            .value_count = 1,
            .p_values = @ptrCast("VK_DBG_LAYER_ACTION_LOG_MSG"),
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
            .p_values = &vk.TRUE,
        },
        vk.LayerSettingEXT {
            .p_layer_name = "VK_LAYER_KHRONOS_validation",
            .p_setting_name = "duplicate_message_limit",
            .type = .uint32_ext,
            .value_count = 1,
            .p_values = &@as(u32, 1),
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

    var pdevices_count: u32 = undefined;
    _ = try vki.enumeratePhysicalDevices(instance, &pdevices_count, null);

    var pdevices: [pdevices_cap]vk.PhysicalDevice = undefined;
    _ = try vki.enumeratePhysicalDevices(instance, &pdevices_count, &pdevices);
    
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

        var propss_count: u32 = undefined;
        vki.getPhysicalDeviceQueueFamilyProperties(pdevice, &propss_count, null);

        var propss: [propss_cap]vk.QueueFamilyProperties = undefined;
        vki.getPhysicalDeviceQueueFamilyProperties(pdevice, &propss_count, &propss);

        _graphics_family = null;
        _present_family = null;
        var queue_family: u32 = 0;

        pdevice: while (queue_family != propss_count) {
            const props = propss[queue_family];
            if (props.queue_flags.graphics_bit) {
                _graphics_family = queue_family;
            }

            const present = try vki.getPhysicalDeviceSurfaceSupportKHR(pdevice, queue_family, surface) == vk.TRUE;
            if (present and (_graphics_family == null or _graphics_family.? != queue_family)) {
                _present_family = queue_family;
            }

            if (_graphics_family != null and _present_family != null) {
                var surface_formats_count: u32 = undefined;
                _ = try vki.getPhysicalDeviceSurfaceFormatsKHR(pdevice, surface, &surface_formats_count, null);
                if (surface_formats_count == 0) {
                    break :pdevice;
                }

                var surface_formats: [surface_formats_cap]vk.SurfaceFormatKHR = undefined;
                _ = try vki.getPhysicalDeviceSurfaceFormatsKHR(pdevice, surface, &surface_formats_count, &surface_formats);

                var present_modes_count: u32 = undefined;
                _ = try vki.getPhysicalDeviceSurfacePresentModesKHR(pdevice, surface, &present_modes_count, null);
                if (present_modes_count == 0) {
                    break :pdevice;
                }

                var present_modes: [present_modes_cap]vk.PresentModeKHR = undefined;
                _ = try vki.getPhysicalDeviceSurfacePresentModesKHR(pdevice, surface, &present_modes_count, &present_modes);
                
                var extensions_count: u32 = undefined;
                _ = try vki.enumerateDeviceExtensionProperties(pdevice, null, &extensions_count, null);

                var extensions: [extensions_cap]vk.ExtensionProperties = undefined;
                _ = try vki.enumerateDeviceExtensionProperties(pdevice, null, &extensions_count, &extensions);

                required_ext: for (required_device_extensions) |required_ext| {
                    for (0..extensions_count) |ext_i| {
                        const ext = extensions[ext_i];

                        if (std.mem.eql(u8, std.mem.span(required_ext), std.mem.sliceTo(&ext.extension_name, 0))) {
                            continue :required_ext;
                        }
                    }
                    break :pdevice;
                }

                surface_format = chooseSurfaceFormat(&surface_formats, surface_formats_count);
                present_mode = choosePresentMode(&present_modes, present_modes_count);

                break :outer;
            }
            queue_family += 1;
        }

        pdevice_i += 1;
        if (pdevice_i == pdevices_count) {
            return error.SuitablePhysicalDeviceNotFound;
        }
    }

    const pdevice_mem_props = vki.getPhysicalDeviceMemoryProperties(pdevice);

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
            .p_enabled_features = &vk.PhysicalDeviceFeatures{
                .sampler_anisotropy = vk.TRUE,
            },
        },
        null
    );
    const vkd = try DeviceDispatch.load(device, vki.dispatch.vkGetDeviceProcAddr);
    defer vkd.destroyDevice(device, null);

    var surface_capabilities = try vki.getPhysicalDeviceSurfaceCapabilitiesKHR(pdevice, surface);
    var swapchain_images_count: u32 = surface_capabilities.min_image_count + 1;
    if (surface_capabilities.max_image_count != 0 and swapchain_images_count > surface_capabilities.max_image_count) {
        swapchain_images_count = surface_capabilities.max_image_count;
    }
    
    setExtent(&extent, &window, surface_capabilities);

    const depth_image_info = vk.ImageCreateInfo {
        .format = .d32_sfloat,
        .image_type = .@"2d",
        .tiling = .optimal,
        .initial_layout = .undefined,
        .usage = .{ 
            .depth_stencil_attachment_bit = true,
        },
        .sharing_mode = .exclusive,
        .samples = .{ .@"1_bit" = true },
        .extent = .{
            .width = extent.width,
            .height = extent.height,
            .depth = 1,
        },
        .mip_levels = 1,
        .array_layers = 1,
    };

    const depth_image = try vkd.createImage(
        device, 
        &depth_image_info, 
        null,
    );
    defer vkd.destroyImage(device, depth_image, null);

    const depth_image_memory_requirements = vkd.getImageMemoryRequirements(device, depth_image);
    const depth_image_memory = try vkd.allocateMemory(
        device, 
        &.{
            .allocation_size = depth_image_memory_requirements.size,
            .memory_type_index = findMemoryTypeIndex(
                depth_image_memory_requirements.memory_type_bits, 
                vk.MemoryPropertyFlags {
                    .device_local_bit = true 
                },
                pdevice_mem_props,
            ),
        }, 
        null,
    );
    defer vkd.freeMemory(device, depth_image_memory, null);
    _ = try vkd.bindImageMemory(device, depth_image, depth_image_memory, 0);

    const depth_image_view = try vkd.createImageView(
        device, 
        &.{
            .components = .{
                .r = .identity,
                .g = .identity,
                .b = .identity,
                .a = .identity,
            },
            .format = depth_image_info.format,
            .view_type = .@"2d",
            .image = depth_image,
            .subresource_range = .{
                .aspect_mask = .{ .depth_bit = true },
                .base_array_layer = 0,
                .layer_count = 1,
                .base_mip_level = 0,
                .level_count = 1,
            }
        }, 
        null,
    );
    defer vkd.destroyImageView(device, depth_image_view, null);

    var swapchain = try vkd.createSwapchainKHR(
        device,
        &.{
            .surface = surface,
            .min_image_count = swapchain_images_count,
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

    _ = try vkd.getSwapchainImagesKHR(device, swapchain, &swapchain_images_count, null);
    var swapchain_images: [present_frames_cap]vk.Image = undefined;
    _ = try vkd.getSwapchainImagesKHR(device, swapchain, &swapchain_images_count, &swapchain_images);
    
    const last_render_index = swapchain_images_count - 2;
    const render_frames_count = swapchain_images_count - 1;

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

    const depth_attachment = vk.AttachmentDescription {
        .format = depth_image_info.format,
        .samples = .{ .@"1_bit" = true },
        .load_op = .clear,
        .store_op = .store,
        .stencil_load_op = .dont_care,
        .stencil_store_op = .dont_care,
        .initial_layout = .undefined,
        .final_layout = .depth_stencil_attachment_optimal,
    };

    const color_attachment_ref = vk.AttachmentReference {
        .attachment = 0,
        .layout = .color_attachment_optimal,
    };

    const depth_attachment_ref = vk.AttachmentReference {
        .attachment = 1,
        .layout = .depth_stencil_attachment_optimal,
    };

    const subpass = vk.SubpassDescription {
        .pipeline_bind_point = .graphics,
        .color_attachment_count = 1,
        .p_color_attachments = @ptrCast(&color_attachment_ref),
        .p_depth_stencil_attachment = &depth_attachment_ref,
    };

    const subpass_dependency = vk.SubpassDependency {
        .src_subpass = vk.SUBPASS_EXTERNAL,
        .dst_subpass = 0,
        .src_stage_mask = .{ 
            .color_attachment_output_bit = true,
            .early_fragment_tests_bit = true,
        },
        .src_access_mask = .{
        },
        .dst_stage_mask = .{
            .color_attachment_output_bit = true,
            .early_fragment_tests_bit = true,
        },
        .dst_access_mask = .{
            .color_attachment_write_bit = true,
            .depth_stencil_attachment_write_bit = true,
        },
    };

    const render_pass_attachments = [_]vk.AttachmentDescription {
        color_attachment,
        depth_attachment,
    };
    const render_pass = try vkd.createRenderPass(
        device,
        &.{
            .attachment_count = render_pass_attachments.len,
            .p_attachments = &render_pass_attachments,
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

    const shader_stage_infos = [_]vk.PipelineShaderStageCreateInfo{
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

    const camera_descriptor_binding = vk.DescriptorSetLayoutBinding{
        .binding = 0,     
        .descriptor_type = .uniform_buffer_dynamic,
        .descriptor_count = 1,
        .stage_flags = .{ .vertex_bit = true },
    };
    const image_sampler_descriptor_binding = vk.DescriptorSetLayoutBinding{
        .binding = 1,     
        .descriptor_type = .combined_image_sampler,
        .descriptor_count = 1,
        .stage_flags = .{ .fragment_bit = true },
    };
    const descriptor_bindings = [_]vk.DescriptorSetLayoutBinding {
        camera_descriptor_binding,
        image_sampler_descriptor_binding,
    };
    const descriptor_set_layout = try vkd.createDescriptorSetLayout(
        device, 
        &.{
            .binding_count = descriptor_bindings.len,
            .p_bindings = &descriptor_bindings,
        }, 
        null,
    );
    defer vkd.destroyDescriptorSetLayout(device, descriptor_set_layout, null);

    const descriptor_pool_sizes = [_]vk.DescriptorPoolSize {
        vk.DescriptorPoolSize {
            .type = .uniform_buffer_dynamic,
            .descriptor_count = 1,
        },
        vk.DescriptorPoolSize {
            .type = .combined_image_sampler,
            .descriptor_count = 1,
        },
    };
    const descriptor_pool = try vkd.createDescriptorPool(
        device, 
        &.{
            .pool_size_count = descriptor_pool_sizes.len,
            .p_pool_sizes = &descriptor_pool_sizes,
            .max_sets = 1,
        }, 
        null,
    );
    defer vkd.destroyDescriptorPool(device, descriptor_pool, null);

    var descriptor_set: vk.DescriptorSet = undefined; 
    _ = try vkd.allocateDescriptorSets(
        device, 
        &.{
            .descriptor_pool = descriptor_pool,
            .descriptor_set_count = 1,
            .p_set_layouts = @ptrCast(&descriptor_set_layout),
        },
        @ptrCast(&descriptor_set),
    );

    const vertex_input_state_info = vk.PipelineVertexInputStateCreateInfo{
        .vertex_binding_description_count = 1,
        .p_vertex_binding_descriptions = @ptrCast(&Vertex.binding_description),
        .vertex_attribute_description_count = Vertex.attribute_descriptions.len,
        .p_vertex_attribute_descriptions = &Vertex.attribute_descriptions,
    };

    const input_assembly_state_info = vk.PipelineInputAssemblyStateCreateInfo{
        .topology = .triangle_list,
        .primitive_restart_enable = vk.FALSE,
    };

    const viewport_state_info = vk.PipelineViewportStateCreateInfo{
        .viewport_count = 1,
        .p_viewports = null, // set in createCommandBuffers with cmdSetViewport
        .scissor_count = 1,
        .p_scissors = null, // set in createCommandBuffers with cmdSetScissor
    };

    const rasterization_state_info = vk.PipelineRasterizationStateCreateInfo{
        .depth_clamp_enable = vk.FALSE,
        .rasterizer_discard_enable = vk.FALSE,
        .polygon_mode = .fill,
        .cull_mode = .{ .back_bit = true },
        .front_face = .counter_clockwise,
        .depth_bias_enable = vk.FALSE,
        .depth_bias_constant_factor = 0.0,
        .depth_bias_clamp = 0.0,
        .depth_bias_slope_factor = 0.0,
        .line_width = 1.0,
    };

    const multi_sample_state_info = vk.PipelineMultisampleStateCreateInfo{
        .rasterization_samples = .{ .@"1_bit" = true },
        .sample_shading_enable = vk.FALSE,
        .min_sample_shading = 1,
        .alpha_to_coverage_enable = vk.FALSE,
        .alpha_to_one_enable = vk.FALSE,
    };

    const blend_attachment_states = [_]vk.PipelineColorBlendAttachmentState{
        // present framebuffer attachment
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

    const blend_state_info = vk.PipelineColorBlendStateCreateInfo{
        .logic_op_enable = vk.FALSE,
        .logic_op = .copy,
        .attachment_count = blend_attachment_states.len,
        .p_attachments = &blend_attachment_states,
        .blend_constants = [_]f32{ 0.0, 0.0, 0.0, 0.0, },
    };

    const depth_stencil_info = vk.PipelineDepthStencilStateCreateInfo {
        .depth_test_enable = vk.TRUE,
        .depth_write_enable = vk.TRUE,
        .depth_compare_op = .greater,
        .depth_bounds_test_enable = vk.FALSE,
        .stencil_test_enable = vk.FALSE,
        .front = undefined,
        .back = undefined,
        .min_depth_bounds = 0.0,
        .max_depth_bounds = 1.0,
    };

    const dynamic_states = [_]vk.DynamicState{ 
        .viewport, 
        .scissor, 
    };
    const dynamic_state_info = vk.PipelineDynamicStateCreateInfo{
        .flags = .{},
        .dynamic_state_count = dynamic_states.len,
        .p_dynamic_states = &dynamic_states,
    };

    const pipeline_layout = try vkd.createPipelineLayout(
        device,
        &.{
            .set_layout_count = 1,
            .p_set_layouts = @ptrCast(&descriptor_set_layout),
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
            .stage_count = shader_stage_infos.len,
            .p_stages = &shader_stage_infos,
            .p_vertex_input_state = &vertex_input_state_info,
            .p_input_assembly_state = &input_assembly_state_info,
            .p_tessellation_state = null,
            .p_viewport_state = &viewport_state_info,
            .p_rasterization_state = &rasterization_state_info,
            .p_multisample_state = &multi_sample_state_info,
            .p_depth_stencil_state = &depth_stencil_info,
            .p_color_blend_state = &blend_state_info,
            .p_dynamic_state = &dynamic_state_info,
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

    var command_buffers: [render_frames_cap]vk.CommandBuffer = undefined;
    _ = try vkd.allocateCommandBuffers(
        device,
        &.{
            .command_pool = command_pool,
            .level = .primary,
            .command_buffer_count = swapchain_images_count,
        }, 
        &command_buffers,
    );

    const graphics_queue = vkd.getDeviceQueue(device, graphics_family, 0);
    const present_queue = vkd.getDeviceQueue(device, present_family, 0);

    var image_acquired_array: [render_frames_cap]vk.Semaphore = undefined;
    var frame_begin_array: [render_frames_cap]vk.Fence = undefined;

    var image_rendered_array: [present_frames_cap]vk.Semaphore = undefined;
    var swapchain_image_views: [present_frames_cap]vk.ImageView = undefined;
    var swapchain_framebuffers: [present_frames_cap]vk.Framebuffer = undefined;

    {
        var i: u32 = 0;
        while (i != render_frames_count) {
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
                depth_image_view,
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
    
            frame_begin_array[i] = try vkd.createFence(
                device, 
                &.{
                    .flags = .{
                        .signaled_bit = true,
                    },
                }, 
                null,
            );
    
            i += 1;
        }

    
        image_rendered_array[i] = try vkd.createSemaphore(
            device,
            &.{
                .flags = .{},
            },
            null,
        );

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
            depth_image_view,
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

    defer {
        var i: u32 = 0;
        while (i != render_frames_count) {
            vkd.destroyFramebuffer(device, swapchain_framebuffers[i], null);
            vkd.destroyImageView(device, swapchain_image_views[i], null);
            vkd.destroyFence(device, frame_begin_array[i], null);
            vkd.destroySemaphore(device, image_rendered_array[i], null);
            vkd.destroySemaphore(device, image_acquired_array[i], null);
            i += 1;
        }
        vkd.destroySemaphore(device, image_rendered_array[i], null);
        vkd.destroyFramebuffer(device, swapchain_framebuffers[i], null);
        vkd.destroyImageView(device, swapchain_image_views[i], null);
    }

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

    const cwd = std.fs.cwd();
    const image_file = try cwd.openFile("assets.hex", .{
        .mode = .read_only
    });
    const allocator = std.heap.page_allocator;
    const _image = try img.Image.load(image_file, allocator);
    defer allocator.free(_image.ptr[0 .. _image.size]);

    image_file.close();
    
    const image_info = vk.ImageCreateInfo{
        .format = _image.format,
        .image_type = .@"2d",
        .tiling = .optimal,
        .initial_layout = .undefined,
        .usage = .{ 
            .transfer_dst_bit = true,
            .sampled_bit = true,
        },
        .sharing_mode = .exclusive,
        .samples = .{ .@"1_bit" = true },
        .extent = .{ 
            .width = _image.width,
            .height = _image.height,
            .depth = 1,
        },
        .array_layers = 1,
        .mip_levels = 1,
    };
    const image: vk.Image = try vkd.createImage (
        device, 
        &image_info, 
        null,
    );
    defer vkd.destroyImage(device, image, null);

    const image_memory_requirements = vkd.getImageMemoryRequirements(device, image);
    const image_memory = try vkd.allocateMemory(
        device, 
        &.{
            .allocation_size = image_memory_requirements.size,
            .memory_type_index = findMemoryTypeIndex(
                image_memory_requirements.memory_type_bits, 
                vk.MemoryPropertyFlags {
                    .device_local_bit = true 
                },
                pdevice_mem_props,
            ),
        }, 
        null,
    );
    defer vkd.freeMemory(device, image_memory, null);
    _ = try vkd.bindImageMemory(device, image, image_memory, 0);


    const image_view = try vkd.createImageView(
        device, 
        &.{
            .components = .{
                .r = .identity,
                .g = .identity,
                .b = .identity,
                .a = .identity,
            },
            .image = image,
            .view_type = .@"2d",
            .format = image_info.format,
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
        },
        null,
    );
    defer vkd.destroyImageView(device, image_view, null);

    const sampler = try vkd.createSampler(
        device, 
        &.{
            .mag_filter = .linear,
            .min_filter = .linear,
            .address_mode_u = .repeat,
            .address_mode_v = .repeat,
            .address_mode_w = .repeat,
            .anisotropy_enable = vk.TRUE,
            .max_anisotropy = pdevice_props.limits.max_sampler_anisotropy,
            .border_color = .int_opaque_black,
            .unnormalized_coordinates = vk.FALSE,
            .compare_enable = vk.FALSE,
            .compare_op = .always,
            .mipmap_mode = .linear,
            .mip_lod_bias = 0.0,
            .min_lod = 0.0,
            .max_lod = 0.0,
        },
        null,
    );
    defer vkd.destroySampler(device, sampler, null);

    const staging_buffer_info = vk.BufferCreateInfo {
        .size = vertex_buffer_info.size + _image.size + index_buffer_info.size,
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

    const uniform_buffer_stride = @max(@sizeOf(Camera), pdevice_props.limits.min_uniform_buffer_offset_alignment);
    const uniform_buffer_info = vk.BufferCreateInfo {
        .size = uniform_buffer_stride * render_frames_count,
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
    _ = try vkd.bindBufferMemory(
        device, 
        uniform_buffer, 
        uniform_memory, 
        0,
    );

    const uniform_ptr: [*]u8 = @ptrCast(
        (try vkd.mapMemory(
            device, 
            uniform_memory, 
            0, 
            uniform_buffer_info.size, 
            .{},
        )).?
    );
    defer vkd.unmapMemory(device, uniform_memory);

    const descriptor_writes = [_]vk.WriteDescriptorSet {
        vk.WriteDescriptorSet {
            .descriptor_type = camera_descriptor_binding.descriptor_type,
            .descriptor_count = camera_descriptor_binding.descriptor_count,
            .dst_set = descriptor_set,
            .dst_binding = camera_descriptor_binding.binding,
            .dst_array_element = 0,
            .p_buffer_info = @ptrCast(&vk.DescriptorBufferInfo{
                .buffer = uniform_buffer,
                .offset = 0,
                .range = uniform_buffer_stride,
            }),
            .p_image_info = undefined,
            .p_texel_buffer_view = undefined,
        },
        vk.WriteDescriptorSet {
            .descriptor_type = image_sampler_descriptor_binding.descriptor_type,
            .descriptor_count = image_sampler_descriptor_binding.descriptor_count,
            .dst_set = descriptor_set,
            .dst_binding = image_sampler_descriptor_binding.binding,
            .dst_array_element = 0,
            .p_buffer_info = undefined,
            .p_image_info = @ptrCast(&vk.DescriptorImageInfo {
                .image_layout = .shader_read_only_optimal,
                .image_view = image_view,
                .sampler = sampler,
            }),
            .p_texel_buffer_view = undefined,
        },
    };

    vkd.updateDescriptorSets(
        device, 
        descriptor_writes.len, 
        &descriptor_writes, 
        0, 
        null,
    );

    const staging_ptr: [*]u8 = @ptrCast(
        (try vkd.mapMemory(
            device, 
            staging_memmory, 
            0, 
            staging_buffer_info.size, 
            .{},
        )).?
    );
    const staging_image_offset = vertex_buffer_info.size;
    const staging_index_offset = staging_image_offset + _image.size;
    @memcpy(staging_ptr[0 .. staging_image_offset], @as([*]const u8, @ptrCast(&vertices)));
    @memcpy(staging_ptr[staging_image_offset .. staging_index_offset], _image.ptr[0 .. _image.size]);
    @memcpy(staging_ptr[staging_index_offset .. staging_buffer_info.size], @as([*]const u8, @ptrCast(&indices)));
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
                .src_offset = staging_index_offset,
                .dst_offset = 0,
                .size = index_buffer_info.size,
            }),
        );


        const layouts = [_]vk.ImageLayout {
            .undefined,
            .transfer_dst_optimal,
            .shader_read_only_optimal,
        };
        const accesses = [_]vk.AccessFlags {
            .{},
            .{ .transfer_write_bit = true },
            .{ .shader_read_bit = true },
        };
        const stages = [_]vk.PipelineStageFlags{
            .{ .top_of_pipe_bit = true },
            .{ .transfer_bit = true },
            .{ .fragment_shader_bit = true },
        };

        var i: u8 = 0;
        transitionImageLayout(
            &vkd, 
            image, 
            command_buffer, 
            layouts[i], 
            layouts[i + 1], 
            stages[i], 
            stages[i + 1], 
            accesses[i], 
            accesses[i + 1]
        );
        i += 1;

        vkd.cmdCopyBufferToImage(
            command_buffer, 
            staging_buffer, 
            image, 
            .transfer_dst_optimal, 
            1, 
            @ptrCast(&vk.BufferImageCopy{
                .buffer_offset = staging_image_offset,
                .buffer_row_length = 0,
                .buffer_image_height = 0,
                .image_subresource = .{
                    .aspect_mask = .{ .color_bit = true },
                    .mip_level = 0,
                    .base_array_layer = 0,
                    .layer_count = 1,
                },
                .image_offset = .{
                    .x = 0, .y = 0, .z = 0,
                },
                .image_extent = image_info.extent,
            })
        );

        transitionImageLayout(
            &vkd, 
            image, 
            command_buffer, 
            layouts[i], 
            layouts[i + 1], 
            stages[i], 
            stages[i + 1], 
            accesses[i], 
            accesses[i + 1]
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

    var width_to_height = @as(f32, @floatFromInt(extent.width)) / @as(f32, @floatFromInt(extent.height));
    
    var last_time: f32 = @floatCast(glfw.getTime());

    var camera_z_to_y: f32 = 0.0;
    var camera_z_to_x: f32 = 0.0;
    var camera: Camera = .{
        .position = .{
            .x = 0.0,
            .y = 0.0,
            .z = 0.0,
        },
        .near_z = 0.1,
        .cos_z_to_y = @cos(camera_z_to_y),
        .sin_z_to_y = @sin(camera_z_to_y),
        .cos_z_to_x = @cos(camera_z_to_x),
        .sin_z_to_x = @sin(camera_z_to_x),
        .x_scale = 2.0 / 3.0,
        .y_scale = 2.0 / 3.0 * width_to_height,
    };

    var render_index: u32 = 0;

    while (true) {
        if (window.shouldClose() or window.getKey(.escape) == .press) {
            break;
        }

        const image_acquired = image_acquired_array[render_index];
        const frame_begin = frame_begin_array[render_index];

        const command_buffer = command_buffers[render_index];
        const uniform_camera_ptr: *Camera = @ptrCast(@alignCast(uniform_ptr + render_index * uniform_buffer_stride)); 

        _ = try vkd.waitForFences(
            device, 
            1, 
            @ptrCast(&frame_begin), 
            vk.TRUE,
            std.math.maxInt(u64)
        );

        const present_index = (try vkd.acquireNextImageKHR(
            device,
            swapchain,
            std.math.maxInt(u64),
            image_acquired,
            .null_handle
        )).image_index;

        const swapchain_framebuffer = swapchain_framebuffers[present_index];
        const image_rendered = image_rendered_array[present_index];

        
        _ = try vkd.resetFences(
            device, 
            1,
            @ptrCast(&frame_begin)
        );

        _ = try vkd.resetCommandBuffer(command_buffer, .{});

        uniform_camera_ptr.* = camera;

        _ = try vkd.beginCommandBuffer(
            command_buffer, 
            &.{},
        );

        const clear_values = [_]vk.ClearValue {
            .{
                .color = .{
                    .float_32 = [4]f32 {0.0, 0.0, 0.0, 0.0,},
                },
            },
            .{
                .depth_stencil = .{ .depth = 1.0, .stencil = 0 }
            },
        };
        
        vkd.cmdBeginRenderPass(
            command_buffer, 
            &vk.RenderPassBeginInfo {
                .render_pass = render_pass,
                .framebuffer = swapchain_framebuffer,
                .render_area = vk.Rect2D {
                    .extent = extent,
                    .offset = .{
                        .x = 0,
                        .y = 0,
                    },
                },
                .clear_value_count = clear_values.len,
                .p_clear_values = &clear_values,
            }, 
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
                .x = 0.0,
                .y = 0.0,
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
            @ptrCast(&vertex_buffer), 
            &.{0},
        );

        vkd.cmdBindIndexBuffer(
            command_buffer, 
            index_buffer, 
            0, 
            .uint16,
        );

        vkd.cmdBindDescriptorSets(
            command_buffer, 
            .graphics, 
            pipeline_layout, 
            0, 
            1, 
            @ptrCast(&descriptor_set), 
            1,
            @ptrCast(&(uniform_buffer_stride * render_index)),
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
            frame_begin,
        );

        const result = try vkd.queuePresentKHR(
            present_queue,
            &.{
                .swapchain_count = 1,
                .p_swapchains = @ptrCast(&swapchain),
                .p_image_indices = @ptrCast(&present_index),
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
                        .min_image_count = swapchain_images_count,
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
                _ = try vkd.getSwapchainImagesKHR(device, swapchain, &swapchain_images_count, null);
                _ = try vkd.getSwapchainImagesKHR(device, swapchain, &swapchain_images_count, &swapchain_images);

                for (0..swapchain_images_count) |i| {
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
                        depth_image_view
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

                width_to_height = @as(f32, @floatFromInt(extent.width)) / @as(f32, @floatFromInt(extent.height));
                camera.y_scale = camera.x_scale * width_to_height;
                continue;
            },
            .success => {},
            else => return error.AcquireNextImageFailed,
        }

        if (render_index == last_render_index) {
            render_index = 0;
        } else {
            render_index += 1;
        }

        {
            const w = window.getKey(.w) == .press;
            const s = window.getKey(.s) == .press;
            const d = window.getKey(.d) == .press;
            const a = window.getKey(.a) == .press;

            const up = window.getKey(.up) == .press;
            const down = window.getKey(.down) == .press;
            const right = window.getKey(.right) == .press;
            const left = window.getKey(.left) == .press;

            const left_shift = window.getKey(.left_shift) == .press;
            const space = window.getKey(.space) == .press;

            const forward_z = camera.cos_z_to_y * camera.cos_z_to_x;
            const forward_x = camera.cos_z_to_y * camera.sin_z_to_x;

            if (w and !s) {
                camera.position.z += forward_z * target_dt; 
                camera.position.x += forward_x * target_dt;
            } else if (!w and s) {
                camera.position.z -= forward_z * target_dt;
                camera.position.x -= forward_x * target_dt; 
            }

            if (d and !a) {
                camera.position.z -= camera.sin_z_to_x * target_dt;
                camera.position.x += camera.cos_z_to_x * target_dt;
            } else if (!d and a) {
                camera.position.z += camera.sin_z_to_x * target_dt;
                camera.position.x -= camera.cos_z_to_x * target_dt; 
            }

            if (left_shift and !space) {
                camera.position.y += target_dt;
            } else if (!left_shift and space) {
                camera.position.y -= target_dt;
            }

            if (down and !up) {
                camera_z_to_y += target_dt;
            } else if (!down and up) {
                camera_z_to_y -= target_dt;
            }

            if (right and !left) {
                camera_z_to_x += target_dt;
            } else if (!right and left) {
                camera_z_to_x -= target_dt;
            }

            camera.cos_z_to_y = @cos(camera_z_to_y);
            camera.sin_z_to_y = @sin(camera_z_to_y);
            camera.cos_z_to_x = @cos(camera_z_to_x);
            camera.sin_z_to_x = @sin(camera_z_to_x);
        }

        var time: f32 = undefined;
        while (true) {
            glfw.pollEvents();
            time = @floatCast(glfw.getTime());
            if (time - last_time > target_dt) {
                break;
            }
        }
        last_time = time;
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

fn transitionImageLayout(
    vkd: *const DeviceDispatch,
    image: vk.Image,
    command_buffer: vk.CommandBuffer,
    old_layout: vk.ImageLayout,
    new_layout: vk.ImageLayout,
    src_stage: vk.PipelineStageFlags,
    dst_stage: vk.PipelineStageFlags,
    src_access: vk.AccessFlags,
    dst_access: vk.AccessFlags,
) void {

    const barrier = vk.ImageMemoryBarrier {
        .old_layout = old_layout,
        .new_layout = new_layout,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresource_range = .{
            .aspect_mask = .{ .color_bit = true },
            .base_mip_level = 0,
            .level_count = 1,
            .base_array_layer = 0,
            .layer_count = 1,
        },
        
        .src_access_mask = src_access,
        .dst_access_mask = dst_access,
    };


    vkd.cmdPipelineBarrier(
        command_buffer, 
        src_stage, 
        dst_stage, 
        .{}, 
        0, 
        null, 
        0, 
        null, 
        1, 
        @ptrCast(&barrier),
    );
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