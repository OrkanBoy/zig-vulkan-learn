#version 450

layout(location = 0) in vec3 v_position;
layout(location = 1) in vec3 v_color;
layout(location = 2) in vec2 v_tex_coord;

layout(set = 0, binding = 0) uniform UBO {
    vec3 position;
    float near_z;
    float cos_z_to_y, sin_z_to_y;
    float cos_z_to_x, sin_z_to_x;
    float x_scale, y_scale;
} u_camera;

layout(location = 0) out vec3 color;
layout(location = 1) out vec2 tex_coord;

vec3 apply_affine(
    vec3 pos,
    vec4 affine_0,
    vec4 affine_1,
    vec4 affine_2
) {
    return vec3(
        dot(pos, affine_0.xyz) + affine_0.w,
        dot(pos, affine_1.xyz) + affine_1.w,
        dot(pos, affine_2.xyz) + affine_2.w
    );
}

void main() {
    gl_Position.xyz = v_position;
    
    gl_Position.xyz -= u_camera.position;

    float x, y, z;

    z = gl_Position.z;
    x = gl_Position.x;
    gl_Position.z =  z * u_camera.cos_z_to_x + x * u_camera.sin_z_to_x;
    gl_Position.x = -z * u_camera.sin_z_to_x + x * u_camera.cos_z_to_x;

    z = gl_Position.z;
    y = gl_Position.y;
    gl_Position.z =  z * u_camera.cos_z_to_y + y * u_camera.sin_z_to_y;
    gl_Position.y = -z * u_camera.sin_z_to_y + y * u_camera.cos_z_to_y;

    gl_Position.z += u_camera.near_z;

    gl_Position.x *= u_camera.x_scale;
    gl_Position.y *= u_camera.y_scale;

    // shader divides by .w instead of .z
    gl_Position.w = gl_Position.z;
    gl_Position.z = u_camera.near_z;
    color = v_color;
    tex_coord = v_tex_coord;
}