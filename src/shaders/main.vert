#version 450

layout(location = 0) in vec3 v_position;
layout(location = 1) in vec3 v_color;

layout(set = 0, binding = 0) uniform UBO {
    vec4 _0;
    vec4 _1;
    vec4 _2;
    float near_z;
} u_view;

layout(location = 0) out vec3 color;

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
    vec3 view_space_pos = apply_affine(
        v_position,
        u_view._0,
        u_view._1,
        u_view._2
    );
    gl_Position = vec4(view_space_pos.xy, u_view.near_z, view_space_pos.z);
    color = v_color;
}