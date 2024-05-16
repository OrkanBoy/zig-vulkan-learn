#version 450

layout(location = 0) in vec3 f_color;
layout(location = 1) in vec2 f_tex_coord;

layout(location = 0) out vec4 color;

layout(set = 0, binding = 1) uniform sampler2D u_sampler;

void main() {
    color = texture(u_sampler, f_tex_coord);
}