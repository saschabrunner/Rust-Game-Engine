#version 450

layout(location = 0) in vec3 in_color;

// location is the index of the framebuffer we render to
layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(in_color, 1.0);
}