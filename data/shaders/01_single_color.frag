#version 450

// location is the index of the framebuffer we render to
layout(location = 0) out vec4 fragment_color;

void main() {
    fragment_color = vec4(0.9, 0.5, 1.0, 1.0);
}