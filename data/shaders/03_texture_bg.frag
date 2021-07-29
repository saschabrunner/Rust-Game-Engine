// displays a texture while applying the vertex color to the transparent bits

#version 450

layout(set = 1, binding = 0) uniform sampler2D tex_sampler;

layout(location = 0) in vec3 in_color;
layout(location = 1) in vec2 in_uv;

// location is the index of the framebuffer we render to
layout(location = 0) out vec4 out_color;

void main() {
    vec4 texture_color = texture(tex_sampler, in_uv);
    float bg_alpha = 1.0 - texture_color.a;
    vec3 final_color = (texture_color.rgb * texture_color.a) + (in_color * bg_alpha);
    out_color = vec4(final_color, 1.0);
}