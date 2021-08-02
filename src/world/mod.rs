mod game_object;

pub use game_object::GameObject;

// not dead code, we use memcpy on these structs
#[allow(dead_code)]
pub struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
    uv: [f32; 2],
}

#[cfg_attr(rustfmt, rustfmt_skip)]
pub const CUBE_VERTICES: [Vertex; 24] = [
    Vertex { position: [-0.5, -0.5,  0.5], color: [1.0, 0.0, 0.0], uv: [0.0, 0.0] },    // front
    Vertex { position: [-0.5,  0.5,  0.5], color: [0.0, 1.0, 0.0], uv: [0.0, 1.0] },
    Vertex { position: [ 0.5,  0.5,  0.5], color: [0.0, 0.0, 1.0], uv: [1.0, 1.0] },
    Vertex { position: [ 0.5, -0.5,  0.5], color: [1.0, 1.0, 1.0], uv: [1.0, 0.0] },

    Vertex { position: [ 0.5, -0.5, -0.5], color: [1.0, 0.0, 0.0], uv: [0.0, 0.0] },    // back
    Vertex { position: [ 0.5,  0.5, -0.5], color: [0.0, 1.0, 0.0], uv: [0.0, 1.0] },
    Vertex { position: [-0.5,  0.5, -0.5], color: [0.0, 0.0, 1.0], uv: [1.0, 1.0] },
    Vertex { position: [-0.5, -0.5, -0.5], color: [1.0, 1.0, 1.0], uv: [1.0, 0.0] },

    Vertex { position: [-0.5, -0.5, -0.5], color: [1.0, 0.0, 0.0], uv: [0.0, 0.0] },    // left
    Vertex { position: [-0.5,  0.5, -0.5], color: [0.0, 1.0, 0.0], uv: [0.0, 1.0] },
    Vertex { position: [-0.5,  0.5,  0.5], color: [0.0, 0.0, 1.0], uv: [1.0, 1.0] },
    Vertex { position: [-0.5, -0.5,  0.5], color: [1.0, 1.0, 1.0], uv: [1.0, 0.0] },

    Vertex { position: [ 0.5, -0.5,  0.5], color: [1.0, 0.0, 0.0], uv: [0.0, 0.0] },    // right
    Vertex { position: [ 0.5,  0.5,  0.5], color: [0.0, 1.0, 0.0], uv: [0.0, 1.0] },
    Vertex { position: [ 0.5,  0.5, -0.5], color: [0.0, 0.0, 1.0], uv: [1.0, 1.0] },
    Vertex { position: [ 0.5, -0.5, -0.5], color: [1.0, 1.0, 1.0], uv: [1.0, 0.0] },

    Vertex { position: [-0.5, -0.5, -0.5], color: [1.0, 0.0, 0.0], uv: [0.0, 0.0] },    // top
    Vertex { position: [-0.5, -0.5,  0.5], color: [0.0, 1.0, 0.0], uv: [0.0, 1.0] },
    Vertex { position: [ 0.5, -0.5,  0.5], color: [0.0, 0.0, 1.0], uv: [1.0, 1.0] },
    Vertex { position: [ 0.5, -0.5, -0.5], color: [1.0, 1.0, 1.0], uv: [1.0, 0.0] },

    Vertex { position: [-0.5,  0.5,  0.5], color: [1.0, 0.0, 0.0], uv: [0.0, 0.0] },    // bottom
    Vertex { position: [-0.5,  0.5, -0.5], color: [0.0, 1.0, 0.0], uv: [0.0, 1.0] },
    Vertex { position: [ 0.5,  0.5, -0.5], color: [0.0, 0.0, 1.0], uv: [1.0, 1.0] },
    Vertex { position: [ 0.5,  0.5,  0.5], color: [1.0, 1.0, 1.0], uv: [1.0, 0.0] },
];

// these indices must be counter-clockwise for backface culling to work
#[cfg_attr(rustfmt, rustfmt_skip)]
pub const CUBE_INDICES: [u16; 36] = [
     0,  1,  2,  0,  2,  3, // front
     4,  5,  6,  4,  6,  7, // back
     8,  9, 10,  8, 10, 11, // left
    12, 13, 14, 12, 14, 15, // right
    16, 17, 18, 16, 18, 19, // top
    20, 21, 22, 20, 22, 23, // bottom
];

#[cfg_attr(rustfmt, rustfmt_skip)]
pub const QUAD_VERTICES: [Vertex; 4] = [
    Vertex { position: [-0.5, -0.5,  0.5], color: [1.0, 0.0, 0.0], uv: [0.0, 0.0] },
    Vertex { position: [-0.5,  0.5,  0.5], color: [0.0, 1.0, 0.0], uv: [0.0, 40.0] },
    Vertex { position: [ 0.5,  0.5,  0.5], color: [0.0, 0.0, 1.0], uv: [40.0, 40.0] },
    Vertex { position: [ 0.5, -0.5,  0.5], color: [1.0, 1.0, 1.0], uv: [40.0, 0.0] },
];

// these indices must be counter-clockwise for backface culling to work
#[cfg_attr(rustfmt, rustfmt_skip)]
pub const QUAD_INDICES: [u16; 6] = [
     0,  1,  2,  0,  2,  3,
];
