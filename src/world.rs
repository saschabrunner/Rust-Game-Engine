use crate::{
    device::Device,
    memory::{self, Buffer},
    Material,
};

use std::{iter, mem::size_of, rc::Rc};

use gfx_hal::{
    buffer::{self, SubRange},
    command::CommandBuffer,
    device::Device as _,
    memory as gfx_mem,
    pso::{Descriptor, DescriptorPool, DescriptorSetWrite},
    queue::QueueGroup,
    IndexType,
};

use nalgebra::{Matrix4, Similarity3};

// not dead code, we use memcpy on these structs
#[allow(dead_code)]
pub struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
    uv: [f32; 2],
}

#[allow(dead_code)]
struct Transformations {
    model: Matrix4<f32>,
    view: Matrix4<f32>,
    projection: Matrix4<f32>,
}

pub struct GameObject<B: gfx_hal::Backend> {
    device: Rc<Device<B>>,
    transform: Similarity3<f32>,
    vertex_buffer: Buffer<B>, // todo: make this a reference shared accross objects (mesh struct?)
    index_buffer: Buffer<B>,  // todo: make this a reference shared accross objects (mesh struct?)
    index_count: usize,       // todo: make this a reference shared accross objects (mesh struct?)
    mvp_uniform_buffers: Vec<Buffer<B>>, // per frame in flight // todo: split this up, so vp part can be shared accross objects
    mvp_descriptor_sets: Vec<B::DescriptorSet>, // per frame in flight, will not be freed after allocation, only containing mvp buffers // todo: move vp into seperate set, provided by camera instead of by every object and bind before material
    material: Material<B>, // todo: make this a reference shared accross objects
}

impl<B: gfx_hal::Backend> GameObject<B> {
    pub unsafe fn new(
        device: &Rc<Device<B>>,
        queue_group: &mut QueueGroup<B>,
        command_pool: &mut B::CommandPool,
        mvp_descriptor_set_layout: &B::DescriptorSetLayout,
        descriptor_pool: &mut B::DescriptorPool,
        frames_in_flight: usize,
        vertices: &[Vertex],
        indices: &[u16],
        initial_transform: &Similarity3<f32>,
        material: Material<B>, // todo: don't take ownership, reuse
    ) -> GameObject<B> {
        let vertex_buffer = Self::create_vertex_buffer(device, queue_group, command_pool, vertices);
        let index_buffer = Self::create_index_buffer(device, queue_group, command_pool, indices);
        let mvp_uniform_buffers = Self::create_uniform_buffers(device, frames_in_flight);
        let mvp_descriptor_sets = Self::create_descriptor_sets(
            device,
            mvp_descriptor_set_layout,
            descriptor_pool,
            &mvp_uniform_buffers,
            frames_in_flight,
        );

        GameObject {
            device: device.clone(),
            transform: initial_transform.clone(),
            vertex_buffer,
            index_buffer,
            index_count: indices.len(),
            mvp_uniform_buffers,
            mvp_descriptor_sets,
            material,
        }
    }

    fn create_vertex_buffer(
        device: &Rc<Device<B>>,
        queue_group: &mut QueueGroup<B>,
        command_pool: &mut B::CommandPool,
        vertices: &[Vertex],
    ) -> Buffer<B> {
        let buffer_stride = size_of::<Vertex>();
        let buffer_len = vertices.len() * buffer_stride;
        // buffer needs to have size that's a multiple of non_coherent_atom_size for it to be mapped
        let padded_buffer_len = ((buffer_len + device.limits().non_coherent_atom_size - 1)
            / device.limits().non_coherent_atom_size
            * device.limits().non_coherent_atom_size) as u64;

        let mut staging_buffer = unsafe {
            Buffer::new(
                device,
                padded_buffer_len,
                buffer::Usage::TRANSFER_SRC,
                gfx_mem::Properties::CPU_VISIBLE,
            )
        };

        // copy data into staging buffer
        unsafe {
            memory::copy_into_buffer(
                device,
                vertices.as_ptr() as *const u8,
                &mut staging_buffer,
                buffer_len,
            )
        };

        let buffer = unsafe {
            Buffer::new(
                device,
                buffer_len as u64,
                buffer::Usage::TRANSFER_DST | buffer::Usage::VERTEX,
                gfx_mem::Properties::DEVICE_LOCAL,
            )
        };

        // copy staging buffer into final buffer
        memory::copy_buffer(
            device,
            queue_group,
            &staging_buffer,
            &buffer,
            command_pool,
            buffer_len as u64,
        );

        buffer
    }

    fn create_index_buffer(
        device: &Rc<Device<B>>,
        queue_group: &mut QueueGroup<B>,
        command_pool: &mut B::CommandPool,
        indices: &[u16],
    ) -> Buffer<B> {
        let buffer_stride = size_of::<u16>();
        let buffer_len = indices.len() * buffer_stride;
        let padded_buffer_len = ((buffer_len + device.limits().non_coherent_atom_size - 1)
            / device.limits().non_coherent_atom_size
            * device.limits().non_coherent_atom_size) as u64;

        let mut staging_buffer = unsafe {
            Buffer::new(
                device,
                padded_buffer_len,
                buffer::Usage::TRANSFER_SRC,
                gfx_mem::Properties::CPU_VISIBLE,
            )
        };

        unsafe {
            memory::copy_into_buffer(
                device,
                indices.as_ptr() as *const u8,
                &mut staging_buffer,
                buffer_len,
            )
        };

        let buffer = unsafe {
            Buffer::new(
                device,
                buffer_len as u64,
                buffer::Usage::TRANSFER_DST | buffer::Usage::INDEX,
                gfx_mem::Properties::DEVICE_LOCAL,
            )
        };

        memory::copy_buffer(
            device,
            queue_group,
            &staging_buffer,
            &buffer,
            command_pool,
            buffer_len as u64,
        );

        buffer
    }

    fn create_uniform_buffers(device: &Rc<Device<B>>, frames_in_flight: usize) -> Vec<Buffer<B>> {
        let mut uniform_buffers = Vec::with_capacity(frames_in_flight);
        let padded_length =
            ((size_of::<Transformations>() + device.limits().non_coherent_atom_size - 1)
                / device.limits().non_coherent_atom_size
                * device.limits().non_coherent_atom_size) as u64;
        for _ in 0..frames_in_flight {
            uniform_buffers.push(unsafe {
                Buffer::new(
                    device,
                    padded_length,
                    buffer::Usage::UNIFORM,
                    gfx_mem::Properties::CPU_VISIBLE,
                )
            });
        }

        uniform_buffers
    }

    fn create_descriptor_sets(
        device: &Rc<Device<B>>,
        descriptor_set_layout: &B::DescriptorSetLayout,
        descriptor_pool: &mut B::DescriptorPool,
        uniform_buffers: &[Buffer<B>],
        frames_in_flight: usize,
    ) -> Vec<B::DescriptorSet> {
        let mut descriptor_sets = Vec::with_capacity(frames_in_flight);
        let mut cloned_layouts = Vec::with_capacity(frames_in_flight);
        for _ in 0..frames_in_flight {
            cloned_layouts.push(&*descriptor_set_layout);
        }
        unsafe { descriptor_pool.allocate(cloned_layouts.into_iter(), &mut descriptor_sets) }
            .unwrap();

        for i in 0..frames_in_flight {
            unsafe {
                device
                    .logical_device()
                    .write_descriptor_set(DescriptorSetWrite {
                        set: &mut descriptor_sets[i],
                        binding: 0,
                        array_offset: 0,
                        descriptors: vec![Descriptor::Buffer(
                            uniform_buffers[i].buffer(),
                            SubRange::WHOLE,
                        )]
                        .into_iter(),
                    })
            }
        }

        descriptor_sets
    }

    pub fn get_transform(&mut self) -> &mut Similarity3<f32> {
        &mut self.transform
    }

    /// Copies buffers and records a draw call for this object.
    ///
    /// # Safety
    ///
    /// - `pipeline_layout` must be compatible with this object's descriptor sets.
    pub unsafe fn draw(
        &mut self,
        command_buffer: &mut B::CommandBuffer,
        pipeline_layout: &B::PipelineLayout,
        camera_view: &Matrix4<f32>,
        camera_projection: &Matrix4<f32>,
        frame_id: usize,
    ) {
        // copy uniform buffers
        let transformations = Transformations {
            model: self.transform.to_homogeneous(),
            view: camera_view.clone(),
            projection: camera_projection.clone(),
        };
        memory::copy_into_buffer(
            &self.device,
            &transformations as *const Transformations as *const u8,
            &mut self.mvp_uniform_buffers[frame_id],
            size_of::<Transformations>(),
        );

        // bind resources
        command_buffer.bind_vertex_buffers(
            0,
            iter::once((self.vertex_buffer.buffer(), SubRange::WHOLE)),
        );
        command_buffer.bind_index_buffer(
            self.index_buffer.buffer(),
            SubRange::WHOLE,
            IndexType::U16,
        );
        command_buffer.bind_graphics_descriptor_sets(
            pipeline_layout,
            0,
            vec![
                &self.mvp_descriptor_sets[frame_id],
                self.material.get_descriptor_set(frame_id),
            ]
            .into_iter(),
            iter::empty(),
        );

        // draw
        command_buffer.draw_indexed(0..self.index_count as u32, 0, 0..1);
    }
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
