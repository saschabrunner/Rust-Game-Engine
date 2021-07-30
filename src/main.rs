mod camera;
mod device;
mod memory;

use device::Device;

use std::{
    borrow::Borrow,
    io::Cursor,
    iter,
    mem::{size_of, ManuallyDrop},
    ops::Range,
    rc::Rc,
    time::Instant,
};

use gfx_hal::{
    adapter::{Adapter, PhysicalDevice},
    buffer::{self, SubRange},
    command::{
        ClearColor, ClearDepthStencil, ClearValue, CommandBuffer, CommandBufferFlags, Level,
        RenderAttachmentInfo, SubpassContents,
    },
    device::Device as _,
    format::{Aspects, ChannelType, Format, ImageFeature, Swizzle},
    image::{
        self as gfximg, Extent, FramebufferAttachment, Layout, SamplerDesc, SubresourceRange,
        ViewCapabilities, ViewKind,
    },
    memory::{self as gfx_mem, SparseFlags},
    pass::{
        Attachment, AttachmentLoadOp, AttachmentOps, AttachmentStoreOp, Subpass, SubpassDependency,
        SubpassDesc,
    },
    pool::{CommandPool, CommandPoolCreateFlags},
    pso::{
        self, AttributeDesc, BlendState, BufferDescriptorFormat, BufferDescriptorType,
        ColorBlendDesc, ColorMask, DepthStencilDesc, DepthTest, Descriptor, DescriptorPool,
        DescriptorPoolCreateFlags, DescriptorRangeDesc, DescriptorSetLayoutBinding,
        DescriptorSetWrite, DescriptorType, EntryPoint, Face, GraphicsPipelineDesc,
        ImageDescriptorType, InputAssemblerDesc, PipelineStage, Primitive, PrimitiveAssemblerDesc,
        Rasterizer, Rect, ShaderStageFlags, Specialization, VertexBufferDesc, VertexInputRate,
        Viewport,
    },
    queue::{family::QueueFamilyId, Queue, QueueGroup},
    window::{Extent2D, PresentationSurface, Surface, SwapchainConfig},
    IndexType, Instance,
};

use nalgebra::{Matrix4, Point3, Similarity3, UnitQuaternion, Vector3};

use winit::{
    dpi::LogicalSize,
    event::{DeviceEvent, ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::ControlFlow,
};

use camera::{Camera, CameraDesc};

// not dead code, we use memcpy on these structs
#[allow(dead_code)]
struct Vertex {
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

struct Buffer<B: gfx_hal::Backend> {
    device: Rc<Device<B>>,
    buffer: ManuallyDrop<B::Buffer>,
    memory: ManuallyDrop<B::Memory>, // this should be replaced by memory from an allocator
}

impl<B: gfx_hal::Backend> Buffer<B> {
    unsafe fn new(
        device: &Rc<Device<B>>,
        buffer_len: u64,
        usage: buffer::Usage,
        properties: gfx_mem::Properties,
    ) -> Self {
        let mut buffer = ManuallyDrop::new(
            device
                .logical_device()
                .create_buffer(buffer_len, usage, SparseFlags::empty())
                .unwrap(),
        );

        let buffer_requirements = device.logical_device().get_buffer_requirements(&buffer);

        let buffer_memory_type = device.find_memory_type(buffer_requirements.type_mask, properties);

        let memory = device
            .logical_device()
            .allocate_memory(buffer_memory_type, buffer_requirements.size)
            .unwrap();
        device
            .logical_device()
            .bind_buffer_memory(&memory, 0, &mut buffer)
            .unwrap();
        let memory = ManuallyDrop::new(memory);

        Buffer {
            device: device.clone(),
            buffer,
            memory,
        }
    }
}

impl<B: gfx_hal::Backend> Drop for Buffer<B> {
    fn drop(&mut self) {
        // wait for device to become idle
        self.device.logical_device().wait_idle().unwrap();

        unsafe {
            self.device
                .logical_device()
                .destroy_buffer(ManuallyDrop::take(&mut self.buffer));
            self.device
                .logical_device()
                .free_memory(ManuallyDrop::take(&mut self.memory));
        }
    }
}

struct Image<B: gfx_hal::Backend> {
    device: Rc<Device<B>>,
    image: ManuallyDrop<B::Image>,
    memory: ManuallyDrop<B::Memory>, // this should be replaced by memory from an allocator
}

impl<B: gfx_hal::Backend> Image<B> {
    unsafe fn new(
        device: &Rc<Device<B>>,
        width: u32,
        height: u32,
        format: Format,
        tiling: gfximg::Tiling,
        usage: gfximg::Usage,
        properties: gfx_mem::Properties,
    ) -> Self {
        let mut image = ManuallyDrop::new(
            device
                .logical_device()
                .create_image(
                    gfximg::Kind::D2(width, height, 1, 1),
                    1,
                    format,
                    tiling,
                    usage,
                    SparseFlags::empty(),
                    ViewCapabilities::empty(),
                )
                .unwrap(),
        );

        let image_requirements = device.logical_device().get_image_requirements(&image);
        let memory_type = device.find_memory_type(image_requirements.type_mask, properties);

        let memory = device
            .logical_device()
            .allocate_memory(memory_type, image_requirements.size)
            .unwrap();
        device
            .logical_device()
            .bind_image_memory(&memory, 0, &mut image)
            .unwrap();
        let memory = ManuallyDrop::new(memory);
        Image {
            device: device.clone(),
            image,
            memory,
        }
    }

    fn transition_image_layout(
        &self,
        queue_group: &mut QueueGroup<B>,
        command_pool: &mut B::CommandPool,
        format: Format,
        layouts: Range<Layout>,
    ) {
        let mut command_fence = self
            .device
            .logical_device()
            .create_fence(false)
            .expect("Out of memory creating fence");

        let aspects = if layouts.end == Layout::DepthStencilAttachmentOptimal {
            let mut val = Aspects::DEPTH;

            if Format::is_stencil(format) {
                val |= Aspects::STENCIL;
            }
            val
        } else {
            Aspects::COLOR
        };

        let (stages, accesses) =
            if layouts.start == Layout::Undefined && layouts.end == Layout::TransferDstOptimal {
                Some((
                    PipelineStage::TOP_OF_PIPE..PipelineStage::TRANSFER,
                    gfximg::Access::empty()..gfximg::Access::TRANSFER_WRITE,
                ))
            } else if layouts.start == Layout::TransferDstOptimal
                && layouts.end == Layout::ShaderReadOnlyOptimal
            {
                Some((
                    PipelineStage::TRANSFER..PipelineStage::FRAGMENT_SHADER,
                    gfximg::Access::TRANSFER_WRITE..gfximg::Access::SHADER_READ,
                ))
            } else if layouts.start == Layout::Undefined
                && layouts.end == Layout::DepthStencilAttachmentOptimal
            {
                Some((
                    PipelineStage::TOP_OF_PIPE..PipelineStage::EARLY_FRAGMENT_TESTS,
                    gfximg::Access::DEPTH_STENCIL_ATTACHMENT_READ
                        ..gfximg::Access::DEPTH_STENCIL_ATTACHMENT_WRITE,
                ))
            } else {
                None
            }
            .expect(&format!(
                "Unsupported combination of layouts {:?} and {:?}",
                layouts.start, layouts.end
            ));

        unsafe {
            let mut command_buffer = command_pool.allocate_one(Level::Primary);
            command_buffer.begin_primary(CommandBufferFlags::ONE_TIME_SUBMIT);
            command_buffer.pipeline_barrier(
                // our transition can happen in between any of these stages,
                // optimally we would want this gap to be as wide as possible
                stages.start..stages.end,
                gfx_mem::Dependencies::empty(),
                iter::once(gfx_mem::Barrier::Image {
                    // our data starts with those access flags and this layout
                    // and transitions to the latter ones in the range
                    states: (accesses.start, layouts.start)..(accesses.end, layouts.end),
                    target: &*self.image,
                    range: SubresourceRange {
                        aspects: aspects,
                        level_start: 0,
                        level_count: None,
                        layer_start: 0,
                        layer_count: None,
                    },
                    families: None, // used only for ownership transfer
                }),
            );

            command_buffer.finish();
            queue_group.queues[0].submit(
                iter::once(&command_buffer),
                iter::empty(),
                iter::empty(),
                Some(&mut command_fence),
            );

            self.device
                .logical_device()
                .wait_for_fence(&command_fence, !0)
                .unwrap();
            command_pool.free(iter::once(command_buffer));
            self.device.logical_device().destroy_fence(command_fence);
        }
    }
}

impl<B: gfx_hal::Backend> Drop for Image<B> {
    fn drop(&mut self) {
        // wait for device to become idle
        self.device.logical_device().wait_idle().unwrap();

        unsafe {
            self.device
                .logical_device()
                .destroy_image(ManuallyDrop::take(&mut self.image));
            self.device
                .logical_device()
                .free_memory(ManuallyDrop::take(&mut self.memory));
        }
    }
}

struct ImageView<B: gfx_hal::Backend> {
    device: Rc<Device<B>>,
    image: Image<B>,
    image_view: ManuallyDrop<B::ImageView>,
}

impl<B: gfx_hal::Backend> ImageView<B> {
    pub fn new(
        device: &Rc<Device<B>>,
        image: Image<B>,
        format: Format,
        aspects: Aspects,
        usage: gfximg::Usage,
    ) -> Self {
        let iv = ManuallyDrop::new(
            unsafe {
                device.logical_device().create_image_view(
                    &image.image,
                    ViewKind::D2,
                    format,
                    Swizzle::NO,
                    usage,
                    SubresourceRange {
                        aspects: aspects,
                        level_start: 0,
                        level_count: None,
                        layer_start: 0,
                        layer_count: None,
                    },
                )
            }
            .unwrap(),
        );

        ImageView {
            device: device.clone(),
            image: image,
            image_view: iv,
        }
    }
}

impl<B: gfx_hal::Backend> Drop for ImageView<B> {
    fn drop(&mut self) {
        // wait for device to become idle
        self.device.logical_device().wait_idle().unwrap();

        unsafe {
            self.device
                .logical_device()
                .destroy_image_view(ManuallyDrop::take(&mut self.image_view));
        }
        // drop of image will be called here
    }
}

struct Material<B: gfx_hal::Backend> {
    device: Rc<Device<B>>,
    texture: ImageView<B>, // currently a material is only made out of a texture
    descriptor_sets: Vec<B::DescriptorSet>, // per frame in flight, will not be freed after allocation
                                            // todo: different shader pipelines
                                            // todo: recycle descriptor sets
}

impl<B: gfx_hal::Backend> Material<B> {
    pub unsafe fn new(
        device: &Rc<Device<B>>,
        queue_group: &mut QueueGroup<B>,
        command_pool: &mut B::CommandPool,
        material_descriptor_set_layout: &B::DescriptorSetLayout,
        descriptor_pool: &mut B::DescriptorPool,
        frames_in_flight: usize,
        image_png: &[u8], // todo: replace this with a file path or something else and let this function load the data
    ) -> Material<B> {
        let image = image::load(Cursor::new(image_png), image::ImageFormat::Png)
            .unwrap()
            .to_rgba8();

        let texture = Self::create_texture(device, queue_group, command_pool, &image);
        let descriptor_sets = Self::create_descriptor_sets(
            device,
            material_descriptor_set_layout,
            descriptor_pool,
            &texture,
            frames_in_flight,
        );

        // todo: get image file path as param and load it into texture (imageview<b>)
        Material {
            device: device.clone(),
            texture,
            descriptor_sets,
        }
    }

    fn create_texture(
        device: &Rc<Device<B>>,
        queue_group: &mut QueueGroup<B>,
        command_pool: &mut B::CommandPool,
        raw_image: &image::RgbaImage,
    ) -> ImageView<B> {
        let buffer_len = raw_image.width() * raw_image.height() * 4;
        let padded_buffer_len = ((buffer_len as usize + device.limits().non_coherent_atom_size - 1)
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
            memory::copy_into_memory(
                device,
                raw_image.as_ptr(),
                &mut staging_buffer.memory,
                buffer_len as usize,
            )
        };

        let texture_image = unsafe {
            Image::new(
                &device,
                raw_image.width(),
                raw_image.height(),
                Format::Rgba8Srgb,
                gfximg::Tiling::Optimal,
                gfximg::Usage::TRANSFER_DST | gfximg::Usage::SAMPLED,
                gfx_mem::Properties::DEVICE_LOCAL,
            )
        };

        // transition to transfer destination layout
        texture_image.transition_image_layout(
            queue_group,
            command_pool,
            Format::Rgba8Srgb,
            Layout::Undefined..Layout::TransferDstOptimal,
        );

        // copy buffer to image
        memory::copy_buffer_to_image(
            device,
            queue_group,
            &staging_buffer.buffer,
            &texture_image.image,
            command_pool,
            raw_image.width(),
            raw_image.height(),
        );

        // transition image to shader read layout
        texture_image.transition_image_layout(
            queue_group,
            command_pool,
            Format::Rgba8Srgb,
            Layout::TransferDstOptimal..Layout::ShaderReadOnlyOptimal,
        );

        ImageView::new(
            device,
            texture_image,
            Format::Rgba8Srgb,
            Aspects::COLOR,
            gfximg::Usage::SAMPLED,
        )
    }

    fn create_descriptor_sets(
        device: &Rc<Device<B>>,
        descriptor_set_layout: &B::DescriptorSetLayout,
        descriptor_pool: &mut B::DescriptorPool,
        image_view: &ImageView<B>,
        frames_in_flight: usize,
    ) -> Vec<B::DescriptorSet> {
        // right now we create one descriptor set per frame in flight, even though it's not necessary yet
        // the image view we store can be shared because it doesn't change between frames
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
                        descriptors: vec![
                            // instead of a combined image sampler, we could also use a separate image and a sampler descriptor
                            // however, in this case we are not even using CombinedImageSampler, because the sampler is defined
                            // immutable, therefore we only provide the image view
                            Descriptor::Image(
                                &*image_view.image_view,
                                Layout::ShaderReadOnlyOptimal,
                            ),
                        ]
                        .into_iter(),
                    })
            }
        }

        descriptor_sets
    }

    pub fn get_descriptor_set<'a>(&'a self, frame_id: usize) -> &'a B::DescriptorSet {
        &self.descriptor_sets[frame_id]
    }
}

struct GameObject<B: gfx_hal::Backend> {
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
            memory::copy_into_memory(
                device,
                vertices.as_ptr() as *const u8,
                &mut staging_buffer.memory,
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
            &staging_buffer.buffer,
            &buffer.buffer,
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
            memory::copy_into_memory(
                device,
                indices.as_ptr() as *const u8,
                &mut staging_buffer.memory,
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
            &staging_buffer.buffer,
            &buffer.buffer,
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
                            &*uniform_buffers[i].buffer,
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
        memory::copy_into_memory(
            &self.device,
            &transformations as *const Transformations as *const u8,
            &mut self.mvp_uniform_buffers[frame_id].memory,
            size_of::<Transformations>(),
        );

        // bind resources
        command_buffer.bind_vertex_buffers(
            0,
            iter::once((&*self.vertex_buffer.buffer, SubRange::WHOLE)),
        );
        command_buffer.bind_index_buffer(
            &*self.index_buffer.buffer,
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

struct Pipeline<B: gfx_hal::Backend> {
    device: Rc<Device<B>>,
    pipeline: ManuallyDrop<B::GraphicsPipeline>,
    viewport: Viewport,
}

impl<B: gfx_hal::Backend> Pipeline<B> {
    pub unsafe fn new(
        // todo: add more configuration options
        // perhaps introduce a config struct which implements Default
        device: &Rc<Device<B>>,
        pipeline_layout: &B::PipelineLayout,
        subpass: Subpass<B>,
        vertex_shader: &[u32],
        fragment_shader: &[u32],
        surface_extent: Extent2D,
    ) -> Self {
        let vertex_shader_module = device
            .logical_device()
            .create_shader_module(vertex_shader)
            .expect("Failed to create vertex shader module");

        let fragment_shader_module = device
            .logical_device()
            .create_shader_module(fragment_shader)
            .expect("Failed to create fragment shader module");

        let (vs_entry, fs_entry) = (
            EntryPoint {
                entry: "main",
                module: &vertex_shader_module,
                specialization: Specialization::default(), // specializations can be used for constants later
            },
            EntryPoint {
                entry: "main",
                module: &fragment_shader_module,
                specialization: Specialization::default(),
            },
        );

        // https://vulkan-tutorial.com/Drawing_a_triangle/Graphics_pipeline_basics/Fixed_functions
        let primitive_assembler = PrimitiveAssemblerDesc::Vertex {
            buffers: &[VertexBufferDesc {
                binding: 0,
                stride: size_of::<Vertex>() as u32,
                rate: VertexInputRate::Vertex,
            }],
            attributes: &[
                AttributeDesc {
                    location: 0,
                    binding: 0, // is equal to the number in the VertexBufferDesc
                    element: pso::Element {
                        format: Format::Rgb32Sfloat,
                        offset: 0,
                    },
                },
                AttributeDesc {
                    location: 1,
                    binding: 0,
                    element: pso::Element {
                        format: Format::Rgb32Sfloat,
                        offset: 12, // an offset_of library or offset_from may be considered here
                    },
                },
                AttributeDesc {
                    location: 2,
                    binding: 0,
                    element: pso::Element {
                        format: Format::Rg32Sfloat,
                        offset: 24,
                    },
                },
            ],
            input_assembler: InputAssemblerDesc::new(Primitive::TriangleList),
            vertex: vs_entry,
            tessellation: None,
            geometry: None,
        };

        let mut pipeline_desc = GraphicsPipelineDesc::new(
            primitive_assembler,
            Rasterizer {
                cull_face: Face::BACK,
                ..Rasterizer::FILL // this is the struct update syntax
            },
            Some(fs_entry),
            pipeline_layout,
            // every subpass requires its own pipeline
            // so we refer to a specific supbpass of the renderpass
            subpass,
        );

        pipeline_desc.depth_stencil = DepthStencilDesc {
            depth: Some(DepthTest {
                fun: gfx_hal::pso::Comparison::Less,
                write: true,
            }),
            depth_bounds: false,
            stencil: None,
        };

        pipeline_desc.blender.targets.push(ColorBlendDesc {
            mask: ColorMask::ALL,
            blend: Some(BlendState::ALPHA),
        });

        let pipeline = ManuallyDrop::new(
            device
                .logical_device()
                .create_graphics_pipeline(&pipeline_desc, None)
                .expect("Failed to create graphics pipeline"),
        );

        // shader modules can be destroyed now, since the pipeline compiled
        // them to machine code and we don't need the spir-v intermediate anymore
        device
            .logical_device()
            .destroy_shader_module(vertex_shader_module);
        device
            .logical_device()
            .destroy_shader_module(fragment_shader_module);

        let viewport = Viewport {
            rect: Rect {
                x: 0,
                y: 0,
                w: surface_extent.width as i16,
                h: surface_extent.height as i16,
            },
            depth: 0.0..1.0,
        };

        Pipeline {
            device: device.clone(),
            pipeline,
            viewport,
        }
    }

    pub fn update_dimensions(&mut self, surface_extent: Extent2D) {
        self.viewport.rect.w = surface_extent.width as i16;
        self.viewport.rect.h = surface_extent.height as i16;
    }

    pub unsafe fn bind(&self, command_buffer: &mut B::CommandBuffer) {
        // viewports and scissors are part of the command buffer, known as dynamic viewports and scissors in vulkan
        command_buffer.set_viewports(0, iter::once(self.viewport.clone()));
        command_buffer.set_scissors(0, iter::once(self.viewport.rect));
        command_buffer.bind_graphics_pipeline(&self.pipeline);
    }
}

impl<B: gfx_hal::Backend> Drop for Pipeline<B> {
    fn drop(&mut self) {
        // wait for device to become idle
        self.device.logical_device().wait_idle().unwrap();
        unsafe {
            self.device
                .logical_device()
                .destroy_graphics_pipeline(ManuallyDrop::take(&mut self.pipeline));
        }
    }
}

struct FrameResources<B: gfx_hal::Backend> {
    device: Rc<Device<B>>,
    submission_complete_semaphore: ManuallyDrop<B::Semaphore>,
    submission_complete_fence: ManuallyDrop<B::Fence>,
    command_pool: ManuallyDrop<B::CommandPool>,
    command_buffer: ManuallyDrop<B::CommandBuffer>,
    depth_image_view: ImageView<B>,
}

impl<B: gfx_hal::Backend> FrameResources<B> {
    pub unsafe fn new(
        device: &Rc<Device<B>>,
        queue_family: QueueFamilyId,
        surface_extent: Extent2D,
        depth_format: Format,
    ) -> Self {
        let mut command_pool = ManuallyDrop::new(
            device
                .logical_device()
                .create_command_pool(queue_family, CommandPoolCreateFlags::empty())
                .expect("Out of memory creating command pool"),
        );

        let command_buffer = ManuallyDrop::new(command_pool.allocate_one(Level::Primary));

        let submission_complete_semaphore = ManuallyDrop::new(
            device
                .logical_device()
                .create_semaphore()
                .expect("Out of memory creating semaphore"),
        );

        let submission_complete_fence = ManuallyDrop::new(
            device
                .logical_device()
                .create_fence(true)
                .expect("Out of memory creating fence"),
        );

        let depth_image_view = Self::create_depth_image_view(device, surface_extent, depth_format);

        FrameResources {
            device: device.clone(),
            submission_complete_semaphore,
            submission_complete_fence,
            command_pool,
            command_buffer,
            depth_image_view,
        }
    }

    pub unsafe fn update_dimensions(&mut self, surface_extent: Extent2D, depth_format: Format) {
        self.depth_image_view =
            Self::create_depth_image_view(&self.device, surface_extent, depth_format);
    }

    unsafe fn create_depth_image_view(
        device: &Rc<Device<B>>,
        surface_extent: Extent2D,
        depth_format: Format,
    ) -> ImageView<B> {
        let depth_image = Image::new(
            device,
            surface_extent.width,
            surface_extent.height,
            depth_format,
            gfximg::Tiling::Optimal,
            gfximg::Usage::DEPTH_STENCIL_ATTACHMENT,
            gfx_mem::Properties::DEVICE_LOCAL,
        );

        ImageView::new(
            device,
            depth_image,
            depth_format,
            Aspects::DEPTH, // todo: stencil aspect
            gfximg::Usage::DEPTH_STENCIL_ATTACHMENT,
        )
    }
}

impl<B: gfx_hal::Backend> Drop for FrameResources<B> {
    fn drop(&mut self) {
        // wait for device to become idle
        self.device.logical_device().wait_idle().unwrap();

        unsafe {
            self.command_pool
                .free(iter::once(ManuallyDrop::take(&mut self.command_buffer)));

            self.device
                .logical_device()
                .destroy_command_pool(ManuallyDrop::take(&mut self.command_pool));

            self.device
                .logical_device()
                .destroy_semaphore(ManuallyDrop::take(&mut self.submission_complete_semaphore));

            self.device
                .logical_device()
                .destroy_fence(ManuallyDrop::take(&mut self.submission_complete_fence));
        }
        // drop of depth_image_view will be called here
    }
}

struct Renderer<B: gfx_hal::Backend> {
    camera: Camera,
    surface: ManuallyDrop<B::Surface>,
    surface_format: Format,
    depth_format: Format,
    surface_extent: Extent2D,
    //viewport: Viewport,
    render_pass: ManuallyDrop<B::RenderPass>,
    framebuffer: ManuallyDrop<B::Framebuffer>,
    //pipeline: ManuallyDrop<B::GraphicsPipeline>,
    pipeline: ManuallyDrop<Pipeline<B>>,
    pipeline_layout: ManuallyDrop<B::PipelineLayout>,
    //pipeline_cache: ManuallyDrop<B::PipelineCache> // todo: should I add this?
    //submission_complete_semaphores: Vec<B::Semaphore>,
    //submission_complete_fences: Vec<B::Fence>,
    //command_pools: Vec<B::CommandPool>,
    //command_buffers: Vec<B::CommandBuffer>,
    //texture_image_view: ManuallyDrop<B::ImageView>,
    //depth_image_views: Vec<ImageView<B>>,
    //vertex_buffer: ManuallyDrop<Buffer<B>>, // todo: consider not wrapping my types in manuallydrop and using automatic drop order
    //index_buffer: ManuallyDrop<Buffer<B>>,
    //uniform_buffers: Vec<ManuallyDrop<Buffer<B>>>,
    //floor_vertex_buffer: ManuallyDrop<Buffer<B>>,
    //floor_index_buffer: ManuallyDrop<Buffer<B>>,
    //floor_uniform_buffers: Vec<ManuallyDrop<Buffer<B>>>,
    //vertex_buffer: ManuallyDrop<B::Buffer>,
    //index_buffer: ManuallyDrop<B::Buffer>,
    //uniform_buffers: Vec<ManuallyDrop<B::Buffer>>,
    //floor_vertex_buffer: ManuallyDrop<B::Buffer>,
    //floor_index_buffer: ManuallyDrop<B::Buffer>,
    //floor_uniform_buffers: Vec<ManuallyDrop<B::Buffer>>,
    //texture_image: ManuallyDrop<B::Image>,
    frame_resources: Vec<FrameResources<B>>,
    texture_sampler: ManuallyDrop<B::Sampler>,
    //textures: Vec<ImageView<B>>, // Vec can be cleared to drop elements, no ManuallyDrop required
    //depth_image: ManuallyDrop<B::Image>, //todo: per frame in flight
    //vertex_buffer_memory: ManuallyDrop<B::Memory>,
    //index_buffer_memory: ManuallyDrop<B::Memory>,
    //uniform_buffers_memory: Vec<ManuallyDrop<B::Memory>>,
    //floor_vertex_buffer_memory: ManuallyDrop<B::Memory>,
    //floor_index_buffer_memory: ManuallyDrop<B::Memory>,
    //floor_uniform_buffers_memory: Vec<ManuallyDrop<B::Memory>>,
    //texture_memory: ManuallyDrop<B::Memory>,
    //depth_image_memory: ManuallyDrop<B::Memory>,
    //descriptor_set_layout: ManuallyDrop<B::DescriptorSetLayout>,
    mvp_descriptor_set_layout: ManuallyDrop<B::DescriptorSetLayout>, // todo: remove model from this and add it to separate ds layout in GameObject, then move this to camera perhaps
    material_descriptor_set_layout: ManuallyDrop<B::DescriptorSetLayout>, // currently there's only one possible layout for materials (a single image)
    descriptor_pool: ManuallyDrop<B::DescriptorPool>,
    game_objects: Vec<GameObject<B>>,
    //descriptor_sets: Vec<B::DescriptorSet>,
    //floor_descriptor_sets: Vec<B::DescriptorSet>,
    frames_in_flight: usize,
    frame: u64,
    start_time: Instant,
    last_frame: Instant,
    // these are dropped in the order they are declared, very important!
    // currently assuming that during drop, this is the only Rc to this device!
    // every other struct that has a handle on this device should have been dropped at this point already
    // alternatively, a global function could be created to get a more short lived reference to the device
    // then Rc wouldn't be needed at all
    device: Rc<Device<B>>,
    //adapter: Adapter<B>,
    queue_group: QueueGroup<B>,
    instance: B::Instance,
}

impl<B: gfx_hal::Backend> Renderer<B> {
    pub fn new(instance: B::Instance, mut surface: B::Surface, adapter: Adapter<B>) -> Renderer<B> {
        let mut device = unsafe { Device::<B>::new(adapter, &surface) };
        let mut queue_group = device.queue_groups().pop().unwrap();
        let device = Rc::new(device); // Rc can only be created now because of mutable borrow above
        let surface_format = Self::find_surface_format(device.physical_device(), &surface);
        let depth_format = Self::find_depth_format(device.physical_device());

        let default_surface_extent = Extent2D {
            width: DEFAULT_WINDOW_SIZE[0],
            height: DEFAULT_WINDOW_SIZE[1],
        };
        let capabilities = surface.capabilities(device.physical_device());
        let swapchain_config =
            SwapchainConfig::from_caps(&capabilities, surface_format, default_surface_extent);
        let framebuffer_attachment = swapchain_config.framebuffer_attachment();

        // update surface extent, since the swapchain might enforce different dimensions
        let surface_extent = swapchain_config.extent;

        unsafe {
            surface
                .configure_swapchain(device.logical_device(), swapchain_config)
                .expect("Failed to configure swapchain");
        }

        // limit amount of frames calculated simultaneously to avoid the CPU outrunning the GPU
        let frames_in_flight = 3;
        let mut frame_resources = Vec::with_capacity(frames_in_flight);
        for _ in 0..frames_in_flight {
            frame_resources.push(unsafe {
                FrameResources::new(&device, queue_group.family, surface_extent, depth_format)
            });
        }

        let texture_sampler = Self::create_texture_sampler(&device);
        let mut descriptor_pool = Self::create_descriptor_pool(&device, frames_in_flight); // todo: determine the size of the pool somehow
                                                                                           // todo: consider adding associated functions to Material and GameObject to generate the set layout and cache it in Renderer
                                                                                           // alternatively: create MaterialManager and GameObjectManager
                                                                                           // these managers will cache the set layout and they will OWN all the instances of Material/GameObject
                                                                                           // that way, Renderer can retrieve the layouts, the manager will create instances and always ensure the correct layout is used
                                                                                           // and the manager will be able to recycle sets (command pool does not allow freeing individual sets because we don't use the necessary flag)
                                                                                           // alteralternatively: use offsets and don't allocate so many sets: https://www.reddit.com/r/vulkan/comments/g9aeo0/struggling_with_a_descriptor_set_strategy/
        let mvp_descriptor_set_layout = Self::create_mvp_descriptor_set_layout(&device);
        let material_descriptor_set_layout =
            Self::create_material_descriptor_set_layout(&device, &texture_sampler);

        let material_ground = unsafe {
            Material::new(
                &device,
                &mut queue_group,
                &mut frame_resources[0].command_pool,
                &material_descriptor_set_layout,
                &mut descriptor_pool,
                frames_in_flight,
                include_bytes!("../data/textures/Ground037_1K-PNG/Ground037_1K_Color.png"),
            )
        };

        let material_planks = unsafe {
            Material::new(
                &device,
                &mut queue_group,
                &mut frame_resources[0].command_pool,
                &material_descriptor_set_layout,
                &mut descriptor_pool,
                frames_in_flight,
                include_bytes!("../data/textures/Planks016_1K-PNG/Planks016_1K_Color.png"),
            )
        };

        let cube = unsafe {
            GameObject::new(
                &device,
                &mut queue_group,
                &mut frame_resources[0].command_pool,
                &mvp_descriptor_set_layout,
                &mut descriptor_pool,
                frames_in_flight,
                &CUBE_VERTICES,
                &CUBE_INDICES,
                &Similarity3::<f32>::new(Vector3::zeros(), Vector3::zeros(), 1.0),
                material_planks,
            )
        };

        let floor = unsafe {
            GameObject::new(
                &device,
                &mut queue_group,
                &mut frame_resources[0].command_pool,
                &mvp_descriptor_set_layout,
                &mut descriptor_pool,
                frames_in_flight,
                &QUAD_VERTICES,
                &QUAD_INDICES,
                &Similarity3::<f32>::new(
                    Vector3::new(0.0, 110.0, 0.0),
                    Vector3::x() * std::f32::consts::FRAC_PI_2,
                    200.0,
                ),
                material_ground,
            )
        };

        // same pipeline layout can be used between pipelines
        // similar pipeline layouts are advantagous, where the same descriptor set layouts
        // are at the same slots across layouts
        // https://developer.nvidia.com/vulkan-shader-resource-binding
        let pipeline_layout = ManuallyDrop::new(unsafe {
            device
                .logical_device()
                .create_pipeline_layout(
                    vec![
                        &*mvp_descriptor_set_layout,
                        &*material_descriptor_set_layout,
                    ]
                    .into_iter(),
                    iter::empty(),
                )
                .expect("Out of memory creating pipeline layout")
        });

        // shaders are compiled by build script
        let vertex_shader = gfx_auxil::read_spirv(Cursor::new(include_bytes!(
            "../target/data/shaders/03_model_view_projection_3d.vert.spv"
        )))
        .unwrap();
        let fragment_shader = gfx_auxil::read_spirv(Cursor::new(include_bytes!(
            "../target/data/shaders/03_texture_bg.frag.spv"
        )))
        .unwrap();

        let render_pass = Self::create_render_pass(&device, surface_format, depth_format);

        let pipeline = ManuallyDrop::new(unsafe {
            Pipeline::new(
                &device,
                &pipeline_layout,
                Subpass {
                    index: 0,
                    main_pass: &render_pass,
                },
                &vertex_shader,
                &fragment_shader,
                surface_extent,
            )
        });

        // framebuffer connects attachments to images
        // attachments can be thought of as the slots
        // and the images fill those slots.
        // gfx-rs uses imageless framebuffers,
        // we don't have to recreate them every frame.
        let depth_attachment = FramebufferAttachment {
            usage: gfximg::Usage::DEPTH_STENCIL_ATTACHMENT,
            view_caps: ViewCapabilities::empty(),
            format: depth_format,
        };
        let framebuffer = ManuallyDrop::new(unsafe {
            device
                .logical_device()
                .create_framebuffer(
                    &render_pass,
                    vec![framebuffer_attachment, depth_attachment].into_iter(),
                    Extent {
                        width: surface_extent.width,
                        height: surface_extent.height,
                        depth: 1,
                    },
                )
                .unwrap()
        });

        let camera = Camera::new(
            &CameraDesc {
                position: Point3::new(0.0, 0.0, -5.0),
                ..CameraDesc::default()
            },
            surface_extent,
        );

        Renderer {
            camera,
            surface: ManuallyDrop::new(surface),
            surface_format,
            depth_format,
            surface_extent,
            render_pass,
            framebuffer,
            pipeline,
            pipeline_layout,
            frame_resources,
            texture_sampler,
            mvp_descriptor_set_layout,
            material_descriptor_set_layout,
            descriptor_pool,
            game_objects: vec![cube, floor],
            frames_in_flight,
            frame: 0,
            start_time: Instant::now(),
            last_frame: Instant::now(),
            device,
            queue_group,
            instance,
        }
    }

    pub fn recreate_swapchain(&mut self) {
        let capabilities = self.surface.capabilities(self.device.physical_device());
        let swapchain_config =
            SwapchainConfig::from_caps(&capabilities, self.surface_format, self.surface_extent);

        // update surface extent, since the swapchain might enforce different dimensions
        self.surface_extent = swapchain_config.extent;

        unsafe {
            self.device.logical_device().wait_idle().unwrap();

            // destroy resources about to be recreated
            self.device
                .logical_device()
                .destroy_framebuffer(ManuallyDrop::take(&mut self.framebuffer));

            let depth_attachment = FramebufferAttachment {
                usage: gfximg::Usage::DEPTH_STENCIL_ATTACHMENT,
                view_caps: ViewCapabilities::empty(),
                format: self.depth_format,
            };

            // recreate framebuffer
            self.framebuffer = ManuallyDrop::new(
                self.device
                    .logical_device()
                    .create_framebuffer(
                        &self.render_pass,
                        vec![swapchain_config.framebuffer_attachment(), depth_attachment]
                            .into_iter(),
                        self.surface_extent.to_extent(),
                    )
                    .unwrap(),
            );

            // reconfigure swapchain
            self.surface
                .configure_swapchain(self.device.logical_device(), swapchain_config)
                .expect("Failed to configure swapchain");
        }

        // recreate depth buffers
        for single_frame_resources in self.frame_resources.iter_mut() {
            unsafe {
                single_frame_resources.update_dimensions(self.surface_extent, self.depth_format);
            }
        }

        self.pipeline.update_dimensions(self.surface_extent);
        self.camera.set_surface_extent(self.surface_extent);
    }

    pub fn render(&mut self) {
        let frame_id = (self.frame % self.frames_in_flight as u64) as usize;
        let frame_resources = &mut self.frame_resources[frame_id];
        unsafe {
            // wait for the previous submission to finish before resetting command pool and preparing new frame
            let render_timeout_ns = 1_000_000_000;
            self.device
                .logical_device()
                .wait_for_fence(
                    &frame_resources.submission_complete_fence,
                    render_timeout_ns,
                )
                .expect("Out of memory or device lost waiting for fence");
            self.device
                .logical_device()
                .reset_fence(&mut frame_resources.submission_complete_fence)
                .expect("Out of memory resetting fence");
            frame_resources.command_pool.reset(false);
        }

        let surface_image = unsafe {
            let acquire_timeout_ns = 1_000_000_000;
            match self.surface.acquire_image(acquire_timeout_ns) {
                Ok((image, _)) => image,
                Err(_) => {
                    self.recreate_swapchain();
                    return;
                }
            }
        };

        // calculate delta time and update time
        let current_time = Instant::now();
        let delta_time = (current_time - self.last_frame).as_secs_f32();
        self.last_frame = current_time;

        // rotate and move cube with time
        let transform = self.game_objects[0].get_transform();
        transform.isometry.rotation = transform.isometry.rotation
            * UnitQuaternion::from_euler_angles(0.0, -delta_time, delta_time);

        let mut command_buffer = &mut *frame_resources.command_buffer;
        unsafe {
            command_buffer.begin_primary(CommandBufferFlags::ONE_TIME_SUBMIT);
            self.pipeline.bind(&mut command_buffer);

            command_buffer.begin_render_pass(
                &self.render_pass,
                &self.framebuffer,
                self.pipeline.viewport.rect,
                // clear value is part of each render attachment info
                vec![
                    RenderAttachmentInfo {
                        image_view: surface_image.borrow(), // borrow returns borrow of diff type
                        clear_value: ClearValue {
                            color: ClearColor {
                                float32: [0.1, 0.25, 0.8, 1.0],
                            },
                        },
                    },
                    RenderAttachmentInfo {
                        image_view: &*frame_resources.depth_image_view.image_view, // can only borrow SwapchainImage
                        clear_value: ClearValue {
                            depth_stencil: ClearDepthStencil {
                                depth: 1.0,
                                stencil: 0,
                            },
                        },
                    },
                ]
                .into_iter(),
                SubpassContents::Inline,
            );

            // draw all objects
            let view = self.camera.get_view();
            let projection = self.camera.get_projection();
            for game_object in &mut self.game_objects {
                game_object.draw(
                    command_buffer,
                    &self.pipeline_layout,
                    &view,
                    &projection,
                    frame_id,
                )
            }

            command_buffer.end_render_pass();
            command_buffer.finish();

            // we don't need a wait semaphore here, since gfx-rs synchronizes acquire image internally already
            self.queue_group.queues[0].submit(
                iter::once(&*command_buffer),
                iter::empty(),
                iter::once(&*frame_resources.submission_complete_semaphore),
                Some(&mut frame_resources.submission_complete_fence),
            );
            let result = self.queue_group.queues[0].present(
                &mut self.surface,
                surface_image,
                Some(&mut frame_resources.submission_complete_semaphore),
            );

            if result.is_err() {
                self.recreate_swapchain();
            }
        }

        self.frame += 1;
    }

    fn find_surface_format(physical_device: &B::PhysicalDevice, surface: &B::Surface) -> Format {
        // get the formats that the surface and the adapter support together
        let supported_formats = surface.supported_formats(physical_device).unwrap_or(vec![]);
        // if supported_formats is empty, it means we can choose freely
        let default_format = *supported_formats.get(0).unwrap_or(&Format::Rgba8Srgb);
        // try to find an srgb format, otherwise fallback to the first format we found above
        supported_formats
            .into_iter()
            .find(|format| format.base_format().1 == ChannelType::Srgb)
            .unwrap_or(default_format)
    }

    fn find_compatible_format<'a, I: Iterator<Item = &'a Format>>(
        physical_device: &B::PhysicalDevice,
        candidates: I,
        tiling: gfximg::Tiling,
        features: ImageFeature,
    ) -> Option<Format> {
        for candidate in candidates {
            let properties = physical_device.format_properties(Some(*candidate));
            match tiling {
                gfximg::Tiling::Optimal => {
                    if properties.optimal_tiling & features == features {
                        return Some(*candidate);
                    }
                }
                gfximg::Tiling::Linear => {
                    if properties.linear_tiling & features == features {
                        return Some(*candidate);
                    }
                }
            }
        }

        None
    }

    fn find_depth_format(physical_device: &B::PhysicalDevice) -> Format {
        // could probably be rewritten by iterating through all of the available formats and calling
        // is_depth on them
        Self::find_compatible_format(
            physical_device,
            vec![
                Format::D24UnormS8Uint, // nvidia recommends 24bit depth formats https://developer.nvidia.com/blog/vulkan-dos-donts/
                Format::D32Sfloat,
                Format::D32SfloatS8Uint,
            ]
            .iter(),
            gfximg::Tiling::Optimal,
            ImageFeature::DEPTH_STENCIL_ATTACHMENT,
        )
        .expect("Couldn't finad a compatible depth format.")
    }

    fn create_mvp_descriptor_set_layout(
        device: &Rc<Device<B>>,
    ) -> ManuallyDrop<B::DescriptorSetLayout> {
        ManuallyDrop::new(
            unsafe {
                device.logical_device().create_descriptor_set_layout(
                    vec![DescriptorSetLayoutBinding {
                        binding: 0,
                        ty: DescriptorType::Buffer {
                            ty: BufferDescriptorType::Uniform,
                            format: BufferDescriptorFormat::Structured {
                                dynamic_offset: false,
                            },
                        },
                        count: 1,
                        stage_flags: ShaderStageFlags::VERTEX,
                        immutable_samplers: false,
                    }]
                    .into_iter(),
                    iter::empty(),
                )
            }
            .expect("Out of memory creating descriptor set layout"),
        )
    }

    fn create_material_descriptor_set_layout(
        device: &Rc<Device<B>>,
        sampler: &B::Sampler,
    ) -> ManuallyDrop<B::DescriptorSetLayout> {
        ManuallyDrop::new(
            unsafe {
                device.logical_device().create_descriptor_set_layout(
                    vec![DescriptorSetLayoutBinding {
                        binding: 0,
                        ty: DescriptorType::Image {
                            ty: ImageDescriptorType::Sampled { with_sampler: true },
                        },
                        count: 1,
                        stage_flags: ShaderStageFlags::FRAGMENT,
                        immutable_samplers: true,
                    }]
                    .into_iter(),
                    iter::once(sampler),
                )
            }
            .expect("Out of memory creating descriptor set layout"),
        )
    }

    fn create_descriptor_pool(
        device: &Rc<Device<B>>,
        frames_in_flight: usize,
    ) -> ManuallyDrop<B::DescriptorPool> {
        ManuallyDrop::new(
            unsafe {
                device.logical_device().create_descriptor_pool(
                    frames_in_flight * 2 * 2, // total amount of sets we will allocate // todo: determine size in a more elegant way?
                    vec![
                        DescriptorRangeDesc {
                            ty: DescriptorType::Buffer {
                                ty: BufferDescriptorType::Uniform,
                                format: BufferDescriptorFormat::Structured {
                                    dynamic_offset: false,
                                },
                            },
                            // how many descriptors of this type will exist (technically indepependent of max_sets)
                            count: frames_in_flight * 2,
                        },
                        DescriptorRangeDesc {
                            ty: DescriptorType::Image {
                                ty: ImageDescriptorType::Sampled { with_sampler: true },
                            },
                            count: frames_in_flight * 2,
                        },
                    ]
                    .into_iter(),
                    DescriptorPoolCreateFlags::empty(),
                )
            }
            .expect("Out of memory creating descriptor pool"),
        )
    }

    fn create_texture_sampler(device: &Rc<Device<B>>) -> ManuallyDrop<B::Sampler> {
        // this uses no anisotropy, has to be changed if required
        ManuallyDrop::new(
            unsafe {
                device.logical_device().create_sampler(&SamplerDesc::new(
                    gfximg::Filter::Linear,
                    gfximg::WrapMode::Tile,
                ))
            }
            .unwrap(),
        )
    }

    fn create_render_pass(
        device: &Rc<Device<B>>,
        surface_color_format: Format,
        depth_format: Format,
    ) -> ManuallyDrop<B::RenderPass> {
        // https://vulkan-tutorial.com/en/Drawing_a_triangle/Graphics_pipeline_basics/Render_passes
        let color_attachment = Attachment {
            format: Some(surface_color_format),
            samples: 1, // no multi-sampling (yet)
            ops: AttachmentOps::new(AttachmentLoadOp::Clear, AttachmentStoreOp::Store), // clear at start, store at end
            stencil_ops: AttachmentOps::DONT_CARE, // no stencil usage yet
            layouts: Layout::Undefined..Layout::Present, // don't mind layout at the start, but want to present at end
        };

        let depth_attachment = Attachment {
            format: Some(depth_format),
            samples: 1,
            ops: AttachmentOps::new(AttachmentLoadOp::Clear, AttachmentStoreOp::DontCare),
            stencil_ops: AttachmentOps::DONT_CARE,
            layouts: Layout::Undefined..Layout::DepthStencilAttachmentOptimal,
        };

        let subpass = SubpassDesc {
            colors: &[(0, Layout::ColorAttachmentOptimal)],
            depth_stencil: Some(&(1, Layout::DepthStencilAttachmentOptimal)),
            inputs: &[],
            resolves: &[],
            preserves: &[],
        };

        let dependency = SubpassDependency {
            passes: None..Some(0),
            stages: PipelineStage::COLOR_ATTACHMENT_OUTPUT | PipelineStage::EARLY_FRAGMENT_TESTS
                ..PipelineStage::COLOR_ATTACHMENT_OUTPUT | PipelineStage::EARLY_FRAGMENT_TESTS,
            accesses: gfximg::Access::empty()
                ..gfximg::Access::COLOR_ATTACHMENT_WRITE
                    | gfximg::Access::DEPTH_STENCIL_ATTACHMENT_WRITE,
            flags: gfx_mem::Dependencies::empty(),
        };

        ManuallyDrop::new(unsafe {
            device
                .logical_device()
                .create_render_pass(
                    vec![color_attachment, depth_attachment].into_iter(),
                    iter::once(subpass),
                    iter::once(dependency),
                )
                .expect("Out of memory creating render pass")
        })
    }
}

impl<B: gfx_hal::Backend> Drop for Renderer<B> {
    fn drop(&mut self) {
        // wait for device to become idle
        self.device.logical_device().wait_idle().unwrap();

        unsafe {
            // drop per frame resources
            self.frame_resources.clear();

            self.device
                .logical_device()
                .destroy_sampler(ManuallyDrop::take(&mut self.texture_sampler));

            // drop game objects
            self.game_objects.clear();
            ManuallyDrop::drop(&mut self.pipeline);

            self.device
                .logical_device()
                .destroy_descriptor_pool(ManuallyDrop::take(&mut self.descriptor_pool));
            self.device
                .logical_device()
                .destroy_descriptor_set_layout(ManuallyDrop::take(
                    &mut self.mvp_descriptor_set_layout,
                ));
            self.device
                .logical_device()
                .destroy_descriptor_set_layout(ManuallyDrop::take(
                    &mut self.material_descriptor_set_layout,
                ));

            self.device
                .logical_device()
                .destroy_render_pass(ManuallyDrop::take(&mut self.render_pass));
            self.device
                .logical_device()
                .destroy_framebuffer(ManuallyDrop::take(&mut self.framebuffer));
            self.surface
                .unconfigure_swapchain(&self.device.logical_device());
            self.device
                .logical_device()
                .destroy_pipeline_layout(ManuallyDrop::take(&mut self.pipeline_layout));
            self.instance
                .destroy_surface(ManuallyDrop::take(&mut self.surface));

            // at this point, every member in the struct gets dropped in order.
            // however, when a ManuallyDrop is dropped, it's contents it points to won't be dropped!
            // this is correct however, since we already passed ownership of all those values
            // to these destroy_ functions and they then dropped those values for us already.
        }
    }
}

const APPLICATION_NAME: &'static str = "Rust Game Engine";
const DEFAULT_WINDOW_SIZE: [u32; 2] = [512, 512];

#[cfg_attr(rustfmt, rustfmt_skip)]
const CUBE_VERTICES: [Vertex; 24] = [
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
const CUBE_INDICES: [u16; 36] = [
     0,  1,  2,  0,  2,  3, // front
     4,  5,  6,  4,  6,  7, // back
     8,  9, 10,  8, 10, 11, // left
    12, 13, 14, 12, 14, 15, // right
    16, 17, 18, 16, 18, 19, // top
    20, 21, 22, 20, 22, 23, // bottom
];

#[cfg_attr(rustfmt, rustfmt_skip)]
const QUAD_VERTICES: [Vertex; 4] = [
    Vertex { position: [-0.5, -0.5,  0.5], color: [1.0, 0.0, 0.0], uv: [0.0, 0.0] },
    Vertex { position: [-0.5,  0.5,  0.5], color: [0.0, 1.0, 0.0], uv: [0.0, 40.0] },
    Vertex { position: [ 0.5,  0.5,  0.5], color: [0.0, 0.0, 1.0], uv: [40.0, 40.0] },
    Vertex { position: [ 0.5, -0.5,  0.5], color: [1.0, 1.0, 1.0], uv: [40.0, 0.0] },
];

// these indices must be counter-clockwise for backface culling to work
#[cfg_attr(rustfmt, rustfmt_skip)]
const QUAD_INDICES: [u16; 6] = [
     0,  1,  2,  0,  2,  3,
];

fn main() {
    let event_loop = winit::event_loop::EventLoop::new();
    let logical_window_size: LogicalSize<u32> = DEFAULT_WINDOW_SIZE.into();

    let window = winit::window::WindowBuilder::new()
        .with_title(APPLICATION_NAME)
        .with_inner_size(logical_window_size)
        .build(&event_loop)
        .expect("Failed to create window");

    window.set_cursor_grab(true).unwrap();
    window.set_cursor_visible(false);

    let instance = backend::Instance::create(APPLICATION_NAME, 1).expect("Backend not supported");

    // surface creation is unsafe because raw window handle might be invalid pointer
    let surface = unsafe {
        instance
            .create_surface(&window)
            .expect("Failed to create surface for window")
    };

    let adapter: Adapter<backend::Backend> = find_adapter(&instance);

    let mut renderer = Renderer::new(instance, surface, adapter);

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
            WindowEvent::Resized(dimensions) => {
                renderer.surface_extent = Extent2D {
                    width: dimensions.width,
                    height: dimensions.height,
                };
                renderer.recreate_swapchain();
            }
            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                renderer.surface_extent = Extent2D {
                    width: new_inner_size.width,
                    height: new_inner_size.height,
                };
                renderer.recreate_swapchain();
            }
            WindowEvent::KeyboardInput { input, .. } => match input {
                KeyboardInput {
                    state: ElementState::Pressed,
                    virtual_keycode: Some(key),
                    ..
                } => match key {
                    // todo: build key tracking input system around it? DeviceEvent::Key instead maybe?
                    VirtualKeyCode::W => renderer.camera.translate(0.0, 1.0),
                    VirtualKeyCode::A => renderer.camera.translate(-1.0, 0.0),
                    VirtualKeyCode::S => renderer.camera.translate(0.0, -1.0),
                    VirtualKeyCode::D => renderer.camera.translate(1.0, 0.0),
                    VirtualKeyCode::Escape => *control_flow = ControlFlow::Exit,
                    _ => (),
                },
                _ => (),
            },
            _ => (),
        },
        Event::DeviceEvent { event, .. } => match event {
            DeviceEvent::MouseMotion { delta } => {
                renderer.camera.rotate(delta.0 as f32, delta.1 as f32)
            }
            _ => (),
        },
        Event::MainEventsCleared => window.request_redraw(),
        Event::RedrawRequested(_) => {
            renderer.render();
        }
        _ => (),
    });
}

// todo: this might be needed again for proper dpi handling
// fn get_window_sizes<T>(event_loop: &EventLoop<T>) -> (LogicalSize<u32>, PhysicalSize<u32>) {
//     // get either the primary monitor's dpi or the
//     // monitor with the highest dpi (on Wayland for example)
//     let primary_monitor = event_loop.primary_monitor();
//     let dpi = match primary_monitor {
//         Some(primary_monitor) => primary_monitor.scale_factor(),
//         None => event_loop
//             .available_monitors()
//             .map(|monitor| monitor.scale_factor())
//             .max_by(|a, b| a.partial_cmp(b).unwrap())
//             .unwrap_or(0.0),
//     };

//     let logical: LogicalSize<u32> = DEFAULT_WINDOW_SIZE.into();
//     let physical: PhysicalSize<u32> = logical.to_physical(dpi);

//     (logical, physical)
// }

fn find_adapter<B: gfx_hal::Backend>(instance: &B::Instance) -> Adapter<B> {
    // multiple adapters might exist, like an internal and a dedicated GPU
    // an algorithm could be implemented here to select a different GPU
    instance.enumerate_adapters().remove(0)
}
