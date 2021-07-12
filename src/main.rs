mod camera;

use std::{
    borrow::Borrow,
    io::Cursor,
    iter,
    mem::{size_of, ManuallyDrop},
    ops::Range,
    ptr,
    time::Instant,
};

use gfx_hal::{
    adapter::{Adapter, MemoryType, PhysicalDevice},
    buffer::{self, SubRange},
    command::{
        BufferCopy, BufferImageCopy, ClearColor, ClearDepthStencil, ClearValue, CommandBuffer,
        CommandBufferFlags, Level, RenderAttachmentInfo, SubpassContents,
    },
    device::Device,
    format::{Aspects, ChannelType, Format, ImageFeature, Swizzle},
    image::{
        self as gfximg, Extent, FramebufferAttachment, Layout, SamplerDesc, SubresourceLayers,
        SubresourceRange, ViewCapabilities, ViewKind,
    },
    memory::{self, Segment, SparseFlags},
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
    queue::{Queue, QueueFamily, QueueGroup},
    window::{Extent2D, PresentationSurface, Surface, SwapchainConfig},
    IndexType, Instance, MemoryTypeId,
};

use nalgebra::{Matrix4, Point3, Vector3};

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

struct ModelTransform<B: gfx_hal::Backend> {
    model: Matrix4<f32>,
    buffer: Buffer<B>,
}

struct ViewProjectionTransform<B: gfx_hal::Backend> {
    view: Matrix4<f32>,
    projection: Matrix4<f32>,
    buffer: Buffer<B>,
}

struct Buffer<B: gfx_hal::Backend> {
    buffer: ManuallyDrop<B::Buffer>,
    memory: ManuallyDrop<B::Memory>,
}

struct Image<B: gfx_hal::Backend> {
    image: ManuallyDrop<B::Image>,
    memory: ManuallyDrop<B::Memory>,
}

struct Texture<B: gfx_hal::Backend> {
    image: Image<B>,
    image_view: ManuallyDrop<B::ImageView>,
    sampler: ManuallyDrop<B::Sampler>, // TODO: make this sharable between textures
}

struct Model<B: gfx_hal::Backend> {
    vertex_buffer: Buffer<B>,
    index_buffer: Buffer<B>,
    model_transforms: Vec<ModelTransform<B>>, // unique, per frame in flight
    view_projection_transform: Vec<ViewProjectionTransform<B>>, // shared, per frame in flight
}

struct Material<B: gfx_hal::Backend> {
    descriptor_sets: Vec<B::DescriptorSet>, // dependent on frames in flight
                                            // todo: different shader pipelines
}

struct Renderer<B: gfx_hal::Backend> {
    camera: Camera,
    surface: ManuallyDrop<B::Surface>,
    surface_format: Format,
    depth_format: Format,
    surface_extent: Extent2D,
    viewport: Viewport,
    render_pass: ManuallyDrop<B::RenderPass>,
    framebuffer: ManuallyDrop<B::Framebuffer>,
    pipeline: ManuallyDrop<B::GraphicsPipeline>,
    pipeline_layout: ManuallyDrop<B::PipelineLayout>,
    //pipeline_cache: ManuallyDrop<B::PipelineCache> // todo: should I add this?
    submission_complete_semaphores: Vec<B::Semaphore>,
    submission_complete_fences: Vec<B::Fence>,
    command_pools: Vec<B::CommandPool>,
    command_buffers: Vec<B::CommandBuffer>,
    //texture_image_view: ManuallyDrop<B::ImageView>,
    depth_image_view: ManuallyDrop<B::ImageView>,
    //texture_sampler: ManuallyDrop<B::Sampler>,
    vertex_buffer: ManuallyDrop<B::Buffer>,
    index_buffer: ManuallyDrop<B::Buffer>,
    uniform_buffers: Vec<ManuallyDrop<B::Buffer>>,
    floor_vertex_buffer: ManuallyDrop<B::Buffer>,
    floor_index_buffer: ManuallyDrop<B::Buffer>,
    floor_uniform_buffers: Vec<ManuallyDrop<B::Buffer>>,
    //texture_image: ManuallyDrop<B::Image>,
    texture: Texture<B>,
    depth_image: ManuallyDrop<B::Image>, //todo: per frame in flight
    vertex_buffer_memory: ManuallyDrop<B::Memory>,
    index_buffer_memory: ManuallyDrop<B::Memory>,
    uniform_buffers_memory: Vec<ManuallyDrop<B::Memory>>,
    floor_vertex_buffer_memory: ManuallyDrop<B::Memory>,
    floor_index_buffer_memory: ManuallyDrop<B::Memory>,
    floor_uniform_buffers_memory: Vec<ManuallyDrop<B::Memory>>,
    //texture_memory: ManuallyDrop<B::Memory>,
    depth_image_memory: ManuallyDrop<B::Memory>,
    descriptor_set_layout: ManuallyDrop<B::DescriptorSetLayout>,
    descriptor_pool: ManuallyDrop<B::DescriptorPool>,
    descriptor_sets: Vec<B::DescriptorSet>,
    floor_descriptor_sets: Vec<B::DescriptorSet>,
    frames_in_flight: usize,
    frame: u64,
    start_time: Instant,
    // these are dropped in the order they are declared, very important!
    device: B::Device,
    adapter: Adapter<B>,
    queue_group: QueueGroup<B>,
    instance: B::Instance,
}

impl<B: gfx_hal::Backend> Renderer<B> {
    pub fn new(instance: B::Instance, mut surface: B::Surface, adapter: Adapter<B>) -> Renderer<B> {
        let physical_device = &adapter.physical_device;
        let limits = physical_device.properties().limits;
        let memory_types = physical_device.memory_properties().memory_types;
        // in the future I might also want to consider the memory_heaps field in the memory properties
        // this will be useful to determine wheter the type belongs to the GPUs VRAM or might be slow system RAM

        let queue_family = Self::find_queue_family(&adapter, &surface);

        let mut gpu = unsafe {
            physical_device
                .open(&[(queue_family, &[1.0])], gfx_hal::Features::empty())
                .expect("Failed to open device")
        };

        let device = gpu.device;
        let mut queue_group = gpu.queue_groups.pop().unwrap();

        // limit amount of frames calculated simultaneously to avoid the CPU outrunning the GPU
        let frames_in_flight = 3;

        let mut command_pools = Vec::with_capacity(frames_in_flight);
        let mut command_buffers = Vec::with_capacity(frames_in_flight);
        let mut submission_complete_semaphores = Vec::with_capacity(frames_in_flight);
        let mut submission_complete_fences = Vec::with_capacity(frames_in_flight);

        for _ in 0..frames_in_flight {
            unsafe {
                // a command pool allocates command buffers and is supposed to be used from a single thread only
                let mut command_pool = device
                    .create_command_pool(queue_group.family, CommandPoolCreateFlags::empty())
                    .expect("Out of memory creating command pool");

                command_buffers.push(command_pool.allocate_one(Level::Primary));
                command_pools.push(command_pool);
                submission_complete_semaphores.push(
                    device
                        .create_semaphore()
                        .expect("Out of memory creating semaphore"),
                );
                submission_complete_fences.push(
                    device
                        .create_fence(true)
                        .expect("Out of memory creating fence"),
                );
            }
        }

        let surface_format = Self::find_surface_format(physical_device, &surface);
        let depth_format = Self::find_depth_format(physical_device);

        let default_surface_extent = Extent2D {
            width: DEFAULT_WINDOW_SIZE[0],
            height: DEFAULT_WINDOW_SIZE[1],
        };
        let capabilities = surface.capabilities(physical_device);
        let swapchain_config =
            SwapchainConfig::from_caps(&capabilities, surface_format, default_surface_extent);
        let framebuffer_attachment = swapchain_config.framebuffer_attachment();

        // update surface extent, since the swapchain might enforce different dimensions
        let surface_extent = swapchain_config.extent;

        let viewport = Viewport {
            rect: Rect {
                x: 0,
                y: 0,
                w: surface_extent.width as i16,
                h: surface_extent.height as i16,
            },
            depth: 0.0..1.0,
        };

        unsafe {
            surface
                .configure_swapchain(&device, swapchain_config)
                .expect("Failed to configure swapchain");
        }

        let image = image::load(
            Cursor::new(include_bytes!(
                "../data/textures/Ground037_1K-PNG/Ground037_1K_Color.png"
            )),
            image::ImageFormat::Png,
        )
        .unwrap()
        .to_rgba8();
        // let (texture_image, texture_memory) = Self::create_texture_image(
        //     &device,
        //     &mut queue_group,
        //     &mut command_pools[0],
        //     &limits,
        //     &memory_types,
        //     &image,
        // );

        // let texture_image_view = Self::create_image_view(
        //     &device,
        //     &texture_image,
        //     Format::Rgba8Srgb,
        //     Aspects::COLOR,
        //     gfximg::Usage::SAMPLED,
        // );
        // let texture_sampler = Self::create_texture_sampler(&device);

        let texture = Self::create_texture(
            &device,
            &mut queue_group,
            &mut command_pools[0],
            &limits,
            &memory_types,
            &image,
        );

        let descriptor_set_layout = Self::create_descriptor_set_layout(&device);
        let (uniform_buffers, uniform_buffers_memory) =
            Self::create_uniform_buffers(&device, &limits, &memory_types, frames_in_flight);
        let mut descriptor_pool = Self::create_descriptor_pool(&device, frames_in_flight);
        let descriptor_sets = Self::create_descriptor_sets(
            &device,
            &descriptor_set_layout,
            &mut descriptor_pool,
            &uniform_buffers,
            &texture.image_view, // todo: make single arg
            &texture.sampler,
            frames_in_flight,
        );

        let (depth_image, depth_image_memory) = Self::create_image(
            &device,
            &memory_types,
            surface_extent.width,
            surface_extent.height,
            depth_format,
            gfximg::Tiling::Optimal,
            gfximg::Usage::DEPTH_STENCIL_ATTACHMENT,
            memory::Properties::DEVICE_LOCAL,
        );

        let depth_image_view = Self::create_image_view(
            &device,
            &depth_image,
            depth_format,
            Aspects::DEPTH, // todo: stencil aspect
            gfximg::Usage::DEPTH_STENCIL_ATTACHMENT,
        );

        let render_pass = Self::create_render_pass(&device, surface_format, depth_format);

        let (vertex_buffer, vertex_buffer_memory) = Self::create_vertex_buffer(
            &device,
            &mut queue_group,
            &mut command_pools[0],
            &limits,
            &memory_types,
            &CUBE_VERTICES,
        );

        let (index_buffer, index_buffer_memory) = Self::create_index_buffer(
            &device,
            &mut queue_group,
            &mut command_pools[0],
            &limits,
            &memory_types,
            &CUBE_INDICES,
        );

        let (floor_vertex_buffer, floor_vertex_buffer_memory) = Self::create_vertex_buffer(
            &device,
            &mut queue_group,
            &mut command_pools[0],
            &limits,
            &memory_types,
            &QUAD_VERTICES,
        );
        let (floor_index_buffer, floor_index_buffer_memory) = Self::create_index_buffer(
            &device,
            &mut queue_group,
            &mut command_pools[0],
            &limits,
            &memory_types,
            &QUAD_INDICES,
        );
        let (floor_uniform_buffers, floor_uniform_buffers_memory) =
            Self::create_uniform_buffers(&device, &limits, &memory_types, frames_in_flight);
        let floor_descriptor_sets = Self::create_descriptor_sets(
            &device,
            &descriptor_set_layout,
            &mut descriptor_pool,
            &floor_uniform_buffers,
            &texture.image_view, // todo: make single arg
            &texture.sampler,
            frames_in_flight,
        );

        let pipeline_layout = ManuallyDrop::new(unsafe {
            device
                .create_pipeline_layout(iter::once(&*descriptor_set_layout), iter::empty())
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

        let pipeline = unsafe {
            Self::make_pipeline(
                &device,
                &render_pass,
                &*pipeline_layout,
                &vertex_shader,
                &fragment_shader,
            )
        };

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
            viewport,
            render_pass,
            framebuffer,
            pipeline,
            pipeline_layout,
            submission_complete_semaphores,
            submission_complete_fences,
            command_pools,
            command_buffers,
            //texture_image_view,
            texture,
            depth_image_view,
            //texture_sampler,
            vertex_buffer,
            index_buffer,
            uniform_buffers,
            floor_vertex_buffer,
            floor_index_buffer,
            floor_uniform_buffers,
            //texture_image,
            depth_image,
            vertex_buffer_memory,
            index_buffer_memory,
            uniform_buffers_memory,
            floor_vertex_buffer_memory,
            floor_index_buffer_memory,
            floor_uniform_buffers_memory,
            //texture_memory,
            depth_image_memory,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
            floor_descriptor_sets,
            frames_in_flight,
            frame: 0,
            start_time: Instant::now(),
            device,
            adapter,
            queue_group,
            instance,
        }
    }

    pub fn recreate_swapchain(&mut self) {
        let capabilities = self.surface.capabilities(&self.adapter.physical_device);
        let swapchain_config =
            SwapchainConfig::from_caps(&capabilities, self.surface_format, self.surface_extent);

        // update surface extent, since the swapchain might enforce different dimensions
        self.surface_extent = swapchain_config.extent;

        self.viewport.rect.w = self.surface_extent.width as i16;
        self.viewport.rect.h = self.surface_extent.height as i16;

        unsafe {
            self.device.wait_idle().unwrap();

            // destroy resources about to be recreated
            self.device
                .destroy_framebuffer(ManuallyDrop::take(&mut self.framebuffer));
            self.device
                .destroy_image_view(ManuallyDrop::take(&mut self.depth_image_view));
            self.device
                .destroy_image(ManuallyDrop::take(&mut self.depth_image));
            self.device
                .free_memory(ManuallyDrop::take(&mut self.depth_image_memory));

            let depth_attachment = FramebufferAttachment {
                usage: gfximg::Usage::DEPTH_STENCIL_ATTACHMENT,
                view_caps: ViewCapabilities::empty(),
                format: self.depth_format,
            };

            // recreate framebuffer
            self.framebuffer = ManuallyDrop::new(
                self.device
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
                .configure_swapchain(&self.device, swapchain_config)
                .expect("Failed to configure swapchain");
        }

        // recreate depth buffer
        let (depth_image, depth_image_memory) = Self::create_image(
            &self.device,
            &self
                .adapter
                .physical_device
                .memory_properties()
                .memory_types,
            self.surface_extent.width,
            self.surface_extent.height,
            self.depth_format,
            gfximg::Tiling::Optimal,
            gfximg::Usage::DEPTH_STENCIL_ATTACHMENT,
            memory::Properties::DEVICE_LOCAL,
        );

        let depth_image_view = Self::create_image_view(
            &self.device,
            &depth_image,
            self.depth_format,
            Aspects::DEPTH, // todo: stencil aspect
            gfximg::Usage::DEPTH_STENCIL_ATTACHMENT,
        );

        self.depth_image = depth_image;
        self.depth_image_memory = depth_image_memory;
        self.depth_image_view = depth_image_view;

        self.camera.set_surface_extent(self.surface_extent);
    }

    pub fn render(&mut self) {
        let frame_id = (self.frame % self.frames_in_flight as u64) as usize;
        unsafe {
            // wait for the previous submission to finish before resetting command pool and preparing new frame
            let render_timeout_ns = 1_000_000_000;
            self.device
                .wait_for_fence(
                    &self.submission_complete_fences[frame_id],
                    render_timeout_ns,
                )
                .expect("Out of memory or device lost waiting for fence");
            self.device
                .reset_fence(&mut self.submission_complete_fences[frame_id])
                .expect("Out of memory resetting fence");
            self.command_pools[frame_id].reset(false);
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

        // rotate and move model with time
        let time = self.start_time.elapsed().as_secs_f32();
        let model = Matrix4::<f32>::new_translation(&Vector3::<f32>::new(
            (time * 0.5).sin() * 5.0,
            (time * 0.7).sin() * 4.0,
            (time * 0.8).sin() * 4.0,
        )) * Matrix4::<f32>::from_euler_angles(0.0, -time, time);

        let view = self.camera.get_view();
        let projection = self.camera.get_projection();

        let transformations = Transformations {
            model,
            view,
            projection,
        };

        // copy data into uniform buffer
        unsafe {
            Self::copy_into_memory(
                &self.device,
                &transformations as *const Transformations as *const u8,
                &mut self.uniform_buffers_memory[frame_id],
                size_of::<Transformations>(),
            )
        };

        let command_buffer = &mut self.command_buffers[frame_id];
        unsafe {
            command_buffer.begin_primary(CommandBufferFlags::ONE_TIME_SUBMIT);
            // viewports and scissors are part of the command buffer, different to Vulkan
            command_buffer.set_viewports(0, iter::once(self.viewport.clone()));
            command_buffer.set_scissors(0, iter::once(self.viewport.rect));

            command_buffer.bind_graphics_pipeline(&self.pipeline);
            command_buffer
                .bind_vertex_buffers(0, iter::once((&*self.vertex_buffer, SubRange::WHOLE)));
            command_buffer.bind_index_buffer(&*self.index_buffer, SubRange::WHOLE, IndexType::U16);
            command_buffer.bind_graphics_descriptor_sets(
                &self.pipeline_layout,
                0,
                iter::once(&self.descriptor_sets[frame_id]),
                iter::empty(),
            );

            command_buffer.begin_render_pass(
                &self.render_pass,
                &self.framebuffer,
                self.viewport.rect,
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
                        image_view: &*self.depth_image_view, // borrow only works for SwapchainImage
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
            command_buffer.draw_indexed(0..CUBE_INDICES.len() as u32, 0, 0..1);
            // render a second model, the floor
            {
                let model = Matrix4::<f32>::new_translation(&Vector3::new(0.0, 110.0, 0.0))
                    * Matrix4::<f32>::from_euler_angles(std::f32::consts::PI / 2.0, 0.0, 0.0)
                    * Matrix4::<f32>::new_scaling(200.0);

                // TODO: create a global descriptor set with view and projection and one model specific set for every model each
                let transformations = Transformations {
                    model,
                    view,
                    projection,
                };

                Self::copy_into_memory(
                    &self.device,
                    &transformations as *const Transformations as *const u8,
                    &mut self.floor_uniform_buffers_memory[frame_id],
                    size_of::<Transformations>(),
                );

                command_buffer.bind_vertex_buffers(
                    0,
                    iter::once((&*self.floor_vertex_buffer, SubRange::WHOLE)),
                );
                command_buffer.bind_index_buffer(
                    &*self.floor_index_buffer,
                    SubRange::WHOLE,
                    IndexType::U16,
                );
                command_buffer.bind_graphics_descriptor_sets(
                    &self.pipeline_layout,
                    0,
                    iter::once(&self.floor_descriptor_sets[frame_id]),
                    iter::empty(),
                );

                command_buffer.draw_indexed(0..QUAD_INDICES.len() as u32, 0, 0..1);
            }

            command_buffer.end_render_pass();
            command_buffer.finish();

            // we don't need a wait semaphore here, since gfx-rs synchronizes acquire image internally already
            self.queue_group.queues[0].submit(
                iter::once(&*command_buffer),
                iter::empty(),
                iter::once(&self.submission_complete_semaphores[frame_id]),
                Some(&mut self.submission_complete_fences[frame_id]),
            );
            let result = self.queue_group.queues[0].present(
                &mut self.surface,
                surface_image,
                Some(&mut self.submission_complete_semaphores[frame_id]),
            );

            if result.is_err() {
                self.recreate_swapchain();
            }
        }

        self.frame += 1;
    }

    fn find_queue_family<'a>(adapter: &'a Adapter<B>, surface: &B::Surface) -> &'a B::QueueFamily {
        adapter
            .queue_families
            .iter()
            .find(|family| {
                surface.supports_queue_family(family) && family.queue_type().supports_graphics()
            })
            .expect("No compatible queue family found")
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
                Format::D32Sfloat,
                Format::D24UnormS8Uint,
                Format::D32SfloatS8Uint,
            ]
            .iter(),
            gfximg::Tiling::Optimal,
            ImageFeature::DEPTH_STENCIL_ATTACHMENT,
        )
        .expect("Couldn't finad a compatible depth format.")
    }

    fn create_buffer(
        device: &B::Device,
        memory_types: &[MemoryType],
        buffer_len: u64,
        usage: buffer::Usage,
        properties: memory::Properties,
    ) -> (ManuallyDrop<B::Buffer>, ManuallyDrop<B::Memory>) {
        let mut buffer = ManuallyDrop::new(
            unsafe { device.create_buffer(buffer_len, usage, SparseFlags::empty()) }.unwrap(),
        );

        let buffer_requirements = unsafe { device.get_buffer_requirements(&buffer) };

        let buffer_memory_type =
            Self::find_memory_type(&memory_types, buffer_requirements.type_mask, properties);

        let memory = unsafe {
            let memory = device
                .allocate_memory(buffer_memory_type, buffer_requirements.size)
                .unwrap();
            device.bind_buffer_memory(&memory, 0, &mut buffer).unwrap();
            ManuallyDrop::new(memory)
        };

        (buffer, memory)
    }

    fn create_image(
        device: &B::Device,
        memory_types: &[MemoryType],
        width: u32,
        height: u32,
        format: Format,
        tiling: gfximg::Tiling,
        usage: gfximg::Usage,
        properties: memory::Properties,
    ) -> (ManuallyDrop<B::Image>, ManuallyDrop<B::Memory>) {
        let mut image = ManuallyDrop::new(
            unsafe {
                device.create_image(
                    gfximg::Kind::D2(width, height, 1, 1),
                    1,
                    format,
                    tiling,
                    usage,
                    SparseFlags::empty(),
                    ViewCapabilities::empty(),
                )
            }
            .unwrap(),
        );

        let image_requirements = unsafe { device.get_image_requirements(&image) };
        let memory_type =
            Self::find_memory_type(&memory_types, image_requirements.type_mask, properties);
        let memory = unsafe {
            let memory = device
                .allocate_memory(memory_type, image_requirements.size)
                .unwrap();
            device.bind_image_memory(&memory, 0, &mut image).unwrap();
            ManuallyDrop::new(memory)
        };

        (image, memory)
    }

    fn transition_image_layout(
        device: &B::Device,
        queue_group: &mut QueueGroup<B>,
        command_pool: &mut B::CommandPool,
        image: &B::Image,
        format: Format,
        layouts: Range<Layout>,
    ) {
        let mut command_fence = device
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
                memory::Dependencies::empty(),
                iter::once(memory::Barrier::Image {
                    // our data starts with those access flags and this layout
                    // and transitions to the latter ones in the range
                    states: (accesses.start, layouts.start)..(accesses.end, layouts.end),
                    target: image,
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

            device.wait_for_fence(&command_fence, !0).unwrap();
            command_pool.free(iter::once(command_buffer));
            device.destroy_fence(command_fence);
        }
    }

    fn copy_buffer_to_image(
        device: &B::Device,
        queue_group: &mut QueueGroup<B>,
        source: &B::Buffer,
        destination: &B::Image,
        command_pool: &mut B::CommandPool,
        width: u32,
        height: u32,
    ) {
        let mut copy_fence = device
            .create_fence(false)
            .expect("Out of memory creating fence");

        unsafe {
            let mut command_buffer = command_pool.allocate_one(Level::Primary);
            command_buffer.begin_primary(CommandBufferFlags::ONE_TIME_SUBMIT);

            command_buffer.copy_buffer_to_image(
                &source,
                &destination,
                Layout::TransferDstOptimal,
                iter::once(BufferImageCopy {
                    buffer_offset: 0,
                    buffer_width: width,
                    buffer_height: height,
                    image_layers: SubresourceLayers {
                        aspects: Aspects::COLOR,
                        level: 0,
                        layers: 0..1,
                    },
                    image_offset: gfximg::Offset::ZERO,
                    image_extent: Extent {
                        width: width,
                        height: height,
                        depth: 1,
                    },
                }),
            );
            command_buffer.finish();
            queue_group.queues[0].submit(
                iter::once(&command_buffer),
                iter::empty(),
                iter::empty(),
                Some(&mut copy_fence),
            );

            device.wait_for_fence(&copy_fence, !0).unwrap();
            command_pool.free(iter::once(command_buffer));
            device.destroy_fence(copy_fence);
        }
    }

    /// Copies into a CPU visible buffer, undefined behavior if memory is not CPU visible!
    unsafe fn copy_into_memory(
        device: &B::Device,
        source: *const u8,
        destination: &mut B::Memory,
        len: usize,
    ) {
        let mapping = device.map_memory(destination, Segment::ALL).unwrap();
        ptr::copy_nonoverlapping(source, mapping, len);

        // manually flush the memory, so that it is ensured to be available to the GPU.
        // if instead limited buffer_memory_type to a type that contains the property
        // memory::Properties::COHERENT, we wouldn't have to explicitly flush, possible
        // optimization for the future.
        // either way however, the memory is not guaranteed to be visible on the GPU yet
        // until we submit our queue, according to the specification.
        device
            .flush_mapped_memory_ranges(iter::once((&*destination, Segment::ALL)))
            .unwrap();

        device.unmap_memory(destination);
    }

    fn copy_buffer(
        device: &B::Device,
        queue_group: &mut QueueGroup<B>,
        source: &B::Buffer,
        destination: &B::Buffer,
        command_pool: &mut B::CommandPool,
        len: u64,
    ) {
        let mut copy_fence = device
            .create_fence(false)
            .expect("Out of memory creating fence");

        unsafe {
            let mut command_buffer = command_pool.allocate_one(Level::Primary);
            command_buffer.begin_primary(CommandBufferFlags::ONE_TIME_SUBMIT);
            command_buffer.copy_buffer(
                &*source,
                &*destination,
                iter::once(BufferCopy {
                    src: 0,
                    dst: 0,
                    size: len as u64,
                }),
            );
            command_buffer.finish();
            queue_group.queues[0].submit(
                iter::once(&command_buffer),
                iter::empty(),
                iter::empty(),
                Some(&mut copy_fence),
            );

            device.wait_for_fence(&copy_fence, !0).unwrap();
            command_pool.free(iter::once(command_buffer));
            device.destroy_fence(copy_fence);
        }
    }

    fn create_vertex_buffer(
        device: &B::Device,
        queue_group: &mut QueueGroup<B>,
        command_pool: &mut B::CommandPool,
        limits: &gfx_hal::Limits,
        memory_types: &[MemoryType],
        vertices: &[Vertex],
    ) -> (ManuallyDrop<B::Buffer>, ManuallyDrop<B::Memory>) {
        let buffer_stride = size_of::<Vertex>();
        let buffer_len = vertices.len() * buffer_stride;
        // buffer needs to have size that's a multiple of non_coherent_atom_size for it to be mapped
        let padded_buffer_len = ((buffer_len + limits.non_coherent_atom_size - 1)
            / limits.non_coherent_atom_size
            * limits.non_coherent_atom_size) as u64;

        let (staging_buffer, mut staging_memory) = Self::create_buffer(
            device,
            memory_types,
            padded_buffer_len,
            buffer::Usage::TRANSFER_SRC,
            memory::Properties::CPU_VISIBLE,
        );

        // copy data into staging buffer
        unsafe {
            Self::copy_into_memory(
                device,
                vertices.as_ptr() as *const u8,
                &mut staging_memory,
                buffer_len,
            )
        };

        let (buffer, memory) = Self::create_buffer(
            device,
            memory_types,
            buffer_len as u64,
            buffer::Usage::TRANSFER_DST | buffer::Usage::VERTEX,
            memory::Properties::DEVICE_LOCAL,
        );

        // copy staging buffer into final buffer
        Self::copy_buffer(
            device,
            queue_group,
            &staging_buffer,
            &buffer,
            command_pool,
            buffer_len as u64,
        );

        unsafe {
            device.destroy_buffer(ManuallyDrop::into_inner(staging_buffer));
            device.free_memory(ManuallyDrop::into_inner(staging_memory));
        }

        (buffer, memory)
    }

    fn create_index_buffer(
        device: &B::Device,
        queue_group: &mut QueueGroup<B>,
        command_pool: &mut B::CommandPool,
        limits: &gfx_hal::Limits,
        memory_types: &[MemoryType],
        indices: &[u16],
    ) -> (ManuallyDrop<B::Buffer>, ManuallyDrop<B::Memory>) {
        let buffer_stride = size_of::<u16>();
        let buffer_len = indices.len() * buffer_stride;
        let padded_buffer_len = ((buffer_len + limits.non_coherent_atom_size - 1)
            / limits.non_coherent_atom_size
            * limits.non_coherent_atom_size) as u64;

        let (staging_buffer, mut staging_memory) = Self::create_buffer(
            device,
            memory_types,
            padded_buffer_len,
            buffer::Usage::TRANSFER_SRC,
            memory::Properties::CPU_VISIBLE,
        );

        unsafe {
            Self::copy_into_memory(
                device,
                indices.as_ptr() as *const u8,
                &mut staging_memory,
                buffer_len,
            )
        };

        let (buffer, memory) = Self::create_buffer(
            device,
            memory_types,
            buffer_len as u64,
            buffer::Usage::TRANSFER_DST | buffer::Usage::INDEX,
            memory::Properties::DEVICE_LOCAL,
        );

        Self::copy_buffer(
            device,
            queue_group,
            &staging_buffer,
            &buffer,
            command_pool,
            buffer_len as u64,
        );

        unsafe {
            device.destroy_buffer(ManuallyDrop::into_inner(staging_buffer));
            device.free_memory(ManuallyDrop::into_inner(staging_memory));
        }

        (buffer, memory)
    }

    fn create_descriptor_set_layout(device: &B::Device) -> ManuallyDrop<B::DescriptorSetLayout> {
        ManuallyDrop::new(
            unsafe {
                device.create_descriptor_set_layout(
                    vec![
                        DescriptorSetLayoutBinding {
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
                        },
                        DescriptorSetLayoutBinding {
                            binding: 1,
                            ty: DescriptorType::Image {
                                ty: ImageDescriptorType::Sampled { with_sampler: true },
                            },
                            count: 1,
                            stage_flags: ShaderStageFlags::FRAGMENT,
                            immutable_samplers: false,
                        },
                    ]
                    .into_iter(),
                    iter::empty(),
                )
            }
            .expect("Out of memory creating descriptor set layout"),
        )
    }

    fn create_uniform_buffers(
        device: &B::Device,
        limits: &gfx_hal::Limits,
        memory_types: &[MemoryType],
        frames_in_flight: usize,
    ) -> (Vec<ManuallyDrop<B::Buffer>>, Vec<ManuallyDrop<B::Memory>>) {
        let mut uniform_buffers = Vec::with_capacity(frames_in_flight);
        let mut uniform_buffers_memory = Vec::with_capacity(frames_in_flight);
        let padded_length = ((size_of::<Transformations>() + limits.non_coherent_atom_size - 1)
            / limits.non_coherent_atom_size
            * limits.non_coherent_atom_size) as u64;
        for _ in 0..frames_in_flight {
            let (uniform_buffer, uniform_buffer_memory) = Self::create_buffer(
                &device,
                &memory_types,
                padded_length,
                buffer::Usage::UNIFORM,
                memory::Properties::CPU_VISIBLE,
            );
            uniform_buffers.push(uniform_buffer);
            uniform_buffers_memory.push(uniform_buffer_memory);
        }

        (uniform_buffers, uniform_buffers_memory)
    }

    fn create_descriptor_pool(
        device: &B::Device,
        frames_in_flight: usize,
    ) -> ManuallyDrop<B::DescriptorPool> {
        // max_sets and count are only equal because every descriptor set contains one of each descriptor at the moment
        // this can change in the future and does not have to be equal
        ManuallyDrop::new(
            unsafe {
                device.create_descriptor_pool(
                    frames_in_flight * 2, // total amount of sets we will allocate // todo: make nice
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

    fn create_descriptor_sets(
        device: &B::Device,
        descriptor_set_layout: &B::DescriptorSetLayout,
        descriptor_pool: &mut B::DescriptorPool,
        uniform_buffers: &Vec<ManuallyDrop<B::Buffer>>,
        image_view: &B::ImageView,
        sampler: &B::Sampler,
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
                device.write_descriptor_set(DescriptorSetWrite {
                    set: &mut descriptor_sets[i],
                    binding: 0,
                    array_offset: 0,
                    descriptors: vec![
                        Descriptor::Buffer(&*uniform_buffers[i], SubRange::WHOLE),
                        // instead of a combined image sampler, we could also use a separate image and a sampler descriptor
                        Descriptor::CombinedImageSampler(
                            image_view,
                            Layout::ShaderReadOnlyOptimal,
                            sampler,
                        ),
                    ]
                    .into_iter(),
                })
            }
        }

        descriptor_sets
    }

    // fn create_texture_image(
    //     device: &B::Device,
    //     queue_group: &mut QueueGroup<B>,
    //     command_pool: &mut B::CommandPool,
    //     limits: &gfx_hal::Limits,
    //     memory_types: &[MemoryType],
    //     raw_image: &image::RgbaImage,
    // ) -> (ManuallyDrop<B::Image>, ManuallyDrop<B::Memory>) {
    //     let buffer_len = raw_image.width() * raw_image.height() * 4;
    //     let padded_buffer_len = ((buffer_len as usize + limits.non_coherent_atom_size - 1)
    //         / limits.non_coherent_atom_size
    //         * limits.non_coherent_atom_size) as u64;

    //     let (staging_buffer, mut staging_memory) = Self::create_buffer(
    //         &device,
    //         &memory_types,
    //         padded_buffer_len,
    //         buffer::Usage::TRANSFER_SRC,
    //         memory::Properties::CPU_VISIBLE,
    //     );

    //     unsafe {
    //         Self::copy_into_memory(
    //             &device,
    //             raw_image.as_ptr(),
    //             &mut staging_memory,
    //             buffer_len as usize,
    //         )
    //     };

    //     let (texture_image, texture_memory) = Self::create_image(
    //         &device,
    //         &memory_types,
    //         raw_image.width(),
    //         raw_image.height(),
    //         Format::Rgba8Srgb,
    //         gfximg::Tiling::Optimal,
    //         gfximg::Usage::TRANSFER_DST | gfximg::Usage::SAMPLED,
    //         memory::Properties::DEVICE_LOCAL,
    //     );

    //     // transition to transfer destination layout
    //     Self::transition_image_layout(
    //         &device,
    //         queue_group,
    //         command_pool,
    //         &texture_image,
    //         Format::Rgba8Srgb,
    //         Layout::Undefined..Layout::TransferDstOptimal,
    //     );

    //     // copy buffer to image
    //     Self::copy_buffer_to_image(
    //         &device,
    //         queue_group,
    //         &staging_buffer,
    //         &texture_image,
    //         command_pool,
    //         raw_image.width(),
    //         raw_image.height(),
    //     );

    //     // transition image to shader read layout
    //     Self::transition_image_layout(
    //         &device,
    //         queue_group,
    //         command_pool,
    //         &texture_image,
    //         Format::Rgba8Srgb,
    //         Layout::TransferDstOptimal..Layout::ShaderReadOnlyOptimal,
    //     );

    //     unsafe {
    //         device.destroy_buffer(ManuallyDrop::into_inner(staging_buffer));
    //         device.free_memory(ManuallyDrop::into_inner(staging_memory));
    //     }

    //     (texture_image, texture_memory)
    // }

    fn create_texture(
        device: &B::Device,
        queue_group: &mut QueueGroup<B>,
        command_pool: &mut B::CommandPool,
        limits: &gfx_hal::Limits,
        memory_types: &[MemoryType],
        raw_image: &image::RgbaImage,
    ) -> Texture<B> {
        let buffer_len = raw_image.width() * raw_image.height() * 4;
        let padded_buffer_len = ((buffer_len as usize + limits.non_coherent_atom_size - 1)
            / limits.non_coherent_atom_size
            * limits.non_coherent_atom_size) as u64;

        let (staging_buffer, mut staging_memory) = Self::create_buffer(
            &device,
            &memory_types,
            padded_buffer_len,
            buffer::Usage::TRANSFER_SRC,
            memory::Properties::CPU_VISIBLE,
        );

        unsafe {
            Self::copy_into_memory(
                &device,
                raw_image.as_ptr(),
                &mut staging_memory,
                buffer_len as usize,
            )
        };

        // todo: refactor create_image to return Image<B>
        let (texture_image, texture_memory) = Self::create_image(
            &device,
            &memory_types,
            raw_image.width(),
            raw_image.height(),
            Format::Rgba8Srgb,
            gfximg::Tiling::Optimal,
            gfximg::Usage::TRANSFER_DST | gfximg::Usage::SAMPLED,
            memory::Properties::DEVICE_LOCAL,
        );

        // transition to transfer destination layout
        Self::transition_image_layout(
            &device,
            queue_group,
            command_pool,
            &texture_image,
            Format::Rgba8Srgb,
            Layout::Undefined..Layout::TransferDstOptimal,
        );

        // copy buffer to image
        Self::copy_buffer_to_image(
            &device,
            queue_group,
            &staging_buffer,
            &texture_image,
            command_pool,
            raw_image.width(),
            raw_image.height(),
        );

        // transition image to shader read layout
        Self::transition_image_layout(
            &device,
            queue_group,
            command_pool,
            &texture_image,
            Format::Rgba8Srgb,
            Layout::TransferDstOptimal..Layout::ShaderReadOnlyOptimal,
        );

        unsafe {
            device.destroy_buffer(ManuallyDrop::into_inner(staging_buffer));
            device.free_memory(ManuallyDrop::into_inner(staging_memory));
        }

        let texture_image_view = Self::create_image_view(
            &device,
            &texture_image,
            Format::Rgba8Srgb,
            Aspects::COLOR,
            gfximg::Usage::SAMPLED,
        );
        let texture_sampler = Self::create_texture_sampler(&device);

        Texture {
            image: Image {
                image: texture_image,
                memory: texture_memory,
            },
            image_view: texture_image_view,
            sampler: texture_sampler,
        }
    }

    fn create_image_view(
        device: &B::Device,
        image: &B::Image,
        format: Format,
        aspects: Aspects,
        usage: gfximg::Usage,
    ) -> ManuallyDrop<B::ImageView> {
        ManuallyDrop::new(
            unsafe {
                device.create_image_view(
                    image,
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
        )
    }

    fn create_texture_sampler(device: &B::Device) -> ManuallyDrop<B::Sampler> {
        // this uses no anisotropy, has to be changed if required
        ManuallyDrop::new(
            unsafe {
                device.create_sampler(&SamplerDesc::new(
                    gfximg::Filter::Linear,
                    gfximg::WrapMode::Tile,
                ))
            }
            .unwrap(),
        )
    }

    fn find_memory_type(
        memory_types: &[MemoryType],
        requirements_type_mask: u32,
        required_properties: memory::Properties,
    ) -> MemoryTypeId {
        memory_types
            .iter()
            .enumerate()
            .position(|(id, memory_type)| {
                // shift one bit and use bitwise and to find the first memory type that
                // is usable and has the required properties
                requirements_type_mask & (1 << id) != 0
                    && memory_type.properties.contains(required_properties)
            })
            .unwrap()
            .into()
    }

    fn create_render_pass(
        device: &B::Device,
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
            flags: memory::Dependencies::empty(),
        };

        ManuallyDrop::new(unsafe {
            device
                .create_render_pass(
                    vec![color_attachment, depth_attachment].into_iter(),
                    iter::once(subpass),
                    iter::once(dependency),
                )
                .expect("Out of memory creating render pass")
        })
    }

    unsafe fn make_pipeline(
        device: &B::Device,
        render_pass: &B::RenderPass,
        pipeline_layout: &B::PipelineLayout,
        vertex_shader: &[u32],
        fragment_shader: &[u32],
    ) -> ManuallyDrop<B::GraphicsPipeline> {
        let vertex_shader_module = device
            .create_shader_module(vertex_shader)
            .expect("Failed to create vertex shader module");

        let fragment_shader_module = device
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
            Subpass {
                index: 0,
                main_pass: render_pass,
            },
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

        let pipeline = device
            .create_graphics_pipeline(&pipeline_desc, None)
            .expect("Failed to create graphics pipeline");

        // shader modules can be destroyed now, since the pipeline compiled
        // them to machine code and we don't need the spir-v intermediate anymore
        device.destroy_shader_module(vertex_shader_module);
        device.destroy_shader_module(fragment_shader_module);

        ManuallyDrop::new(pipeline)
    }
}

impl<B: gfx_hal::Backend> Drop for Renderer<B> {
    fn drop(&mut self) {
        // wait for device to become idle
        self.device.wait_idle().unwrap();

        unsafe {
            // "drain" the vector, so that each value is dropped as we're iterating over it.
            // we need to drop the values before we destroy further elements of the renderer
            for command_pool in self.command_pools.drain(..) {
                self.device.destroy_command_pool(command_pool);
            }

            for submission_complete_semaphore in self.submission_complete_semaphores.drain(..) {
                self.device.destroy_semaphore(submission_complete_semaphore);
            }

            for submission_complete_fence in self.submission_complete_fences.drain(..) {
                self.device.destroy_fence(submission_complete_fence);
            }

            // todo: move drop implementation into specific structs
            self.device
                .destroy_image_view(ManuallyDrop::take(&mut self.texture.image_view));
            self.device
                .destroy_sampler(ManuallyDrop::take(&mut self.texture.sampler));

            self.device
                .destroy_image_view(ManuallyDrop::take(&mut self.depth_image_view));

            self.device
                .destroy_buffer(ManuallyDrop::take(&mut self.vertex_buffer));
            self.device
                .destroy_buffer(ManuallyDrop::take(&mut self.index_buffer));
            for uniform_buffer in self.uniform_buffers.iter_mut() {
                self.device
                    .destroy_buffer(ManuallyDrop::take(uniform_buffer));
            }
            self.device
                .destroy_buffer(ManuallyDrop::take(&mut self.floor_vertex_buffer));
            self.device
                .destroy_buffer(ManuallyDrop::take(&mut self.floor_index_buffer));
            for floor_uniform_buffer in self.floor_uniform_buffers.iter_mut() {
                self.device
                    .destroy_buffer(ManuallyDrop::take(floor_uniform_buffer));
            }

            self.device
                .destroy_image(ManuallyDrop::take(&mut self.texture.image.image));
            self.device
                .destroy_image(ManuallyDrop::take(&mut self.depth_image));

            self.device
                .free_memory(ManuallyDrop::take(&mut self.vertex_buffer_memory));
            self.device
                .free_memory(ManuallyDrop::take(&mut self.index_buffer_memory));
            for uniform_buffer_memory in self.uniform_buffers_memory.iter_mut() {
                self.device
                    .free_memory(ManuallyDrop::take(uniform_buffer_memory));
            }
            self.device
                .free_memory(ManuallyDrop::take(&mut self.floor_vertex_buffer_memory));
            self.device
                .free_memory(ManuallyDrop::take(&mut self.floor_index_buffer_memory));
            for floor_uniform_buffer_memory in self.floor_uniform_buffers_memory.iter_mut() {
                self.device
                    .free_memory(ManuallyDrop::take(floor_uniform_buffer_memory));
            }
            self.device
                .free_memory(ManuallyDrop::take(&mut self.texture.image.memory));
            self.device
                .free_memory(ManuallyDrop::take(&mut self.depth_image_memory));

            self.device
                .destroy_descriptor_pool(ManuallyDrop::take(&mut self.descriptor_pool));
            self.device
                .destroy_descriptor_set_layout(ManuallyDrop::take(&mut self.descriptor_set_layout));

            self.device
                .destroy_render_pass(ManuallyDrop::take(&mut self.render_pass));
            self.device
                .destroy_framebuffer(ManuallyDrop::take(&mut self.framebuffer));
            self.surface.unconfigure_swapchain(&self.device);
            self.device
                .destroy_graphics_pipeline(ManuallyDrop::take(&mut self.pipeline));
            self.device
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
