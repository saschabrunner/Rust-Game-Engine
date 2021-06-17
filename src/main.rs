use std::{
    borrow::Borrow,
    io::Cursor,
    iter,
    mem::{size_of, ManuallyDrop},
    ptr,
};

use gfx_hal::{
    adapter::{Adapter, MemoryType, PhysicalDevice},
    buffer::{SubRange, Usage},
    command::{
        ClearColor, ClearValue, CommandBuffer, CommandBufferFlags, Level, RenderAttachmentInfo,
        SubpassContents,
    },
    device::Device,
    format::{ChannelType, Format},
    image::{Extent, Layout},
    memory::{self, Segment, SparseFlags},
    pass::{Attachment, AttachmentLoadOp, AttachmentOps, AttachmentStoreOp, Subpass, SubpassDesc},
    pool::{CommandPool, CommandPoolCreateFlags},
    pso::{
        self, AttributeDesc, BlendState, ColorBlendDesc, ColorMask, EntryPoint, Face,
        GraphicsPipelineDesc, InputAssemblerDesc, Primitive, PrimitiveAssemblerDesc, Rasterizer,
        Rect, Specialization, VertexBufferDesc, VertexInputRate, Viewport,
    },
    queue::{Queue, QueueFamily, QueueGroup},
    window::{Extent2D, PresentationSurface, Surface, SwapchainConfig},
    IndexType, Instance, MemoryTypeId,
};

use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::ControlFlow,
};

// not dead code, we use memcpy on this struct
#[allow(dead_code)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 3],
}

struct Renderer<B: gfx_hal::Backend> {
    surface: ManuallyDrop<B::Surface>,
    surface_format: Format,
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
    vertex_buffer: ManuallyDrop<B::Buffer>,
    index_buffer: ManuallyDrop<B::Buffer>,
    vertex_buffer_memory: ManuallyDrop<B::Memory>,
    index_buffer_memory: ManuallyDrop<B::Memory>,
    frames_in_flight: usize,
    frame: u64,
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
        let queue_group = gpu.queue_groups.pop().unwrap();

        let (vertex_buffer, vertex_buffer_memory) =
            Self::create_vertex_buffer(&device, &limits, &memory_types);

        let (index_buffer, index_buffer_memory) =
            Self::create_index_buffer(&device, &limits, &memory_types);

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
        let render_pass = Self::create_render_pass(&device, surface_format);

        // used for uniforms and push constants in the future
        let pipeline_layout = ManuallyDrop::new(unsafe {
            device
                .create_pipeline_layout(iter::empty(), iter::empty())
                .expect("Out of memory creating pipeline layout")
        });

        // shaders are compiled by build script
        let vertex_shader = gfx_auxil::read_spirv(Cursor::new(include_bytes!(
            "../target/data/shaders/02_pos_color.vert.spv"
        )))
        .unwrap();
        let fragment_shader = gfx_auxil::read_spirv(Cursor::new(include_bytes!(
            "../target/data/shaders/02_vertex_color.frag.spv"
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

        let default_surface_extent = Extent2D {
            width: DEFAULT_WINDOW_SIZE[0],
            height: DEFAULT_WINDOW_SIZE[1],
        };
        let capabilities = surface.capabilities(&adapter.physical_device);
        let swapchain_config =
            SwapchainConfig::from_caps(&capabilities, surface_format, default_surface_extent);

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

        let framebuffer_attachment = swapchain_config.framebuffer_attachment();

        unsafe {
            surface
                .configure_swapchain(&device, swapchain_config)
                .expect("Failed to configure swapchain");
        }

        // framebuffer connects attachments to images
        // attachments can be thought of as the slots
        // and the images fill those slots.
        // gfx-rs uses imageless framebuffers,
        // we don't have to recreate them every frame.
        let framebuffer = ManuallyDrop::new(unsafe {
            device
                .create_framebuffer(
                    &render_pass,
                    iter::once(framebuffer_attachment),
                    Extent {
                        width: surface_extent.width,
                        height: surface_extent.height,
                        depth: 1,
                    },
                )
                .unwrap()
        });

        Renderer {
            surface: ManuallyDrop::new(surface),
            surface_format,
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
            vertex_buffer,
            index_buffer,
            vertex_buffer_memory,
            index_buffer_memory,
            frames_in_flight,
            frame: 0,
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

            // re-create framebuffer
            self.device
                .destroy_framebuffer(ManuallyDrop::take(&mut self.framebuffer));
            self.framebuffer = ManuallyDrop::new(
                self.device
                    .create_framebuffer(
                        &self.render_pass,
                        iter::once(swapchain_config.framebuffer_attachment()),
                        self.surface_extent.to_extent(),
                    )
                    .unwrap(),
            );

            // reconfigure swapchain
            self.surface
                .configure_swapchain(&self.device, swapchain_config)
                .expect("Failed to configure swapchain");
        }
    }

    pub fn render(&mut self) {
        let surface_image = unsafe {
            let acquire_timeout_ns = 1_000_000_000;
            match self.surface.acquire_image(acquire_timeout_ns) {
                Ok((image, _)) => image,
                Err(_) => {
                    self.recreate_swapchain();
                    //should_configure_swapchain = true;
                    return;
                }
            }
        };

        let frame_id = (self.frame % self.frames_in_flight as u64) as usize;
        unsafe {
            // wait for the last submission to finish before resetting command pool and preparing new frame
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

        let command_buffer = &mut self.command_buffers[frame_id];
        unsafe {
            command_buffer.begin_primary(CommandBufferFlags::ONE_TIME_SUBMIT);
            // viewports and scissors are part of the command buffer, different to Vulkan
            command_buffer.set_viewports(0, iter::once(self.viewport.clone()));
            command_buffer.set_scissors(0, iter::once(self.viewport.rect));

            command_buffer.begin_render_pass(
                &self.render_pass,
                &self.framebuffer,
                self.viewport.rect,
                // clear value is part of each render attachment info
                iter::once(RenderAttachmentInfo {
                    image_view: surface_image.borrow(), // borrow returns borrow of diff type
                    clear_value: ClearValue {
                        color: ClearColor {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    },
                }),
                SubpassContents::Inline,
            );

            command_buffer.bind_graphics_pipeline(&self.pipeline);
            command_buffer
                .bind_vertex_buffers(0, iter::once((&*self.vertex_buffer, SubRange::WHOLE)));
            command_buffer.bind_index_buffer(&*self.index_buffer, SubRange::WHOLE, IndexType::U16);
            command_buffer.draw_indexed(0..INDICES.len() as u32, 0, 0..1);
            command_buffer.end_render_pass();
            command_buffer.finish();

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

    fn create_vertex_buffer(
        device: &B::Device,
        limits: &gfx_hal::Limits,
        memory_types: &[MemoryType],
    ) -> (ManuallyDrop<B::Buffer>, ManuallyDrop<B::Memory>) {
        let buffer_stride = size_of::<Vertex>();
        let buffer_len = VERTICES.len() * buffer_stride;
        // buffer needs to have size that's a multiple of non_coherent_atom_size for it to be mapped
        let padded_buffer_len = (buffer_len + limits.non_coherent_atom_size - 1)
            / limits.non_coherent_atom_size
            * limits.non_coherent_atom_size;

        let mut buffer = ManuallyDrop::new(
            unsafe {
                device.create_buffer(
                    padded_buffer_len as u64,
                    Usage::VERTEX,
                    SparseFlags::empty(),
                )
            }
            .unwrap(),
        );

        let buffer_requirements = unsafe { device.get_buffer_requirements(&buffer) };

        let buffer_memory_type = Self::find_memory_type(
            &memory_types,
            buffer_requirements.type_mask,
            memory::Properties::CPU_VISIBLE,
        );

        // allocate memory, map it, copy data into it and unmap it
        let memory = unsafe {
            let mut memory = device
                .allocate_memory(buffer_memory_type, buffer_requirements.size)
                .unwrap();
            device.bind_buffer_memory(&memory, 0, &mut buffer).unwrap();
            let mapping = device.map_memory(&mut memory, Segment::ALL).unwrap();
            ptr::copy_nonoverlapping(VERTICES.as_ptr() as *const u8, mapping, buffer_len);

            // manually flush the memory, so that it is ensured to be available to the GPU.
            // if instead limited buffer_memory_type to a type that contains the property
            // memory::Properties::COHERENT, we wouldn't have to explicitly flush.
            // either way however, the memory is not guaranteed to be visible on the GPU yet
            // until we submit our queue, according to the specification.
            device
                .flush_mapped_memory_ranges(iter::once((&memory, Segment::ALL)))
                .unwrap();

            device.unmap_memory(&mut memory);
            ManuallyDrop::new(memory)
        };

        (buffer, memory)
    }

    fn create_index_buffer(
        device: &B::Device,
        limits: &gfx_hal::Limits,
        memory_types: &[MemoryType],
    ) -> (ManuallyDrop<B::Buffer>, ManuallyDrop<B::Memory>) {
        let buffer_stride = size_of::<u16>();
        let buffer_len = INDICES.len() * buffer_stride;
        let padded_buffer_len = (buffer_len + limits.non_coherent_atom_size - 1)
            / limits.non_coherent_atom_size
            * limits.non_coherent_atom_size;

        let mut buffer = ManuallyDrop::new(
            unsafe {
                device.create_buffer(padded_buffer_len as u64, Usage::INDEX, SparseFlags::empty())
            }
            .unwrap(),
        );

        let buffer_requirements = unsafe { device.get_buffer_requirements(&buffer) };

        let buffer_memory_type = Self::find_memory_type(
            &memory_types,
            buffer_requirements.type_mask,
            memory::Properties::CPU_VISIBLE,
        );

        let memory = unsafe {
            let mut memory = device
                .allocate_memory(buffer_memory_type, buffer_requirements.size)
                .unwrap();
            device.bind_buffer_memory(&memory, 0, &mut buffer).unwrap();
            let mapping = device.map_memory(&mut memory, Segment::ALL).unwrap();
            ptr::copy_nonoverlapping(INDICES.as_ptr() as *const u8, mapping, buffer_len);

            device
                .flush_mapped_memory_ranges(iter::once((&memory, Segment::ALL)))
                .unwrap();

            device.unmap_memory(&mut memory);
            ManuallyDrop::new(memory)
        };

        (buffer, memory)
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
    ) -> ManuallyDrop<B::RenderPass> {
        // https://vulkan-tutorial.com/en/Drawing_a_triangle/Graphics_pipeline_basics/Render_passes
        let color_attachment = Attachment {
            format: Some(surface_color_format),
            samples: 1, // no multi-sampling (yet)
            ops: AttachmentOps::new(AttachmentLoadOp::Clear, AttachmentStoreOp::Store), // clear at start, store at end
            stencil_ops: AttachmentOps::DONT_CARE, // no stencil usage yet
            layouts: Layout::Undefined..Layout::Present, // don't mind layout at the start, but want to present at end
        };

        let subpass = SubpassDesc {
            colors: &[(0, Layout::ColorAttachmentOptimal)],
            depth_stencil: None,
            inputs: &[],
            resolves: &[],
            preserves: &[],
        };

        ManuallyDrop::new(unsafe {
            device
                .create_render_pass(
                    iter::once(color_attachment),
                    iter::once(subpass),
                    iter::empty(),
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
                        format: Format::Rg32Sfloat,
                        offset: 0,
                    },
                },
                AttributeDesc {
                    location: 1,
                    binding: 0,
                    element: pso::Element {
                        format: Format::Rgb32Sfloat,
                        offset: 8, // an offset_of library or offset_from may be considered here
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

            self.device
                .destroy_buffer(ManuallyDrop::take(&mut self.vertex_buffer));
            self.device
                .destroy_buffer(ManuallyDrop::take(&mut self.index_buffer));
            self.device
                .free_memory(ManuallyDrop::take(&mut self.vertex_buffer_memory));
            self.device
                .free_memory(ManuallyDrop::take(&mut self.index_buffer_memory));
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
const VERTICES: [Vertex; 4] = [
    Vertex { position: [-0.5, -0.5], color: [1.0, 1.0, 1.0] },
    Vertex { position: [-0.5,  0.5], color: [0.0, 1.0, 1.0] },
    Vertex { position: [ 0.5,  0.5], color: [1.0, 0.0, 1.0] },
    Vertex { position: [ 0.5, -0.5], color: [0.0, 1.0, 1.0] },
];

const INDICES: [u16; 6] = [0, 1, 2, 0, 2, 3];

fn main() {
    let event_loop = winit::event_loop::EventLoop::new();
    let logical_window_size: LogicalSize<u32> = DEFAULT_WINDOW_SIZE.into();

    let window = winit::window::WindowBuilder::new()
        .with_title(APPLICATION_NAME)
        .with_inner_size(logical_window_size)
        .build(&event_loop)
        .expect("Failed to create window");

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
