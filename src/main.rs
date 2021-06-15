use std::{borrow::Borrow, iter, mem::ManuallyDrop};

use gfx_hal::{
    adapter::{Adapter, PhysicalDevice},
    command::{
        ClearColor, ClearValue, CommandBuffer, CommandBufferFlags, Level, RenderAttachmentInfo,
        SubpassContents,
    },
    device::Device,
    format::{ChannelType, Format},
    image::{Extent, Layout},
    pass::{Attachment, AttachmentLoadOp, AttachmentOps, AttachmentStoreOp, Subpass, SubpassDesc},
    pool::{CommandPool, CommandPoolCreateFlags},
    pso::{
        BlendState, ColorBlendDesc, ColorMask, EntryPoint, Face, GraphicsPipelineDesc,
        InputAssemblerDesc, Primitive, PrimitiveAssemblerDesc, Rasterizer, Rect, Specialization,
        Viewport,
    },
    queue::{Queue, QueueFamily},
    window::{Extent2D, PresentationSurface, Surface, SwapchainConfig},
    Instance,
};

use shaderc::ShaderKind;

use winit::{
    dpi::{LogicalSize, PhysicalSize},
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

struct Resources<B: gfx_hal::Backend> {
    instance: B::Instance,
    surface: B::Surface,
    device: B::Device,
    render_passes: Vec<B::RenderPass>,
    pipeline_layouts: Vec<B::PipelineLayout>,
    pipelines: Vec<B::GraphicsPipeline>,
    framebuffers: Vec<B::Framebuffer>,
    command_pool: B::CommandPool,
    submission_complete_fence: B::Fence,
    rendering_complete_semaphore: B::Semaphore,
}

// we create a tuple struct that wraps Resources in a ManuallyDrop
// so we can take ownership and destroy the resources using the
// destroy_ functions, which require ownership instead of just
// a borrow
struct ResourceHolder<B: gfx_hal::Backend>(ManuallyDrop<Resources<B>>);

impl<B: gfx_hal::Backend> Drop for ResourceHolder<B> {
    fn drop(&mut self) {
        // todo: this implementation is not complete and results in a crash
        unsafe {
            let Resources {
                instance,
                mut surface,
                device,
                command_pool,
                render_passes,
                pipeline_layouts,
                pipelines,
                framebuffers,
                submission_complete_fence,
                rendering_complete_semaphore,
            } = ManuallyDrop::take(&mut self.0); // we "read" the value behind this ManuallyDrop

            device.destroy_command_pool(command_pool);
            device.destroy_semaphore(rendering_complete_semaphore);
            device.destroy_fence(submission_complete_fence);
            for render_pass in render_passes {
                device.destroy_render_pass(render_pass);
            }
            for framebuffer in framebuffers {
                device.destroy_framebuffer(framebuffer);
            }
            surface.unconfigure_swapchain(&device);
            for pipeline in pipelines {
                device.destroy_graphics_pipeline(pipeline);
            }
            for pipeline_layout in pipeline_layouts {
                device.destroy_pipeline_layout(pipeline_layout);
            }
            instance.destroy_surface(surface);
        }
    }
}

const APPLICATION_NAME: &'static str = "Rust Game Engine";
const DEFAULT_WINDOW_SIZE: [u32; 2] = [512, 512];

fn main() {
    let event_loop = winit::event_loop::EventLoop::new();
    let (logical_window_size, physical_window_size) = get_window_sizes(&event_loop);

    let mut surface_extent = Extent2D {
        width: physical_window_size.width,
        height: physical_window_size.height,
    };

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
    let (device, mut queue_group) = {
        let queue_family = find_queue_family(&adapter, &surface);

        let mut gpu = unsafe {
            adapter
                .physical_device
                .open(&[(queue_family, &[1.0])], gfx_hal::Features::empty())
                .expect("Failed to open device")
        };

        (gpu.device, gpu.queue_groups.pop().unwrap())
    };

    let (command_pool, mut command_buffer) = unsafe {
        // a command pool allocates command buffers and is supposed to be used from a single thread only
        let mut command_pool = device
            .create_command_pool(queue_group.family, CommandPoolCreateFlags::empty())
            .expect("Out of memory creating command pool");

        // mutable access to command_pool required here
        let command_buffer = command_pool.allocate_one(Level::Primary);

        // Even though command_buffer isn't declared mut,
        // because we're handing over ownership, the new
        // declaration may be mut!
        // command_pool however is no longer needed mutable
        // so we hand it over into a non mut variable.
        (command_pool, command_buffer)
    };

    let surface_color_format = find_surface_color_format(&adapter, &surface);

    let render_pass = create_render_pass::<backend::Backend>(&device, surface_color_format);

    // used for uniforms and push constants in the future
    let pipeline_layout = unsafe {
        device
            .create_pipeline_layout(iter::empty(), iter::empty())
            .expect("Out of memory creating pipeline layout")
    };

    let vertex_shader = include_str!("../data/shaders/01_triangle.vert");
    let fragment_shader = include_str!("../data/shaders/01_single_color.frag");

    let pipeline = unsafe {
        make_pipeline::<backend::Backend>(
            &device,
            &render_pass,
            &pipeline_layout,
            vertex_shader,
            fragment_shader,
        )
    };

    // framebuffer connects attachments to images
    // attachments can be thought of as the slots
    // and the images fill those slots
    let framebuffer = unsafe {
        // todo: clean up the following code
        // store and reuse the capabilities later and so on
        let caps = surface.capabilities(&adapter.physical_device);

        let swapchain_config =
            SwapchainConfig::from_caps(&caps, surface_color_format, surface_extent);

        let framebuffer_attachment = swapchain_config.framebuffer_attachment();

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
    };

    let submission_complete_fence = device
        .create_fence(true)
        .expect("Out of memory creating fence");
    let rendering_complete_semaphore = device
        .create_semaphore()
        .expect("Out of memory creating semaphore");

    let mut resource_holder: ResourceHolder<backend::Backend> =
        ResourceHolder(ManuallyDrop::new(Resources {
            instance,
            surface,
            device,
            command_pool,
            render_passes: vec![render_pass],
            pipeline_layouts: vec![pipeline_layout],
            pipelines: vec![pipeline],
            framebuffers: vec![framebuffer],
            submission_complete_fence,
            rendering_complete_semaphore,
        }));

    // flag to keep check if swapchain needs to be reconfigured next refresh
    let mut should_configure_swapchain = true;

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized(dimensions) => {
                    surface_extent = Extent2D {
                        width: dimensions.width,
                        height: dimensions.height,
                    };
                    should_configure_swapchain = true;
                }
                WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                    surface_extent = Extent2D {
                        width: new_inner_size.width,
                        height: new_inner_size.height,
                    };
                    should_configure_swapchain = true;
                }
                _ => (),
            },
            Event::MainEventsCleared => window.request_redraw(),
            Event::RedrawRequested(_) => {
                let res: &mut Resources<_> = &mut resource_holder.0; // alternative &mut Resources<backend::Backend>
                let render_pass = &res.render_passes[0];
                let pipeline = &res.pipelines[0];

                unsafe {
                    // wait for the last submission to finish before resetting command pool and preparing new frame
                    let render_timeout_ns = 1_000_000_000;
                    res.device
                        .wait_for_fence(&res.submission_complete_fence, render_timeout_ns)
                        .expect("Out of memory or device lost waiting for fence");
                    res.device
                        .reset_fence(&mut res.submission_complete_fence)
                        .expect("Out of memory resetting fence");
                    res.command_pool.reset(false);
                }

                if should_configure_swapchain {
                    let caps = res.surface.capabilities(&adapter.physical_device);

                    let swapchain_config =
                        SwapchainConfig::from_caps(&caps, surface_color_format, surface_extent);

                    // update surface extent, since the swapchain might enforce different dimensions
                    surface_extent = swapchain_config.extent;

                    unsafe {
                        res.surface
                            .configure_swapchain(&res.device, swapchain_config)
                            .expect("Failed to configure swapchain");
                    }

                    should_configure_swapchain = false;
                }

                let surface_image = unsafe {
                    let acquire_timeout_ns = 1_000_000_000;

                    match res.surface.acquire_image(acquire_timeout_ns) {
                        Ok((image, _)) => image,
                        Err(_) => {
                            should_configure_swapchain = true;
                            return;
                        }
                    }
                };

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
                    command_buffer.begin_primary(CommandBufferFlags::ONE_TIME_SUBMIT);
                    // viewports and scissors are part of the command buffer, different to Vulkan
                    command_buffer.set_viewports(0, iter::once(viewport.clone()));
                    command_buffer.set_scissors(0, iter::once(viewport.rect));

                    command_buffer.begin_render_pass(
                        render_pass,
                        &res.framebuffers[0], // todo: switch to single framebuffer in gfx
                        viewport.rect,
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

                    command_buffer.bind_graphics_pipeline(pipeline);
                    command_buffer.draw(0..3, 0..1);
                    command_buffer.end_render_pass();
                    command_buffer.finish();
                }

                unsafe {
                    queue_group.queues[0].submit(
                        iter::once(&command_buffer),
                        iter::empty(),
                        iter::once(&res.rendering_complete_semaphore),
                        Some(&mut res.submission_complete_fence),
                    );

                    let result = queue_group.queues[0].present(
                        &mut res.surface,
                        surface_image,
                        Some(&mut res.rendering_complete_semaphore),
                    );

                    should_configure_swapchain |= result.is_err();
                }
            }
            _ => (),
        }
    });
}

fn get_window_sizes<T>(event_loop: &EventLoop<T>) -> (LogicalSize<u32>, PhysicalSize<u32>) {
    // get either the primary monitor's dpi or the
    // monitor with the highest dpi (on Wayland for example)
    let primary_monitor = event_loop.primary_monitor();
    let dpi = match primary_monitor {
        Some(primary_monitor) => primary_monitor.scale_factor(),
        None => event_loop
            .available_monitors()
            .map(|monitor| monitor.scale_factor())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0),
    };

    let logical: LogicalSize<u32> = DEFAULT_WINDOW_SIZE.into();
    let physical: PhysicalSize<u32> = logical.to_physical(dpi);

    (logical, physical)
}

fn find_adapter<B: gfx_hal::Backend>(instance: &B::Instance) -> Adapter<B> {
    // multiple adapters might exist, like an internal and a dedicated GPU
    // an algorithm could be implemented here to select a different GPU
    instance.enumerate_adapters().remove(0)
}

fn find_queue_family<'a, B: gfx_hal::Backend>(
    adapter: &'a Adapter<B>,
    surface: &B::Surface,
) -> &'a B::QueueFamily {
    adapter
        .queue_families
        .iter()
        .find(|family| {
            surface.supports_queue_family(family) && family.queue_type().supports_graphics()
        })
        .expect("No compatible queue family found")
}

fn find_surface_color_format<B: gfx_hal::Backend>(
    adapter: &Adapter<B>,
    surface: &B::Surface,
) -> Format {
    // get the formats that the surface and the adapter support together
    let supported_formats = surface
        .supported_formats(&adapter.physical_device)
        .unwrap_or(vec![]);

    // if supported_formats is empty, it means we can choose freely
    let default_format = *supported_formats.get(0).unwrap_or(&Format::Rgba8Srgb);

    // try to find an srgb format, otherwise fallback to the first format we found above
    supported_formats
        .into_iter()
        .find(|format| format.base_format().1 == ChannelType::Srgb)
        .unwrap_or(default_format)
}

fn create_render_pass<B: gfx_hal::Backend>(
    device: &B::Device,
    surface_color_format: Format,
) -> B::RenderPass {
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
    unsafe {
        device
            .create_render_pass(
                iter::once(color_attachment),
                iter::once(subpass),
                iter::empty(),
            )
            .expect("Out of memory creating render pass")
    }
}

fn compile_shader(glsl: &str, shader_kind: ShaderKind) -> Vec<u32> {
    let mut compiler = shaderc::Compiler::new().unwrap();

    let compiled_shader = compiler
        .compile_into_spirv(glsl, shader_kind, "unnamed", "main", None)
        .expect("Failed to compile shader");

    compiled_shader.as_binary().to_vec()
}

unsafe fn make_pipeline<B: gfx_hal::Backend>(
    device: &B::Device,
    render_pass: &B::RenderPass,
    pipeline_layout: &B::PipelineLayout,
    vertex_shader: &str,
    fragment_shader: &str,
) -> B::GraphicsPipeline {
    let vertex_shader_module = device
        .create_shader_module(&compile_shader(vertex_shader, ShaderKind::Vertex))
        .expect("Failed to create vertex shader module");

    let fragment_shader_module = device
        .create_shader_module(&compile_shader(fragment_shader, ShaderKind::Fragment))
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
        buffers: &[],
        attributes: &[],
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
    pipeline
}
