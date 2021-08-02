use crate::{
    camera::{Camera, CameraDesc},
    device::Device,
    image::{Image, ImageView},
    render::Material,
    world::{self, GameObject, Vertex},
};

use std::{
    borrow::Borrow,
    io::Cursor,
    iter,
    mem::{size_of, ManuallyDrop},
    rc::Rc,
    time::Instant,
};

use gfx_hal::{
    adapter::{Adapter, PhysicalDevice},
    command::{
        ClearColor, ClearDepthStencil, ClearValue, CommandBuffer, CommandBufferFlags, Level,
        RenderAttachmentInfo, SubpassContents,
    },
    device::Device as _,
    format::{Aspects, ChannelType, Format, ImageFeature},
    image::{
        self as gfx_img, Extent, FramebufferAttachment, Layout, SamplerDesc, ViewCapabilities,
    },
    memory as gfx_mem,
    pass::{
        Attachment, AttachmentLoadOp, AttachmentOps, AttachmentStoreOp, Subpass, SubpassDependency,
        SubpassDesc,
    },
    pool::{CommandPool, CommandPoolCreateFlags},
    pso::{
        self, AttributeDesc, BlendState, BufferDescriptorFormat, BufferDescriptorType,
        ColorBlendDesc, ColorMask, DepthStencilDesc, DepthTest, DescriptorPoolCreateFlags,
        DescriptorRangeDesc, DescriptorSetLayoutBinding, DescriptorType, EntryPoint, Face,
        GraphicsPipelineDesc, ImageDescriptorType, InputAssemblerDesc, PipelineStage, Primitive,
        PrimitiveAssemblerDesc, Rasterizer, Rect, ShaderStageFlags, Specialization,
        VertexBufferDesc, VertexInputRate, Viewport,
    },
    queue::{family::QueueFamilyId, Queue, QueueGroup},
    window::{Extent2D, PresentationSurface, Surface, SwapchainConfig},
    Instance,
};

use nalgebra::{Point3, Similarity3, Translation3, UnitQuaternion, Vector3};

pub struct Renderer<B: gfx_hal::Backend> {
    camera: Camera,
    surface: ManuallyDrop<B::Surface>,
    surface_format: Format,
    depth_format: Format,
    surface_extent: Extent2D,
    render_pass: ManuallyDrop<B::RenderPass>,
    framebuffer: ManuallyDrop<B::Framebuffer>,
    pipeline: ManuallyDrop<Pipeline<B>>,
    pipeline_layout: ManuallyDrop<B::PipelineLayout>,
    frame_resources: Vec<FrameResources<B>>,
    texture_sampler: ManuallyDrop<B::Sampler>,
    mvp_descriptor_set_layout: ManuallyDrop<B::DescriptorSetLayout>, // todo: remove model from this and add it to separate ds layout in GameObject, then move this to camera perhaps
    material_descriptor_set_layout: ManuallyDrop<B::DescriptorSetLayout>, // currently there's only one possible layout for materials (a single image)
    descriptor_pool: ManuallyDrop<B::DescriptorPool>,
    game_objects: Vec<GameObject<B>>,
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
            width: crate::DEFAULT_WINDOW_SIZE[0],
            height: crate::DEFAULT_WINDOW_SIZE[1],
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
                include_bytes!("../../data/textures/Ground037_1K-PNG/Ground037_1K_Color.png"),
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
                include_bytes!("../../data/textures/Planks016_1K-PNG/Planks016_1K_Color.png"),
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
                &world::CUBE_VERTICES,
                &world::CUBE_INDICES,
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
                &world::QUAD_VERTICES,
                &world::QUAD_INDICES,
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
            "../../target/data/shaders/03_model_view_projection_3d.vert.spv"
        )))
        .unwrap();
        let fragment_shader = gfx_auxil::read_spirv(Cursor::new(include_bytes!(
            "../../target/data/shaders/03_texture_bg.frag.spv"
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
            usage: gfx_img::Usage::DEPTH_STENCIL_ATTACHMENT,
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

    pub fn recreate_swapchain(&mut self, new_surface_extent: Extent2D) {
        let capabilities = self.surface.capabilities(self.device.physical_device());
        let swapchain_config =
            SwapchainConfig::from_caps(&capabilities, self.surface_format, new_surface_extent);

        // update surface extent from the swapchain config, since the swapchain might enforce different dimensions
        self.surface_extent = swapchain_config.extent;

        unsafe {
            self.device.logical_device().wait_idle().unwrap();

            // destroy resources about to be recreated
            self.device
                .logical_device()
                .destroy_framebuffer(ManuallyDrop::take(&mut self.framebuffer));

            let depth_attachment = FramebufferAttachment {
                usage: gfx_img::Usage::DEPTH_STENCIL_ATTACHMENT,
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
                    self.recreate_swapchain(self.surface_extent);
                    return;
                }
            }
        };

        // calculate delta time and update time
        let current_time = Instant::now();
        let delta_time = (current_time - self.last_frame).as_secs_f32();
        let application_time = (current_time - self.start_time).as_secs_f32();
        self.last_frame = current_time;

        // rotate and move cube with time
        let transform = self.game_objects[0].get_transform();
        transform.isometry.rotation *=
            UnitQuaternion::from_euler_angles(0.0, -delta_time, delta_time);
        transform.isometry.translation = Translation3::<f32>::new(
            (application_time * 0.5).sin() * 5.0,
            (application_time * 0.7).sin() * 4.0,
            (application_time * 0.8).sin() * 4.0,
        );

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
                        image_view: frame_resources.depth_image_view.image_view(), // can only borrow SwapchainImage
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
                self.recreate_swapchain(self.surface_extent);
            }
        }

        self.frame += 1;
    }

    /// Getter for input system to move the camera.
    /// TODO: Refactor camera into different system.
    pub fn camera(&mut self) -> &mut Camera {
        &mut self.camera
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
        tiling: gfx_img::Tiling,
        features: ImageFeature,
    ) -> Option<Format> {
        for candidate in candidates {
            let properties = physical_device.format_properties(Some(*candidate));
            match tiling {
                gfx_img::Tiling::Optimal => {
                    if properties.optimal_tiling & features == features {
                        return Some(*candidate);
                    }
                }
                gfx_img::Tiling::Linear => {
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
            gfx_img::Tiling::Optimal,
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
                    gfx_img::Filter::Linear,
                    gfx_img::WrapMode::Tile,
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
            accesses: gfx_img::Access::empty()
                ..gfx_img::Access::COLOR_ATTACHMENT_WRITE
                    | gfx_img::Access::DEPTH_STENCIL_ATTACHMENT_WRITE,
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
            gfx_img::Tiling::Optimal,
            gfx_img::Usage::DEPTH_STENCIL_ATTACHMENT,
            gfx_mem::Properties::DEVICE_LOCAL,
        );

        ImageView::new(
            device,
            depth_image,
            depth_format,
            Aspects::DEPTH, // todo: stencil aspect
            gfx_img::Usage::DEPTH_STENCIL_ATTACHMENT,
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
