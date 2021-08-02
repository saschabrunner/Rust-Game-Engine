use crate::device::Device;

use std::{iter, mem::ManuallyDrop, ops::Range, rc::Rc};

use gfx_hal::{
    command::{CommandBuffer, CommandBufferFlags, Level},
    device::Device as _,
    format::{Aspects, Format, Swizzle},
    image::{self as gfximg, Layout, SubresourceRange, ViewCapabilities, ViewKind},
    memory::{self as gfx_mem, SparseFlags},
    pool::CommandPool,
    pso::PipelineStage,
    queue::{Queue, QueueGroup},
};

pub struct Image<B: gfx_hal::Backend> {
    device: Rc<Device<B>>,
    image: ManuallyDrop<B::Image>,
    memory: ManuallyDrop<B::Memory>, // this should be replaced by memory from an allocator
}

impl<B: gfx_hal::Backend> Image<B> {
    pub unsafe fn new(
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

    pub fn transition_image_layout(
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

    /// Get a reference to the raw gfx-rs image.
    pub fn image(&self) -> &B::Image {
        &self.image
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

pub struct ImageView<B: gfx_hal::Backend> {
    device: Rc<Device<B>>,
    _image: Image<B>,
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
            _image: image,
            image_view: iv,
        }
    }

    /// Get a reference to the raw gfx-rs image view.
    pub fn image_view(&self) -> &B::ImageView {
        &self.image_view
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
