use crate::{
    device::Device,
    image::{Image, ImageView},
    memory::{self, Buffer},
};

use std::{io::Cursor, rc::Rc};

use gfx_hal::{
    buffer as gfx_buf,
    device::Device as _,
    format::{Aspects, Format},
    image::{self as gfx_img, Layout},
    memory as gfx_mem,
    pso::{Descriptor, DescriptorPool, DescriptorSetWrite},
    queue::QueueGroup,
};

pub struct Material<B: gfx_hal::Backend> {
    _device: Rc<Device<B>>,
    _texture: ImageView<B>, // currently a material is only made out of a texture
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
        let image = image_lib::load(Cursor::new(image_png), image_lib::ImageFormat::Png)
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
            _device: device.clone(),
            _texture: texture,
            descriptor_sets,
        }
    }

    fn create_texture(
        device: &Rc<Device<B>>,
        queue_group: &mut QueueGroup<B>,
        command_pool: &mut B::CommandPool,
        raw_image: &image_lib::RgbaImage,
    ) -> ImageView<B> {
        let buffer_len = raw_image.width() * raw_image.height() * 4;
        let padded_buffer_len = ((buffer_len as usize + device.limits().non_coherent_atom_size - 1)
            / device.limits().non_coherent_atom_size
            * device.limits().non_coherent_atom_size) as u64;

        let mut staging_buffer = unsafe {
            Buffer::new(
                device,
                padded_buffer_len,
                gfx_buf::Usage::TRANSFER_SRC,
                gfx_mem::Properties::CPU_VISIBLE,
            )
        };

        unsafe {
            memory::copy_into_buffer(
                device,
                raw_image.as_ptr(),
                &mut staging_buffer,
                buffer_len as usize,
            )
        };

        let texture_image = unsafe {
            Image::new(
                &device,
                raw_image.width(),
                raw_image.height(),
                Format::Rgba8Srgb,
                gfx_img::Tiling::Optimal,
                gfx_img::Usage::TRANSFER_DST | gfx_img::Usage::SAMPLED,
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
            &staging_buffer,
            &texture_image,
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
            gfx_img::Usage::SAMPLED,
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
                                image_view.image_view(),
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
