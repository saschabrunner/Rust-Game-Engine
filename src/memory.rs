use crate::device::Device;
use crate::image::Image;

use std::{iter, mem::ManuallyDrop, ptr, rc::Rc};

use gfx_hal::{
    buffer,
    command::{BufferCopy, BufferImageCopy, CommandBuffer, CommandBufferFlags, Level},
    device::Device as _,
    format::Aspects,
    image::{self as gfximg, Extent, Layout, SubresourceLayers},
    memory::{self as gfx_mem, Segment, SparseFlags},
    pool::CommandPool,
    queue::{Queue, QueueGroup},
};

pub struct Buffer<B: gfx_hal::Backend> {
    device: Rc<Device<B>>,
    buffer: ManuallyDrop<B::Buffer>,
    memory: ManuallyDrop<B::Memory>, // this should be replaced by memory from an allocator
}

impl<B: gfx_hal::Backend> Buffer<B> {
    pub unsafe fn new(
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

    /// Get a reference to the raw gfx-rs buffer.
    pub fn buffer(&self) -> &B::Buffer {
        &self.buffer
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

pub fn copy_buffer_to_image<B: gfx_hal::Backend>(
    device: &Rc<Device<B>>,
    queue_group: &mut QueueGroup<B>,
    source: &Buffer<B>,
    destination: &Image<B>,
    command_pool: &mut B::CommandPool,
    width: u32,
    height: u32,
) {
    let mut copy_fence = device
        .logical_device()
        .create_fence(false)
        .expect("Out of memory creating fence");

    unsafe {
        let mut command_buffer = command_pool.allocate_one(Level::Primary);
        command_buffer.begin_primary(CommandBufferFlags::ONE_TIME_SUBMIT);

        command_buffer.copy_buffer_to_image(
            &source.buffer,
            destination.image(),
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

        device
            .logical_device()
            .wait_for_fence(&copy_fence, !0)
            .unwrap();
        command_pool.free(iter::once(command_buffer));
        device.logical_device().destroy_fence(copy_fence);
    }
}

/// Copies into a CPU visible buffer, undefined behavior if memory is not CPU visible!
pub unsafe fn copy_into_buffer<B: gfx_hal::Backend>(
    device: &Rc<Device<B>>,
    source: *const u8,
    destination: &mut Buffer<B>,
    len: usize,
) {
    // todo: consider keeping the memory mapped for uniform buffers
    let mapping = device
        .logical_device()
        .map_memory(&mut destination.memory, Segment::ALL)
        .unwrap();
    ptr::copy_nonoverlapping(source, mapping, len);

    // manually flush the memory, so that it is ensured to be available to the GPU.
    // if instead limited buffer_memory_type to a type that contains the property
    // memory::Properties::COHERENT, we wouldn't have to explicitly flush, possible
    // optimization for the future.
    // either way however, the memory is not guaranteed to be visible on the GPU yet
    // until we submit our queue, according to the specification.
    device
        .logical_device()
        .flush_mapped_memory_ranges(iter::once((&*destination.memory, Segment::ALL)))
        .unwrap();

    device
        .logical_device()
        .unmap_memory(&mut destination.memory);
}

pub fn copy_buffer<B: gfx_hal::Backend>(
    device: &Rc<Device<B>>,
    queue_group: &mut QueueGroup<B>,
    source: &Buffer<B>,
    destination: &Buffer<B>,
    command_pool: &mut B::CommandPool,
    len: u64,
) {
    let mut copy_fence = device
        .logical_device()
        .create_fence(false)
        .expect("Out of memory creating fence");

    unsafe {
        let mut command_buffer = command_pool.allocate_one(Level::Primary);
        command_buffer.begin_primary(CommandBufferFlags::ONE_TIME_SUBMIT);
        command_buffer.copy_buffer(
            &*source.buffer,
            &*destination.buffer,
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

        device
            .logical_device()
            .wait_for_fence(&copy_fence, !0)
            .unwrap();
        command_pool.free(iter::once(command_buffer));
        device.logical_device().destroy_fence(copy_fence);
    }
}
