use crate::device::Device;

use std::{iter, ptr, rc::Rc};

use gfx_hal::{
    command::{BufferCopy, BufferImageCopy, CommandBuffer, CommandBufferFlags, Level},
    device::Device as _,
    format::Aspects,
    image::{self as gfximg, Extent, Layout, SubresourceLayers},
    memory::Segment,
    pool::CommandPool,
    queue::{Queue, QueueGroup},
};

pub fn copy_buffer_to_image<B: gfx_hal::Backend>(
    device: &Rc<Device<B>>,
    queue_group: &mut QueueGroup<B>,
    source: &B::Buffer,
    destination: &B::Image,
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

        device
            .logical_device()
            .wait_for_fence(&copy_fence, !0)
            .unwrap();
        command_pool.free(iter::once(command_buffer));
        device.logical_device().destroy_fence(copy_fence);
    }
}

/// Copies into a CPU visible buffer, undefined behavior if memory is not CPU visible!
pub unsafe fn copy_into_memory<B: gfx_hal::Backend>(
    device: &Rc<Device<B>>,
    source: *const u8,
    destination: &mut B::Memory,
    len: usize,
) {
    // todo: consider keeping the memory mapped for uniform buffers
    let mapping = device
        .logical_device()
        .map_memory(destination, Segment::ALL)
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
        .flush_mapped_memory_ranges(iter::once((&*destination, Segment::ALL)))
        .unwrap();

    device.logical_device().unmap_memory(destination);
}

pub fn copy_buffer<B: gfx_hal::Backend>(
    device: &Rc<Device<B>>,
    queue_group: &mut QueueGroup<B>,
    source: &B::Buffer,
    destination: &B::Buffer,
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

        device
            .logical_device()
            .wait_for_fence(&copy_fence, !0)
            .unwrap();
        command_pool.free(iter::once(command_buffer));
        device.logical_device().destroy_fence(copy_fence);
    }
}
