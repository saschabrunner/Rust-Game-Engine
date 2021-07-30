use gfx_hal::{
    adapter::{Adapter, Gpu, MemoryType, PhysicalDevice},
    memory as gfx_mem,
    queue::{QueueFamily, QueueGroup},
    window::Surface,
    MemoryTypeId,
};

pub struct Device<B: gfx_hal::Backend> {
    gpu: Gpu<B>,
    limits: gfx_hal::Limits,
    // in the future I might also want to consider the memory_heaps field in the memory properties
    // this will be useful to determine wheter the type belongs to the GPUs VRAM or might be slow system RAM
    memory_types: Vec<MemoryType>,
    adapter: Adapter<B>,
}

impl<B: gfx_hal::Backend> Device<B> {
    pub unsafe fn new(adapter: Adapter<B>, surface: &B::Surface) -> Self {
        let queue_family = Self::find_queue_family(&adapter, surface);
        let physical_device = &adapter.physical_device;

        let gpu = physical_device
            .open(&[(queue_family, &[1.0])], gfx_hal::Features::empty())
            .expect("Failed to open device");

        Device {
            gpu,
            limits: physical_device.properties().limits,
            memory_types: physical_device.memory_properties().memory_types,
            adapter,
        }
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

    pub fn find_memory_type(
        &self,
        requirements_type_mask: u32,
        required_properties: gfx_mem::Properties,
    ) -> MemoryTypeId {
        self.memory_types()
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

    pub fn logical_device(&self) -> &B::Device {
        &self.gpu.device
    }

    pub fn queue_groups(&mut self) -> &mut Vec<QueueGroup<B>> {
        &mut self.gpu.queue_groups
    }

    pub fn physical_device(&self) -> &B::PhysicalDevice {
        &self.adapter.physical_device
    }

    pub fn limits(&self) -> &gfx_hal::Limits {
        &self.limits
    }

    pub fn memory_types(&self) -> &[MemoryType] {
        &self.memory_types
    }
}
