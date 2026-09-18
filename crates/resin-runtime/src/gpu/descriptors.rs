//! Pipeline-owned layouts and immutable descriptor sets owned by a recording.
use super::{ResinAllocation, ResinCommandBuffer, ResinGpu, ResinStatus, root_range, vk_status};
use ash::{Device, vk};
use std::{
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
    slice,
};

pub(super) struct Layout {
    device: Device,
    set: vk::DescriptorSetLayout,
    pub pipeline: vk::PipelineLayout,
    pub count: u32,
}

impl Layout {
    pub fn create(
        gpu: &ResinGpu,
        modules: &[(&[u8], bool)],
    ) -> Result<Option<Rc<Self>>, ResinStatus> {
        let mut count = None;
        for (module, allow_writes) in modules {
            let words = super::pipeline::spirv_words(module)?;
            let interface = reflect(&words)?;
            if interface.writes && !allow_writes {
                return Err(ResinStatus::Unsupported);
            }
            if let Some(bindings) = interface.count {
                count = Some(count.unwrap_or(0).max(bindings));
            } else if !gpu.device_addresses {
                return Err(ResinStatus::Unsupported);
            }
        }
        let Some(count) = count else {
            return Ok(None);
        };
        if count > gpu.storage_bindings {
            return Err(ResinStatus::Unsupported);
        }
        let bindings = (0..count)
            .map(|binding| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::ALL)
            })
            .collect::<Vec<_>>();
        let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let set =
            unsafe { gpu.device.create_descriptor_set_layout(&info, None) }.map_err(vk_status)?;
        let push = root_range();
        let info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(slice::from_ref(&set))
            .push_constant_ranges(slice::from_ref(&push));
        let pipeline =
            unsafe { gpu.device.create_pipeline_layout(&info, None) }.map_err(|error| {
                unsafe { gpu.device.destroy_descriptor_set_layout(set, None) };
                vk_status(error)
            })?;
        Ok(Some(Rc::new(Self {
            device: gpu.device.clone(),
            set,
            pipeline,
            count,
        })))
    }
}
impl Drop for Layout {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline_layout(self.pipeline, None);
            self.device.destroy_descriptor_set_layout(self.set, None);
        }
    }
}

// Logical-addressing modules use the descriptor ABI. Optimizers can eliminate
// unused bindings, so only the active prefix needs a layout; source slots keep
// their binding numbers. Physical-addressing modules retain the legacy ABI.
struct Interface {
    count: Option<u32>,
    writes: bool,
}

fn reflect(words: &[u32]) -> Result<Interface, ResinStatus> {
    let mut logical = false;
    let mut sets = BTreeMap::new();
    let mut bindings = BTreeMap::new();
    let mut variables = BTreeMap::new();
    let mut readonly = BTreeSet::new();
    let mut offset = 5;
    while offset < words.len() {
        let size = (words[offset] >> 16) as usize;
        if size == 0 || size > words.len() - offset {
            return Err(ResinStatus::InvalidArgument);
        }
        let args = &words[offset + 1..offset + size];
        match (words[offset] & 0xffff, args) {
            (14, [model, _]) => logical = *model == 0,
            (71, [target, 24]) => {
                readonly.insert(*target);
            }
            (71, [target, 34, set]) => {
                sets.insert(*target, *set);
            }
            (71, [target, 33, binding]) => {
                bindings.insert(*target, *binding);
            }
            (59, [_, id, storage, ..]) => {
                variables.insert(*id, *storage);
            }
            _ => (),
        }
        offset += size;
    }
    if !logical {
        return Ok(Interface {
            count: None,
            writes: false,
        });
    }
    let mut count = 1u32; // Reserved read-only constants channel, even if unused.
    for (id, binding) in bindings {
        if sets.get(&id) != Some(&0) || variables.get(&id) != Some(&12) {
            return Err(ResinStatus::Unsupported);
        }
        count = count.max(binding.checked_add(1).ok_or(ResinStatus::Unsupported)?);
    }
    let writes = variables
        .iter()
        .any(|(id, storage)| *storage == 12 && !readonly.contains(id));
    Ok(Interface {
        count: Some(count),
        writes,
    })
}

/// One immutable set per recorded command. A pool amortizes allocations across
/// consecutive dispatches using the same layout and is freed with the recording.
pub(super) struct Pool {
    device: Device,
    handle: vk::DescriptorPool,
    layout: Rc<Layout>,
    remaining: u32,
}
impl Drop for Pool {
    fn drop(&mut self) {
        unsafe { self.device.destroy_descriptor_pool(self.handle, None) };
    }
}
impl Pool {
    fn create(layout: Rc<Layout>) -> Result<Self, ResinStatus> {
        let size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: layout
                .count
                .checked_mul(32)
                .ok_or(ResinStatus::Unsupported)?,
        };
        let info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(32)
            .pool_sizes(slice::from_ref(&size));
        let handle =
            unsafe { layout.device.create_descriptor_pool(&info, None) }.map_err(vk_status)?;
        Ok(Self {
            device: layout.device.clone(),
            handle,
            layout,
            remaining: 32,
        })
    }
}

impl ResinCommandBuffer {
    pub(crate) fn uses_descriptors(&self) -> bool {
        self.resources.is_some()
    }

    pub(crate) fn bind_resources(
        &mut self,
        buffers: &[vk::DescriptorBufferInfo],
    ) -> Result<(), ResinStatus> {
        let layout = self
            .resources
            .as_ref()
            .ok_or(ResinStatus::InvalidArgument)?
            .clone();
        if buffers.len() < layout.count as usize {
            return Err(ResinStatus::InvalidArgument);
        }
        if self
            .descriptor_pools
            .last()
            .is_none_or(|pool| pool.remaining == 0 || !Rc::ptr_eq(&pool.layout, &layout))
        {
            self.descriptor_pools.push(Pool::create(layout.clone())?);
        }
        let pool = self.descriptor_pools.last_mut().unwrap();
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool.handle)
            .set_layouts(slice::from_ref(&layout.set));
        let set = unsafe { self.device.allocate_descriptor_sets(&info) }.map_err(vk_status)?[0];
        pool.remaining -= 1;
        let writes = buffers
            .iter()
            .take(layout.count as usize)
            .enumerate()
            .map(|(binding, buffer)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(binding as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(slice::from_ref(buffer))
            })
            .collect::<Vec<_>>();
        unsafe {
            self.device.update_descriptor_sets(&writes, &[]);
            self.device.cmd_bind_descriptor_sets(
                self.handle,
                if self.graphics {
                    vk::PipelineBindPoint::GRAPHICS
                } else {
                    vk::PipelineBindPoint::COMPUTE
                },
                layout.pipeline,
                0,
                &[set],
                &[],
            );
        }
        Ok(())
    }
}

impl ResinGpu {
    pub(crate) fn descriptor_range(
        &self,
        allocation: &ResinAllocation,
        offset: usize,
        bytes: usize,
    ) -> Result<(vk::DescriptorBufferInfo, u64), ResinStatus> {
        // Empty views bind one valid placeholder byte; no shader access is allowed
        // by their zero logical length, including an empty tail at a slab's end.
        let absolute = allocation
            .buffer_offset
            .checked_add(if bytes == 0 { 0 } else { offset as u64 })
            .ok_or(ResinStatus::InvalidArgument)?;
        let aligned = absolute / self.storage_alignment * self.storage_alignment;
        let relative = absolute - aligned;
        let range = relative
            .checked_add(bytes.max(1) as u64)
            .ok_or(ResinStatus::InvalidArgument)?;
        if range > self.storage_range {
            return Err(ResinStatus::Unsupported);
        }
        Ok((
            vk::DescriptorBufferInfo {
                buffer: allocation.buffer,
                offset: aligned,
                range,
            },
            relative,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn descriptor_reflection_rejects_malformed_words_and_foreign_sets() {
        let header = [0x07230203, 0x10600, 0, 10, 0];
        for tail in [
            vec![0],
            vec![5 << 16 | 71, 1],
            vec![
                3 << 16 | 14,
                0,
                1,
                4 << 16 | 71,
                1,
                34,
                2,
                4 << 16 | 71,
                1,
                33,
                0,
            ],
        ] {
            assert!(reflect(&[header.as_slice(), &tail].concat()).is_err());
        }
    }
    #[test]
    fn descriptor_reflection_counts_binding_slots_not_alias_variables() {
        let words = [
            0x07230203,
            0x10600,
            0,
            10,
            0,
            3 << 16 | 14,
            0,
            1,
            4 << 16 | 71,
            1,
            34,
            0,
            4 << 16 | 71,
            1,
            33,
            3,
            4 << 16 | 59,
            2,
            1,
            12,
            4 << 16 | 71,
            4,
            34,
            0,
            4 << 16 | 71,
            4,
            33,
            3,
            4 << 16 | 59,
            2,
            4,
            12,
        ];
        let interface = reflect(&words).unwrap();
        assert_eq!(interface.count, Some(4));
        assert!(interface.writes);
    }
}
