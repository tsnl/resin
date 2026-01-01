use super::*;

//
// API:
//

pub struct Draw2dRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,

    target_size_wh: [u32; 2],

    quad_batch_bind_group_layout: wgpu::BindGroupLayout,
    quad_batch_sampler: wgpu::Sampler,
}
impl Draw2dRenderer {
    pub fn create(device: wgpu::Device, queue: wgpu::Queue, target_size_wh: [u32; 2]) -> Arc<Self> {
        let quad_batch_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Draw2dRenderer.QuadBatch.BindGroupLayout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });
        let quad_batch_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Draw2dRenderer.QuadBatch.Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        Arc::new(Self {
            device,
            queue,
            target_size_wh,
            quad_batch_bind_group_layout,
            quad_batch_sampler,
        })
    }
}

pub struct Draw2dFrame {
    renderer: Arc<Draw2dRenderer>,

    output_image: Rgba32FloatTexture,

    quad_batch_map: HashMap<wgpu::Texture, QuadBatch>,
}
impl Draw2dFrame {
    pub fn new(renderer: Arc<Draw2dRenderer>) -> Self {
        let output_image = Rgba32FloatTexture::new(
            &renderer.device,
            renderer.target_size_wh,
            "Draw2dFrame.OutputImage",
        );
        let quad_batch_map = Default::default();
        Self {
            renderer,
            output_image,
            quad_batch_map,
        }
    }
}

pub struct Draw2dQuad {
    pub dst_xy_px: [u16; 2],
    pub dst_wh_px: Option<[u16; 2]>,
    pub src_xy_px: Option<[u16; 4]>,
    pub src_wh_px: Option<[u16; 2]>,
    pub fill_texture: Option<wgpu::Texture>,
    pub fill_color_rgba: [u8; 4],
    pub border_thickness_px: [u8; 4],
    pub border_color_rgba: [u8; 4],
}
impl Default for Draw2dQuad {
    fn default() -> Self {
        Self {
            dst_xy_px: [0, 0],
            dst_wh_px: None,
            src_xy_px: None,
            src_wh_px: None,
            fill_texture: None,
            fill_color_rgba: [0xFF; 4],
            border_thickness_px: [0; 4],
            border_color_rgba: [0x00; 4],
        }
    }
}

//
// Implementation:
//

struct QuadBatch {
    atlas: wgpu::Texture,
    dev_buffer: StorageBuffer<PodQuad>,
    host_buffer: StagingBuffer<PodQuad>,
    bind_group: wgpu::BindGroup,
    capacity: usize,
}
impl QuadBatch {
    fn new(
        device: &wgpu::Device,
        atlas: &wgpu::Texture,
        min_capacity: usize,
        quad_batch_bind_group_layout: &wgpu::BindGroupLayout,
        quad_batch_sampler: &wgpu::Sampler,
    ) -> Self {
        let capacity = min_capacity.next_power_of_two();
        let atlas = atlas.clone();
        let dev_buffer =
            StorageBuffer::<PodQuad>::new(device, capacity, "Draw2dFrame.QuadBatch.StorageBuffer");
        let host_buffer =
            StagingBuffer::<PodQuad>::new(device, capacity, "Draw2dFrame.QuadBatch.StagingBuffer");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Draw2dFrame.QuadBatch.BindGroup"),
            layout: quad_batch_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dev_buffer.wgpu_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(quad_batch_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        &atlas.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
            ],
        });
        Self {
            atlas,
            dev_buffer,
            host_buffer,
            bind_group,
            capacity,
        }
    }
    fn update(
        &mut self,
        device: &wgpu::Device,
        command_encoder: &mut wgpu::CommandEncoder,
        quad_batch_bind_group_layout: &wgpu::BindGroupLayout,
        quad_batch_sampler: &wgpu::Sampler,
        data: &[PodQuad],
    ) {
        // Ensure data fits in buffer, reallocating if necessary:
        self.realloc_if_needed(
            device,
            data.len(),
            quad_batch_bind_group_layout,
            quad_batch_sampler,
        );

        // Write data to device buffer via staging buffer:
        self.host_buffer.write(data);
        self.host_buffer.copy_to(&self.dev_buffer, command_encoder);
    }
    fn realloc_if_needed(
        &mut self,
        device: &wgpu::Device,
        target_capacity: usize,
        quad_batch_bind_group_layout: &wgpu::BindGroupLayout,
        quad_batch_sampler: &wgpu::Sampler,
    ) {
        debug_assert!(self.capacity.is_power_of_two());
        if target_capacity > self.capacity {
            *self = Self::new(
                device,
                &self.atlas,
                target_capacity,
                quad_batch_bind_group_layout,
                quad_batch_sampler,
            );
        }
        if target_capacity < self.capacity / 4 {
            *self = Self::new(
                device,
                &self.atlas,
                (self.capacity / 2).max(1),
                quad_batch_bind_group_layout,
                quad_batch_sampler,
            );
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PodQuad {
    dst_xy_ndc: [i16; 2],         // 2x2=4: 0..4
    dst_wh_ndc: [i16; 2],         // 2x2=4: 4..8
    src_xy_uv: [u16; 4],          // 2x4=8: 8..16
    src_wh_uv: [u16; 2],          // 2x2=4: 16..20
    fill_color_rgba: [u8; 4],     // 1x4=4: 20..24
    border_thickness_px: [u8; 4], // 1x4=4: 24..28
    border_color_rgba: [u8; 4],   // 1x4=4: 28..32
}
