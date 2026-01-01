use simd_math::SimdUVec2;
use wgpu::include_wgsl;

use super::*;

//
// API:
//

pub struct Draw2dRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,

    target_size_wh: [u16; 2],

    bind_group_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,

    pipeline: wgpu::RenderPipeline,

    default_white_texture: Rgba32FloatTexture,
    sampler: wgpu::Sampler,
}
impl Draw2dRenderer {
    pub fn create(device: wgpu::Device, queue: wgpu::Queue, target_size_wh: [u16; 2]) -> Arc<Self> {
        // Layout:
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Draw2dRenderer.PipelineLayout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        // Pipeline:
        let shader_module = device.create_shader_module(include_wgsl!("draw_2d.wgsl"));
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Draw2dRenderer.Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba32Float,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        // Bind resources:
        let default_white_texture = {
            let texture =
                Rgba32FloatTexture::new(&device, [1, 1], "Draw2dRenderer.DefaultWhiteTexture");
            queue.write_texture(
                texture.texel_copy_texture_info(),
                bytemuck::cast_slice(&[1.0_f32; 4]),
                texture.texel_copy_buffer_layout(),
                texture.extent_3d(),
            );
            texture
        };
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Draw2dRenderer.QuadBatch.Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // Done:
        Arc::new(Self {
            device,
            queue,
            target_size_wh,
            bind_group_layout,
            pipeline_layout,
            pipeline,
            default_white_texture,
            sampler,
        })
    }
    pub fn record(
        &self,
        quads: &[Draw2dQuad],
        frame: &mut Draw2dFrame,
        command_encoder: &mut wgpu::CommandEncoder,
    ) {
        frame.record(
            &self.device,
            &self.bind_group_layout,
            &self.sampler,
            &self.default_white_texture,
            &self.pipeline,
            self.target_size_wh,
            command_encoder,
            quads,
        );
    }
}

/// Draw2dFrame encapsulates a single frame in flight for 2D drawing, including mutable state.
/// Users are responsible for ensuring that a Draw2dFrame instance is not used across multiple frames in flight.
pub struct Draw2dFrame {
    output_image: Rgba32FloatTexture,

    quad_group_cache: HashMap<wgpu::Texture, QuadGroup>,
}
impl Draw2dFrame {
    pub fn new(device: &wgpu::Device, target_size_wh: [u16; 2]) -> Self {
        let output_image =
            Rgba32FloatTexture::new(device, target_size_wh, "Draw2dFrame.OutputImage");
        let quad_batch_cache = Default::default();
        Self {
            output_image,
            quad_group_cache: quad_batch_cache,
        }
    }
    fn record(
        &mut self,
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        default_white_texture: &Rgba32FloatTexture,
        pipeline: &wgpu::RenderPipeline,
        target_size_wh: [u16; 2],
        command_encoder: &mut wgpu::CommandEncoder,
        quads: &[Draw2dQuad],
    ) {
        // TODO: Resize output image if needed, recreate pipeline if size changed.

        // Organize quads into batches:
        let quad_batch_list = QuadBatchList::new(quads, target_size_wh);

        // Upload group data to GPU, using cached QuadGroup instances where possible:
        let group_bind_groups = {
            let mut res: HashMap<wgpu::Texture, wgpu::BindGroup> = Default::default();
            for (texture, group_quads) in quad_batch_list.bind_groups {
                let texture = texture.unwrap_or(default_white_texture.wgpu_texture().clone());
                let bind_group = self.acquire_quad_group(
                    &device,
                    command_encoder,
                    &texture,
                    &group_quads,
                    bind_group_layout,
                    sampler,
                );
                res.insert(texture.clone(), bind_group);
            }
            res
        };

        // Draw:
        for (texture, draw_range) in quad_batch_list.draw_ranges.iter() {
            let texture = texture
                .clone()
                .unwrap_or(default_white_texture.wgpu_texture().clone());
            let bind_group = group_bind_groups.get(&texture).unwrap();

            {
                let mut rp = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Draw2dFrame.RenderPass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self
                            .output_image
                            .wgpu_texture()
                            .create_view(&Default::default()),
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                rp.set_pipeline(pipeline);
                rp.set_bind_group(0, bind_group, &[]);
                rp.draw(0..6, (draw_range.start as u32)..(draw_range.end as u32));
            }
        }
    }
    fn acquire_quad_group(
        &mut self,
        device: &wgpu::Device,
        command_encoder: &mut wgpu::CommandEncoder,
        key: &wgpu::Texture,
        quads: &[PodQuad],
        bind_group_layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        self.quad_group_cache
            .entry(key.clone())
            .or_insert_with(|| {
                QuadGroup::new(device, &key, quads.len(), bind_group_layout, sampler)
            })
            .update(device, command_encoder, bind_group_layout, sampler, quads)
    }
}

pub struct Draw2dQuad {
    pub dst_xy_px: [u16; 2],
    pub dst_wh_px: Option<[u16; 2]>,
    pub src_xy_px: Option<[u16; 2]>,
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

struct QuadBatchList {
    draw_ranges: Vec<(Option<wgpu::Texture>, Range<usize>)>,
    bind_groups: HashMap<Option<wgpu::Texture>, Vec<PodQuad>>,
}
impl QuadBatchList {
    fn new(quads: &[Draw2dQuad], framebuffer_size_wh: [u16; 2]) -> Self {
        // First, identify contiguous runs of quads that share the same texture atlas:
        let mut runs = vec![0..1];
        for (i, quad) in quads.iter().enumerate().skip(1) {
            let prev_quad = &quads[i - 1];
            if quad.fill_texture.as_ref() == prev_quad.fill_texture.as_ref() {
                let last_run = runs.last_mut().unwrap();
                last_run.end += 1;
            } else {
                runs.push(i..(i + 1));
            }
        }

        // Transform runs in global space into runs in group space while assembling the group quads:
        let mut bind_groups = HashMap::default();
        let mut draw_calls = Vec::default();
        for run in runs.into_iter() {
            let image = quads[run.start].fill_texture.clone();

            let group_quads: &mut Vec<_> = bind_groups.entry(image.clone()).or_default();
            let group_offset = group_quads.len();
            let group_length = run.len();
            group_quads.extend(
                quads[run.clone()]
                    .iter()
                    .map(|quad| PodQuad::new(quad, framebuffer_size_wh)),
            );

            draw_calls.push((image, group_offset..group_offset + group_length))
        }

        // Done:
        Self {
            draw_ranges: draw_calls,
            bind_groups,
        }
    }
}

/// QuadGroup manages GPU resources for several quads that share the same texture atlas.
/// Each group is subdivided into "batches" that can be drawn in a single draw call.
struct QuadGroup {
    atlas: wgpu::Texture,
    dev_buffer: StorageBuffer<PodQuad>,
    host_buffer: StagingBuffer<PodQuad>,
    bind_group: wgpu::BindGroup,
    capacity: usize,
}
impl QuadGroup {
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
    ) -> wgpu::BindGroup {
        // Ensure data fits in buffer, reallocating if necessary:
        self.realloc_if_needed(
            device,
            data.len(),
            quad_batch_bind_group_layout,
            quad_batch_sampler,
        );

        // Write data to device buffer via staging buffer:
        self.host_buffer.write(data);
        self.host_buffer
            .copy_to_buffer(&self.dev_buffer, command_encoder);

        // Done: return the latest bind group
        self.bind_group.clone()
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
    src_xy_uv: [u16; 2],          // 2x2=4: 8..12
    src_wh_uv: [u16; 2],          // 2x2=4: 12..16
    fill_color_rgba: [u8; 4],     // 1x4=4: 16..20
    border_thickness_px: [u8; 4], // 1x4=4: 20..24
    border_color_rgba: [u8; 4],   // 1x4=4: 24..28
    _rsv: [u8; 4],                // 1x4=4: 28..32
}
impl PodQuad {
    fn new(original: &Draw2dQuad, framebuffer_size_wh: [u16; 2]) -> Self {
        macro_rules! texture_size {
            () => {
                original
                    .fill_texture
                    .as_ref()
                    .map(|original| {
                        [
                            original.width().try_into().unwrap(),
                            original.height().try_into().unwrap(),
                        ]
                    })
                    .unwrap_or([1, 1])
            };
        }
        macro_rules! ndc {
            ($a:expr) => {
                [
                    ($a[0] as f32 / framebuffer_size_wh[0] as f32) * 2.0 - 1.0,
                    -($a[1] as f32 / framebuffer_size_wh[1] as f32) * 2.0 + 1.0,
                ]
            };
        }
        macro_rules! uv {
            ($a:expr) => {{
                let tex_wh = texture_size!();
                [
                    $a[0] as f32 / tex_wh[0] as f32,
                    $a[1] as f32 / tex_wh[1] as f32,
                ]
            }};
        }
        macro_rules! fx_i16 {
            ($v:expr) => {
                [fp32_to_fx_i16($v[0]), fp32_to_fx_i16($v[1])]
            };
        }
        macro_rules! fx_u16 {
            ($v:expr) => {
                [fp32_to_fx_u16($v[0]), fp32_to_fx_u16($v[1])]
            };
        }

        let dst_xy_ndc = fx_i16!(ndc!(original.dst_xy_px));
        let dst_wh_ndc = fx_i16!(ndc!(original.dst_wh_px.unwrap_or(framebuffer_size_wh)));
        let src_xy_uv = fx_u16!(uv!(original.src_xy_px.unwrap_or([0, 0])));
        let src_wh_uv = fx_u16!(uv!(original.src_wh_px.unwrap_or(texture_size!())));
        let fill_color_rgba = original.fill_color_rgba;
        let border_thickness_px = original.border_thickness_px;
        let border_color_rgba = original.border_color_rgba;

        Self {
            dst_xy_ndc,
            dst_wh_ndc,
            src_xy_uv,
            src_wh_uv,
            fill_color_rgba,
            border_thickness_px,
            border_color_rgba,
            _rsv: [0; 4],
        }
    }
}
