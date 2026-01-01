use wgpu::include_wgsl;

use super::*;

//
// API:
//

pub struct Draw2dRenderer {
    device: wgpu::Device,

    target_size_wh: [u16; 2],

    bind_group_layout: wgpu::BindGroupLayout,

    pipeline: wgpu::RenderPipeline,

    default_white_texture: Rgba8UnormTexture,
    sampler: wgpu::Sampler,
}
impl Draw2dRenderer {
    pub fn create(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target_size_wh: [u16; 2],
    ) -> Arc<Self> {
        let device = device.clone();

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
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Cw,
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
                Rgba8UnormTexture::new(&device, [32, 32], "Draw2dRenderer.DefaultWhiteTexture");
            queue.write_texture(
                texture.texel_copy_texture_info(),
                bytemuck::cast_slice(&[0xFF_u8; 4 * 32 * 32]),
                texture.texel_copy_buffer_layout(),
                texture.size(),
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
            target_size_wh,
            bind_group_layout,
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
    output_image: Rgba8UnormTexture,
    quad_group_cache: HashMap<wgpu::Texture, QuadGroup>,
}
impl Draw2dFrame {
    pub fn new(device: &wgpu::Device, target_size_wh: [u16; 2]) -> Self {
        let output_image =
            Rgba8UnormTexture::new(device, target_size_wh, "Draw2dFrame.OutputImage");
        let quad_batch_cache = Default::default();
        Self {
            output_image,
            quad_group_cache: quad_batch_cache,
        }
    }
    pub fn output_image(&self) -> &Rgba8UnormTexture {
        &self.output_image
    }
    fn record(
        &mut self,
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        default_white_texture: &Rgba8UnormTexture,
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
                    device,
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

            for (texture, draw_range) in quad_batch_list.draw_ranges.iter() {
                let texture = texture
                    .clone()
                    .unwrap_or(default_white_texture.wgpu_texture().clone());
                let bind_group = group_bind_groups.get(&texture).unwrap();

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
            .or_insert_with(|| QuadGroup::new(device, key, quads.len(), bind_group_layout, sampler))
            .update(device, command_encoder, bind_group_layout, sampler, quads)
    }
}

pub struct Draw2dQuad {
    pub dst_xy_px: [u16; 2],
    pub dst_wh_px: Option<[u16; 2]>,
    pub src_xy_px: Option<[u16; 2]>,
    pub src_wh_px: Option<[u16; 2]>,
    pub fill_texture: Option<wgpu::Texture>,
    pub fill_color_rgba: [f32; 4],
    pub border_thickness_px: [u16; 4],
    pub border_color_rgba: [f32; 4],
}
impl Default for Draw2dQuad {
    fn default() -> Self {
        Self {
            dst_xy_px: [0, 0],
            dst_wh_px: None,
            src_xy_px: None,
            src_wh_px: None,
            fill_texture: None,
            fill_color_rgba: [1.0; 4],
            border_thickness_px: [0; 4],
            border_color_rgba: [1.0; 4],
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
    device_buffer: StorageBuffer<PodQuad>,
    staging_buffer: StagingBuffer<PodQuad>,
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
        let device_buffer =
            StorageBuffer::<PodQuad>::new(device, capacity, "Draw2dFrame.QuadBatch.DeviceBuffer");
        let staging_buffer =
            StagingBuffer::<PodQuad>::new(device, capacity, "Draw2dFrame.QuadBatch.StagingBuffer");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Draw2dFrame.QuadBatch.BindGroup"),
            layout: quad_batch_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: device_buffer.wgpu_buffer().as_entire_binding(),
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
            device_buffer,
            staging_buffer,
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
        self.staging_buffer.write(data);
        self.staging_buffer
            .copy_to_buffer(&self.device_buffer, command_encoder);

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
    dst_xy_ndc: [f32; 2],           // 4x2=8: 0..8
    dst_wh_ndc: [f32; 2],           // 4x2=8: 8..16
    src_xy_uv: [f32; 2],            // 4x2=8: 16..24
    src_wh_uv: [f32; 2],            // 4x2=8: 24..32
    fill_color_rgba: [f32; 4],      // 4x4=16: 32..48
    border_thickness_ndc: [f32; 4], // 4x4=16: 48..64
    border_color_rgba: [f32; 4],    // 4x4=16: 64..80
    _rsv: [f32; 4],                 // 4x4=16: 80..96
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
        macro_rules! ndc2_xy {
            ($a:expr) => {{
                [
                    ($a[0] as f32 / framebuffer_size_wh[0] as f32) * 2.0 - 1.0,
                    -($a[1] as f32 / framebuffer_size_wh[1] as f32) * 2.0 + 1.0,
                ]
            }};
        }
        macro_rules! ndc2_wh {
            ($a:expr) => {{
                [
                    ($a[0] as f32 / framebuffer_size_wh[0] as f32) * 2.0,
                    -($a[1] as f32 / framebuffer_size_wh[1] as f32) * 2.0,
                ]
            }};
        }
        macro_rules! ndc_trbl {
            ($a:expr) => {
                [
                    -($a[0] as f32 / framebuffer_size_wh[1] as f32) * 2.0,
                    ($a[1] as f32 / framebuffer_size_wh[0] as f32) * 2.0,
                    -($a[2] as f32 / framebuffer_size_wh[1] as f32) * 2.0,
                    ($a[3] as f32 / framebuffer_size_wh[0] as f32) * 2.0,
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

        let dst_xy_ndc = ndc2_xy!(original.dst_xy_px);
        let dst_wh_ndc = ndc2_wh!(original.dst_wh_px.unwrap_or(framebuffer_size_wh));
        let src_xy_uv = uv!(original.src_xy_px.unwrap_or([0, 0]));
        let src_wh_uv = uv!(original.src_wh_px.unwrap_or(texture_size!()));
        let fill_color_rgba = original.fill_color_rgba;
        let border_thickness_ndc = ndc_trbl!(original.border_thickness_px);
        let border_color_rgba = original.border_color_rgba;

        Self {
            dst_xy_ndc,
            dst_wh_ndc,
            src_xy_uv,
            src_wh_uv,
            fill_color_rgba,
            border_thickness_ndc,
            border_color_rgba,
            _rsv: [0.0; 4],
        }
    }
}

//
// Tests:
//

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;

    #[test]
    fn test_pod_quad_size() {
        assert_eq!(std::mem::size_of::<PodQuad>(), 96);
    }

    #[test]
    fn basic_render_test() {
        let wgpu_instance = wgpu::Instance::new(&Default::default());
        let adapter =
            pollster::block_on(wgpu_instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }));
        let (device, queue) =
            pollster::block_on(adapter.unwrap().request_device(&wgpu::DeviceDescriptor {
                label: Some("TestDevice"),
                ..Default::default()
            }))
            .unwrap();

        let rainbow_image = image::open("tests/data/rainbow-512x512.png").unwrap();
        let rainbow_texture = Rgba8UnormTexture::new(&device, [512, 512], "TestRainbowImage");
        queue.write_texture(
            rainbow_texture.texel_copy_texture_info(),
            rainbow_image.as_bytes(),
            rainbow_texture.texel_copy_buffer_layout(),
            rainbow_texture.size(),
        );
        // FIXME: need to convert loaded textures to linear colorspace for correct alpha blending

        // TODO: add a test-case to ensure our sorting logic works as expected

        let renderer = Draw2dRenderer::create(&device, &queue, [1024, 1024]);
        let mut frame = Draw2dFrame::new(&device, [1024, 1024]);
        let readback_buffer =
            ReadbackBuffer::<[u8; 4]>::new(&device, 1024 * 1024, "TestReadbackBuffer");

        let mut command_encoder =
            renderer
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("TestCommandEncoder"),
                });
        {
            let quads = vec![
                Draw2dQuad {
                    dst_xy_px: [10, 10],
                    dst_wh_px: Some([497, 497]),
                    fill_color_rgba: [1.0, 0.0, 0.0, 1.0],
                    ..Default::default()
                },
                Draw2dQuad {
                    dst_xy_px: [517, 10],
                    dst_wh_px: Some([497, 497]),
                    fill_texture: Some(rainbow_texture.wgpu_texture().clone()),
                    fill_color_rgba: [1.0, 1.0, 1.0, 1.0],
                    ..Default::default()
                },
            ];
            renderer.record(&quads, &mut frame, &mut command_encoder);

            readback_buffer.copy_from_texture(&frame.output_image, &mut command_encoder);
        }
        queue.submit(iter::once(command_encoder.finish()));

        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

        image::DynamicImage::ImageRgba8(
            image::RgbaImage::from_raw(
                1024,
                1024,
                bytemuck::cast_vec(readback_buffer.read().into_vec()),
            )
            .unwrap(),
        )
        .save("test_output.png")
        .unwrap();
    }
}
