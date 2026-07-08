//! `rasterize` executor for the wgpu backend: a render pipeline drawing
//! non-indexed triangles (vertex pulling from the clip-position storage
//! buffer) into an RGBA32Float visibility target with a D32 depth buffer,
//! then copied into the output storage buffer.
//!
//! wgpu's NDC (y up) maps row 0 of the framebuffer to NDC y = +1, matching
//! the node's row-0-is-top convention with no flip. `copy_texture_to_buffer`
//! requires 256-byte row alignment, so the texture is always copied into a
//! row-padded intermediate and repacked densely into the program's output
//! buffer by a tiny generated compute shader — one code path regardless of
//! width, and the padded path stays exercised by every test.

use std::borrow::Cow;

use super::error::WgpuRuntimeError;
use super::program::WgpuRasterStep;
use super::runtime::RunState;
use crate::backends::hwraster;

/// Dense-repack compute shader: padded rows → `[H, W, 4]` (one pixel per
/// thread; constants baked into the text like the rest of the codegen).
fn repack_wgsl(width: u32, height: u32, padded_row_floats: u32) -> String {
    let pixels = width * height;
    format!(
        "@group(0) @binding(0) var<storage, read> src: array<f32>;\n\
         @group(0) @binding(1) var<storage, read_write> dst: array<f32>;\n\
         @compute @workgroup_size(64)\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
             let pixel = gid.x;\n\
             if pixel >= {pixels}u {{ return; }}\n\
             let src_base = (pixel / {width}u) * {padded_row_floats}u + (pixel % {width}u) * 4u;\n\
             let dst_base = pixel * 4u;\n\
             dst[dst_base] = src[src_base];\n\
             dst[dst_base + 1u] = src[src_base + 1u];\n\
             dst[dst_base + 2u] = src[src_base + 2u];\n\
             dst[dst_base + 3u] = src[src_base + 3u];\n\
         }}\n"
    )
}

pub(super) fn execute(
    state: &mut RunState<'_>,
    step: &WgpuRasterStep,
) -> Result<(), WgpuRuntimeError> {
    let device = &state.ctx.device;
    let buffers = state.buffers;

    let extent = wgpu::Extent3d {
        width: step.width,
        height: step.height,
        depth_or_array_layers: 1,
    };
    let texture = |label, format, usage| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    };
    let color = texture(
        "resin-raster-color",
        wgpu::TextureFormat::Rgba32Float,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
    );
    let depth = texture(
        "resin-raster-depth",
        wgpu::TextureFormat::Depth32Float,
        wgpu::TextureUsages::RENDER_ATTACHMENT,
    );

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("resin-raster-shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(hwraster::RASTER_WGSL)),
    });
    // `layout: None`: the storage bindings (read-only, VERTEX visibility)
    // are inferred from the shader.
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("resin-raster-pipeline"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some(hwraster::VS_ENTRY),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..wgpu::PrimitiveState::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            // Strict '<': the first-drawn (lowest prim) triangle wins ties.
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some(hwraster::FS_ENTRY),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba32Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview: None,
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("resin-raster-geometry"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffers[step.positions_buffer_index].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffers[step.triangles_buffer_index].as_entire_binding(),
            },
        ],
    });

    let row_bytes = u64::from(step.width) * 16;
    let padded_row_bytes = row_bytes.next_multiple_of(u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT));
    let padded = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("resin-raster-padded"),
        size: padded_row_bytes * u64::from(step.height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let repack_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("resin-raster-repack"),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(repack_wgsl(
            step.width,
            step.height,
            (padded_row_bytes / 4) as u32,
        ))),
    });
    let repack_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("resin-raster-repack"),
        layout: None,
        module: &repack_shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let repack_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("resin-raster-repack"),
        layout: &repack_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: padded.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffers[step.output_buffer_index].as_entire_binding(),
            },
        ],
    });

    let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
    let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());

    let encoder = state.encoder();
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("resin-raster-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Background pixels stay all-zero.
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..step.triangle_count * 3, 0..1);
    }
    encoder.copy_texture_to_buffer(
        color.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &padded,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row_bytes as u32),
                rows_per_image: None,
            },
        },
        extent,
    );
    {
        let pixels = u64::from(step.width) * u64::from(step.height);
        let groups_x = pixels.div_ceil(64);
        // The naive 1-D repack dispatch caps at ~4.19M pixels (2048×2048);
        // error clearly instead of tripping driver validation.
        if groups_x > u64::from(crate::backends::hwtrace::MAX_WORKGROUPS_PER_DIM) {
            return Err(WgpuRuntimeError::Message(format!(
                "rasterize: {pixels} pixels need {groups_x} repack workgroups, exceeding the \
                 {} per-dimension dispatch limit",
                crate::backends::hwtrace::MAX_WORKGROUPS_PER_DIM
            )));
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("resin-raster-repack"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&repack_pipeline);
        pass.set_bind_group(0, &repack_bind_group, &[]);
        pass.dispatch_workgroups(groups_x as u32, 1, 1);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn repack_wgsl_validates() {
        let wgsl = super::repack_wgsl(50, 3, 224);
        let module = naga::front::wgsl::parse_str(&wgsl)
            .unwrap_or_else(|e| panic!("parse: {e}\n---\n{wgsl}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::default(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("validate: {e:?}\n---\n{wgsl}"));
    }
}
