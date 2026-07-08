//! `trace_rays` executor for the wgpu backend.
//!
//! Two modes (selected at dispatch time):
//! - **hardware**: BLAS/TLAS build + WGSL ray query, when the device has
//!   `EXPERIMENTAL_RAY_QUERY` / `EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE`;
//! - **compute fallback**: host BVH build over (possibly device-produced)
//!   geometry + a WGSL traversal kernel, on any adapter.

use std::borrow::Cow;

use wgpu::util::DeviceExt;

use super::error::WgpuRuntimeError;
use super::program::WgpuTraceStep;
use super::runtime::RunState;
use crate::backends::hwtrace;

pub(super) fn execute(
    state: &mut RunState<'_>,
    step: &WgpuTraceStep,
) -> Result<(), WgpuRuntimeError> {
    if step.ray_count == 0 {
        return Ok(());
    }
    // An acceleration structure cannot be built over zero triangles; the
    // fallback's empty BVH handles that case on any device.
    if state.ctx.supports_ray_tracing() && step.triangle_count > 0 && !hwtrace::force_fallback()
    {
        execute_hardware(state, step)
    } else {
        execute_fallback(state, step)
    }
}

fn execute_hardware(
    state: &mut RunState<'_>,
    step: &WgpuTraceStep,
) -> Result<(), WgpuRuntimeError> {
    let device = &state.ctx.device;
    let vertex_buffer = &state.buffers[step.vertices_buffer_index()];
    let index_buffer = &state.buffers[step.triangles_buffer_index()];

    // Geometry is graph data, so the BLAS/TLAS are rebuilt per invoke.
    let size_desc = wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: step.vertex_count,
        index_format: Some(wgpu::IndexFormat::Uint32),
        index_count: Some(step.triangle_count * 3),
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };
    let blas = device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("resin-trace-blas"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![size_desc.clone()],
        },
    );
    let tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("resin-trace-tlas"),
        max_instances: 1,
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
    });
    let mut package = wgpu::TlasPackage::new(tlas);
    let identity = [
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0,
    ];
    *package
        .get_mut_single(0)
        .expect("max_instances = 1") = Some(wgpu::TlasInstance::new(&blas, identity, 0, 0xff));

    state.encoder().build_acceleration_structures(
        std::iter::once(&wgpu::BlasBuildEntry {
            blas: &blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size: &size_desc,
                vertex_buffer,
                first_vertex: 0,
                vertex_stride: 12,
                index_buffer: Some(index_buffer),
                first_index: Some(0),
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }),
        std::iter::once(&package),
    );

    let pipeline = create_pipeline(
        device,
        &hwtrace::ray_query_kernel(step.ray_count),
        "resin-trace-rq",
    );
    let mut entries = ray_bindings(state, step);
    entries.push(wgpu::BindGroupEntry {
        binding: 5,
        resource: wgpu::BindingResource::AccelerationStructure(package.tlas()),
    });
    let bind_group = state.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("resin-trace-rq"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    dispatch(state, &pipeline, &bind_group, step.ray_count)?;

    // The Blas/TlasPackage must outlive the recorded build; submit before
    // this frame's locals drop.
    state.flush();
    Ok(())
}

fn execute_fallback(
    state: &mut RunState<'_>,
    step: &WgpuTraceStep,
) -> Result<(), WgpuRuntimeError> {
    // Geometry may have been produced on-device by earlier steps: read it
    // back (flushing pending work) and build the BVH on the host.
    let vertex_bytes = state.read_buffer_bytes(step.vertices_buffer_index())?;
    let triangle_bytes = state.read_buffer_bytes(step.triangles_buffer_index())?;
    let vertices = hwtrace::bytes_to_f32(&vertex_bytes[..step.vertex_count as usize * 12]);
    let triangles = hwtrace::bytes_to_u32(&triangle_bytes[..step.triangle_count as usize * 12]);
    let bvh = hwtrace::build_bvh(&vertices, &triangles);

    let device = &state.ctx.device;
    let upload = |label: &str, words: &[u32]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: &hwtrace::u32s_to_bytes(words),
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let tris_buffer = upload("resin-trace-tris", &bvh.packed_tris);
    let nodes_buffer = upload("resin-trace-nodes", &bvh.nodes);

    let pipeline = create_pipeline(
        device,
        &hwtrace::fallback_kernel(step.ray_count),
        "resin-trace-bvh",
    );
    let mut entries = ray_bindings(state, step);
    entries.push(wgpu::BindGroupEntry {
        binding: 5,
        resource: state.buffers[step.vertices_buffer_index()].as_entire_binding(),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: 6,
        resource: tris_buffer.as_entire_binding(),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: 7,
        resource: nodes_buffer.as_entire_binding(),
    });
    let bind_group = state.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("resin-trace-bvh"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    dispatch(state, &pipeline, &bind_group, step.ray_count)?;
    Ok(())
}

/// Bindings shared by both kernels: 0 = output, 1..4 = ray buffers.
fn ray_bindings<'a>(
    state: &'a RunState<'_>,
    step: &WgpuTraceStep,
) -> Vec<wgpu::BindGroupEntry<'a>> {
    let mut entries = vec![wgpu::BindGroupEntry {
        binding: 0,
        resource: state.buffers[step.output_buffer_index].as_entire_binding(),
    }];
    for (binding, &arg) in step.arg_buffer_indices[..4].iter().enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: (binding + 1) as u32,
            resource: state.buffers[arg].as_entire_binding(),
        });
    }
    entries
}

fn create_pipeline(device: &wgpu::Device, wgsl: &str, label: &str) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(wgsl)),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn dispatch(
    state: &mut RunState<'_>,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    ray_count: u32,
) -> Result<(), WgpuRuntimeError> {
    let groups_x = hwtrace::dispatch_x(ray_count).map_err(WgpuRuntimeError::Message)?;
    let mut pass = state
        .encoder()
        .begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("resin-trace"),
            timestamp_writes: None,
        });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(groups_x, 1, 1);
    Ok(())
}
