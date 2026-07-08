//! Naive WebGPU runtime: create buffers, bind, dispatch, read back.
//!
//! No persistent program cache beyond the shared device — each invoke rebuilds
//! GPU objects from the lowered [`WgpuProgram`] specs.

use std::borrow::Cow;
use std::sync::OnceLock;

use wgpu::util::DeviceExt;

use super::error::WgpuRuntimeError;
use super::program::{WgpuBufferSpec, WgpuPipelineSpec, WgpuProgram};

pub struct WgpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

pub fn shared_context() -> Result<&'static WgpuContext, WgpuRuntimeError> {
    static CTX: OnceLock<Result<WgpuContext, String>> = OnceLock::new();
    match CTX.get_or_init(create_context) {
        Ok(ctx) => Ok(ctx),
        Err(msg) => Err(WgpuRuntimeError::Message(msg.clone())),
    }
}

fn create_context() -> Result<WgpuContext, String> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter_options = wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    };
    let adapter = pollster::block_on(instance.request_adapter(&adapter_options))
        .or_else(|| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                force_fallback_adapter: true,
                ..adapter_options
            }))
        })
        .ok_or_else(|| "no suitable GPU adapter found".to_string())?;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("resin"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    ))
    .map_err(|e| format!("request_device: {e}"))?;

    Ok(WgpuContext { device, queue })
}

/// Run one program: upload params → dispatch queue → densify sinks to host.
pub fn run_program(
    ctx: &WgpuContext,
    program: &WgpuProgram,
    param_bytes: &[(usize, &[u8])],
    sink_out: &mut [(usize, &mut [u8])],
) -> Result<(), WgpuRuntimeError> {
    let buffers: Vec<wgpu::Buffer> = program
        .buffers
        .iter()
        .map(|spec| create_buffer(&ctx.device, spec))
        .collect::<Result<_, _>>()?;

    for &(index, bytes) in param_bytes {
        let spec = program
            .buffers
            .get(index)
            .ok_or_else(|| WgpuRuntimeError::Message(format!("bad param buffer {index}")))?;
        if bytes.len() as u64 != spec.nbytes {
            return Err(WgpuRuntimeError::Message(format!(
                "param buffer {index} size mismatch: expected {}, got {}",
                spec.nbytes,
                bytes.len()
            )));
        }
        ctx.queue.write_buffer(&buffers[index], 0, bytes);
    }

    let pipelines: Vec<wgpu::ComputePipeline> = program
        .pipelines
        .iter()
        .map(|spec| create_pipeline(&ctx.device, spec))
        .collect::<Result<_, _>>()?;

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("resin-run"),
        });

    for dispatch in &program.queue {
        let pipe_spec = &program.pipelines[dispatch.pipeline_index];
        let pipeline = &pipelines[dispatch.pipeline_index];

        if pipe_spec.clear_output_before_dispatch {
            encoder.clear_buffer(
                &buffers[dispatch.output_buffer_index],
                0,
                Some(program.buffers[dispatch.output_buffer_index].byte_len()),
            );
        }

        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: buffers[dispatch.output_buffer_index].as_entire_binding(),
        }];
        for (binding, &view_index) in dispatch.arg_view_indices.iter().enumerate() {
            let view = &program.buffer_views[view_index];
            entries.push(wgpu::BindGroupEntry {
                binding: (binding + 1) as u32,
                resource: buffers[view.buffer_index].as_entire_binding(),
            });
        }

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resin-bind-group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("resin-compute"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let [x, y, z] = pipe_spec.dispatch_size;
        if x > 0 {
            pass.dispatch_workgroups(x, y, z);
        }
        drop(pass);
    }

    ctx.queue.submit(Some(encoder.finish()));

    for entry in sink_out.iter_mut() {
        let (view_index, host_out) = entry;
        let view = &program.buffer_views[*view_index];
        let buf_spec = &program.buffers[view.buffer_index];
        let raw = read_buffer(ctx, &buffers[view.buffer_index], buf_spec.byte_len())?;
        let densified = densify_view(&raw, view.offset, &view.shape, &view.pitch)?;
        if densified.len() != host_out.len() {
            return Err(WgpuRuntimeError::Message(format!(
                "sink size mismatch: densified {} vs host {}",
                densified.len(),
                host_out.len()
            )));
        }
        host_out.copy_from_slice(&densified);
    }

    Ok(())
}

fn create_buffer(
    device: &wgpu::Device,
    spec: &WgpuBufferSpec,
) -> Result<wgpu::Buffer, WgpuRuntimeError> {
    let size = spec.byte_len();
    let usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;

    if let Some(init) = &spec.init {
        if init.len() as u64 != spec.nbytes {
            return Err(WgpuRuntimeError::Message(format!(
                "buffer init size mismatch: expected {}, got {}",
                spec.nbytes,
                init.len()
            )));
        }
        // Pad to wgpu min size if needed.
        let mut contents = init.to_vec();
        if contents.len() < size as usize {
            contents.resize(size as usize, 0);
        }
        Ok(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("resin-buffer"),
            contents: &contents,
            usage,
        }))
    } else {
        Ok(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resin-buffer"),
            size,
            usage,
            mapped_at_creation: false,
        }))
    }
}

fn create_pipeline(
    device: &wgpu::Device,
    spec: &WgpuPipelineSpec,
) -> Result<wgpu::ComputePipeline, WgpuRuntimeError> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("resin-shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(spec.wgsl.as_str())),
    });

    Ok(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("resin-pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some(spec.entry_point.as_str()),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    }))
}

fn read_buffer(
    ctx: &WgpuContext,
    buffer: &wgpu::Buffer,
    size: u64,
) -> Result<Vec<u8>, WgpuRuntimeError> {
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("resin-read-staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("resin-read"),
        });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    ctx.queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _ = ctx.device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .map_err(|_| WgpuRuntimeError::BufferMapFailed)?
        .map_err(|_| WgpuRuntimeError::BufferMapFailed)?;

    let mapped = slice.get_mapped_range();
    Ok(mapped.to_vec())
}

/// Host densify of a pitched view (same semantics as CPU gather).
fn densify_view(
    buffer: &[u8],
    offset: u32,
    shape: &[u32],
    pitch: &[u32],
) -> Result<Vec<u8>, WgpuRuntimeError> {
    let count: usize = if shape.is_empty() {
        1
    } else {
        shape.iter().map(|&d| d as usize).product()
    };
    let mut out = vec![0u8; count * 4];
    let mut coords = vec![0u32; shape.len()];
    for linear in 0..count {
        let mut rem = linear;
        for axis in (0..shape.len()).rev() {
            let dim = shape[axis] as usize;
            coords[axis] = if dim == 0 {
                0
            } else {
                (rem % dim) as u32
            };
            if dim != 0 {
                rem /= dim;
            }
        }
        let mut idx = offset as usize;
        for (c, p) in coords.iter().zip(pitch.iter()) {
            idx += (*c as usize) * (*p as usize);
        }
        let start = idx * 4;
        let end = start + 4;
        if end > buffer.len() {
            return Err(WgpuRuntimeError::Message(format!(
                "view densify OOB: index {idx}, buffer {} bytes",
                buffer.len()
            )));
        }
        out[linear * 4..linear * 4 + 4].copy_from_slice(&buffer[start..end]);
    }
    Ok(out)
}
