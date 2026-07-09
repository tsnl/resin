//! WebGPU runtime: create buffers, bind, dispatch, read back.
//!
//! Naive: each invoke rebuilds GPU objects from the lowered program. Only the
//! device/queue pair is shared (and cached process-wide).

use std::borrow::Cow;
use std::sync::OnceLock;

use wgpu::util::DeviceExt;

use super::WgpuProgram;
use crate::ir::{Accessor, element_count};
use crate::jit::Error;

pub struct Context {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

pub fn shared_context() -> Result<&'static Context, Error> {
    static CTX: OnceLock<Result<Context, String>> = OnceLock::new();
    match CTX.get_or_init(create_context) {
        Ok(ctx) => Ok(ctx),
        Err(msg) => Err(Error::Wgpu(msg.clone())),
    }
}

fn create_context() -> Result<Context, String> {
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

    Ok(Context { device, queue })
}

/// Upload params → run the dispatch queue → densify sinks back to the host.
pub fn run(
    ctx: &Context,
    program: &WgpuProgram,
    params: &[&crate::jit::Array],
    outputs: &mut [&mut crate::jit::Array],
) -> Result<(), Error> {
    let ir = &program.ir;
    let buffers: Vec<wgpu::Buffer> = ir
        .buffers
        .iter()
        .map(|spec| {
            let usage = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC;
            let size = byte_len(spec.len());
            match &spec.init {
                Some(init) => {
                    let mut contents = f32s_to_bytes(init);
                    contents.resize(size as usize, 0); // wgpu minimum size
                    ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("resin-buffer"),
                        contents: &contents,
                        usage,
                    })
                }
                None => ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("resin-buffer"),
                    size,
                    usage,
                    mapped_at_creation: false,
                }),
            }
        })
        .collect();

    for (array, &buffer_ref) in params.iter().zip(&ir.params) {
        let expected = ir.buffer(buffer_ref).len();
        if array.data().len() != expected {
            return Err(Error::Size { expected, got: array.data().len() });
        }
        ctx.queue
            .write_buffer(&buffers[buffer_ref.0], 0, &f32s_to_bytes(array.data()));
    }

    let pipelines: Vec<wgpu::ComputePipeline> = program
        .pipelines
        .iter()
        .map(|spec| {
            let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("resin-shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&spec.wgsl)),
            });
            ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("resin-pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        })
        .collect();

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("resin-run") });

    for (dispatch, &pipeline_index) in ir.queue.iter().zip(&program.pipeline_of) {
        let spec = &program.pipelines[pipeline_index];
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: buffers[ir.view(dispatch.output).buffer.0].as_entire_binding(),
        }];
        for (i, &arg) in dispatch.args.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: (i + 1) as u32,
                resource: buffers[ir.view(arg).buffer.0].as_entire_binding(),
            });
        }
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resin-bind-group"),
            layout: &pipelines[pipeline_index].get_bind_group_layout(0),
            entries: &entries,
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("resin-compute"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines[pipeline_index]);
        pass.set_bind_group(0, &bind_group, &[]);
        let [x, y, z] = spec.workgroups;
        if x > 0 {
            pass.dispatch_workgroups(x, y, z);
        }
    }

    ctx.queue.submit(Some(encoder.finish()));

    for (array, &sink) in outputs.iter_mut().zip(&ir.sinks) {
        let view = ir.view(sink);
        let raw = read_buffer(ctx, &buffers[view.buffer.0], byte_len(ir.buffer(view.buffer).len()))?;
        densify(&bytes_to_f32s(&raw), &view.accessor, array.data_mut())?;
    }
    Ok(())
}

fn byte_len(elements: usize) -> u64 {
    (elements as u64 * 4).max(4)
}

fn f32s_to_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytes_to_f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn read_buffer(ctx: &Context, buffer: &wgpu::Buffer, size: u64) -> Result<Vec<u8>, Error> {
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("resin-read-staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("resin-read") });
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
        .map_err(|_| Error::Wgpu("buffer map callback dropped".into()))?
        .map_err(|e| Error::Wgpu(format!("buffer map failed: {e:?}")))?;

    let mapped = slice.get_mapped_range();
    Ok(mapped.to_vec())
}

/// Densify a pitched view into a row-major host slice (same semantics as the
/// CPU backend's sink gather).
fn densify(buffer: &[f32], accessor: &Accessor, out: &mut [f32]) -> Result<(), Error> {
    let s = accessor.strided();
    let count = element_count(&s.shape);
    if out.len() != count {
        return Err(Error::Size { expected: count, got: out.len() });
    }
    let mut coords = vec![0; s.rank()];
    for (linear, slot) in out.iter_mut().enumerate() {
        let mut rem = linear;
        for axis in (0..s.rank()).rev() {
            coords[axis] = rem % s.shape[axis];
            rem /= s.shape[axis];
        }
        *slot = buffer[s.index(&coords)];
    }
    Ok(())
}
