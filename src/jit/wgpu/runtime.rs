//! WebGPU runtime: create buffers, bind, dispatch, read back.
//!
//! Naive: each invoke rebuilds GPU objects from the lowered program. Only the
//! device/queue pair is shared (and cached process-wide).

use std::borrow::Cow;
use std::sync::OnceLock;

use wgpu::util::DeviceExt;

use super::WgpuProgram;
use crate::ir::{Accessor, BufferData, Kernel, element_count};
use crate::jit::{Array, ArrayData, Error};
use crate::ops::ElementType;

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
    params: &[&Array],
    outputs: &mut [&mut Array],
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
                    let mut contents = buffer_data_to_bytes(init);
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
        let expected_type = ir.buffer(buffer_ref).element_type;
        if array.element_type() != expected_type {
            return Err(Error::ElementType {
                expected: expected_type,
                got: array.element_type(),
            });
        }
        if array.as_data().len() != expected {
            return Err(Error::Size { expected, got: array.as_data().len() });
        }
        ctx.queue
            .write_buffer(&buffers[buffer_ref.0], 0, &array_data_to_bytes(array.as_data()));
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
        // Scatter remaps write sparsely into a cleared target (matches CPU).
        if let Kernel::Remap { info } = &dispatch.kernel
            && info.is_scatter()
        {
            let out_buf = &buffers[ir.view(dispatch.output).buffer.0];
            let nbytes = byte_len(ir.buffer(ir.view(dispatch.output).buffer).len());
            encoder.clear_buffer(out_buf, 0, Some(nbytes));
        }
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
        let etype = ir.buffer(view.buffer).element_type;
        let raw = read_buffer(ctx, &buffers[view.buffer.0], byte_len(ir.buffer(view.buffer).len()))?;
        match (etype, array.as_data_mut()) {
            (ElementType::F32, ArrayData::F32(dst)) => {
                densify_f32(&bytes_to_f32s(&raw), &view.accessor, dst)?;
            }
            (ElementType::U32, ArrayData::U32(dst)) => {
                densify_u32(&bytes_to_u32s(&raw), &view.accessor, dst)?;
            }
            _ => {
                return Err(Error::ElementType {
                    expected: etype,
                    got: array.element_type(),
                });
            }
        }
    }
    Ok(())
}

fn byte_len(elements: usize) -> u64 {
    (elements as u64 * 4).max(4)
}

fn buffer_data_to_bytes(data: &BufferData) -> Vec<u8> {
    match data {
        BufferData::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        BufferData::U32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    }
}

fn array_data_to_bytes(data: &ArrayData) -> Vec<u8> {
    match data {
        ArrayData::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        ArrayData::U32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    }
}

fn bytes_to_f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn bytes_to_u32s(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
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

fn densify_f32(buffer: &[f32], accessor: &Accessor, out: &mut [f32]) -> Result<(), Error> {
    let count = element_count(&accessor.shape);
    if out.len() != count {
        return Err(Error::Size { expected: count, got: out.len() });
    }
    let mut coords = vec![0; accessor.rank()];
    for (linear, slot) in out.iter_mut().enumerate() {
        let mut rem = linear;
        for axis in (0..accessor.rank()).rev() {
            coords[axis] = rem % accessor.shape[axis];
            rem /= accessor.shape[axis];
        }
        *slot = buffer[accessor.index(&coords)];
    }
    Ok(())
}

fn densify_u32(buffer: &[u32], accessor: &Accessor, out: &mut [u32]) -> Result<(), Error> {
    let count = element_count(&accessor.shape);
    if out.len() != count {
        return Err(Error::Size { expected: count, got: out.len() });
    }
    let mut coords = vec![0; accessor.rank()];
    for (linear, slot) in out.iter_mut().enumerate() {
        let mut rem = linear;
        for axis in (0..accessor.rank()).rev() {
            coords[axis] = rem % accessor.shape[axis];
            rem /= accessor.shape[axis];
        }
        *slot = buffer[accessor.index(&coords)];
    }
    Ok(())
}
