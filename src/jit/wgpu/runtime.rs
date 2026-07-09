//! WebGPU runtime: create arena heaps, bind, dispatch, read back.
//!
//! After arena packing the IR has one buffer per element type. Each invoke
//! allocates those few GPU buffers, uploads param **regions**, binds the heaps
//! each shader needs (≤2), and densifies sink views back to the host.

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
    let usage = wgpu::BufferUsages::STORAGE
        | wgpu::BufferUsages::COPY_DST
        | wgpu::BufferUsages::COPY_SRC;

    // One GPU buffer per arena (typically 1–2).
    let mut host_shadow: Vec<Vec<u8>> = ir
        .buffers
        .iter()
        .map(|spec| {
            let size = byte_len(spec.len()) as usize;
            match &spec.init {
                Some(init) => {
                    let mut bytes = buffer_data_to_bytes(init);
                    bytes.resize(size, 0);
                    bytes
                }
                None => vec![0u8; size],
            }
        })
        .collect();

    // Paint param views into the arena shadows.
    for (array, &param) in params.iter().zip(&ir.params) {
        let view = ir.view(param);
        let etype = ir.buffer(view.buffer).element_type;
        if array.element_type() != etype {
            return Err(Error::ElementType {
                expected: etype,
                got: array.element_type(),
            });
        }
        let count = element_count(&view.accessor.shape);
        if array.as_data().len() != count {
            return Err(Error::Size {
                expected: count,
                got: array.as_data().len(),
            });
        }
        write_dense_region(
            &mut host_shadow[view.buffer.0],
            &view.accessor,
            array.as_data(),
        )?;
    }

    let buffers: Vec<wgpu::Buffer> = host_shadow
        .iter()
        .map(|bytes| {
            ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("resin-arena"),
                contents: bytes,
                usage,
            })
        })
        .collect();

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
        // Scatter remaps write sparsely into a cleared **region** of the arena.
        if let Kernel::Remap { info } = &dispatch.kernel
            && info.is_scatter()
        {
            clear_dense_region_gpu(
                ctx,
                &mut encoder,
                &buffers,
                ir,
                dispatch.output,
            )?;
        }

        let entries: Vec<wgpu::BindGroupEntry> = spec
            .heap_buffers
            .iter()
            .enumerate()
            .map(|(i, &buf)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffers[buf.0].as_entire_binding(),
            })
            .collect();

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
        let arena_len = ir.buffer(view.buffer).len();
        let raw = read_buffer(ctx, &buffers[view.buffer.0], byte_len(arena_len))?;
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

/// Zero a dense output view inside its arena (byte-ranged clear).
fn clear_dense_region_gpu(
    ctx: &Context,
    encoder: &mut wgpu::CommandEncoder,
    buffers: &[wgpu::Buffer],
    ir: &crate::ir::Program,
    output: crate::ir::BufferViewRef,
) -> Result<(), Error> {
    let view = ir.view(output);
    let acc = &view.accessor;
    // Dense view: contiguous element range [offset, offset+count).
    if !acc.is_dense() {
        return Err(Error::Wgpu("scatter clear requires dense output view".into()));
    }
    let count = element_count(&acc.shape) as u64;
    let offset_bytes = (acc.offset as u64) * 4;
    let size_bytes = count * 4;
    if size_bytes > 0 {
        encoder.clear_buffer(&buffers[view.buffer.0], offset_bytes, Some(size_bytes));
    }
    let _ = ctx;
    Ok(())
}

fn write_dense_region(
    shadow: &mut [u8],
    accessor: &Accessor,
    data: &ArrayData,
) -> Result<(), Error> {
    let count = element_count(&accessor.shape);
    if data.len() != count {
        return Err(Error::Size {
            expected: count,
            got: data.len(),
        });
    }
    let mut coords = vec![0; accessor.rank()];
    match data {
        ArrayData::F32(src) => {
            for (linear, &value) in src.iter().enumerate() {
                decode(linear, &accessor.shape, &mut coords);
                let i = accessor.index(&coords);
                let bytes = value.to_le_bytes();
                shadow[i * 4..i * 4 + 4].copy_from_slice(&bytes);
            }
        }
        ArrayData::U32(src) => {
            for (linear, &value) in src.iter().enumerate() {
                decode(linear, &accessor.shape, &mut coords);
                let i = accessor.index(&coords);
                let bytes = value.to_le_bytes();
                shadow[i * 4..i * 4 + 4].copy_from_slice(&bytes);
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

fn bytes_to_f32s(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn bytes_to_u32s(raw: &[u8]) -> Vec<u32> {
    raw.chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn read_buffer(ctx: &Context, buffer: &wgpu::Buffer, size: u64) -> Result<Vec<u8>, Error> {
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("resin-readback"),
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
    ctx.device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .map_err(|_| Error::Wgpu("readback channel closed".into()))?
        .map_err(|e| Error::Wgpu(format!("map_async: {e}")))?;
    let data = slice.get_mapped_range().to_vec();
    staging.unmap();
    Ok(data)
}

fn densify_f32(buffer: &[f32], accessor: &Accessor, out: &mut [f32]) -> Result<(), Error> {
    let count = element_count(&accessor.shape);
    if out.len() != count {
        return Err(Error::Size { expected: count, got: out.len() });
    }
    let mut coords = vec![0; accessor.rank()];
    for (linear, slot) in out.iter_mut().enumerate() {
        decode(linear, &accessor.shape, &mut coords);
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
        decode(linear, &accessor.shape, &mut coords);
        *slot = buffer[accessor.index(&coords)];
    }
    Ok(())
}

fn decode(linear: usize, shape: &[usize], coords: &mut [usize]) {
    debug_assert_eq!(coords.len(), shape.len());
    let mut rem = linear;
    for axis in (0..shape.len()).rev() {
        coords[axis] = rem % shape[axis];
        rem /= shape[axis];
    }
}
