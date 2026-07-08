//! Naive WebGPU runtime: create buffers, bind, dispatch, read back.
//!
//! No persistent program cache beyond the shared device — each invoke rebuilds
//! GPU objects from the lowered [`WgpuProgram`] specs. Compute steps batch
//! into one command encoder; hardware steps (`trace_rays` / `rasterize`) may
//! flush the encoder when they need host-side work (e.g. a BVH build for the
//! compute fallback).

use std::borrow::Cow;
use std::collections::HashSet;
use std::sync::OnceLock;

use wgpu::util::DeviceExt;

use super::error::WgpuRuntimeError;
use super::program::{WgpuBufferSpec, WgpuPipelineSpec, WgpuProgram, WgpuStep};

/// Features required for the hardware ray-query path.
pub(super) fn ray_tracing_features() -> wgpu::Features {
    wgpu::Features::EXPERIMENTAL_RAY_QUERY
        | wgpu::Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE
}

pub struct WgpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// Features actually enabled on the device (ray tracing is requested
    /// opportunistically).
    pub features: wgpu::Features,
}

impl WgpuContext {
    pub fn supports_ray_tracing(&self) -> bool {
        self.features.contains(ray_tracing_features())
    }
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

    // Ray tracing is opportunistic: enabled when the adapter offers it, so
    // trace_rays can pick the hardware path at dispatch time. The features
    // are experimental, so a driver that advertises but fails to enable them
    // must not take down plain compute — retry featureless.
    let rt = ray_tracing_features();
    let mut required_features = if adapter.features().contains(rt) {
        rt
    } else {
        wgpu::Features::empty()
    };
    let request = |features: wgpu::Features| {
        pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("resin"),
                required_features: features,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
    };
    let (device, queue) = match request(required_features) {
        Ok(pair) => pair,
        Err(_) if !required_features.is_empty() => {
            required_features = wgpu::Features::empty();
            request(required_features).map_err(|e| format!("request_device: {e}"))?
        }
        Err(e) => return Err(format!("request_device: {e}")),
    };

    Ok(WgpuContext {
        device,
        queue,
        features: required_features,
    })
}

/// Mutable execution state threaded through the step executors.
pub(super) struct RunState<'a> {
    pub ctx: &'a WgpuContext,
    pub program: &'a WgpuProgram,
    pub buffers: &'a [wgpu::Buffer],
    encoder: Option<wgpu::CommandEncoder>,
}

impl RunState<'_> {
    /// Current command encoder (created on demand).
    pub fn encoder(&mut self) -> &mut wgpu::CommandEncoder {
        self.encoder.get_or_insert_with(|| {
            self.ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("resin-run"),
                })
        })
    }

    /// Submit all recorded work. Step executors call this before host-side
    /// readback (e.g. building a BVH from device-produced geometry).
    pub fn flush(&mut self) {
        if let Some(encoder) = self.encoder.take() {
            self.ctx.queue.submit(Some(encoder.finish()));
        }
    }

    /// Blocking read of a whole device buffer (flushes pending work).
    pub fn read_buffer_bytes(&mut self, buffer_index: usize) -> Result<Vec<u8>, WgpuRuntimeError> {
        self.flush();
        read_buffer(
            self.ctx,
            &self.buffers[buffer_index],
            self.program.buffers[buffer_index].byte_len(),
        )
    }
}

/// Run one program: upload params → execute steps → densify sinks to host.
pub fn run_program(
    ctx: &WgpuContext,
    program: &WgpuProgram,
    param_bytes: &[(usize, &[u8])],
    sink_out: &mut [(usize, &mut [u8])],
) -> Result<(), WgpuRuntimeError> {
    // Geometry consumed by acceleration-structure builds needs BLAS_INPUT.
    let blas_input_buffers: HashSet<usize> = if ctx.supports_ray_tracing() {
        program.trace_geometry_buffer_indices().collect()
    } else {
        HashSet::new()
    };

    let buffers: Vec<wgpu::Buffer> = program
        .buffers
        .iter()
        .enumerate()
        .map(|(index, spec)| {
            let extra = if blas_input_buffers.contains(&index) {
                wgpu::BufferUsages::BLAS_INPUT
            } else {
                wgpu::BufferUsages::empty()
            };
            create_buffer(&ctx.device, spec, extra)
        })
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

    let mut state = RunState {
        ctx,
        program,
        buffers: &buffers,
        encoder: None,
    };

    for step in &program.queue {
        match step {
            WgpuStep::Compute(dispatch) => {
                let pipe_spec = &program.pipelines[dispatch.pipeline_index];
                let pipeline = &pipelines[dispatch.pipeline_index];
                encode_compute(&mut state, dispatch, pipe_spec, pipeline)?;
            }
            WgpuStep::TraceRays(step) => super::trace::execute(&mut state, step)?,
            WgpuStep::Rasterize(step) => super::raster::execute(&mut state, step)?,
        }
    }

    state.flush();

    for entry in sink_out.iter_mut() {
        let (view_index, host_out) = entry;
        let view = &program.buffer_views[*view_index];
        let buf_spec = &program.buffers[view.buffer_index];
        let raw = read_buffer(ctx, &buffers[view.buffer_index], buf_spec.byte_len())?;
        let densified = crate::backends::densify_view(&raw, view.offset, &view.shape, &view.pitch)
            .map_err(WgpuRuntimeError::Message)?;
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

fn encode_compute(
    state: &mut RunState<'_>,
    dispatch: &super::program::WgpuDispatch,
    pipe_spec: &WgpuPipelineSpec,
    pipeline: &wgpu::ComputePipeline,
) -> Result<(), WgpuRuntimeError> {
    let program = state.program;
    let buffers = state.buffers;
    let device = &state.ctx.device;

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

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("resin-bind-group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });

    let encoder = state.encoder();
    if pipe_spec.clear_output_before_dispatch {
        encoder.clear_buffer(
            &buffers[dispatch.output_buffer_index],
            0,
            Some(program.buffers[dispatch.output_buffer_index].byte_len()),
        );
    }

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
    Ok(())
}

fn create_buffer(
    device: &wgpu::Device,
    spec: &WgpuBufferSpec,
    extra_usage: wgpu::BufferUsages,
) -> Result<wgpu::Buffer, WgpuRuntimeError> {
    let size = spec.byte_len();
    let usage = wgpu::BufferUsages::STORAGE
        | wgpu::BufferUsages::COPY_DST
        | wgpu::BufferUsages::COPY_SRC
        | extra_usage;

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

pub(super) fn read_buffer(
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

