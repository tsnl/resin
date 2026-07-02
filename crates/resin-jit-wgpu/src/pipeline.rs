//! Compiled graph factory and per-instance pipelines.
//!
//! - [`PipelineFactory`]: shared immutable program (shaders/compute pipelines) + schema.
//! - [`Pipeline`]: one executable instance with its own device buffers (not re-entrant for
//!   overlapping GPU work; create multiple instances for triple-buffering).
//! - [`GpuFuture`]: completion token for a submit (host `wait`; backends may later attach
//!   native sync objects such as Vulkan semaphores).

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::Arc;

use resin_dsl::View;
use resin_ir::IrProgram;
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::lower::build_wgpu_program;
use crate::program::{
    WgpuBufferSpec, WgpuComputePipelineSpec, WgpuCopy, WgpuDispatch, WgpuProgram, WgpuQueueOp,
};
use crate::WgslKernelConfig;

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("config error: {0}")]
    Config(String),
    #[error("program error: {0}")]
    Program(String),
    #[error("buffer map failed")]
    BufferMapFailed,
    #[error("backend error: {0}")]
    Backend(String),
    #[error("{0}")]
    Compile(String),
}

impl From<wgpu::RequestDeviceError> for PipelineError {
    fn from(e: wgpu::RequestDeviceError) -> Self {
        Self::Backend(e.to_string())
    }
}

/// Shared GPU device/queue (optional handle for creating many factories).
#[derive(Clone)]
pub struct DeviceContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl DeviceContext {
    pub fn from_config(config: &DeviceConfig) -> Result<Self, PipelineError> {
        let (device, queue) = match &config.device_name {
            None => request_default_device()?,
            Some(name) => request_named_device(name)?,
        };
        Ok(Self { device, queue })
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}

/// Device selection (replaces former `InterpConfig`).
#[derive(Debug, Clone, Default)]
pub struct DeviceConfig {
    pub device_name: Option<String>,
}

impl DeviceConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_device_name(device_name: impl Into<String>) -> Self {
        Self {
            device_name: Some(device_name.into()),
        }
    }
}

struct SharedProgram {
    ctx: DeviceContext,
    program: WgpuProgram,
    /// Immutable GPU compute pipelines (shareable across instances).
    pipelines: Vec<wgpu::ComputePipeline>,
    /// Named inputs (IR params), in registration order.
    input_names: Vec<String>,
    /// Named outputs (IR sinks).
    output_names: Vec<String>,
}

/// Blueprint: compiled graph + shared shader/pipeline state. Not executable alone.
#[derive(Clone)]
pub struct PipelineFactory {
    shared: Arc<SharedProgram>,
}

impl PipelineFactory {
    /// Construct a factory from an already-lowered [`WgpuProgram`].
    pub fn from_program(ctx: DeviceContext, program: WgpuProgram) -> Result<Self, PipelineError> {
        let pipelines = program
            .pipelines
            .iter()
            .map(|spec| create_pipeline(&ctx.device, spec))
            .collect::<Result<Vec<_>, _>>()?;
        let input_names: Vec<String> = program.param_buffers.keys().cloned().collect();
        let output_names: Vec<String> = program.sinks.keys().cloned().collect();
        Ok(Self {
            shared: Arc::new(SharedProgram {
                ctx,
                program,
                pipelines,
                input_names,
                output_names,
            }),
        })
    }

    /// Allocate a fresh instance (own buffers + bind groups). Safe to run in parallel
    /// with other instances from the same factory (separate buffer sets).
    pub fn create(&self) -> Result<Pipeline, PipelineError> {
        let buffers = self
            .shared
            .program
            .buffers
            .iter()
            .map(|spec| create_buffer(&self.shared.ctx.device, spec))
            .collect::<Result<Vec<_>, _>>()?;
        let prepared_queue = prepare_queue(
            &self.shared.ctx.device,
            &self.shared.program,
            &buffers,
            &self.shared.pipelines,
        )?;
        Ok(Pipeline {
            shared: Arc::clone(&self.shared),
            buffers,
            prepared_queue,
        })
    }

    pub fn input_names(&self) -> &[String] {
        &self.shared.input_names
    }

    pub fn output_names(&self) -> &[String] {
        &self.shared.output_names
    }
}

/// One executable instance: exclusive device buffers for this in-flight work.
pub struct Pipeline {
    shared: Arc<SharedProgram>,
    buffers: Vec<wgpu::Buffer>,
    prepared_queue: Vec<PreparedQueueOp>,
}

impl Pipeline {
    /// Write a named input parameter (must match a compile-time param).
    pub fn write_input(&self, name: &str, data: &[u8]) -> Result<(), PipelineError> {
        let idx = *self
            .shared
            .program
            .param_buffers
            .get(name)
            .ok_or_else(|| PipelineError::Program(format!("unknown input {name:?}")))?;
        self.write_buffer_index(idx, data)
    }

    /// Read a named output sink after the corresponding [`GpuFuture`] has been waited on
    /// (or after [`Self::call`]).
    pub fn read_output(&self, name: &str) -> Result<Vec<u8>, PipelineError> {
        let view_idx = *self
            .shared
            .program
            .sinks
            .get(name)
            .ok_or_else(|| PipelineError::Program(format!("unknown output {name:?}")))?;
        let buf_idx = self.shared.program.buffer_views[view_idx].buffer_index;
        self.read_buffer_index(buf_idx)
    }

    /// Upload all inputs, submit GPU work, return a completion future (does not wait).
    pub fn submit<'a>(
        &mut self,
        inputs: impl IntoIterator<Item = (&'a str, &'a [u8])>,
    ) -> Result<GpuFuture, PipelineError> {
        for (name, data) in inputs {
            self.write_input(name, data)?;
        }
        self.encode_and_submit()?;
        Ok(GpuFuture {
            device: self.shared.ctx.device.clone(),
        })
    }

    /// Upload inputs, submit, wait, read all outputs (simple synchronous helper).
    pub fn call<'a>(
        &mut self,
        inputs: impl IntoIterator<Item = (&'a str, &'a [u8])>,
    ) -> Result<BTreeMap<String, Vec<u8>>, PipelineError> {
        let fut = self.submit(inputs)?;
        fut.wait();
        let mut out = BTreeMap::new();
        for name in &self.shared.output_names {
            out.insert(name.clone(), self.read_output(name)?);
        }
        Ok(out)
    }

    /// Submit with inputs already written via [`Self::write_input`].
    pub fn submit_written(&mut self) -> Result<GpuFuture, PipelineError> {
        self.encode_and_submit()?;
        Ok(GpuFuture {
            device: self.shared.ctx.device.clone(),
        })
    }

    fn encode_and_submit(&self) -> Result<(), PipelineError> {
        let mut encoder =
            self.shared
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("resin-pipeline-run"),
                });

        for prepared in &self.prepared_queue {
            match prepared {
                PreparedQueueOp::Dispatch(d) => {
                    debug_assert_ne!(d.dispatch_size, [0, 0, 0]);
                    if d.clear_output_before_dispatch {
                        encoder.clear_buffer(
                            &self.buffers[d.output_buffer_index],
                            0,
                            Some(d.output_byte_len),
                        );
                    }
                    let pipeline = &self.shared.pipelines[d.pipeline_index];
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("resin-compute-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &d.bind_group, &[]);
                    pass.dispatch_workgroups(
                        d.dispatch_size[0],
                        d.dispatch_size[1],
                        d.dispatch_size[2],
                    );
                }
                PreparedQueueOp::Copy(c) => {
                    encoder.copy_buffer_to_buffer(
                        &self.buffers[c.source_buffer_index],
                        c.src_offset,
                        &self.buffers[c.output_buffer_index],
                        0,
                        c.copy_bytes,
                    );
                }
            }
        }
        self.shared.ctx.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    fn write_buffer_index(&self, index: usize, data: &[u8]) -> Result<(), PipelineError> {
        let spec = self
            .shared
            .program
            .buffers
            .get(index)
            .ok_or_else(|| PipelineError::Program(format!("invalid buffer id {index}")))?;
        if data.len() as u64 != spec.byte_len() {
            return Err(PipelineError::Program(format!(
                "write size mismatch: expected {} bytes, got {}",
                spec.byte_len(),
                data.len()
            )));
        }
        let buffer = self
            .buffers
            .get(index)
            .ok_or_else(|| PipelineError::Program(format!("invalid buffer id {index}")))?;
        self.shared.ctx.queue.write_buffer(buffer, 0, data);
        Ok(())
    }

    fn read_buffer_index(&self, index: usize) -> Result<Vec<u8>, PipelineError> {
        let buffer = self
            .buffers
            .get(index)
            .ok_or_else(|| PipelineError::Program(format!("invalid buffer id {index}")))?;
        let size = buffer.size();
        let staging = self
            .shared
            .ctx
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("resin-read-staging"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        let mut encoder =
            self.shared
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("resin-read"),
                });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.shared.ctx.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.shared.ctx.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|_| PipelineError::BufferMapFailed)?
            .map_err(|_| PipelineError::BufferMapFailed)?;
        let data = {
            let mapped = buffer_slice.get_mapped_range();
            mapped.to_vec()
        };
        staging.unmap();
        Ok(data)
    }
}

/// GPU work completion token. Host-centric today; Vulkan backends can attach semaphores.
pub struct GpuFuture {
    device: wgpu::Device,
}

impl GpuFuture {
    /// Block the host until submitted work is complete.
    pub fn wait(self) {
        self.device.poll(wgpu::Maintain::Wait);
    }

    /// Backend escape hatch for native sync objects (semaphores, etc.).
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }
}

/// Trace-time compile: register params + sinks, lower, build a [`PipelineFactory`].
pub fn compile(
    params: &[(&str, &View)],
    sinks: &[(&str, &View)],
    device: DeviceConfig,
    config: Option<WgslKernelConfig>,
) -> Result<PipelineFactory, PipelineError> {
    let mut ir = IrProgram::new();
    for (name, view) in params {
        ir.register_param(*name, view)
            .map_err(PipelineError::Compile)?;
    }
    for (name, view) in sinks {
        ir.build_sink(*name, view).map_err(PipelineError::Compile)?;
    }
    ir.seal_params().map_err(PipelineError::Compile)?;
    let artifact = build_wgpu_program(&ir, config);
    let ctx = DeviceContext::from_config(&device)?;
    PipelineFactory::from_program(ctx, artifact)
}

// --- device acquisition ---

fn request_default_device() -> Result<(wgpu::Device, wgpu::Queue), PipelineError> {
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
        .ok_or_else(|| PipelineError::Backend("no suitable GPU adapter found".into()))?;
    request_device_from_adapter(adapter)
}

fn request_named_device(device_name: &str) -> Result<(wgpu::Device, wgpu::Queue), PipelineError> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .enumerate_adapters(wgpu::Backends::all())
        .into_iter()
        .find(|adapter| adapter.get_info().name == device_name)
        .ok_or_else(|| {
            PipelineError::Config(format!(
                "no GPU adapter found with device_name '{device_name}'"
            ))
        })?;
    request_device_from_adapter(adapter)
}

fn request_device_from_adapter(
    adapter: wgpu::Adapter,
) -> Result<(wgpu::Device, wgpu::Queue), PipelineError> {
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("resin"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    ))?;
    Ok((device, queue))
}

// --- prepare bind groups (per instance; pipelines shared) ---

struct PreparedDispatch {
    pipeline_index: usize,
    bind_group: wgpu::BindGroup,
    dispatch_size: [u32; 3],
    clear_output_before_dispatch: bool,
    output_buffer_index: usize,
    output_byte_len: u64,
}

struct PreparedCopy {
    source_buffer_index: usize,
    output_buffer_index: usize,
    src_offset: u64,
    copy_bytes: u64,
}

enum PreparedQueueOp {
    Dispatch(PreparedDispatch),
    Copy(PreparedCopy),
}

fn prepare_queue(
    device: &wgpu::Device,
    program: &WgpuProgram,
    buffers: &[wgpu::Buffer],
    pipelines: &[wgpu::ComputePipeline],
) -> Result<Vec<PreparedQueueOp>, PipelineError> {
    program
        .queue
        .iter()
        .map(|op| match op {
            WgpuQueueOp::Dispatch(dispatch) => Ok(PreparedQueueOp::Dispatch(prepare_dispatch(
                device, program, buffers, pipelines, dispatch,
            )?)),
            WgpuQueueOp::Copy(copy) => Ok(PreparedQueueOp::Copy(prepare_copy(program, copy)?)),
        })
        .collect()
}

fn prepare_dispatch(
    device: &wgpu::Device,
    program: &WgpuProgram,
    buffers: &[wgpu::Buffer],
    pipelines: &[wgpu::ComputePipeline],
    dispatch: &WgpuDispatch,
) -> Result<PreparedDispatch, PipelineError> {
    let compute = &program.pipelines[dispatch.pipeline_index];
    let pipeline = &pipelines[dispatch.pipeline_index];
    let mut entries = vec![wgpu::BindGroupEntry {
        binding: 0,
        resource: buffers[dispatch.output_buffer_index].as_entire_binding(),
    }];
    for (binding, &view_index) in dispatch.arg_buffer_view_indices.iter().enumerate() {
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
    Ok(PreparedDispatch {
        pipeline_index: dispatch.pipeline_index,
        bind_group,
        dispatch_size: compute.dispatch_size,
        clear_output_before_dispatch: compute.clear_output_before_dispatch,
        output_buffer_index: dispatch.output_buffer_index,
        output_byte_len: program.buffers[dispatch.output_buffer_index].byte_len(),
    })
}

fn prepare_copy(program: &WgpuProgram, copy: &WgpuCopy) -> Result<PreparedCopy, PipelineError> {
    let source_view = &program.buffer_views[copy.source_buffer_view_index];
    let output_spec = program
        .buffers
        .get(copy.output_buffer_index)
        .ok_or_else(|| {
            PipelineError::Program(format!(
                "invalid output buffer index {}",
                copy.output_buffer_index
            ))
        })?;
    let accessor = &source_view.accessor;
    let element_nbytes = output_spec.etype.nbytes() as u64;
    let copy_bytes: u64 =
        accessor.shape.iter().map(|&d| d as u64).product::<u64>() * element_nbytes;
    if output_spec.byte_len() != copy_bytes {
        return Err(PipelineError::Program(format!(
            "copy size mismatch: output {} bytes, view {} bytes",
            output_spec.byte_len(),
            copy_bytes
        )));
    }
    Ok(PreparedCopy {
        source_buffer_index: source_view.buffer_index,
        output_buffer_index: copy.output_buffer_index,
        src_offset: accessor.offset as u64 * element_nbytes,
        copy_bytes,
    })
}

fn create_buffer(
    device: &wgpu::Device,
    spec: &WgpuBufferSpec,
) -> Result<wgpu::Buffer, PipelineError> {
    let size = spec.byte_len().max(4);
    let usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
    if let Some(init) = &spec.init {
        if init.len() as u64 != spec.byte_len() {
            return Err(PipelineError::Program(format!(
                "buffer init size mismatch: expected {} bytes, got {}",
                spec.byte_len(),
                init.len()
            )));
        }
        Ok(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("resin-buffer"),
                contents: init,
                usage,
            }),
        )
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
    spec: &WgpuComputePipelineSpec,
) -> Result<wgpu::ComputePipeline, PipelineError> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("resin-shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(spec.wgsl.as_str())),
    });
    Ok(
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("resin-pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some(spec.entry_point.as_str()),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        }),
    )
}
