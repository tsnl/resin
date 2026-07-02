//! Compiled graph factory and per-instance pipelines.
//!
//! - [`PipelineFactory`]: shared immutable program (shaders/compute pipelines) + schema.
//! - [`Pipeline`]: one executable instance with its own device buffer pool — admitted
//!   params, intermediate tensors, and sink backing stores are all allocated in
//!   [`PipelineFactory::create`]. [`Pipeline::call`] binds a dataloader's
//!   `Tree<Leaf = WgpuBuffer>` (or [`GpuLeaf`]) into the param slots, runs the graph,
//!   and leaves results in the sink buffers. Not re-entrant for overlapping GPU work;
//!   create multiple instances for triple-buffering.
//! - [`GpuFuture`]: completion token for a submit (host `wait`; backends may later attach
//!   native sync objects such as Vulkan semaphores).

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::Arc;

use resin_core::{format_param_path, Tree};
use resin_dsl::View;
use resin_ir::IrProgram;
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::lower::build_wgpu_program;
use crate::program::{
    WgpuBufferSpec, WgpuComputePipelineSpec, WgpuCopy, WgpuDispatch, WgpuProgram, WgpuQueueOp,
};
use crate::session::{BufferView, GpuLeaf, Session, WgpuBuffer, WgpuSession};
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

/// Back-compat alias for [`crate::session::WgpuSession`].
pub type DeviceContext = WgpuSession;

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
    session: WgpuSession,
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
    pub fn from_program(session: WgpuSession, program: WgpuProgram) -> Result<Self, PipelineError> {
        let pipelines = program
            .pipelines
            .iter()
            .map(|spec| create_pipeline(session.device(), spec))
            .collect::<Result<Vec<_>, _>>()?;
        let input_names: Vec<String> = program.param_buffers.keys().cloned().collect();
        let output_names: Vec<String> = program.sinks.keys().cloned().collect();
        Ok(Self {
            shared: Arc::new(SharedProgram {
                session,
                program,
                pipelines,
                input_names,
                output_names,
            }),
        })
    }

    pub fn session(&self) -> &WgpuSession {
        &self.shared.session
    }

    /// Allocate a fresh instance (own buffers + bind groups). Safe to run in parallel
    /// with other instances from the same factory (separate buffer sets).
    pub fn create(&self) -> Result<Pipeline, PipelineError> {
        let buffers = self
            .shared
            .program
            .buffers
            .iter()
            .map(|spec| create_buffer(self.shared.session.device(), spec))
            .collect::<Result<Vec<_>, _>>()?;
        let prepared_queue = prepare_queue(
            self.shared.session.device(),
            &self.shared.program,
            &buffers,
            &self.shared.pipelines,
        )?;
        Ok(Pipeline {
            shared: Arc::clone(&self.shared),
            buffers,
            prepared_queue,
            pending_binds: Vec::new(),
        })
    }

    pub fn input_names(&self) -> &[String] {
        &self.shared.input_names
    }

    pub fn output_names(&self) -> &[String] {
        &self.shared.output_names
    }
}

/// Inputs for [`Pipeline::call`] / [`Pipeline::submit`]: bind leaves into admitted param slots.
///
/// Prefer `Tree<Leaf = WgpuBuffer>` / [`GpuLeaf`] from a dataloader. Host byte trees and
/// name/byte pairs remain for tests.
pub trait PipelineInput {
    fn bind_inputs(self, pipe: &mut Pipeline) -> Result<(), PipelineError>;
}

/// `Tree` of host bytes (tests and legacy demos).
#[derive(Debug, Clone, Copy)]
pub struct TreeInputs<'a, T>(pub &'a T);

impl<T> PipelineInput for TreeInputs<'_, T>
where
    T: Tree<Leaf = Vec<u8>>,
{
    fn bind_inputs(self, pipe: &mut Pipeline) -> Result<(), PipelineError> {
        for (path, bytes) in self.0.flatten() {
            pipe.write_input(&format_param_path(&path), bytes)?;
        }
        Ok(())
    }
}

/// One leaf in a GPU input tree (whole buffer or view).
pub trait GpuTreeLeaf {
    fn bind_into(&self, pipe: &mut Pipeline, name: &str) -> Result<(), PipelineError>;
}

impl GpuTreeLeaf for WgpuBuffer {
    fn bind_into(&self, pipe: &mut Pipeline, name: &str) -> Result<(), PipelineError> {
        pipe.bind_param_buffer(name, self)
    }
}

impl GpuTreeLeaf for GpuLeaf {
    fn bind_into(&self, pipe: &mut Pipeline, name: &str) -> Result<(), PipelineError> {
        pipe.bind_param_leaf(name, self)
    }
}

/// `Tree` of GPU buffers/views for [`Pipeline::call`] (same as [`PipelineInput`] on `&T`).
#[derive(Debug, Clone, Copy)]
pub struct TreeGpuInputs<'a, T>(pub &'a T);

impl<T> PipelineInput for TreeGpuInputs<'_, T>
where
    T: Tree,
    T::Leaf: GpuTreeLeaf,
{
    fn bind_inputs(self, pipe: &mut Pipeline) -> Result<(), PipelineError> {
        <&T as PipelineInput>::bind_inputs(self.0, pipe)
    }
}

impl<T> PipelineInput for &T
where
    T: Tree,
    T::Leaf: GpuTreeLeaf,
{
    fn bind_inputs(self, pipe: &mut Pipeline) -> Result<(), PipelineError> {
        for (path, leaf) in self.flatten() {
            leaf.bind_into(pipe, &format_param_path(&path))?;
        }
        Ok(())
    }
}

fn write_named_inputs<'a>(
    pipe: &mut Pipeline,
    pairs: impl IntoIterator<Item = (&'a str, &'a [u8])>,
) -> Result<(), PipelineError> {
    for (name, data) in pairs {
        pipe.write_input(name, data)?;
    }
    Ok(())
}

impl<const N: usize> PipelineInput for [(&str, &[u8]); N] {
    fn bind_inputs(self, pipe: &mut Pipeline) -> Result<(), PipelineError> {
        write_named_inputs(pipe, self)
    }
}

impl PipelineInput for Vec<(&str, &[u8])> {
    fn bind_inputs(self, pipe: &mut Pipeline) -> Result<(), PipelineError> {
        write_named_inputs(pipe, self)
    }
}

struct PendingBind {
    dst_idx: usize,
    src: WgpuBuffer,
    src_offset: u64,
    byte_len: u64,
}

/// `wgpu::Buffer` clones share one allocation; pointer identity on the wrapper is not enough.
fn same_gpu_buffer(a: &WgpuBuffer, b: &WgpuBuffer) -> bool {
    a == b
}

/// One executable instance: exclusive device buffers for this in-flight work.
pub struct Pipeline {
    shared: Arc<SharedProgram>,
    buffers: Vec<wgpu::Buffer>,
    prepared_queue: Vec<PreparedQueueOp>,
    pending_binds: Vec<PendingBind>,
}

impl Pipeline {
    pub fn session(&self) -> &WgpuSession {
        &self.shared.session
    }

    fn param_buffer(&self, name: &str) -> Result<&WgpuBuffer, PipelineError> {
        let idx = *self
            .shared
            .program
            .param_buffers
            .get(name)
            .ok_or_else(|| PipelineError::Program(format!("unknown input {name:?}")))?;
        self.buffers
            .get(idx)
            .ok_or_else(|| PipelineError::Program(format!("invalid buffer id {idx}")))
    }

    /// Mirror compile-time param paths into this instance's admitted param buffers.
    pub fn param_tree<M: Tree<Leaf = View>>(
        &self,
        template: &M,
    ) -> Result<M::Map<WgpuBuffer>, PipelineError> {
        let leaves = template
            .flatten()
            .map(|(path, _)| {
                let name = format_param_path(&path);
                self.param_buffer(&name).cloned().map(|buf| (path, buf))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(<M::Map<WgpuBuffer> as Tree>::unflatten(leaves))
    }

    /// Mirror compile-time sink paths into this instance's sink backing buffers.
    pub fn sink_tree<M: Tree<Leaf = View>>(
        &self,
        template: &M,
    ) -> Result<M::Map<WgpuBuffer>, PipelineError> {
        let leaves = template
            .flatten()
            .map(|(path, _)| {
                let name = format_param_path(&path);
                self.sink_buffer(&name).cloned().map(|buf| (path, buf))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(<M::Map<WgpuBuffer> as Tree>::unflatten(leaves))
    }

    fn sink_buffer(&self, name: &str) -> Result<&WgpuBuffer, PipelineError> {
        let view_idx = *self
            .shared
            .program
            .sinks
            .get(name)
            .ok_or_else(|| PipelineError::Program(format!("unknown output {name:?}")))?;
        let buf_idx = self.shared.program.buffer_views[view_idx].buffer_index;
        self.buffers
            .get(buf_idx)
            .ok_or_else(|| PipelineError::Program(format!("invalid buffer id {buf_idx}")))
    }

    /// Bind a GPU leaf into a pipeline-owned param slot (device copy when needed).
    pub fn bind_param_leaf(&mut self, name: &str, leaf: &GpuLeaf) -> Result<(), PipelineError> {
        match leaf {
            GpuLeaf::Buffer(src) => self.bind_param_buffer(name, src),
            GpuLeaf::View(view) => self.bind_param_view(name, view),
        }
    }

    /// Bind a whole buffer into a pipeline-owned param slot.
    pub fn bind_param_buffer(&mut self, name: &str, src: &WgpuBuffer) -> Result<(), PipelineError> {
        self.queue_param_bind(name, src, 0, src.size())
    }

    fn bind_param_view(&mut self, name: &str, view: &BufferView) -> Result<(), PipelineError> {
        self.queue_param_bind(name, &view.buffer, view.offset, view.byte_len)
    }

    fn queue_param_bind(
        &mut self,
        name: &str,
        src: &WgpuBuffer,
        src_offset: u64,
        byte_len: u64,
    ) -> Result<(), PipelineError> {
        let idx = *self
            .shared
            .program
            .param_buffers
            .get(name)
            .ok_or_else(|| PipelineError::Program(format!("unknown input {name:?}")))?;
        let dst = self
            .buffers
            .get(idx)
            .ok_or_else(|| PipelineError::Program(format!("invalid buffer id {idx}")))?;
        if byte_len != dst.size() {
            return Err(PipelineError::Program(format!(
                "bind {name:?}: size mismatch {byte_len} vs {}",
                dst.size()
            )));
        }
        if src_offset + byte_len > src.size() {
            return Err(PipelineError::Program(format!(
                "bind {name:?}: source range exceeds buffer size {}",
                src.size()
            )));
        }
        if same_gpu_buffer(dst, src) && src_offset == 0 {
            return Ok(());
        }
        self.pending_binds.push(PendingBind {
            dst_idx: idx,
            src: src.clone(),
            src_offset,
            byte_len,
        });
        Ok(())
    }

    /// Device-copy matching leaves from `src` onto `dst` (same tree shape and visit order).
    pub fn copy_tree<S, D>(&mut self, src: &S, dst: &D) -> Result<(), PipelineError>
    where
        S: Tree<Leaf = WgpuBuffer>,
        D: Tree<Leaf = WgpuBuffer>,
    {
        for ((path_s, src_buf), (path_d, dst_buf)) in src.flatten().zip(dst.flatten()) {
            if path_s != path_d {
                return Err(PipelineError::Program(format!(
                    "copy_tree path mismatch: {} vs {}",
                    format_param_path(&path_s),
                    format_param_path(&path_d)
                )));
            }
            self.queue_buffer_copy(dst_buf, src_buf)?;
        }
        Ok(())
    }

    /// Write a named input parameter from host bytes (batch upload).
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

    /// Bind inputs, submit GPU work, return a completion future (does not wait).
    pub fn submit(&mut self, inputs: impl PipelineInput) -> Result<GpuFuture, PipelineError> {
        inputs.bind_inputs(self)?;
        self.encode_and_submit()?;
        Ok(GpuFuture {
            device: self.shared.session.device().clone(),
        })
    }

    /// Bind inputs, run, wait — outputs remain in sink buffers (gather via [`Self::sink_tree`]).
    pub fn call_bound(&mut self, inputs: impl PipelineInput) -> Result<(), PipelineError> {
        inputs.bind_inputs(self)?;
        let fut = self.submit_written()?;
        fut.wait();
        Ok(())
    }

    /// Bind a GPU input tree, run, return the output tree (same shape as compile-time sinks).
    pub fn call_trees<In, Out>(
        &mut self,
        inputs: &In,
        outputs: &Out,
    ) -> Result<Out::Map<WgpuBuffer>, PipelineError>
    where
        In: Tree,
        In::Leaf: GpuTreeLeaf,
        Out: Tree<Leaf = View>,
    {
        self.call_bound(inputs)?;
        self.sink_tree(outputs)
    }

    /// Bind inputs, submit, wait, read all sinks to host bytes (test helper).
    pub fn call(
        &mut self,
        inputs: impl PipelineInput,
    ) -> Result<BTreeMap<String, Vec<u8>>, PipelineError> {
        self.call_bound(inputs)?;
        let mut out = BTreeMap::new();
        for name in &self.shared.output_names {
            out.insert(name.clone(), self.read_output(name)?);
        }
        Ok(out)
    }

    /// Submit queued param binds only (e.g. after [`Self::copy_tree`]).
    pub fn flush_binds(&mut self) -> Result<GpuFuture, PipelineError> {
        self.encode_binds_only()?;
        Ok(GpuFuture {
            device: self.shared.session.device().clone(),
        })
    }

    /// Submit with inputs already bound.
    pub fn submit_written(&mut self) -> Result<GpuFuture, PipelineError> {
        self.encode_and_submit()?;
        Ok(GpuFuture {
            device: self.shared.session.device().clone(),
        })
    }

    fn encode_and_submit(&mut self) -> Result<(), PipelineError> {
        let mut encoder =
            self.shared
                .session
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("resin-pipeline-run"),
                });

        for bind in self.pending_binds.drain(..) {
            let dst = &self.buffers[bind.dst_idx];
            if same_gpu_buffer(&bind.src, dst) && bind.src_offset == 0 {
                continue;
            }
            encoder.copy_buffer_to_buffer(&bind.src, bind.src_offset, dst, 0, bind.byte_len);
        }

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
                    if c.source_buffer_index == c.output_buffer_index && c.src_offset == 0 {
                        continue;
                    }
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
        self.shared.session.queue().submit(Some(encoder.finish()));
        Ok(())
    }

    fn encode_binds_only(&mut self) -> Result<(), PipelineError> {
        if self.pending_binds.is_empty() {
            return Ok(());
        }
        let mut encoder =
            self.shared
                .session
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("resin-pipeline-binds"),
                });
        for bind in self.pending_binds.drain(..) {
            let dst = &self.buffers[bind.dst_idx];
            if same_gpu_buffer(&bind.src, dst) && bind.src_offset == 0 {
                continue;
            }
            encoder.copy_buffer_to_buffer(&bind.src, bind.src_offset, dst, 0, bind.byte_len);
        }
        self.shared.session.queue().submit(Some(encoder.finish()));
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
        self.shared.session.write_buffer(buffer, 0, data)
    }

    fn read_buffer_index(&self, index: usize) -> Result<Vec<u8>, PipelineError> {
        let buffer = self
            .buffers
            .get(index)
            .ok_or_else(|| PipelineError::Program(format!("invalid buffer id {index}")))?;
        self.shared.session.read_buffer(buffer)
    }

    fn buffer_index(&self, buf: &WgpuBuffer) -> Result<usize, PipelineError> {
        self.buffers
            .iter()
            .position(|b| same_gpu_buffer(b, buf))
            .ok_or_else(|| {
                PipelineError::Program("buffer not owned by this pipeline instance".into())
            })
    }

    fn queue_buffer_copy(
        &mut self,
        dst: &WgpuBuffer,
        src: &WgpuBuffer,
    ) -> Result<(), PipelineError> {
        let byte_len = dst.size();
        if byte_len != src.size() {
            return Err(PipelineError::Program(format!(
                "copy_tree size mismatch: {} vs {}",
                src.size(),
                byte_len
            )));
        }
        if same_gpu_buffer(dst, src) {
            return Ok(());
        }
        let dst_idx = self.buffer_index(dst)?;
        if same_gpu_buffer(src, &self.buffers[dst_idx]) {
            return Ok(());
        }
        self.pending_binds.push(PendingBind {
            dst_idx,
            src: src.clone(),
            src_offset: 0,
            byte_len,
        });
        Ok(())
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

/// Trace-time compile from input/output [`Tree`]s of [`View`] leaves.
///
/// Paths become dotted buffer names (`model.layers.0.weight`). Wrap roots with
/// [`resin_core::Named`] so names are non-empty:
///
/// ```ignore
/// use resin_core::{named, Empty};
/// let session = WgpuSession::open(DeviceConfig::default())?;
/// compile(
///     &(named("a", a), named("b", b)),
///     &named("out", out),
///     &session,
///     None,
/// )?;
/// ```
pub fn compile<I, O>(
    inputs: &I,
    outputs: &O,
    session: &WgpuSession,
    config: Option<WgslKernelConfig>,
) -> Result<PipelineFactory, PipelineError>
where
    I: Tree<Leaf = View>,
    O: Tree<Leaf = View>,
{
    let mut ir = IrProgram::new();
    for (path, view) in inputs.flatten() {
        let name = format_param_path(&path);
        if name.is_empty() {
            return Err(PipelineError::Compile(
                "compile input leaf has empty path; wrap with Named".into(),
            ));
        }
        ir.register_param(name, view)
            .map_err(PipelineError::Compile)?;
    }
    for (path, view) in outputs.flatten() {
        let name = format_param_path(&path);
        if name.is_empty() {
            return Err(PipelineError::Compile(
                "compile output leaf has empty path; wrap with Named".into(),
            ));
        }
        ir.build_sink(name, view).map_err(PipelineError::Compile)?;
    }
    ir.seal_params().map_err(PipelineError::Compile)?;
    let artifact = build_wgpu_program(&ir, config);
    PipelineFactory::from_program(session.clone(), artifact)
}

/// Open a session and compile (convenience when the session is not reused).
pub fn compile_open<I, O>(
    inputs: &I,
    outputs: &O,
    device: DeviceConfig,
    config: Option<WgslKernelConfig>,
) -> Result<PipelineFactory, PipelineError>
where
    I: Tree<Leaf = View>,
    O: Tree<Leaf = View>,
{
    let session = WgpuSession::open(&device)?;
    compile(inputs, outputs, &session, config)
}

/// Low-level stringly compile (params/sinks as name slices). Prefer [`compile`].
pub fn compile_named(
    params: &[(&str, &View)],
    sinks: &[(&str, &View)],
    session: &WgpuSession,
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
    PipelineFactory::from_program(session.clone(), artifact)
}

/// Open a session and [`compile_named`].
pub fn compile_named_open(
    params: &[(&str, &View)],
    sinks: &[(&str, &View)],
    device: DeviceConfig,
    config: Option<WgslKernelConfig>,
) -> Result<PipelineFactory, PipelineError> {
    let session = WgpuSession::open(&device)?;
    compile_named(params, sinks, &session, config)
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
