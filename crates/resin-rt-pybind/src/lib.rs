use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use resin_rt::WgpuInterp;
use resin_rt::{request_default_device, WgpuInterpError, WgpuProgram};

fn interp_error(err: WgpuInterpError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[pyfunction]
fn decode_wgpu_program_msgpack(program_msgpack: &[u8]) -> PyResult<()> {
    WgpuProgram::from_msgpack(program_msgpack)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(())
}

#[pyfunction]
fn encode_wgpu_program_msgpack(program_msgpack: &[u8]) -> PyResult<Vec<u8>> {
    let program = WgpuProgram::from_msgpack(program_msgpack)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    program
        .to_msgpack()
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

#[pyclass(name = "WgpuInterp", module = "resin_rt_pybind")]
pub struct PyWgpuInterp {
    inner: WgpuInterp,
}

#[pymethods]
impl PyWgpuInterp {
    /// Convenience constructor that acquires a default GPU device and queue.
    ///
    /// `program_msgpack` is a MessagePack blob produced by `resin_wgpu.WgpuProgram.to_msgpack`.
    #[new]
    fn new(program_msgpack: &[u8]) -> PyResult<Self> {
        let program = WgpuProgram::from_msgpack(program_msgpack)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let (device, queue) = request_default_device().map_err(interp_error)?;
        WgpuInterp::new(&device, &queue, program)
            .map(|inner| Self { inner })
            .map_err(interp_error)
    }

    fn run(&self) -> PyResult<()> {
        self.inner.run().map_err(interp_error)
    }

    fn write_buffer(&self, buffer_index: usize, data: &[u8]) -> PyResult<()> {
        self.inner
            .write_buffer(buffer_index, data)
            .map_err(interp_error)
    }

    fn read_buffer(&self, buffer_index: usize) -> PyResult<Vec<u8>> {
        self.inner.read_buffer(buffer_index).map_err(interp_error)
    }

    fn param_buffer_index(&self, param_id: u64) -> PyResult<usize> {
        self.inner.param_buffer_index(param_id).map_err(interp_error)
    }

    fn copy_buffer_to_buffer(
        &self,
        src_buffer_index: usize,
        dst_buffer_index: usize,
    ) -> PyResult<()> {
        self.inner
            .copy_buffer_to_buffer(src_buffer_index, dst_buffer_index)
            .map_err(interp_error)
    }
}

#[pymodule]
fn resin_rt_pybind(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWgpuInterp>()?;
    m.add_function(wrap_pyfunction!(decode_wgpu_program_msgpack, m)?)?;
    m.add_function(wrap_pyfunction!(encode_wgpu_program_msgpack, m)?)?;
    Ok(())
}