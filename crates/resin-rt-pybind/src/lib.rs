use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use resin_jit_wgpu::{
    create_interp, parse_backend, BufferId, Interp, InterpConfig, InterpError, ProgramId,
    WgpuProgram,
};

fn interp_error(err: InterpError) -> PyErr {
    match err {
        InterpError::Config(message) => PyValueError::new_err(message),
        InterpError::UnsupportedBackend(message) => PyValueError::new_err(message),
        InterpError::Program(message) => PyValueError::new_err(message),
        InterpError::BufferMapFailed => PyValueError::new_err(err.to_string()),
        InterpError::Backend(message) => PyValueError::new_err(message),
    }
}

fn parse_config(py: Python<'_>, config: Option<Bound<'_, PyAny>>) -> PyResult<InterpConfig> {
    let Some(config) = config else {
        return Ok(InterpConfig::Nil);
    };

    let bytes: Vec<u8> = if config.is_instance_of::<PyBytes>() {
        config.extract()?
    } else {
        let msgpack = py.import("msgpack")?;
        msgpack.call_method1("packb", (&config,))?.extract()?
    };

    rmpv::decode::read_value(&mut &bytes[..]).map_err(|err| PyValueError::new_err(err.to_string()))
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

#[pyclass(name = "Interp", module = "resin_rt_pybind")]
pub struct PyInterp {
    inner: Box<dyn Interp>,
}

#[pymethods]
impl PyInterp {
    #[new]
    #[pyo3(signature = (backend, config=None))]
    fn new(py: Python<'_>, backend: &str, config: Option<Bound<'_, PyAny>>) -> PyResult<Self> {
        let backend = parse_backend(backend).map_err(interp_error)?;
        let config = parse_config(py, config)?;
        let inner = create_interp(backend, config).map_err(interp_error)?;
        Ok(Self { inner })
    }

    fn admit(&mut self, program_msgpack: &[u8]) -> PyResult<usize> {
        // Python still ships programs as msgpack; decode once at the boundary.
        let program = WgpuProgram::from_msgpack(program_msgpack)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        self.inner
            .admit_program(program)
            .map(|program_id| program_id.0)
            .map_err(interp_error)
    }

    fn program_count(&self) -> usize {
        self.inner.program_count()
    }

    fn run(&self, program_id: usize) -> PyResult<()> {
        self.inner.run(ProgramId(program_id)).map_err(interp_error)
    }

    fn write_buffer(&self, program_id: usize, buffer_id: usize, data: &[u8]) -> PyResult<()> {
        self.inner
            .write_buffer(ProgramId(program_id), BufferId(buffer_id), data)
            .map_err(interp_error)
    }

    fn read_buffer(&self, program_id: usize, buffer_id: usize) -> PyResult<Vec<u8>> {
        self.inner
            .read_buffer(ProgramId(program_id), BufferId(buffer_id))
            .map_err(interp_error)
    }

    fn copy_buffer_to_buffer(
        &self,
        src_program_id: usize,
        src_buffer_id: usize,
        dst_program_id: usize,
        dst_buffer_id: usize,
    ) -> PyResult<()> {
        self.inner
            .copy_buffer_to_buffer(
                ProgramId(src_program_id),
                BufferId(src_buffer_id),
                ProgramId(dst_program_id),
                BufferId(dst_buffer_id),
            )
            .map_err(interp_error)
    }
}

#[pymodule]
fn resin_rt_pybind(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyInterp>()?;
    m.add_function(wrap_pyfunction!(decode_wgpu_program_msgpack, m)?)?;
    m.add_function(wrap_pyfunction!(encode_wgpu_program_msgpack, m)?)?;
    Ok(())
}
