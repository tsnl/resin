use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
mod resin_runtime_pybind {
    use pyo3::prelude::*;

    /// Formats the sum of two numbers as string.
    #[pyfunction]
    fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
        Ok((a + b).to_string())
    }
}
