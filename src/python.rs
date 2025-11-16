use super::*;

#[pyo3::pymodule]
mod zero {
    use super::*;

    use pyo3::prelude::*;

    #[pyclass(unsendable, name = "Instance")]
    pub struct PythonInstance {
        inner: Instance,
    }

    #[pymethods]
    impl PythonInstance {
        #[new]
        #[pyo3(signature = (require_window_support=true, require_render_support=true, debug_mode=true))]
        fn python_new(
            require_window_support: bool,
            require_render_support: bool,
            debug_mode: bool,
        ) -> Self {
            Self {
                inner: Instance::new(InstanceConfig {
                    require_window_support,
                    require_render_support,
                    debug_mode,
                }),
            }
        }

        #[pyo3(name = "run")]
        fn python_run(&self) {
            self.inner.run();
        }
    }
}
