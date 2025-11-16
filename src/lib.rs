use pyo3::prelude::*;
use zero_prelude::*;

use zero_gpu::{GpuManager, GpuManagerConfig};
use zero_render::{RenderManager, RenderManagerConfig};
use zero_window::{Window, WindowConfig, WindowManager, WindowManagerConfig};

mod instance;
pub use instance::{Instance, InstanceConfig};

//
// Python API
//

#[pymodule]
fn zero(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PythonInstance>()?;
    m.add_class::<PythonWindowManager>()?;
    m.add_class::<PythonWindow>()?;
    m.add_class::<PythonRenderManager>()?;
    Ok(())
}

macro_rules! wrapper_pyclass {
    ($wrapper_name:ident ( $inner:ty ): $name:literal) => {
        #[pyclass(unsendable, name = $name)]
        pub struct $wrapper_name {
            inner: $inner,
        }
        impl From<$inner> for $wrapper_name {
            fn from(inner: $inner) -> Self {
                Self { inner }
            }
        }
    };
}

// PythonInstance
//

wrapper_pyclass!(PythonInstance(Instance): "Instance");

#[pymethods]
impl PythonInstance {
    #[new]
    #[pyo3(signature = (require_window_support=true, require_render_support=true, debug_mode=true))]
    fn new(require_window_support: bool, require_render_support: bool, debug_mode: bool) -> Self {
        Self::from(Instance::new(InstanceConfig {
            require_window_support,
            require_render_support,
            debug_mode,
        }))
    }

    #[getter]
    fn window_manager(&self) -> PythonWindowManager {
        PythonWindowManager::from(self.inner.window_manager().clone())
    }

    #[getter]
    fn render_manager(&self) -> PythonRenderManager {
        PythonRenderManager::from(self.inner.render_manager().clone())
    }
}

// PythonWindowManager
//

wrapper_pyclass!(PythonWindowManager(Arc<WindowManager>): "WindowManager");
wrapper_pyclass!(PythonWindow(Arc<Window>): "Window");

#[pymethods]
impl PythonWindowManager {
    fn create_window(&self, title: String, width: u32, height: u32) -> PythonWindow {
        let window_config = WindowConfig {
            title,
            width,
            height,
        };
        PythonWindow::from(self.inner.create_window(window_config).clone())
    }
    fn update(&self) {
        self.inner.update();
    }
}

#[pymethods]
impl PythonWindow {
    #[getter]
    pub fn should_close(&self) -> bool {
        self.inner.should_close()
    }
    pub fn set_visible(&self, visible: bool) {
        self.inner.set_visible(visible)
    }
}

// PythonRenderManager
//

wrapper_pyclass!(PythonRenderManager(Arc<RenderManager>): "RenderManager");
