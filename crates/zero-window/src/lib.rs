use zero_prelude::*;

use zero_gpu::GpuManager;

mod window_manager;
pub use window_manager::{WindowManager, WindowManagerConfig};

mod window;
pub use window::{Window, WindowConfig};
