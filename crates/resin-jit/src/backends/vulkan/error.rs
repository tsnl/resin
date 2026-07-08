#[derive(Debug, thiserror::Error)]
pub enum VulkanLowerError {
    #[error("IR program validation failed: {0}")]
    InvalidIr(#[from] resin_ir::IrError),
    #[error("unsupported kernel for Vulkan lowering: {0}")]
    UnsupportedKernel(&'static str),
    #[error("WGSL parse error: {0}")]
    WgslParse(String),
    #[error("WGSL validation error: {0}")]
    WgslValidate(String),
    #[error("SPIR-V emission error: {0}")]
    SpvEmit(String),
    #[error("Vulkan lowering error: {0}")]
    Message(String),
}

impl From<crate::backends::wgsl::WgslError> for VulkanLowerError {
    fn from(err: crate::backends::wgsl::WgslError) -> Self {
        use crate::backends::wgsl::WgslError;
        match err {
            WgslError::UnsupportedKernel(kernel) => VulkanLowerError::UnsupportedKernel(kernel),
            WgslError::Message(msg) => VulkanLowerError::Message(msg),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum VulkanRuntimeError {
    #[error("no Vulkan implementation available: {0}")]
    NoVulkan(String),
    #[error("no suitable Vulkan device with a compute queue found")]
    NoDevice,
    #[error("Vulkan call failed: {call}: {result:?}")]
    Call {
        call: &'static str,
        result: ash::vk::Result,
    },
    #[error("runtime error: {0}")]
    Message(String),
}

impl VulkanRuntimeError {
    pub(super) fn call(call: &'static str, result: ash::vk::Result) -> Self {
        Self::Call { call, result }
    }
}

/// Shorthand for mapping `ash` results.
pub(super) trait VkResultExt<T> {
    fn ctx(self, call: &'static str) -> Result<T, VulkanRuntimeError>;
}

impl<T> VkResultExt<T> for Result<T, ash::vk::Result> {
    fn ctx(self, call: &'static str) -> Result<T, VulkanRuntimeError> {
        self.map_err(|result| VulkanRuntimeError::call(call, result))
    }
}
