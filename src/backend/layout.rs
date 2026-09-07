use super::Error;
use crate::ir::{Module, Ty};
pub(super) fn layout(module: &Module, ty: &Ty) -> Result<crate::ir::layout::Layout, Error> {
    crate::ir::layout::layout(&module.types, ty).map_err(|error| Error(error.to_string()))
}
