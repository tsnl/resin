use super::Error;
use crate::lir::{Module, Ty};
pub(crate) fn layout(module: &Module, ty: &Ty) -> Result<crate::types::layout::Layout, Error> {
    crate::types::layout::layout(&module.types, ty).map_err(|error| Error(error.to_string()))
}
