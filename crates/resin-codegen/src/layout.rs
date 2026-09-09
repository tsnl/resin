use super::Error;
use resin_lir::{Module, Ty};
pub(crate) fn layout(
    module: &Module,
    ty: &Ty,
) -> Result<resin_common::types::layout::Layout, Error> {
    resin_common::types::layout::layout(&module.types, ty).map_err(|error| Error(error.to_string()))
}
