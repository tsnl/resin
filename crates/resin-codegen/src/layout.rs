use super::Error;
use resin_common::prelude::*;
use resin_lir::Module;
pub(crate) fn layout(
    module: &Module,
    ty: &Ty,
) -> Result<resin_common::types::layout::Layout, Error> {
    resin_common::types::layout::layout(&module.types, ty).map_err(|error| Error(error.to_string()))
}
