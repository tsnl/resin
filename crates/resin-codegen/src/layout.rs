use super::Error;
use resin_lir::Module;
use resin_types::prelude::*;
pub(crate) fn layout(module: &Module, ty: &Ty) -> Result<resin_types::layout::Layout, Error> {
    resin_types::layout::layout(&module.types, ty).map_err(|error| Error(error.to_string()))
}
