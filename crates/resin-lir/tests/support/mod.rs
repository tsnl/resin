use resin_executor::{Cancellation, Execution};
use std::{num::NonZeroUsize, sync::Arc};

/// Direct language fixtures keep their assertions synchronous while using the
/// same owned async pass as applications. Async behavior has separate tests.
pub fn build_lir(
    source: &resin_hir::Module,
    entries: &[resin_lir::Entry],
    options: &resin_lir::LoweringOptions,
) -> Result<resin_lir::Module, Vec<resin_lir::Error>> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    match runtime.block_on(resin_lir::build_lir(
        Arc::new(source.clone()),
        entries.to_vec(),
        options.clone(),
        &Execution::new(NonZeroUsize::MIN),
        &Cancellation::new(),
    )) {
        Ok(module) => Ok(module),
        Err(resin_lir::BuildError::Diagnostics { errors }) => Err(errors),
        Err(resin_lir::BuildError::Execution { error }) => panic!("{error}"),
    }
}
