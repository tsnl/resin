//! Frontend inspection does not generate or execute target code.
use crate::{
    ast,
    compiler::{Input, Session},
    ir,
};
use std::path::Path;

#[derive(Clone, Copy)]
pub(super) enum Output {
    Cst,
    Ast,
    Ir,
    Check,
}

pub(super) fn run(
    compiler: &mut Session,
    input: &Input,
    output: Output,
    destination: Option<&Path>,
) -> super::Result<i32> {
    let path = crate::analysis::normalize_path(&input.path)?;
    let snapshot = compiler.analyze(&path)?;
    let text = match output {
        Output::Cst => snapshot
            .syntax_tree(&path)
            .ok_or_else(|| format!("cannot read {}", input.path.display()))?
            .root_node()
            .to_sexp(),
        Output::Ast => ast::print::format_program(snapshot.program()?),
        Output::Check => {
            snapshot.program()?;
            "ok".into()
        }
        Output::Ir => ir::format_module(snapshot.module()?),
    };
    super::output::write(format!("{text}\n").as_bytes(), destination)
}
