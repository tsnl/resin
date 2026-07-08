use resin_core::Tree;
use resin_dsl::Tensor;
use resin_ir::{optimize, BufferRef, BufferViewRef, IrProgram};

use crate::error::CompileError;
use crate::jit::Jit;
use crate::lower::lower_program;

pub(crate) fn compile<J, P, O>(
    jit: &J,
    params: &P::Mapped<Tensor>,
    sinks: &O,
) -> Result<J::Artifact, CompileError>
where
    J: Jit,
    P: Tree<J::Tensor>,
    O: Tree<Tensor>,
    P::Mapped<Tensor>: Tree<Tensor>,
    <P::Mapped<Tensor> as Tree<Tensor>>::Mapped<BufferRef>: Tree<BufferRef> + Send + Sync,
    O::Mapped<BufferViewRef>: Tree<BufferViewRef> + Send + Sync,
{
    let program = lower_program(params, sinks)?;
    compile_ir(jit, program)
}

pub(crate) fn compile_ir<J, P, S>(
    jit: &J,
    program: IrProgram<P, S>,
) -> Result<J::Artifact, CompileError>
where
    J: Jit,
    P: Tree<BufferRef> + Send + Sync,
    S: Tree<BufferViewRef> + Send + Sync,
{
    let program = optimize(program);
    jit.lower(&program)
        .map_err(|err| CompileError::Lower(Box::new(err)))
}