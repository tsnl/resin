//! A backend client constructs LIR directly and never depends on frontend state.
use resin_codegen::{self as codegen, Stage};
use resin_lir::{
    BasicBlock, BlockId, Function, FunctionId, Instr, Local, Module, Terminator, Ty, Value,
};
use resin_lir_verifier::VerifiedModule;

fn constant_function(parameter: Ty, result: Ty, value: Value) -> Function {
    Function {
        name: None,
        foreign: None,
        result,
        locals: vec![Local {
            name: None,
            ty: parameter,
        }],
        entry: BlockId::from_index(0),
        blocks: vec![BasicBlock {
            name: None,
            instrs: vec![Instr::Push { value }],
            terminator: Terminator::Return,
        }],
    }
}

fn module() -> Module {
    let pointer = Ty::Pointer {
        pointee: Box::new(Ty::UInt64),
    };
    Module {
        functions: vec![
            constant_function(Ty::Unit, Ty::Int32, Value::Int32 { value: 42 }),
            constant_function(Ty::parameter(&[Ty::UInt64, pointer]), Ty::Unit, Value::Unit),
        ],
        entries: [("main".into(), FunctionId::from_index(0))].into(),
        ..Default::default()
    }
}

#[test]
fn target_trees_outlive_the_verified_input() {
    let (host, shader) = {
        let checked = VerifiedModule::new(module()).unwrap();
        (
            codegen::generate_c(checked.view(), "main", &[]).unwrap(),
            codegen::generate_glsl(checked.view(), FunctionId::from_index(1), Stage::Compute)
                .unwrap(),
        )
    };
    assert!(codegen::print_c(&host).contains("int main("));
    assert!(codegen::print_glsl(&shader).contains("void main()"));
}

#[test]
fn entry_selection_reports_errors_after_module_verification() {
    let checked = VerifiedModule::new(module()).unwrap();
    assert!(codegen::generate_c(checked.view(), "missing", &[]).is_err());
    let error = codegen::generate_glsl(checked.view(), FunctionId::from_index(99), Stage::Compute)
        .unwrap_err();
    assert!(error.to_string().contains("invalid shader function"));
    assert!(
        codegen::generate_glsl(checked.view(), FunctionId::from_index(0), Stage::Compute).is_err()
    );
}

#[test]
fn emission_convenience_operations_still_require_valid_lir() {
    let mut module = module();
    module.functions[0].locals.clear();
    assert!(codegen::emit_c(&module, "main").is_err());
    assert!(
        codegen::emit_glsl_function(&module, FunctionId::from_index(1), Stage::Compute).is_err()
    );
}
