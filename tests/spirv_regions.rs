use support::shaders;
mod support;
use resin_lir::{BasicBlock, BlockId, Instr, Terminator};
use resin_types::prelude::*;
use support::toolchain;

#[test]
fn a_loop_with_an_always_returning_body_has_a_valid_continue_target() {
    let mut module = support::module(
        "export { kernel };\n        fn helper(x: u32) -> u32 { x }\n        @compute_shader fn kernel(i: u64, output: PtrMut<u32>) {\n            output.* = helper(u32(i));\n        }",
    );
    let helper = module
        .functions
        .iter_mut()
        .find(|function| function.name.as_deref() == Some("helper"))
        .unwrap();
    helper.entry = BlockId::from_index(0);
    helper.blocks = vec![
        BasicBlock {
            name: None,
            instrs: vec![Instr::Push {
                value: Value::UInt32 { value: 7 },
            }],
            terminator: Terminator::Loop {
                condition: BlockId::from_index(1),
                body: BlockId::from_index(2),
                next: Some(BlockId::from_index(3)),
            },
        },
        BasicBlock {
            name: None,
            instrs: vec![Instr::Push {
                value: Value::Bool { value: true },
            }],
            terminator: Terminator::LoopTest,
        },
        BasicBlock {
            name: None,
            instrs: vec![],
            terminator: Terminator::Return,
        },
        BasicBlock {
            name: None,
            instrs: vec![],
            terminator: Terminator::Return,
        },
    ];
    let project = support::project::Project::new(&module, None).unwrap();
    for shader in project.generated.shaders() {
        shaders::validate(shader.unoptimized_spirv());
    }
    if let Some(optimizer) = shaders::optimizer() {
        let built = project.build(&toolchain::spirv(&optimizer)).unwrap();
        for shader in project.generated.shaders() {
            shaders::validate(&built.path(shader.spirv().file_name().unwrap()));
        }
    }
}

#[test]
fn elimination_ends_loop_conditions_and_selection_continuations() {
    let abort = "{ let mut value: None; value = None; let mut test: bool; test = value!; test }";
    let conditions = [
        abort.to_string(),
        format!("if (i == u64(0)) {abort} else {abort}"),
        format!("if (i == u64(0)) {abort} else {{ i < u64(4) }}"),
        format!("{{ while ({abort}) {{ output.* = u32(1); }}; i == u64(0) }}"),
        format!("{{ let mut test = if (i == u64(0)) {abort} else {abort}; test }}"),
    ];
    for condition in conditions {
        let module = support::module(&format!(
            "export {{ kernel }};\n            @compute_shader fn kernel(i: u64, output: PtrMut<u32>) {{\n                while ({condition}) {{ output.* = u32(1); }};\n                output.* = u32(2);\n            }}"
        ));
        let project = support::project::Project::new(&module, None).unwrap();
        for shader in project.generated.shaders() {
            shaders::validate(shader.unoptimized_spirv());
        }
        if let Some(optimizer) = shaders::optimizer() {
            project.build(&toolchain::spirv(&optimizer)).unwrap();
        }
    }
}
