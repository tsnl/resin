use support::shaders;
mod support;
#[path = "support/toolchain.rs"]
mod toolchain;
use resin_lir::{BasicBlock, BlockId, Instr, Terminator};
use resin_types::prelude::*;

#[test]
fn a_loop_with_an_always_returning_body_has_a_valid_continue_target() {
    let mut module = support::module(
        "export { kernel };
        def helper(x: uint) -> uint = { x };
        @compute_shader def kernel(i: ulong, output: Ptr<uint>) = {
            output.* := helper(uint(i));
        };",
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
    let abort = "{ var value: None; value := None; var test: bool; test := value!; test }";
    let conditions = [
        abort.to_string(),
        format!("if (i == 0_ul) {abort} else {abort}"),
        format!("if (i == 0_ul) {abort} else {{ i < 4_ul }}"),
        format!("{{ while ({abort}) {{ output.* := 1_ui; }}; i == 0_ul }}"),
        format!("{{ var test = if (i == 0_ul) {abort} else {abort}; test }}"),
    ];
    for condition in conditions {
        let module = support::module(&format!(
            "export {{ kernel }};
            @compute_shader def kernel(i: ulong, output: Ptr<uint>) = {{
                while ({condition}) {{ output.* := 1_ui; }};
                output.* := 2_ui;
            }};"
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
