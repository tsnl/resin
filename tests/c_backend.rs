use std::{ffi::OsString, fs, process::Command};

use resin::{
    backend::c,
    ir,
    toolchain::{self, TempDir},
};
mod support;
use support::module;

fn run_module(module: &ir::Module) -> std::process::Output {
    let source = c::emit(module, "main").unwrap();
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let output = temp
        .path()
        .join(format!("program{}", std::env::consts::EXE_SUFFIX));
    let cc = std::env::var_os("CC")
        .unwrap_or_else(|| OsString::from(resin::toolchain::DEFAULT_C_COMPILER));
    toolchain::compile_c(&source, &output, &cc).unwrap_or_else(|error| panic!("{error}\n{source}"));
    Command::new(output).output().unwrap()
}

fn runs(source: &str, code: i32) {
    let output = run_module(&module(source));
    assert_eq!(
        output.status.code(),
        Some(code),
        "{source}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn inferred_types_lower_to_concrete_c_and_preserve_effect_order() {
    runs(
        "export { main }; def main() -> _ = { var n: _; var p: Ptr<_>; n := 40; p := &n; p.* := p.* + 2; p.* };",
        42,
    );
    runs(
        "export { main }; def wide() -> _ = { var n = 4294967297; var p: Ptr<ulong>; p := &n; p.* }; def main() -> int = { if (wide() == ulong(4294967297)) { 0 } else { 1 } };",
        0,
    );
    runs(
        "export { main }; def main() -> _ = { var n = 0; var pair: { a: _, b: _ }; pair := { b = (n := n + 1), a = (n := n + 1) }; pair.a * 10 + pair.b };",
        21,
    );
    runs(
        "export { main }; def add(n: int) -> int = { n + 1 }; def select() -> (int) -> _ = { add }; def main() -> _ = { select()(41) };",
        42,
    );
}

#[test]
fn typed_pointer_offsets_use_element_sizes() {
    runs(
        "export { main }; def main () -> int = { var values = [10, 20, 30]; var p = Ptr<int> (&values); (p + 1).* := 7; var end = p + uint (2); (end - 1).* + (end + -2).* + end.* };",
        47,
    );
    runs(
        "export { main }; type Payload = { marker: uint, wide: ulong, amount: float32 }; def main () -> int = { var values = [Payload { marker = uint (1), wide = ulong (4294967297), amount = float32 (0.5) }, Payload { marker = uint (2), wide = ulong (8589934593), amount = float32 (1.5) }]; var p = Ptr<Payload> (&values); var q = p + 1; q.amount := q.amount + float32 (2.0); if (q.wide == ulong (8589934593) && q.marker == uint (2) && q.amount == float32 (3.5) && p.amount == float32 (0.5)) { 0 } else { 1 } };",
        0,
    );
}

#[test]
fn numbered_examples_compile_as_strict_c11() {
    for entry in fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/examples")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|ext| ext == "resin")
            && path
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .starts_with("eg")
        {
            let program = resin::ast::load(&path).unwrap();
            let module = ir::generate_program(&program).unwrap();
            let output = run_module(&module);
            assert_eq!(
                output.status.code(),
                Some(0),
                "{}\n{}",
                path.display(),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}

#[test]
fn discarded_branch_results_compile_and_preserve_effects() {
    runs(
        "export { main }; def main () -> int = { var x = 0; if (1 == 1) { x := 1; () } else { () }; if (1 == 2) { (1, 2) } else { (3, 4) }; x };",
        1,
    );
}

#[test]
fn while_rechecks_conditions_and_discards_body_values() {
    runs(
        "export { main }; def main () -> int = { var n = 0; var sum = 0; while ((n := n + 1) <= 4) { sum := sum + n }; sum + n };",
        15,
    );
    runs(
        "export { main }; def main () -> int = { var n = 0; while (1 == 0) { n := 42; }; n };",
        0,
    );
    runs(
        "export { main }; def main () -> int = { var n: int; while ((n := 7) == 0) {}; n };",
        7,
    );
    runs(
        "export { main }; def main () -> int = { var n = 0; while (n < 1000000) { n := n + 1; }; if (n == 1000000) { 0 } else { 1 } };",
        0,
    );
}

#[test]
fn while_nests_with_branches_and_preserves_outer_values() {
    runs(
        "export { main }; def main () -> int = { var i = 0; var total = 0; while (i < 3) { var j = 0; while (j < 4) { if (j < 2) { total := total + 1 } else { total := total + 2 }; j := j + 1; }; i := i + 1; }; total };",
        18,
    );
    runs(
        "export { main }; def main () -> int = { var n = 0; var x = 7; while (n < 3) { var x = 10; n := n + 1; x := 20; }; x + n };",
        10,
    );
    runs(
        "export { main }; def main () -> int = { var n = 0; var r = { first = 9, body = while (n < 3) { n := n + 1; }, last = n }; r.first + r.last };",
        12,
    );
    runs(
        "export { main }; def main () -> int = { var n = 0; while (if (n < 3) { (1 == 1) && ((n := n + 1) < 3) } else { 1 == 0 }) {}; n };",
        3,
    );
}

#[test]
fn ordinary_functions_can_be_passed_and_selected() {
    runs(
        "export { main }; def add (x: int, y: int) -> int = { x + y }; def apply (f: (int, int) -> int, args: (int, int)) -> int = { f(args) }; def main () -> int = { var a = add; var b = if (1 == 1) { a } else { add }; apply(a, (10, 3)) + b(20, 4) };",
        37,
    );
}

#[test]
fn recursive_functions_receive_state_explicitly() {
    runs(
        "export { main }; def fact(n: int, offset: int) -> int = { if (n == 0) { offset } else { n * fact(n - 1, offset) } }; def main() -> int = { var offset = 2; fact(4, offset) };",
        48,
    );
}

#[test]
fn mutual_recursion_needs_no_forward_declaration() {
    runs(
        "export { main }; def f (n: int) -> int = { if (n == 0) { 7 } else { next(n - 1) } }; def next (m: int) -> int = { f(m) }; def main () -> int = { f(3) };",
        7,
    );
}

#[test]
fn calls_are_unary_with_unit_and_tuple_sugar() {
    runs(
        "export { main }; def zero () -> int = { 2 }; def sum (a: int, b: int) -> int = { a + b }; def apply (f: (int, int) -> int, p: (int, int)) -> int = { f(p) }; def main () -> int = { apply(sum, (zero(), 5)) };",
        7,
    );
}

#[test]
fn nominal_records_preserve_source_order_and_field_layout() {
    runs(
        "export { main }; type R = { a: int, b: int }; def main () -> int = { var x = 0; var r = R { b = (x := 1), a = (x := 2) }; r.a * 10 + r.b + x };",
        23,
    );
}

#[test]
fn loaded_values_do_not_change_after_later_stores() {
    runs(
        "export { main }; def main () -> int = { var x = 1; var old = x; x := 2; old * 10 + x };",
        12,
    );
}

#[test]
fn short_circuiting_and_joins_preserve_effects() {
    runs(
        "export { main }; def main () -> int = { var x = 0; var a = (1 == 2) && ((x := 1) == 1); var b = (1 == 1) || ((x := 2) == 2); var n = if (a || b) { 3 } else { 4 }; n + x };",
        3,
    );
}

#[test]
fn nominal_function_types_and_numeric_operations_work() {
    runs(
        "export { main }; type Meters = int; type F = (Meters) -> Meters; def add (x: Meters) -> Meters = { x + Meters (2) }; def main () -> int = { var f = F (add); int (f(Meters (5))) };",
        7,
    );
    runs(
        "export { main }; type Entry = () -> int; def start() -> int = { 23 }; def main() -> int = { var entry = Entry(start); entry() };",
        23,
    );
}

#[test]
fn integer_arithmetic_wraps_at_its_declared_width() {
    runs(
        "export { main }; def main () -> int = { var a = sbyte (127); var b = a + sbyte (1); var c = int (2147483647) + int (1); var d = long (-9223372036854775808) / long (-1); if (b == sbyte (-128) && c == int (-2147483648) && d == long (-9223372036854775808)) { 0 } else { 1 } };",
        0,
    );
}

#[test]
fn signed_right_shift_and_unsigned_multiplication_are_defined() {
    runs(
        "export { main }; def main () -> int = { var a = int (-8) >> int (2); var b = uint (4294967295) * uint (4294967295); if (a == int (-2) && b == uint (1)) { 0 } else { 1 } };",
        0,
    );
}

#[test]
fn invalid_integer_operations_fail_at_runtime() {
    for expression in ["1 / 0", "1 % 0", "1 << 32", "1 >> -1"] {
        let result = run_module(&module(&format!(
            "export {{ main }}; def main () -> int = {{ {expression} }};"
        )));
        assert_eq!(result.status.code(), Some(1));
        assert!(String::from_utf8_lossy(&result.stderr).contains("resin:"));
    }
}

#[test]
fn unsupported_operations_report_backend_errors() {
    let error = c::emit(
        &module("export { main }; def main () -> int = { var r = { x = 1 }; r + r; 0 };"),
        "main",
    )
    .unwrap_err();
    assert!(error.to_string().contains("unsupported builtin"));
    assert!(error.to_string().contains("instruction"));
}

#[test]
fn invalid_ir_is_rejected_before_emitting() {
    assert!(
        c::emit(&ir::Module::default(), "main")
            .unwrap_err()
            .to_string()
            .contains("export { main }")
    );
    let mut m = module("export { main }; def main () -> int = { 0 };");
    m.functions[0].blocks[0]
        .instrs
        .insert(0, ir::Instr::Discard);
    assert!(
        c::emit(&m, "main")
            .unwrap_err()
            .to_string()
            .contains("StackUnderflow")
    );
}

#[test]
fn unused_functions_do_not_fail_strict_compilation() {
    let mut m = module("export { main }; def main () -> int = { 0 };");
    let mut spare = m.functions[0].clone();
    spare.name = Some("unused".into());
    m.functions.push(spare);
    assert!(run_module(&m).status.success());
}

#[test]
fn failed_compilation_preserves_existing_output() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let output = temp.path().join("existing");
    fs::write(&output, b"keep me").unwrap();
    assert!(
        toolchain::compile_c(
            "not C",
            &output,
            std::ffi::OsStr::new(resin::toolchain::DEFAULT_C_COMPILER)
        )
        .is_err()
    );
    assert_eq!(fs::read(&output).unwrap(), b"keep me");
    assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 1);
}

#[test]
fn loops_carry_typed_stack_values_across_edges() {
    use ir::{BasicBlock, BlockId, Instr::*, Local, LocalId, Terminator::*, Ty, Value};
    let mut m = module("export { main }; def main () -> int = { 0 };");
    let f = m
        .functions
        .iter_mut()
        .find(|f| f.name.as_deref() == Some("main"))
        .unwrap();
    let counter = LocalId::from_index(f.locals.len());
    f.locals.push(Local {
        name: None,
        ty: Ty::Int32,
    });
    let int = |n| Push {
        value: Value::Int32 { value: n },
    };
    let op = |name: &str, result| CallBuiltin {
        name: name.into(),
        params: vec![Ty::Int32; 2],
        result,
    };
    f.blocks = vec![
        BasicBlock {
            name: None,
            instrs: vec![
                LocalAddress { local: counter },
                int(3),
                Store,
                Discard,
                int(0),
            ],
            terminator: Break {
                target: BlockId::from_index(1),
            },
        },
        BasicBlock {
            name: None,
            instrs: vec![
                LocalAddress { local: counter },
                Load,
                int(0),
                op(">", Ty::Bool),
            ],
            terminator: Branch {
                then: BlockId::from_index(2),
                els: BlockId::from_index(3),
            },
        },
        BasicBlock {
            name: None,
            instrs: vec![
                LocalAddress { local: counter },
                Load,
                op("+", Ty::Int32),
                LocalAddress { local: counter },
                LocalAddress { local: counter },
                Load,
                int(1),
                op("-", Ty::Int32),
                Store,
                Discard,
            ],
            terminator: Break {
                target: BlockId::from_index(1),
            },
        },
        BasicBlock {
            name: None,
            instrs: vec![],
            terminator: Return,
        },
    ];
    assert_eq!(run_module(&m).status.code(), Some(6));
}

#[test]
fn array_addresses_and_dynamic_bounds_are_executable() {
    use ir::{Instr::*, Local, LocalId, Ty, Value};
    let mut m = module("export { main }; def main () -> int = { 0 };");
    let f = m
        .functions
        .iter_mut()
        .find(|f| f.name.as_deref() == Some("main"))
        .unwrap();
    let array = LocalId::from_index(f.locals.len());
    f.locals.push(Local {
        name: None,
        ty: Ty::Array {
            element: Box::new(Ty::Int32),
            length: 2,
        },
    });
    f.blocks[0].instrs = vec![
        LocalAddress { local: array },
        Push {
            value: Value::Int32 { value: 4 },
        },
        Push {
            value: Value::Int32 { value: 9 },
        },
        MakeArray {
            elements: 2,
            element: Ty::Int32,
        },
        Store,
        Discard,
        LocalAddress { local: array },
        Push {
            value: Value::Int32 { value: 1 },
        },
        AccessDynamic,
        Load,
    ];
    assert_eq!(run_module(&m).status.code(), Some(9));
    let f = m
        .functions
        .iter_mut()
        .find(|f| f.name.as_deref() == Some("main"))
        .unwrap();
    f.blocks[0].instrs[7] = Push {
        value: Value::Int32 { value: -1 },
    };
    let output = run_module(&m);
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("array index"));
}
