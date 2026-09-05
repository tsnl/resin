use std::{ffi::OsString, fs, process::Command};

use resin::{
    backend::c,
    ir,
    toolchain::{self, TempDir},
};
mod support;
use support::module;

fn run_module(module: &ir::Module) -> std::process::Output {
    let source = c::emit(module).unwrap();
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let output = temp.path().join("program");
    let cc = std::env::var_os("CC").unwrap_or_else(|| OsString::from("cc"));
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
fn all_examples_compile_as_strict_c11() {
    for entry in fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/examples")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|ext| ext == "resin") {
            runs(&fs::read_to_string(path).unwrap(), 0);
        }
    }
}

#[test]
fn ordinary_functions_can_be_passed_and_selected() {
    runs(
        "add (x: int, y: int) -> int = { x + y }; apply (f: (int, int) -> int, args: (int, int)) -> int = { f(args) }; main () -> int = { a = add; b = if (1 == 1) { a } else { add }; apply(a, (10, 3)) + b(20, 4) };",
        37,
    );
}

#[test]
fn recursive_functions_can_read_globals() {
    runs(
        "offset = 2; fact (n: int) -> int = { if (n == 0) { offset } else { n * fact(n - 1) } }; main () -> int = { fact(4) };",
        48,
    );
}

#[test]
fn mutual_recursion_needs_no_forward_declaration() {
    runs(
        "f (n: int) -> int = { if (n == 0) { 7 } else { next(n - 1) } }; next (m: int) -> int = { f(m) }; main () -> int = { f(3) };",
        7,
    );
}

#[test]
fn calls_are_unary_with_unit_and_tuple_sugar() {
    runs(
        "zero () -> int = { 2 }; sum (a: int, b: int) -> int = { a + b }; apply (f: (int, int) -> int, p: (int, int)) -> int = { f(p) }; main () -> int = { apply(sum, (zero(), 5)) };",
        7,
    );
}

#[test]
fn nominal_records_preserve_source_order_and_field_layout() {
    runs(
        "R = { a: int, b: int }; main () -> int = { x = 0; r = R { b = (x := 1), a = (x := 2) }; r.a * 10 + r.b + x };",
        23,
    );
}

#[test]
fn loaded_values_do_not_change_after_later_stores() {
    runs(
        "x = 1; main () -> int = { old = x; x := 2; old * 10 + x };",
        12,
    );
}

#[test]
fn short_circuiting_and_joins_preserve_effects() {
    runs(
        "main () -> int = { x = 0; a = (1 == 2) && ((x := 1) == 1); b = (1 == 1) || ((x := 2) == 2); n = if (a || b) { 3 } else { 4 }; n + x };",
        3,
    );
}

#[test]
fn nominal_function_types_and_numeric_operations_work() {
    runs(
        "Meters = int; F = (Meters) -> Meters; add (x: Meters) -> Meters = { x + Meters (2) }; f = F (add); main () -> int = { int (f(Meters (5))) };",
        7,
    );
    runs(
        "Entry = () -> int; start () -> int = { 23 }; main = Entry (start);",
        23,
    );
}

#[test]
fn integer_arithmetic_wraps_at_its_declared_width() {
    runs(
        "main () -> int = { a = sbyte (127); b = a + sbyte (1); c = int (2147483647) + int (1); d = long (-9223372036854775808) / long (-1); if (b == sbyte (-128) && c == int (-2147483648) && d == long (-9223372036854775808)) { 0 } else { 1 } };",
        0,
    );
}

#[test]
fn signed_right_shift_and_unsigned_multiplication_are_defined() {
    runs(
        "main () -> int = { a = int (-8) >> int (2); b = uint (4294967295) * uint (4294967295); if (a == int (-2) && b == uint (1)) { 0 } else { 1 } };",
        0,
    );
}

#[test]
fn invalid_integer_operations_fail_at_runtime() {
    for expression in ["1 / 0", "1 % 0", "1 << 32", "1 >> -1"] {
        let result = run_module(&module(&format!("main () -> int = {{ {expression} }};")));
        assert_eq!(result.status.code(), Some(1));
        assert!(String::from_utf8_lossy(&result.stderr).contains("resin:"));
    }
}

#[test]
fn unsupported_operations_report_backend_errors() {
    let error = c::emit(&module("main () -> int = { r = { x = 1 }; r + r; 0 };")).unwrap_err();
    assert!(error.to_string().contains("unsupported builtin"));
    assert!(error.to_string().contains("instruction"));
}

#[test]
fn invalid_ir_is_rejected_before_emitting() {
    assert!(
        c::emit(&ir::Module::default())
            .unwrap_err()
            .to_string()
            .contains("initializer")
    );
    let mut m = module("main () -> int = { 0 };");
    m.functions[0].blocks[0]
        .instrs
        .insert(0, ir::Instr::Discard);
    assert!(
        c::emit(&m)
            .unwrap_err()
            .to_string()
            .contains("StackUnderflow")
    );
}

#[test]
fn unused_functions_do_not_fail_strict_compilation() {
    let mut m = module("main () -> int = { 0 };");
    let mut spare = m.functions[1].clone();
    spare.name = Some("unused".into());
    m.functions.push(spare);
    assert!(run_module(&m).status.success());
}

#[test]
fn failed_compilation_preserves_existing_output() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let output = temp.path().join("existing");
    fs::write(&output, b"keep me").unwrap();
    assert!(toolchain::compile_c("not C", &output, std::ffi::OsStr::new("cc")).is_err());
    assert_eq!(fs::read(&output).unwrap(), b"keep me");
    assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 1);
}

#[test]
fn loops_carry_typed_stack_values_across_edges() {
    use ir::{BasicBlock, BlockId, Instr::*, Local, LocalId, Terminator::*, Ty, Value};
    let mut m = module("main () -> int = { 0 };");
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
    let mut m = module("main () -> int = { 0 };");
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
