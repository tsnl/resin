use resin::compiler::generate;
use resin::diagnostic::GenerateErrorKind;
use resin::lir::{Instr, Terminator, Ty, format_module};
use resin::lir_verifier::verify;
use resin::types::TypeErrorKind;

fn parse(src: &str) -> resin::ast::SourceFile {
    resin::ast::lower::generate(&resin::cst::Document::reparse(src.to_string(), None))
        .unwrap_or_else(|err| panic!("{err}"))
}

fn compile(src: &str) -> resin::lir::Module {
    generate(&parse(src)).unwrap_or_else(|err| panic!("{err}"))
}

fn compile_err(src: &str) -> GenerateErrorKind {
    generate(&parse(src)).unwrap_err().kind
}

#[test]
fn while_lowers_to_a_back_edge_and_returns_unit() {
    let module = compile(
        "export { main }; def main () -> () = { var i = 0; while (i < 3) { i := i + 1; } };",
    );
    let function = &module.functions[0];
    let condition = function
        .blocks
        .iter()
        .position(|b| b.name.as_deref() == Some("while.cond"))
        .unwrap();
    let body = function
        .blocks
        .iter()
        .find(|b| b.name.as_deref() == Some("while.body"))
        .unwrap();
    assert!(matches!(body.terminator, Terminator::Break { target } if target.index() == condition));
    assert_eq!(function.result, Ty::Unit);
    verify(&module).unwrap();
}

#[test]
fn while_does_not_assume_its_body_ran() {
    for source in [
        "export { main }; def main () -> int = { var x: int; while (1 == 0) { x := 1; }; x };",
        "export { main }; def main () -> () = { var x: int; while (x < 3) { x := 1; }; };",
        "export { main }; def main () -> () = { var x: int; while (1 == 0) { x := x + 1; }; };",
        "export { main }; def main () -> int = { var x: int; while ((1 == 0) && ((x := 1) == 1)) {}; x };",
        "export { main }; def main () -> int = { var x: int; while (1 == 1) { x := 1; }; x };",
    ] {
        assert!(
            matches!(
                compile_err(source),
                GenerateErrorKind::UninitializedValue { .. }
            ),
            "{source}"
        );
    }
    compile("export { main }; def main () -> int = { var x: int; while ((x := 1) == 0) {}; x };");
}

#[test]
fn while_requires_a_boolean_condition_and_keeps_body_bindings_local() {
    assert!(matches!(
        compile_err("export { main }; def main () -> () = { while (1) {} };"),
        GenerateErrorKind::Type(_)
    ));
    assert!(matches!(
        compile_err(
            "export { main }; def main () -> int = { while (1 == 0) { var inner = 1; }; inner };"
        ),
        GenerateErrorKind::UnboundValue { .. }
    ));
    assert!(matches!(
        compile_err("export { main }; def main () -> int = { while (1 == 0) { 42 } };"),
        GenerateErrorKind::Type(TypeErrorKind::TypeMismatch { .. })
    ));
}

#[test]
fn examples_generate_verified_ir() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples");
    let mut found = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("resin") {
            continue;
        }
        found += 1;
        let ast = resin::ast::load(&path).unwrap();
        let module = resin::compiler::generate_program(&ast)
            .unwrap_or_else(|err| panic!("{}: {err}", path.display()));
        verify(&module).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    }
    assert!(
        found >= 9,
        "expected language and GPU examples in examples/"
    );
}

#[test]
fn fibonacci_generates_verified_ir() {
    let src = include_str!("../../../examples/eg001.resin");
    let module = compile(src);
    verify(&module).unwrap();

    assert_eq!(module.functions.len(), 2);
    assert_eq!(module.functions[1].result, Ty::Unit);
    assert_eq!(module.functions[0].result, Ty::Int32);
    assert_eq!(module.functions[0].locals[0].ty, Ty::Int32);
}

#[test]
fn ir_dump_is_an_s_expression_with_names() {
    let dump = format_module(&compile(include_str!("../../../examples/eg001.resin")));
    assert!(dump.starts_with("(module"));
    assert!(dump.contains("main"));
    assert!(dump.contains("fibonacci"));
    assert!(dump.contains("(local f0 int)"));
    assert!(dump.contains("(local-addr n)"));
    assert!(dump.contains("(function-ref fibonacci)"));
    assert!(dump.contains("(block then"));
    assert!(dump.contains("(branch then else)"));
    assert!(!dump.contains("global-addr g"));
}

#[test]
fn recursive_calls_reference_functions_directly() {
    let module = compile(include_str!("../../../examples/eg001.resin"));
    let fib = &module.functions[0];
    assert!(fib.blocks.iter().any(|block| {
        block.instrs.iter().any(|instr| {
            matches!(
                instr,
                Instr::Function { function } if function.index() == 0
            )
        })
    }));
}

#[test]
fn eager_recursion_is_rejected() {
    assert!(matches!(
        compile_err("export { main }; def main() -> () = { var x = x + 1; };"),
        GenerateErrorKind::EagerRecursion { .. }
    ));
}

#[test]
fn recursive_function_result_is_checked() {
    assert!(matches!(
        compile_err("def f (n: int) -> int = { if (n == 0) { () } else { f(n - 1) } };"),
        GenerateErrorKind::Type(TypeErrorKind::TypeMismatch { .. })
    ));
}

#[test]
fn type_mismatch_is_a_type_error() {
    assert!(matches!(
        compile_err("export { main }; def main() -> () = { var x = if (1) { 1 } else { 2 }; };"),
        GenerateErrorKind::Type(TypeErrorKind::ExpectedBoolean { .. })
    ));
}

#[test]
fn linked_list_type_is_finite_through_its_pointer() {
    let module = compile("struct List { value: int, next: Ptr<List> };");
    assert_eq!(
        module.types.iter().filter(|d| d.name().is_some()).count(),
        2
    );
    assert_eq!(module.types[1].name().unwrap().as_ref(), "List");
    let Ty::Record { fields } = module.types[1].body().unwrap() else {
        panic!("expected a record body");
    };
    assert!(matches!(fields[1].ty, Ty::Pointer { .. }));
}

#[test]
fn inline_recursive_type_is_rejected_during_generation() {
    assert!(matches!(
        compile_err("struct Bad { next: Bad };"),
        GenerateErrorKind::Type(TypeErrorKind::RecursiveTypeWithoutIndirection { .. })
    ));
}

#[test]
fn function_values_do_not_capture_local_state() {
    let module = compile(
        "def add (x: int, y: int) -> int = { x + y }; def select () -> (int, int) -> int = { var local = add; local };",
    );
    verify(&module).unwrap();
    assert!(
        module
            .functions
            .iter()
            .flat_map(|f| &f.blocks)
            .flat_map(|b| &b.instrs)
            .any(|i| matches!(i, Instr::Function { .. }))
    );
}

#[test]
fn assignment_and_deref_store_through_an_address() {
    let module = compile("def f (p: Ptr<int>) -> int = { p.* := 1; p.* };");
    verify(&module).unwrap();
    let function = &module.functions[0];
    assert!(function.blocks.iter().any(|block| {
        block
            .instrs
            .iter()
            .any(|instr| matches!(instr, Instr::Store))
    }));
    assert!(function.blocks.iter().any(|block| {
        block
            .instrs
            .iter()
            .any(|instr| matches!(instr, Instr::Load))
    }));
}

#[test]
fn if_joins_then_and_else_values() {
    let module = compile("def f (c: int) -> int = { if (c == 0) { 1 } else { 2 } };");
    verify(&module).unwrap();
    let function = &module.functions[0];
    assert!(
        function
            .blocks
            .iter()
            .any(|block| { matches!(block.terminator, Terminator::Branch { .. }) })
    );
    assert_eq!(function.result, Ty::Int32);
}

#[test]
fn nominal_ascription_wraps_and_unwraps_one_layer() {
    let module = compile(
        r#"
struct Meters { value: int };
def to_meters (n: int) -> Meters = { Meters { value = n } };
def from_meters (m: Meters) -> int = { m.value };
"#,
    );
    verify(&module).unwrap();
    let meters = Ty::Defined {
        definition: resin::lir::TypeId::from_index(1),
    };
    assert_eq!(
        module.functions[0].ty().unwrap(),
        Ty::Function {
            param: Box::new(Ty::Int32),
            result: Box::new(meters.clone()),
        }
    );
    assert_eq!(
        module.functions[1].ty().unwrap(),
        Ty::Function {
            param: Box::new(meters),
            result: Box::new(Ty::Int32),
        }
    );
}

#[test]
fn field_access_autoderefs_a_named_pointer() {
    let module = compile(
        r#"
type P = Ptr<{ x: int }>;
def f (p: P) -> int = { p.x };
"#,
    );
    verify(&module).unwrap();
    assert_eq!(module.functions[0].result, Ty::Int32);
    assert!(module.functions[0].blocks.iter().any(|block| {
        block
            .instrs
            .iter()
            .any(|instr| matches!(instr, Instr::Load))
            && block
                .instrs
                .iter()
                .any(|instr| matches!(instr, Instr::AccessStatic { index: 0 }))
    }));
}

#[test]
fn named_function_type_can_be_called() {
    let module = compile(
        r#"
type Handler = (int) -> int;
def f (h: Handler, n: int) -> int = { h(n) };
"#,
    );
    verify(&module).unwrap();
    assert_eq!(module.functions[0].result, Ty::Int32);
}

#[test]
fn nested_nominal_ascription_does_not_skip_a_layer() {
    assert!(matches!(
        compile_err(
            r#"export { main };

struct Meters { value: int };
struct Distance { value: Meters };

def main() -> () = {
    var x = Distance (1);
};"#
        ),
        GenerateErrorKind::Type(TypeErrorKind::TypeMismatch { .. })
    ));
}

#[test]
fn nested_nominal_ascription_wraps_the_defining_body() {
    let module = compile(
        r#"export { main };

struct Meters { value: int };
struct Distance { value: Meters };

def main() -> () = {
    var x = Distance { value = Meters { value = 1 } };
    var y = x.value.value;
};"#,
    );
    verify(&module).unwrap();
    assert_eq!(
        module.functions[0].locals[1].ty,
        Ty::Defined {
            definition: resin::lir::TypeId::from_index(2),
        }
    );
    assert_eq!(
        module.functions[0]
            .locals
            .iter()
            .find(|local| local.name.as_deref() == Some("y"))
            .unwrap()
            .ty,
        Ty::Int32
    );
}

#[test]
fn nominal_record_ascription_wraps_the_representation() {
    let module = compile(
        r#"
struct List { value: int, next: Ptr<List> };
def nil (p: Ptr<List>) -> List = { List { value = 0, next = p } };
"#,
    );
    verify(&module).unwrap();
    let list = Ty::Defined {
        definition: resin::lir::TypeId::from_index(1),
    };
    assert_eq!(
        module.functions[0].ty().unwrap(),
        Ty::Function {
            param: Box::new(Ty::Pointer {
                pointee: Box::new(list.clone()),
            }),
            result: Box::new(list.clone()),
        }
    );
    assert!(module.functions.iter().any(|function| {
        function.blocks.iter().any(|block| {
            block
                .instrs
                .iter()
                .any(|instr| matches!(instr, Instr::Ascribe { ty } if *ty == list))
        })
    }));
}

#[test]
fn empty_array_needs_an_element_type() {
    assert!(matches!(
        compile_err("export { main }; def main() -> () = { var x = []; };"),
        GenerateErrorKind::Type(TypeErrorKind::EmptyArrayNeedsElementType)
    ));
}

#[test]
fn unbound_value_is_reported() {
    assert!(matches!(
        compile_err("export { main }; def main() -> () = { var x = y; };"),
        GenerateErrorKind::UnboundValue { .. }
    ));
}

#[test]
fn span_and_literal_locals_are_typed() {
    let module = compile(
        r#"export { main };

type Buf = Span<int>;

def main() -> () = {
    var x = 1;
    var buf: Buf;
};"#,
    );
    verify(&module).unwrap();
    assert!(matches!(
        &module.functions[0].locals[2].ty,
        Ty::Span { element } if **element == Ty::Int32
    ));
    assert_eq!(module.functions[0].locals[1].ty, Ty::Int64);
}

#[test]
fn short_circuit_and_compiles() {
    let module = compile("def f (a: int) -> bool = { (a == 0) && (a == 1) };");
    verify(&module).unwrap();
    assert_eq!(module.functions[0].result, Ty::Bool);
}

#[test]
fn pointers_cannot_be_used_in_arithmetic_and_indexing_is_explicit() {
    for body in ["p + 1", "p - 1", "p + p", "-p", "~p", "1 + p", "p(0)"] {
        let source = format!("def bad(p: Ptr<int>) -> Ptr<int> = {{ {body} }};");
        assert!(generate(&parse(&source)).is_err(), "{source}");
    }
    for body in ["xs(0) := 3", "xs(1.5).*", "xs(0, 1).*"] {
        let source = format!("def bad() -> int = {{ var xs = [1, 2]; {body} }};");
        assert!(generate(&parse(&source)).is_err(), "{source}");
    }
    compile("def explicit(p: Ptr<int>) -> Ptr<int> = { Ptr<int>(ulong(p) + ulong(4)) };");
}

#[test]
fn one_armed_if_preserves_conditional_initialization_and_scope() {
    for source in [
        "def main() -> int = { var x: int; if (1 == 1) { x := 1; }; x };",
        "def main() -> int = { if (1 == 1) { var x = 1; }; x };",
        "def main() = { if (1) {} };",
        "def main() = { if (1 == 1) { 42 } };",
    ] {
        assert!(generate(&parse(source)).is_err(), "{source}");
    }
    compile("def main() -> int = { var x: int; if ((x := 2) == 2) {}; x };");
}
