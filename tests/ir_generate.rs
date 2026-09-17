#[allow(dead_code)]
mod support;

use pipeline::generate;
use resin_hir::GenerateErrorKind;
use resin_lir::verify;
use resin_lir::{Instr, Terminator, format_module};
use resin_types::prelude::*;
use support::pipeline;

const FIBONACCI: &str = r#"export { main };
fn fibonacci(n: int) -> int  {
    if (n <= 1) { n } else {
        let mut f0 = fibonacci(n - 1);
        let mut f1 = fibonacci(n - 2);
        f0 + f1
    }
}
fn main()  { fibonacci(10); }
"#;

fn parse(src: &str) -> resin_ast::SourceFile {
    let parsed = support::frontend::ast(&support::frontend::cst(src, None));
    assert!(parsed.errors.is_empty(), "{src}\n{:?}", parsed.errors);
    parsed.file
}

fn compile(src: &str) -> resin_lir::Module {
    generate(&parse(src)).unwrap_or_else(|err| panic!("{err}"))
}

fn compile_err(src: &str) -> GenerateErrorKind {
    generate(&parse(src)).unwrap_err().kind
}

#[test]
fn while_lowers_to_a_structured_loop_and_returns_unit() {
    let module = compile(
        "export { main }; fn main () -> ()  { let mut i = 0; while (i < 3) { i = i + 1; } }",
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
        .position(|b| b.name.as_deref() == Some("while.body"))
        .unwrap();
    assert!(function.blocks.iter().any(|block| matches!(
        block.terminator,
        Terminator::Loop { condition: cond, body: repeated, next: Some(_) }
            if cond.index() == condition && repeated.index() == body
    )));
    assert_eq!(function.blocks[condition].terminator, Terminator::LoopTest);
    assert_eq!(function.blocks[body].terminator, Terminator::Continue);
    assert_eq!(function.result, Ty::Unit);
    verify(&module).unwrap();
}

#[test]
fn while_does_not_assume_its_body_ran() {
    for source in [
        "export { main }; fn main () -> int  { let mut x: int; while (1 == 0) { x = 1; }; x }",
        "export { main }; fn main () -> ()  { let mut x: int; while (x < 3) { x = 1; }; }",
        "export { main }; fn main () -> ()  { let mut x: int; while (1 == 0) { x = x + 1; }; }",
        "export { main }; fn main () -> int  { let mut x: int; while ((1 == 0) && ({ x = 1; x } == 1)) {}; x }",
        "export { main }; fn main () -> int  { let mut x: int; while (1 == 1) { x = 1; }; x }",
    ] {
        assert!(
            matches!(
                compile_err(source),
                GenerateErrorKind::UninitializedValue { .. }
            ),
            "{source}"
        );
    }
    compile(
        "export { main }; fn main () -> int  { let mut x: int; while ({ x = 1; x } == 0) {}; x }",
    );
}

#[test]
fn while_requires_a_boolean_condition_and_keeps_body_bindings_local() {
    assert!(matches!(
        compile_err("export { main }; fn main () -> ()  { while (1) {} }"),
        GenerateErrorKind::Type { kind: _ }
    ));
    assert!(matches!(
        compile_err(
            "export { main }; fn main () -> int  { while (1 == 0) { let mut inner = 1; }; inner }"
        ),
        GenerateErrorKind::UnboundValue { .. }
    ));
    assert!(matches!(
        compile_err("export { main }; fn main () -> int  { while (1 == 0) { 42 } }"),
        GenerateErrorKind::Type {
            kind: TypeErrorKind::TypeMismatch { .. }
        }
    ));
}

#[test]
fn examples_generate_verified_ir() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
    let mut found = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("resin") {
            continue;
        }
        found += 1;
        let module =
            pipeline::file_module(&path).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
        verify(&module).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    }
    assert!(
        found >= 9,
        "expected language and GPU examples in examples/"
    );
}

#[test]
fn fibonacci_generates_verified_ir() {
    let src = FIBONACCI;
    let module = compile(src);
    verify(&module).unwrap();

    assert_eq!(module.functions.len(), 2);
    assert_eq!(module.functions[1].result, Ty::Unit);
    assert_eq!(module.functions[0].result, Ty::Int32);
    assert_eq!(module.functions[0].locals[0].ty, Ty::Int32);
}

#[test]
fn ir_dump_is_an_s_expression_with_names() {
    let dump = format_module(&compile(FIBONACCI));
    assert!(dump.starts_with("(module"));
    assert!(dump.contains("main"));
    assert!(dump.contains("fibonacci"));
    assert!(dump.contains("(local f0 int)"));
    assert!(dump.contains("(local-ref n)"));
    assert!(dump.contains("(function-ref fibonacci)"));
    assert!(dump.contains("(block then"));
    let compact = dump.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(compact.contains("(if (then (block then"), "{dump}");
    assert!(compact.contains("(block join"), "{dump}");
    assert!(!dump.contains("global-addr g"));
}

#[test]
fn recursive_calls_reference_functions_directly() {
    let module = compile(FIBONACCI);
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
        compile_err("export { main }; fn main() -> ()  { let mut x = x + 1; }"),
        GenerateErrorKind::EagerRecursion { .. }
    ));
}

#[test]
fn recursive_function_result_is_checked() {
    assert!(matches!(
        compile_err("fn f (n: int) -> int  { if (n == 0) { () } else { f(n - 1) } }"),
        GenerateErrorKind::Type {
            kind: TypeErrorKind::TypeMismatch { .. }
        }
    ));
}

#[test]
fn type_mismatch_is_a_type_error() {
    assert!(matches!(
        compile_err("export { main }; fn main() -> ()  { let mut x = if (1) { 1 } else { 2 }; }"),
        GenerateErrorKind::Type {
            kind: TypeErrorKind::ExpectedBoolean { .. }
        }
    ));
}

#[test]
fn linked_list_type_is_finite_through_its_pointer() {
    let module = compile("struct List { value: int, next: Ptr<List>, }");
    assert_eq!(
        module.types.iter().filter(|d| d.name().is_some()).count(),
        1
    );
    let list = pipeline::nominal(&module, "List");
    let Ty::Record { fields } = module.types[list.index()].body().unwrap() else {
        panic!("expected a record body");
    };
    assert!(matches!(fields[1].ty, Ty::Pointer { .. }));
}

#[test]
fn inline_recursive_type_is_rejected_during_generation() {
    assert!(matches!(
        compile_err("struct Bad { next: Bad, }"),
        GenerateErrorKind::Type {
            kind: TypeErrorKind::RecursiveTypeWithoutIndirection { .. }
        }
    ));
}

#[test]
fn function_values_do_not_capture_local_state() {
    let module = compile(
        "fn add (x: int, y: int) -> int  { x + y } fn select () -> (int, int) -> int  { let mut local = add; local }",
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
    let module = compile("fn f (p: Ptr<int>) -> int  { p.* = 1; p.* }");
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
    let module = compile("fn f (c: int) -> int  { if (c == 0) { 1 } else { 2 } }");
    verify(&module).unwrap();
    let function = &module.functions[0];
    assert!(
        function
            .blocks
            .iter()
            .any(|block| { matches!(block.terminator, Terminator::If { .. }) })
    );
    assert_eq!(function.result, Ty::Int32);
}

#[test]
fn nominal_ascription_wraps_and_unwraps_one_layer() {
    let module = compile(
        r#"struct Meters { value: int, }
fn to_meters (n: int) -> Meters  { Meters { value = n } }
fn from_meters (m: Meters) -> int  { m.value }
"#,
    );
    verify(&module).unwrap();
    let meters = Ty::Defined {
        definition: pipeline::nominal(&module, "Meters"),
    };
    assert_eq!(
        module.functions[0].ty().unwrap(),
        Ty::Function {
            params: vec![Ty::Int32],
            result: Box::new(meters.clone()),
        }
    );
    assert_eq!(
        module.functions[1].ty().unwrap(),
        Ty::Function {
            params: vec![meters],
            result: Box::new(Ty::Int32),
        }
    );
}

#[test]
fn field_access_autoderefs_a_named_pointer() {
    let module = compile(
        r#"struct FieldsX<T0> { x: T0, }
type P = Ptr<FieldsX<int>>;
fn f (p: P) -> int  { p.x }
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
        r#"type Handler = (int) -> int;
fn f (h: Handler, n: int) -> int  { h(n) }
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

struct Meters { value: int, }
struct Distance { value: Meters, }

fn main() -> ()  {
    let mut x = Distance(1);
}"#
        ),
        GenerateErrorKind::Type {
            kind: TypeErrorKind::TypeMismatch { .. }
        }
    ));
}

#[test]
fn nested_nominal_ascription_wraps_the_defining_body() {
    let module = compile(
        r#"export { main };

struct Meters { value: int, }
struct Distance { value: Meters, }

fn main() -> ()  {
    let mut x = Distance { value = Meters { value = 1 } };
    let mut y = x.value.value;
}"#,
    );
    verify(&module).unwrap();
    assert_eq!(
        module.functions[0]
            .locals
            .iter()
            .find(|local| local.name.as_deref() == Some("x"))
            .unwrap()
            .ty,
        Ty::Defined {
            definition: pipeline::nominal(&module, "Distance"),
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
        r#"struct List { value: int, next: Ptr<List>, }
fn nil (p: Ptr<List>) -> List  { List { value = 0, next = p } }
"#,
    );
    verify(&module).unwrap();
    let list = Ty::Defined {
        definition: pipeline::nominal(&module, "List"),
    };
    assert_eq!(
        module.functions[0].ty().unwrap(),
        Ty::Function {
            params: vec![Ty::Pointer {
                pointee: Box::new(list.clone()),
            }],
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
        compile_err("export { main }; fn main() -> ()  { let mut x = []; }"),
        GenerateErrorKind::Type {
            kind: TypeErrorKind::EmptyArrayNeedsElementType
        }
    ));
}

#[test]
fn unbound_value_is_reported() {
    assert!(matches!(
        compile_err("export { main }; fn main() -> ()  { let mut x = y; }"),
        GenerateErrorKind::UnboundValue { .. }
    ));
}

#[test]
fn span_and_literal_locals_are_typed() {
    let module = compile(
        r#"export { main };

struct Span<T> { data: Ptr<T>, length: ulong, }
type Buf = Span<int>;

fn main() -> ()  {
    let mut x = 1;
    let mut buf: Buf;
}"#,
    );
    verify(&module).unwrap();
    let Ty::Defined { definition } = &module.functions[0].locals[1].ty else {
        panic!("nominal source span")
    };
    assert_eq!(
        module.types[definition.index()].body(),
        Some(&Ty::pointer_length(Ty::Int32))
    );
    assert_eq!(module.functions[0].locals[0].ty, Ty::Int64);
}

#[test]
fn short_circuit_and_compiles() {
    let module = compile("fn f (a: int) -> bool  { (a == 0) && (a == 1) }");
    verify(&module).unwrap();
    assert_eq!(module.functions[0].result, Ty::Bool);
}

#[test]
fn pointers_cannot_be_used_in_arithmetic_and_indexing_is_explicit() {
    for body in ["p + 1", "p - 1", "p + p", "-p", "~p", "1 + p", "p(0)"] {
        let source = format!("fn bad(p: Ptr<int>) -> Ptr<int>  {{ {body} }}");
        assert!(generate(&parse(&source)).is_err(), "{source}");
    }
    for body in ["xs(0).* = 3", "xs(1.5)", "xs(0, 1)"] {
        let source = format!("fn bad() -> int  {{ let mut xs = [1, 2]; {body} }}");
        assert!(generate(&parse(&source)).is_err(), "{source}");
    }
    compile("fn explicit(p: Ptr<int>) -> Ptr<int>  { Ptr<int>(ulong(p) + ulong(4)) }");
}

#[test]
fn one_armed_if_preserves_conditional_initialization_and_scope() {
    for source in [
        "fn main() -> int  { let mut x: int; if (1 == 1) { x = 1; }; x }",
        "fn main() -> int  { if (1 == 1) { let mut x = 1; }; x }",
        "fn main()  { if (1) {} }",
        "fn main()  { if (1 == 1) { 42 } }",
    ] {
        assert!(generate(&parse(source)).is_err(), "{source}");
    }
    compile("fn main() -> int  { let mut x: int; if ({ x = 2; x } == 2) {}; x }");
}
