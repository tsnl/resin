use resin::{
    ast::generate::AstGen,
    ir::{
        GenerateErrorKind, Instr, Terminator, Ty, TypeErrorKind, format_module, generate, verify,
    },
};
use tree_sitter::Parser;

fn parse(src: &str) -> resin::ast::SourceFile {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_resin::LANGUAGE.into())
        .expect("failed to load Resin grammar");
    let tree = parser.parse(src, None).expect("parser returned no tree");
    AstGen::new(src)
        .gen_source_file(tree.root_node())
        .unwrap_or_else(|err| panic!("{err}"))
}

fn compile(src: &str) -> resin::ir::Module {
    generate(&parse(src)).unwrap_or_else(|err| panic!("{err}"))
}

fn compile_err(src: &str) -> GenerateErrorKind {
    generate(&parse(src)).unwrap_err().kind
}

#[test]
fn fibonacci_generates_verified_ir() {
    let src = include_str!("../examples/eg001.resin");
    let module = compile(src);
    verify(&module).unwrap();

    assert_eq!(module.globals.len(), 1);
    assert_eq!(
        module.globals[0].ty,
        Ty::Function {
            params: vec![Ty::Int32],
            result: Box::new(Ty::Int32),
        }
    );
    assert_eq!(module.functions.len(), 2);
    assert_eq!(module.functions[0].result, Ty::Unit);
    assert_eq!(module.functions[1].result, Ty::Int32);
    assert_eq!(module.functions[1].params.len(), 1);
}

#[test]
fn ir_dump_is_an_s_expression_with_names() {
    let dump = format_module(&compile(include_str!("../examples/eg001.resin")));
    assert!(dump.starts_with("(module"));
    assert!(dump.contains("(global fibonacci (func (int) int))"));
    assert!(dump.contains("init"));
    assert!(dump.contains("fibonacci"));
    assert!(dump.contains("(local f0 int)"));
    assert!(dump.contains("(local-addr n)"));
    assert!(dump.contains("(global-addr fibonacci)"));
    assert!(dump.contains("(make-closure fibonacci 0)"));
    assert!(dump.contains("(block then"));
    assert!(dump.contains("(branch then else)"));
    assert!(!dump.contains("global-addr g"));
    assert!(!dump.contains("local-addr l."));
}

#[test]
fn recursive_calls_are_delayed_global_loads() {
    let module = compile(include_str!("../examples/eg001.resin"));
    let fib = &module.functions[1];
    assert!(fib.blocks.iter().any(|block| {
        block.instrs.iter().any(|instr| {
            matches!(
                instr,
                Instr::GlobalAddress { global } if global.index() == 0
            )
        })
    }));
}

#[test]
fn eager_recursion_is_rejected() {
    assert!(matches!(
        compile_err("x = x + 1;"),
        GenerateErrorKind::EagerRecursion { .. }
    ));
}

#[test]
fn recursive_function_needs_a_result_ascription() {
    assert!(matches!(
        compile_err("f = (n: int) => f(n);"),
        GenerateErrorKind::NeedsTypeAnnotation { .. }
    ));
}

#[test]
fn type_mismatch_is_a_type_error() {
    assert!(matches!(
        compile_err("x = if (1) { 1 } else { 2 };"),
        GenerateErrorKind::Type(TypeErrorKind::ExpectedBoolean { .. })
    ));
}

#[test]
fn linked_list_type_is_finite_through_its_pointer() {
    let module = compile("List = { value: int, next: Ptr (List) };");
    assert_eq!(module.types.len(), 1);
    assert_eq!(module.types[0].name.as_ref(), "List");
    let Ty::Record { fields } = &module.types[0].body else {
        panic!("expected a record body");
    };
    assert!(matches!(fields[1].ty, Ty::Pointer { .. }));
}

#[test]
fn inline_recursive_type_is_rejected_during_generation() {
    assert!(matches!(
        compile_err("Bad = { next: Bad };"),
        GenerateErrorKind::Type(TypeErrorKind::RecursiveTypeWithoutIndirection { .. })
    ));
}

#[test]
fn nested_lambda_captures_enclosing_locals() {
    let module = compile("add = (x: int) => (y: int) => { x + y };");
    verify(&module).unwrap();
    let inner = module
        .functions
        .iter()
        .find(|function| function.nonlocals.len() == 1)
        .expect("inner lambda captures x");
    assert_eq!(inner.nonlocals[0].ty, Ty::Int32);
    assert!(module.functions.iter().any(|function| {
        function.blocks.iter().any(|block| {
            block
                .instrs
                .iter()
                .any(|instr| matches!(instr, Instr::MakeClosure { captures, .. } if *captures == 1))
        })
    }));
}

#[test]
fn assignment_and_deref_store_through_an_address() {
    let module = compile("f = (p: Ptr (int)) => { p.* := 1; p.* };");
    verify(&module).unwrap();
    let function = &module.functions[1];
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
    let module = compile("f = (c: int) => if (c == 0) { 1 } else { 2 };");
    verify(&module).unwrap();
    let function = &module.functions[1];
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
Meters = int;
to_meters = (n: int) => Meters (n);
from_meters = (m: Meters) => int (m);
"#,
    );
    verify(&module).unwrap();
    let meters = Ty::Defined {
        definition: resin::ir::TypeId::from_index(0),
    };
    assert_eq!(
        module.globals[0].ty,
        Ty::Function {
            params: vec![Ty::Int32],
            result: Box::new(meters.clone()),
        }
    );
    assert_eq!(
        module.globals[1].ty,
        Ty::Function {
            params: vec![meters],
            result: Box::new(Ty::Int32),
        }
    );
}

#[test]
fn field_access_autoderefs_a_named_pointer() {
    let module = compile(
        r#"
P = Ptr ({ x: int });
f = (p: P) => p.x;
"#,
    );
    verify(&module).unwrap();
    assert_eq!(module.functions[1].result, Ty::Int32);
    assert!(module.functions[1].blocks.iter().any(|block| {
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
Handler = (int) -> int;
f = (h: Handler, n: int) => h(n);
"#,
    );
    verify(&module).unwrap();
    assert_eq!(module.functions[1].result, Ty::Int32);
}

#[test]
fn nested_nominal_ascription_does_not_skip_a_layer() {
    assert!(matches!(
        compile_err(
            r#"
Meters = int;
Distance = Meters;
x = Distance (1);
"#
        ),
        GenerateErrorKind::Type(TypeErrorKind::TypeMismatch { .. })
    ));
}

#[test]
fn nested_nominal_ascription_wraps_the_defining_body() {
    let module = compile(
        r#"
Meters = int;
Distance = Meters;
x = Distance (Meters (1));
y = int (Meters (x));
"#,
    );
    verify(&module).unwrap();
    assert_eq!(
        module.globals[0].ty,
        Ty::Defined {
            definition: resin::ir::TypeId::from_index(1),
        }
    );
    assert_eq!(module.globals[1].ty, Ty::Int32);
}

#[test]
fn nominal_record_ascription_wraps_the_representation() {
    let module = compile(
        r#"
List = { value: int, next: Ptr (List) };
nil = (p: Ptr (List)) => List { value = 0, next = p };
"#,
    );
    verify(&module).unwrap();
    let list = Ty::Defined {
        definition: resin::ir::TypeId::from_index(0),
    };
    assert_eq!(
        module.globals[0].ty,
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
        compile_err("x = [];"),
        GenerateErrorKind::Type(TypeErrorKind::EmptyArrayNeedsElementType)
    ));
}

#[test]
fn unbound_value_is_reported() {
    assert!(matches!(
        compile_err("x = y;"),
        GenerateErrorKind::UnboundValue { .. }
    ));
}

#[test]
fn span_and_literal_globals_are_typed() {
    let module = compile(
        r#"
Buf = Span (int);
x = 1;
"#,
    );
    verify(&module).unwrap();
    assert!(matches!(
        module.types[0].body,
        Ty::Span { ref element } if **element == Ty::Int32
    ));
    assert_eq!(module.globals[0].ty, Ty::Int32);
}

#[test]
fn short_circuit_and_compiles() {
    let module = compile("f = (a: int) => { (a == 0) && (a == 1) };");
    verify(&module).unwrap();
    assert_eq!(module.functions[1].result, Ty::Bool);
}
