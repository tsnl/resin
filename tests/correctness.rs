use resin::{
    ast::{AstGen, SourceFile},
    ir::{self, GenerateErrorKind, Instr, Ty, TypeErrorKind, Value},
};
use tree_sitter::Parser;

fn parse(src: &str) -> Result<SourceFile, resin::ast::AstError> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_resin::LANGUAGE.into())
        .unwrap();
    let tree = parser.parse(src, None).unwrap();
    AstGen::new(src).gen_source_file(tree.root_node())
}

fn compile(src: &str) -> ir::Module {
    ir::generate(&parse(src).unwrap()).unwrap_or_else(|err| panic!("{src}\n{err}"))
}

#[test]
fn unit_and_tuple_calls_have_one_argument_and_one_parameter() {
    let module = compile(
        "f = () => { 1 }; add = (a: int, b: int) => { a + b }; pair = (1, 2); x = f(); y = add(pair); z = add(3, 4);",
    );
    let f = module
        .functions
        .iter()
        .find(|f| f.name.as_deref() == Some("f"))
        .unwrap();
    assert_eq!(f.locals[f.param.index()].ty, Ty::Unit);
    let add = module
        .functions
        .iter()
        .find(|f| f.name.as_deref() == Some("add"))
        .unwrap();
    assert_eq!(
        add.locals[add.param.index()].ty,
        Ty::parameter(&[Ty::Int32, Ty::Int32])
    );
    let instructions = &module.functions[0].blocks[0].instrs;
    assert_eq!(
        instructions
            .iter()
            .filter(|i| matches!(i, Instr::Call))
            .count(),
        3
    );
    assert!(
        instructions
            .iter()
            .any(|i| matches!(i, Instr::Push { value: Value::Unit }))
    );
}

#[test]
fn function_types_accept_unit_tuples_and_higher_order_calls() {
    compile("P = Ptr (()); S = Span ((int, int)); f = (p: P) => p;");
    compile("F = () -> int; f = F (() => { 1 }); x = f();");
    compile(
        "Add = (int, int) -> int; f = Add ((a: int, b: int) => { a + b }); p = (1, 2); x = f(p);",
    );
    compile(
        "apply = (f: (int, int) -> int, p: (int, int)) => f(p); add = (a: int, b: int) => { a + b }; x = apply(add, (1, 2));",
    );
    compile("identity = (p: (int, int)) => p; x = identity(1, 2);");
    compile("Unit = (); x = Unit (()); f = (u: Unit) => (); y = f(x);");
}

#[test]
fn unary_typechecking_rejects_wrong_argument_shapes() {
    for src in [
        "f = () => { 1 }; x = f(1);",
        "f = (a: int, b: int) => a; x = f(1);",
        "f = (a: int) => a; x = f(1, 2);",
    ] {
        assert!(
            matches!(
                ir::generate(&parse(src).unwrap()).unwrap_err().kind,
                GenerateErrorKind::Type(TypeErrorKind::TypeMismatch { .. })
            ),
            "{src}"
        );
    }
}

#[test]
fn nominal_conversion_requires_an_explicit_ascription() {
    for src in [
        "Meters = int; f = (m: Meters) => int (m); x = 1; y = f(x);",
        "Meters = int; f = (m: Meters) => int (m); y = f(1);",
        "Meters = int; f = (x: int) => x; m = Meters (1); y = f(m);",
        "Meters = int; m = Meters (1); n = 2; m := n;",
        "Meters = int; m = Meters (1); n = 2; n := m;",
        "Meters = int; R = { value: Meters }; x = R { value = 1 };",
        "Meters = int; f = (m: Meters, x: int) => x; y = f(1, 2);",
    ] {
        assert!(
            matches!(
                ir::generate(&parse(src).unwrap()).unwrap_err().kind,
                GenerateErrorKind::Type(TypeErrorKind::TypeMismatch { .. })
            ),
            "{src}"
        );
    }
    compile(
        "Meters = int; f = (m: Meters) => int (m); x = 1; y = f(Meters (x)); m = Meters (1); m := Meters (x); x := int (m);",
    );
    compile("Meters = int; R = { value: Meters }; x = R { value = Meters (1) };");
    compile("Meters = int; Distance = Meters; x = Distance (Meters (1)); y = Meters (x);");
}

#[test]
fn record_layout_does_not_reorder_side_effects() {
    let module = compile(
        "R = { a: int, b: int }; f = (x: int) => { r = R { b = (x := 1), a = (x := 2) }; x };",
    );
    let function = module
        .functions
        .iter()
        .find(|f| f.name.as_deref() == Some("f"))
        .unwrap();
    let instrs = &function.blocks[0].instrs;
    let literals: Vec<_> = instrs
        .iter()
        .filter_map(|i| match i {
            Instr::Push {
                value: Value::Int32 { value },
            } => Some(*value),
            _ => None,
        })
        .collect();
    assert_eq!(literals, [1, 2]);
    let fields = instrs
        .iter()
        .find_map(|i| match i {
            Instr::MakeRecord { fields } => {
                Some(fields.iter().map(|f| f.as_ref()).collect::<Vec<_>>())
            }
            _ => None,
        })
        .unwrap();
    assert_eq!(fields, ["a", "b"]);
}

#[test]
fn local_recursion_uses_the_current_closure_without_capturing_its_destination() {
    let module = compile(
        "outer = (x: int) => { f = (n: int) => int { if (n == 0) { x } else { f(n - 1) } }; f(2) };",
    );
    let f = module
        .functions
        .iter()
        .find(|f| f.name.as_deref() == Some("f"))
        .unwrap();
    assert_eq!(
        f.nonlocals
            .iter()
            .map(|n| n.name.as_deref())
            .collect::<Vec<_>>(),
        [Some("x")]
    );
    assert!(
        f.blocks
            .iter()
            .flat_map(|b| &b.instrs)
            .any(|i| matches!(i, Instr::CurrentClosure))
    );
    let outer = module
        .functions
        .iter()
        .find(|f| f.name.as_deref() == Some("outer"))
        .unwrap();
    let instrs = &outer.blocks[0].instrs;
    let construction = instrs
        .iter()
        .position(|i| matches!(i, Instr::MakeClosure { .. }))
        .unwrap();
    let f_local = outer
        .locals
        .iter()
        .position(|l| l.name.as_deref() == Some("f"))
        .unwrap();
    assert!(!instrs[..construction].windows(2).any(|pair| matches!(pair,
        [Instr::LocalAddress { local }, Instr::Load] if local.index() == f_local)));
}

#[test]
fn nested_closures_can_capture_a_recursive_self_value() {
    let module = compile(
        "outer = (x: int) => { f = (n: int) => int { next = (k: int) => f(k); if (n == 0) { x } else { next(n - 1) } }; f(2) };",
    );
    let f = module
        .functions
        .iter()
        .find(|f| f.name.as_deref() == Some("f"))
        .unwrap();
    assert!(
        f.blocks
            .iter()
            .flat_map(|b| &b.instrs)
            .any(|i| matches!(i, Instr::CurrentClosure))
    );
    let next = module
        .functions
        .iter()
        .find(|f| f.name.as_deref() == Some("next"))
        .unwrap();
    assert!(
        next.nonlocals
            .iter()
            .any(|n| n.name.as_deref() == Some("f"))
    );
    compile("outer = () => { f = (f: int) => f; f(1) };");
}

#[test]
fn record_type_members_are_parse_errors_instead_of_panics() {
    assert!(parse("x = { T = int };").is_err());
    assert!(parse("x = { a = 1, T = int };").is_err());
    compile("x = { T = int; T (1) };");
}

#[test]
fn signed_literals_respect_context_and_the_minimum_integer() {
    let module = compile(
        "x = -2147483648; y = long (-1); z = -0x80000000; w = long (1 + 2); hex = -0xdead;",
    );
    assert_eq!(
        module
            .globals
            .iter()
            .map(|g| g.ty.clone())
            .collect::<Vec<_>>(),
        [Ty::Int32, Ty::Int64, Ty::Int32, Ty::Int64, Ty::Int32]
    );
    assert!(
        module.functions[0].blocks[0]
            .instrs
            .iter()
            .any(|i| matches!(
                i,
                Instr::Push {
                    value: Value::Int32 { value: i32::MIN }
                }
            ))
    );
}

#[test]
fn returned_pointers_support_field_assignment() {
    compile("id = (p: Ptr ({ x: int })) => p; f = (p: Ptr ({ x: int })) => { id(p).x := 1 };");
    compile(
        "R = { inner: { x: int } }; id = (p: Ptr (R)) => p; f = (p: Ptr (R)) => { id(p).inner.x := 1 };",
    );
    let src = "id = (r: { x: int }) => r; f = (r: { x: int }) => { id(r).x := 1 };";
    assert!(matches!(
        ir::generate(&parse(src).unwrap()).unwrap_err().kind,
        GenerateErrorKind::NotAPlace
    ));
}

#[test]
fn nested_field_access_evaluates_its_base_once() {
    for src in [
        "id = (r: { inner: { x: int } }) => r; f = (r: { inner: { x: int } }) => id(r).inner.x;",
        "id = (r: { p: Ptr ({ x: int }) }) => r; f = (r: { p: Ptr ({ x: int }) }) => { id(r).p.x := 1 };",
    ] {
        let module = compile(src);
        let f = module
            .functions
            .iter()
            .find(|f| f.name.as_deref() == Some("f"))
            .unwrap();
        assert_eq!(
            f.blocks
                .iter()
                .flat_map(|b| &b.instrs)
                .filter(|i| matches!(i, Instr::Call))
                .count(),
            1,
            "{src}"
        );
    }
}

#[test]
fn uninitialized_reads_and_captures_are_rejected_on_all_paths() {
    for src in [
        "f = () => { x: int; x };",
        "f = () => { x: int; g = () => x; x := 1; g() };",
        "f = () => { x: int; g = () => { x := 1 }; g() };",
        "f = () => { g: () -> int; h = () => g(); g := () => { 1 }; h() };",
        "f = (c: int) => { x: int; if (c == 0) { x := 1 } else { 0 }; x };",
        "f = (c: int) => { x: int; (c == 0) && ((x := 1) == 1); x };",
        "x: int; f = () => { x := 1 }; y = x;",
        "f = () => { r: { x: int }; r.x };",
        "f = () => { r: { x: int }; r.x := 1; r.x };",
    ] {
        assert!(
            matches!(
                ir::generate(&parse(src).unwrap()).unwrap_err().kind,
                GenerateErrorKind::UninitializedValue { .. }
            ),
            "{src}"
        );
    }
    compile("f = () => { x: int; x := 1; g = () => x; g() };");
    compile("f = (c: int) => { x: int; if (c == 0) { x := 1 } else { x := 2 }; x };");
    compile(
        "odd: (int) -> int; even = (n: int) => int { if (n == 0) { 1 } else { odd(n - 1) } }; odd := (n: int) => int { if (n == 0) { 0 } else { even(n - 1) } };",
    );
}
