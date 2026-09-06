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
        "export { main }; def f () -> int = { 1 }; def add (a: int, b: int) -> int = { a + b }; def main() -> () = { var pair = (1, 2); var x = f(); var y = add(pair); var z = add(3, 4); };",
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
    let instructions = &module.functions[module.entries["main"].index()].blocks[0].instrs;
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
    compile("type P = Ptr<()>; type S = Span<(int, int)>; def f (p: P) -> P = { p };");
    compile(
        "export { main }; type F = () -> int; def one () -> int = { 1 }; def main() -> () = { var f = F (one); var x = f(); };",
    );
    compile(
        "export { main }; type Add = (int, int) -> int; def add (a: int, b: int) -> int = { a + b }; def main() -> () = { var f = Add (add); var p = (1, 2); var x = f(p); };",
    );
    compile(
        "export { main }; def apply (f: (int, int) -> int, p: (int, int)) -> int = { f(p) }; def add (a: int, b: int) -> int = { a + b }; def main() -> () = { var x = apply(add, (1, 2)); };",
    );
    compile(
        "export { main }; def identity (p: (int, int)) -> (int, int) = { p }; def main() -> () = { var x = identity(1, 2); };",
    );
    compile(
        "export { main }; type Unit = (); def f (u: Unit) -> () = { () }; def main() -> () = { var x = Unit (()); var y = f(x); };",
    );
}

#[test]
fn type_formers_take_types_between_angle_brackets() {
    let m = compile(
        "export { main }; type Pointer = Ptr<Ptr<int>>; type View = Span<{ value: int, next: Pointer }>; type Callback = Ptr<(int) -> int>; type UnitPointer = Ptr<()>; def main() -> () = { var p = Ptr<int>(ulong(0)); };",
    );
    assert_eq!(
        m.types[0].body(),
        Some(&Ty::Pointer {
            pointee: Box::new(Ty::Pointer {
                pointee: Box::new(Ty::Int32)
            })
        })
    );
    compile(
        "export { main }; def f(x: int) -> int = { x }; type Record = { value: int }; def main() -> () = { var a = f(1); var b = f (2); var r = Record { value = 3 }; };",
    );
    for source in [
        "type P = Ptr(int);",
        "type P = Ptr int;",
        "type P = Ptr<1>;",
        "type P = Ptr<>;",
        "type P = Ptr<int, int>;",
    ] {
        assert!(parse(source).is_err(), "{source}");
    }
}

#[test]
fn unary_typechecking_rejects_wrong_argument_shapes() {
    for src in [
        "export { main }; def f () -> int = { 1 }; def main() -> () = { var x = f(1); };",
        "export { main }; def f (a: int, b: int) -> int = { a }; def main() -> () = { var x = f(1); };",
        "export { main }; def f (a: int) -> int = { a }; def main() -> () = { var x = f(1, 2); };",
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
        "export { main }; type Meters = int; def f (m: Meters) -> int = { int (m) }; def main() -> () = { var x = 1; var y = f(x); };",
        "export { main }; type Meters = int; def f (m: Meters) -> int = { int (m) }; def main() -> () = { var y = f(1); };",
        "export { main }; type Meters = int; def f (x: int) -> int = { x }; def main() -> () = { var m = Meters (1); var y = f(m); };",
        "export { main }; type Meters = int; def main() -> () = { var m = Meters (1); var n = 2; m := n; };",
        "export { main }; type Meters = int; def main() -> () = { var m = Meters (1); var n = 2; n := m; };",
        "export { main }; type Meters = int; type R = { value: Meters }; def main() -> () = { var x = R { value = 1 }; };",
        "export { main }; type Meters = int; def f (m: Meters, x: int) -> int = { x }; def main() -> () = { var y = f(1, 2); };",
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
        "export { main }; type Meters = int; def f (m: Meters) -> int = { int (m) }; def main() -> () = { var x = 1; var y = f(Meters (x)); var m = Meters (1); m := Meters (x); x := int (m); };",
    );
    compile(
        "export { main }; type Meters = int; type R = { value: Meters }; def main() -> () = { var x = R { value = Meters (1) }; };",
    );
    compile(
        "export { main }; type Meters = int; type Distance = Meters; def main() -> () = { var x = Distance (Meters (1)); var y = Meters (x); };",
    );
}

#[test]
fn record_layout_does_not_reorder_side_effects() {
    let module = compile(
        "type R = { a: int, b: int }; def f (x: int) -> int = { var r = R { b = (x := 1), a = (x := 2) }; x };",
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
fn recursion_uses_immutable_function_references() {
    let module = compile(
        "def f (n: int) -> int = { if (n == 0) { 7 } else { next(n - 1) } }; def next (n: int) -> int = { f(n) };",
    );
    for function in &module.functions {
        assert!(
            function
                .blocks
                .iter()
                .flat_map(|b| &b.instrs)
                .any(|i| matches!(i, Instr::Function { .. }))
        );
    }
    let error = ir::generate(
        &parse("export { main }; def f () -> int = { 1 }; def main() -> () = { f := f; };")
            .unwrap(),
    )
    .unwrap_err();
    assert_eq!(error.kind, GenerateErrorKind::NotAPlace);
}

#[test]
fn lambdas_and_nested_definitions_are_parse_errors() {
    for source in [
        "def main() -> () = { var f = (n: int) => n; };",
        "def outer () -> int = { def inner () -> int = { 1 }; inner() };",
        "def missing (n: int) = { n };",
    ] {
        assert!(parse(source).is_err(), "{source}");
    }
}

#[test]
fn record_type_members_are_parse_errors_instead_of_panics() {
    assert!(parse("def main() -> () = { var x = { T = int }; };").is_err());
    assert!(parse("def main() -> () = { var x = { a = 1, T = int }; };").is_err());
    compile("export { main }; def main() -> () = { var x = { type T = int; T (1) }; };");
}

#[test]
fn signed_literals_respect_context_and_the_minimum_integer() {
    let module = compile(
        "export { main }; def main() -> () = { var x = -2147483648; var y = long (-1); var z = -0x80000000; var w = long (1 + 2); var hex = -0xdead; };",
    );
    assert_eq!(
        module.functions[0]
            .locals
            .iter()
            .skip(1)
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
    compile(
        "def id (p: Ptr<{ x: int }>) -> Ptr<{ x: int }> = { p }; def f (p: Ptr<{ x: int }>) -> int = { id(p).x := 1 };",
    );
    compile(
        "type R = { inner: { x: int } }; def id (p: Ptr<R>) -> Ptr<R> = { p }; def f (p: Ptr<R>) -> int = { id(p).inner.x := 1 };",
    );
    let src = "def id (r: { x: int }) -> { x: int } = { r }; def f (r: { x: int }) -> int = { id(r).x := 1 };";
    assert!(matches!(
        ir::generate(&parse(src).unwrap()).unwrap_err().kind,
        GenerateErrorKind::NotAPlace
    ));
}

#[test]
fn nested_field_access_evaluates_its_base_once() {
    for src in [
        "def id (r: { inner: { x: int } }) -> { inner: { x: int } } = { r }; def f (r: { inner: { x: int } }) -> int = { id(r).inner.x };",
        "def id (r: { p: Ptr<{ x: int }> }) -> { p: Ptr<{ x: int }> } = { r }; def f (r: { p: Ptr<{ x: int }> }) -> int = { id(r).p.x := 1 };",
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
fn uninitialized_reads_are_rejected_on_all_paths() {
    for src in [
        "def f () -> int = { var x: int; x };",
        "def f (c: int) -> int = { var x: int; if (c == 0) { x := 1 } else { 0 }; x };",
        "def f (c: int) -> int = { var x: int; (c == 0) && ((x := 1) == 1); x };",
        "export { main }; def main() -> () = { var x: int; var y = x; };",
        "def f () -> int = { var r: { x: int }; r.x };",
        "def f () -> int = { var r: { x: int }; r.x := 1; r.x };",
    ] {
        assert!(
            matches!(
                ir::generate(&parse(src).unwrap()).unwrap_err().kind,
                GenerateErrorKind::UninitializedValue { .. }
            ),
            "{src}"
        );
    }
    compile("def f () -> int = { var x: int; x := 1; x };");
    compile("def f (c: int) -> int = { var x: int; if (c == 0) { x := 1 } else { x := 2 }; x };");
    compile(
        "def even (n: int) -> int = { if (n == 0) { 1 } else { odd(n - 1) } }; def odd (n: int) -> int = { if (n == 0) { 0 } else { even(n - 1) } };",
    );
}
