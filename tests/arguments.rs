use resin_lir::Instr;
use resin_types::prelude::*;

mod support;

fn hir(source: &str) -> Result<resin_hir::Module, resin_hir::GenerateError> {
    resin_hir::generate(&support::parse(source))
}

fn run(source: &str) -> i32 {
    let module = support::module(source);
    let output = support::project::Project::new(&module, Some("main"))
        .unwrap()
        .run();
    assert!(
        output.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.status.code().unwrap()
}

#[test]
fn only_parenthesized_lists_apply_functions() {
    for call in [
        "f [1, 2]",
        "f { value = 1 }",
        "f { 1 }",
        "f {}",
        "f 1",
        "x.method [1]",
    ] {
        let source = format!("def main() = {{ {call}; }};");
        let document = resin_cst::Document::reparse(source.clone(), None);
        assert!(resin_ast::generate(&document).is_err(), "{source}");
        assert!(resin_cst::format_source(&source).is_none(), "{source}");
    }
    for call in [
        "f()",
        "f(())",
        "f(1,)",
        "f((1,))",
        "f((1, 2))",
        "f([1, 2])",
        "f({ 1 })",
        "f(1)(2)",
    ] {
        support::parse(&format!("def main() = {{ {call}; }};"));
    }
}

#[test]
fn calls_distinguish_zero_unit_tuple_and_multiple_arguments() {
    let declarations = "
        def zero() = {};
        def unit(value: ()) = {};
        def tuple(value: (int, int)) = {};
        def pair(first: int, second: int) = {};
        def generic<T>(value: T) = {};
        struct Receiver { def pair(self: Receiver, first: int, second: int) = {}; };
    ";
    for call in [
        "zero(())",
        "unit()",
        "tuple(1, 2)",
        "pair((1, 2))",
        "pair(1)",
        "pair(1, 2, 3)",
        "generic()",
        "generic(1, 2)",
        "Receiver {}.pair((1, 2))",
    ] {
        let source = format!("{declarations} def main() = {{ {call}; }};");
        let error = hir(&source).unwrap_err();
        assert!(error.to_string().contains("argument"), "{source}: {error}");
    }
    hir(&format!("{declarations} def main() = {{ zero(); unit(()); tuple((1, 2)); pair(1, 2,); generic((1, 2)); Receiver {{}}.pair(1, 2); }};")).unwrap();
}

#[test]
fn function_values_preserve_parameter_lists_and_tuple_parameters() {
    assert_eq!(
        run(r#"
        export { main };
        def zero() -> int = { 1 };
        def unit(value: ()) -> int = { 2 };
        def tuple(value: (int, int)) -> int = { value.0 + value.1 };
        def add<T>(a: T, b: T) -> T = { a + b };
        def apply(f: (int, int) -> int, a: int, b: int) -> int = { f(a, b) };
        def main() -> int = {
            var a: () -> int; a := zero;
            var b: (()) -> int; b := unit;
            var c: ((int, int)) -> int; c := tuple;
            a() + b(()) + c((3, 4)) + apply(add::<int>, 5, 6)
        };
    "#),
        21
    );
    for assignment in [
        "var f: () -> int; f := unit;",
        "var f: (int, int) -> int; f := tuple;",
    ] {
        assert!(hir(&format!("def unit(value: ()) -> int = {{ 0 }}; def tuple(value: (int, int)) -> int = {{ 0 }}; def main() = {{ {assignment} }};")).is_err());
    }
}

#[test]
fn arguments_are_evaluated_left_to_right_after_the_callee() {
    assert_eq!(
        run(r#"
        export { main };
        def mark(trace: Ptr<int>, digit: int) -> int = { trace.* := trace.* * 10 + digit };
        def consume(a: int, b: int) = {};
        def callee(trace: Ptr<int>) -> (int, int) -> () = { mark(trace, 1); consume };
        def main() -> int = {
            var trace = 0;
            callee(&trace)(mark(&trace, 2), mark(&trace, 3));
            trace
        };
    "#),
        123
    );
    hir("def consume(a: int, b: int) = {}; def main() = { var value: int; consume(value := 1, value); };").unwrap();
    assert!(hir("def consume(a: int, b: int) = {}; def main() = { var value: int; consume(value, value := 1); };").is_err());
}

#[test]
fn managed_parameters_drop_in_reverse_order_and_failed_arguments_cleanup() {
    assert_eq!(
        run(r#"
        export { main };
        struct Resource { trace: Ptr<int>, digit: int,
            def drop(self: Ptr<Resource>) = { self.trace.* := self.trace.* * 10 + self.digit; };
        };
        struct Failed {};
        def make(trace: Ptr<int>, digit: int) -> Arc<Resource> = { Arc<Resource> { trace = trace, digit = digit } };
        def consume(a: Arc<Resource>, b: Arc<Resource>) = {};
        def fail() -> Result<Arc<Resource>, Failed> = { err(Failed {}) };
        def attempt(trace: Ptr<int>) -> Result<(), Failed> = {
            consume(make(trace, 3), fail()?);
            ok(())
        };
        def main() -> int = {
            var trace = 0;
            consume(make(&trace, 1), make(&trace, 2));
            attempt(&trace);
            trace
        };
    "#),
        213
    );
}

#[test]
fn lir_and_native_signatures_do_not_pack_arguments() {
    let module = support::module(
        "export { main }; def add(a: int, b: int) -> int = { a + b }; def main() -> int = { add(20, 22) };",
    );
    assert_eq!(module.functions[0].parameter_count, 2);
    assert_eq!(
        module.functions[0].locals[..2]
            .iter()
            .map(|local| &local.ty)
            .collect::<Vec<_>>(),
        [&Ty::Int32, &Ty::Int32]
    );
    let calls = module.functions[1]
        .blocks
        .iter()
        .flat_map(|block| &block.instrs)
        .collect::<Vec<_>>();
    assert!(calls.contains(&&Instr::Call { arguments: 2 }));
    assert!(
        !calls
            .iter()
            .any(|instr| matches!(instr, Instr::MakeRecord { .. } | Instr::AccessStatic { .. }))
    );
    let project = support::project::Project::new(&module, Some("main")).unwrap();
    let c = std::fs::read_to_string(project.generated.c_source().unwrap()).unwrap();
    let signature = c.lines().find(|line| line.contains(" r_fn0(")).unwrap();
    assert!(
        signature.contains(" r_arg0, ") && signature.contains(" r_arg1)"),
        "{signature}"
    );
    assert!(c.contains("r_fn1(void)"), "{c}");
    assert_eq!(project.run().status.code(), Some(42));
}

#[test]
fn shaders_emit_individual_parameters_and_zero_argument_calls() {
    let module = support::module(
        "export { kernel }; def seed() -> ulong = { 42 }; def add(a: ulong, b: ulong) -> ulong = { a + b }; @compute_shader def kernel(index: ulong, output: Ptr<ulong>) = { output.* := add(index, seed()); };",
    );
    let project = support::project::Project::new(&module, None).unwrap();
    let shader = project.generated.shaders()[0].unoptimized_spirv();
    support::shaders::validate(shader);
    let bytes = std::fs::read(shader).unwrap();
    // OpFunctionParameter: two kernel inputs and two add inputs; seed has none.
    assert_eq!(support::shaders::instructions(&bytes, 55).count(), 4);
}

#[test]
fn tuple_projection_preserves_places_and_nested_values() {
    assert_eq!(
        run(r#"
        export { main };
        def main() -> int = {
            var pair = ((1, 2), 3);
            pair.0.1 := 20;
            var pointer = &pair.1;
            pointer.* := 21;
            pair.0.0 + pair.0.1 + pair.1
        };
    "#),
        42
    );
}

#[test]
fn verification_rejects_mismatched_argument_counts_and_stack_underflow() {
    let module = support::module(
        "def add(a: int, b: int) -> int = { a + b }; def caller() -> int = { add(1, 2) };",
    );
    let mut wrong_signature = module.clone();
    wrong_signature.functions[0].parameter_count = 1;
    assert!(matches!(
        resin_lir::verify(&wrong_signature).unwrap_err().kind,
        resin_lir::VerifyErrorKind::ArgumentCount {
            expected: 1,
            found: 2
        }
    ));
    for count in [1, 3, usize::MAX] {
        let mut invalid = module.clone();
        for instruction in &mut invalid.functions[1].blocks[0].instrs {
            if let Instr::Call { arguments } = instruction {
                *arguments = count;
            }
        }
        assert!(resin_lir::verify(&invalid).is_err());
    }
}
