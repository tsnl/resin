use resin_hir::{TermKind, Type};

fn compile(source: &str) -> Result<resin_hir::Module, resin_hir::GenerateError> {
    let document = resin_cst::Document::reparse(source.into(), None);
    resin_hir::generate(&resin_ast::generate(&document).unwrap())
}

#[test]
fn method_schemes_keep_owner_binders_before_additional_method_binders() {
    let module = compile("struct Cell<T> { value: T, def choose<U>(self: Cell<T>, value: U) -> U = { value }; }; def main(cell: Cell<int>) -> ulong = { cell.choose::<ulong>(42) };").unwrap();
    let owner = module
        .types
        .iter()
        .find(|definition| definition.name.as_ref() == "Cell")
        .unwrap();
    let method_id = owner.methods["choose"];
    let method = &module.functions[method_id.index()];
    assert_eq!(method.signature.type_params.len(), 2);
    assert_eq!(method.signature.type_params[0].id, owner.type_params[0].id);
    assert_eq!(method.signature.type_params[0].name.val.as_ref(), "T");
    assert_eq!(method.signature.type_params[1].name.val.as_ref(), "U");
    assert_eq!(
        method.signature.result.ty,
        Type::Parameter {
            parameter: method.signature.type_params[1].id
        }
    );
    let main = module
        .functions
        .iter()
        .find(|function| function.name.as_ref() == "main")
        .unwrap();
    let TermKind::Block { tail, .. } = &main.body.as_ref().unwrap().kind else {
        panic!("block")
    };
    let TermKind::Call { func, .. } = &tail.kind else {
        panic!("method call")
    };
    let TermKind::Function {
        function,
        type_args,
    } = &func.kind
    else {
        panic!("resolved method")
    };
    assert_eq!(*function, method_id);
    assert_eq!(type_args, &[Type::Int32, Type::UInt64]);
}

#[test]
fn different_owner_applications_retain_one_polymorphic_method_body() {
    let module = compile("struct Cell<T> { value: T, def read(self: Cell<T>) -> T = { self.value }; }; def small(value: Cell<int>) -> int = { value.read() }; def large(value: Cell<ulong>) -> ulong = { value.read() };").unwrap();
    let owner = module
        .types
        .iter()
        .find(|definition| definition.name.as_ref() == "Cell")
        .unwrap();
    let method = &module.functions[owner.methods["read"].index()];
    assert_eq!(method.signature.type_params.len(), 1);
    assert_eq!(
        method.signature.result.ty,
        Type::Parameter {
            parameter: owner.type_params[0].id
        }
    );
    assert_eq!(
        module
            .functions
            .iter()
            .filter(|function| function.name.as_ref() == "Cell.read")
            .count(),
        1
    );
}

#[test]
fn recursive_method_results_complete_against_their_own_rigid_binders() {
    let module = compile("struct Cell<T> { value: T, def choose<U>(self: Cell<T>, value: U, stop: bool) -> _ = { if (stop) { value } else { self.choose(value, 1 == 1) } }; }; def main() -> int = { Cell<ulong> { value = 7 }.choose(42, 1 == 0) };").unwrap();
    let method = module
        .functions
        .iter()
        .find(|function| function.name.as_ref() == "Cell.choose")
        .unwrap();
    assert_eq!(method.signature.type_params.len(), 2);
    assert_eq!(
        method.signature.result.ty,
        Type::Parameter {
            parameter: method.signature.type_params[1].id
        }
    );
}

#[test]
fn drop_hooks_bind_only_their_owner_parameters() {
    let module =
        compile("struct Cell<T> { value: T, def drop(self: Ptr<Cell<T>>) = {}; };").unwrap();
    let owner = module
        .types
        .iter()
        .find(|definition| definition.name.as_ref() == "Cell")
        .unwrap();
    let drop = &module.functions[owner.drop.unwrap().index()];
    assert_eq!(owner.drop, Some(owner.methods["drop"]));
    assert_eq!(drop.signature.type_params.len(), 1);
    assert_eq!(drop.signature.type_params[0].id, owner.type_params[0].id);
    assert_eq!(drop.signature.result.ty, Type::Unit);
    let Type::Pointer { pointee } = &drop.signature.params[0].annotation.ty else {
        panic!("drop pointer")
    };
    let Type::Defined { arguments, .. } = pointee.as_ref() else {
        panic!("owner")
    };
    assert_eq!(
        arguments,
        &[Type::Parameter {
            parameter: owner.type_params[0].id
        }]
    );
}

#[test]
fn explicit_method_arguments_cannot_replace_or_repeat_owner_arguments() {
    for source in [
        "struct Cell<T> { value: T, def read(self: Cell<T>) -> T = { self.value }; }; def main() = { Cell<int> { value = 1 }.read::<int>(); };",
        "struct Cell<T> { value: T, def choose<U>(self: Cell<T>, value: U) -> U = { value }; }; def main() = { Cell<int> { value = 1 }.choose::<int, ulong>(2); };",
        "struct Cell<T> { value: T, def read(self: Cell<T>) -> T = { self.value }; }; def main() = { Cell<int>.read(Cell<uint> { value = 1 }); };",
        "struct Plain { def read(self: Plain) -> int = { 42 }; }; def main() = { Plain {}.read::<int>(); };",
    ] {
        assert!(compile(source).is_err(), "{source}");
    }
}

#[test]
fn invalid_method_binders_and_destructor_signatures_are_definition_errors() {
    for source in [
        "struct Cell<T> { value: T, def choose<U, U>(self: Cell<T>, value: U) -> U = { value }; };",
        "struct Cell<T> { value: T, def drop<U>(self: Ptr<Cell<T>>) = {}; };",
        "struct Cell<T> { value: T, def drop(self: Ptr<Cell<int>>) = {}; };",
        "struct Cell<T> { value: T, def drop(self: Ptr<Cell<T>>) -> int = { 0 }; };",
        "struct Cell<T> { value: T, def choose<U>(self: Cell<T>, value: U) -> U = { value }; }; def escaped(value: U) = {};",
    ] {
        assert!(compile(source).is_err(), "{source}");
    }
}

#[test]
fn undetermined_method_arguments_require_annotations() {
    let error = compile("struct Factory { def create<T>(self: Factory) -> T = { 41 }; }; def main() = { Factory {}.create(); };").unwrap_err();
    assert!(error.to_string().contains("annotat"), "{error}");
}

fn source_module(name: &str, text: &str) -> resin_ast::SourceModule {
    let source = resin_source::Source::new(name, text);
    let document = resin_cst::Document::reparse(text.into(), None);
    resin_ast::SourceModule {
        source,
        file: resin_ast::generate(&document).unwrap(),
        imports: vec![],
    }
}

fn rejects_local_and_imported_owner(declaration: &str, use_site: &str, expected: &str) {
    let error = compile(&format!("{declaration} {use_site}")).unwrap_err();
    assert!(error.to_string().contains(expected), "{error}");
    let library = source_module(
        "owner.resin",
        &format!("export {{ Managed }}; {declaration}"),
    );
    let mut entry = source_module("main.resin", use_site);
    entry
        .imports
        .push((resin_source::Span { start: 0, end: 0 }, 0));
    let analysis = resin_hir::analyze_program(&resin_ast::Program {
        modules: vec![library, entry],
    });
    assert!(analysis.module.is_none());
    assert!(
        analysis
            .diagnostics
            .iter()
            .any(|error| error.to_string().contains(expected)),
        "{:?}",
        analysis.diagnostics
    );
}

#[test]
fn source_drop_hooks_reject_gpu_elements_before_and_after_importing() {
    let allocator = r#"
        struct Failure {};
        struct Device {
            @gpu_allocator
            def malloc(self: Device, bytes: ulong, alignment: ulong, memory: int) -> Result<GpuPtr<ubyte>, Failure> = { err(Failure {}) };
        };
    "#;
    for (declaration, owner) in [
        (
            "struct Managed { value: int, def drop(self: Ptr<Managed>) = {}; };",
            "Managed",
        ),
        (
            "struct Managed<T> { value: T, def drop(self: Ptr<Managed<T>>) = {}; };",
            "Managed<int>",
        ),
    ] {
        rejects_local_and_imported_owner(
            declaration,
            &format!(
                "{allocator} def main() -> _ = {{ Device {{}}.new({owner} {{ value = 1 }}) }};"
            ),
            "drop",
        );
    }
}

#[test]
fn source_drop_hooks_reject_structural_unwrapping_before_and_after_importing() {
    for (declaration, owner) in [
        (
            "struct Managed { value: int, def drop(self: Ptr<Managed>) = {}; };",
            "Managed",
        ),
        (
            "struct Managed<T> { value: T, def drop(self: Ptr<Managed<T>>) = {}; };",
            "Managed<int>",
        ),
    ] {
        rejects_local_and_imported_owner(
            declaration,
            &format!(
                "type Raw = {{ value: int }}; def unwrap(value: {owner}) -> Raw = {{ Raw(value) }};"
            ),
            "UnwrapManaged",
        );
    }
}
