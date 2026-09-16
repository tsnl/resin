use resin_hir::{TermKind, Type};

mod common;
use common::hir_module;

fn compile(source: &str) -> Result<resin_hir::Module, resin_source::SourceError> {
    hir_module(source)
}

#[test]
fn operation_binders_are_independent_of_struct_binders() {
    let module = compile("struct Cell<T> { value: T,  }\nfn choose<T, U>(self: Cell<T>, value: U) -> U  { value }\n fn main(cell: Cell<int>) -> ulong  { cell:choose::<_, ulong>(42) }").unwrap();
    let owner = module
        .types
        .iter()
        .find(|definition| definition.name.as_ref() == "Cell")
        .unwrap();
    let method_id = resin_types::FunctionId::from_index(
        module
            .functions
            .iter()
            .position(|function| function.name.as_ref() == "choose")
            .unwrap(),
    );
    let method = &module.functions[method_id.index()];
    assert_eq!(method.signature.type_params.len(), 2);
    assert_ne!(method.signature.type_params[0].id, owner.type_params[0].id);
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
    let module = compile("struct Cell<T> { value: T,  }\nfn read<T>(self: Cell<T>) -> T  { self.value }\n fn small(value: Cell<int>) -> int  { value:read() } fn large(value: Cell<ulong>) -> ulong  { value:read() }").unwrap();
    let owner = module
        .types
        .iter()
        .find(|definition| definition.name.as_ref() == "Cell")
        .unwrap();
    let method = module
        .functions
        .iter()
        .find(|function| function.name.as_ref() == "read")
        .unwrap();
    assert_ne!(owner.type_params[0].id, method.signature.type_params[0].id);
    assert_eq!(method.signature.type_params.len(), 1);
    assert_eq!(
        method.signature.result.ty,
        Type::Parameter {
            parameter: method.signature.type_params[0].id
        }
    );
    assert_eq!(
        module
            .functions
            .iter()
            .filter(|function| function.name.as_ref() == "read")
            .count(),
        1
    );
}

#[test]
fn recursive_method_results_complete_against_their_own_rigid_binders() {
    let module = compile("struct Cell<T> { value: T,  }\nfn choose<T, U>(self: Cell<T>, value: U, stop: bool) -> _  { if (stop) { value } else { self:choose(value, 1 == 1) } }\n fn main() -> int  { Cell<ulong> { value = 7 }:choose(42, 1 == 0) }").unwrap();
    let method = module
        .functions
        .iter()
        .find(|function| function.name.as_ref() == "choose")
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
        compile("struct Cell<T> { value: T,  }\nfn drop<T>(self: Ptr<Cell<T>>)  {}\n").unwrap();
    let owner = module
        .types
        .iter()
        .find(|definition| definition.name.as_ref() == "Cell")
        .unwrap();
    let drop = &module.functions[owner.drop.unwrap().index()];
    assert_eq!(drop.name.as_ref(), "drop");
    assert_eq!(drop.signature.type_params.len(), 1);
    assert_ne!(drop.signature.type_params[0].id, owner.type_params[0].id);
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
            parameter: drop.signature.type_params[0].id
        }]
    );
}

#[test]
fn explicit_operation_arguments_match_the_complete_signature() {
    for source in [
        "struct Cell<T> { value: T,  }\nfn read<T>(self: Cell<T>) -> T  { self.value }\n fn main()  { Cell<int> { value = 1 }:read::<int, int>(); }",
        "struct Cell<T> { value: T,  }\nfn choose<T, U>(self: Cell<T>, value: U) -> U  { value }\n fn main()  { Cell<int> { value = 1 }:choose::<ulong>(2); }",
        "struct Cell<T> { value: T,  }\nfn read<T>(self: Cell<T>) -> T  { self.value }\n fn main()  { read::<int>(Cell<uint> { value = 1 }); }",
        "struct Plain {  }\nfn read(self: Plain) -> int  { 42 }\n fn main()  { Plain {}:read::<int>(); }",
    ] {
        assert!(compile(source).is_err(), "{source}");
    }
}

#[test]
fn invalid_method_binders_and_destructor_signatures_are_definition_errors() {
    for source in [
        "struct Cell<T> { value: T,  }\nfn choose<T, U, U>(self: Cell<T>, value: U) -> U  { value }\n",
        "struct Cell<T> { value: T,  }\nfn drop<T, U>(self: Ptr<Cell<T>>)  {}\n",
        "struct Cell<T> { value: T,  }\nfn drop<T>(self: Ptr<Cell<int>>)  {}\n",
        "struct Cell<T> { value: T,  }\nfn drop<T>(self: Ptr<Cell<T>>) -> int  { 0 }\n",
        "struct Cell<T> { value: T,  }\nfn choose<T, U>(self: Cell<T>, value: U) -> U  { value }\n fn escaped(value: U)  {}",
    ] {
        assert!(compile(source).is_err(), "{source}");
    }
}

#[test]
fn undetermined_method_arguments_require_annotations() {
    let error = compile("struct Factory {  }\nfn create<T>(self: Factory) -> T  { 41 }\n fn main()  { Factory {}:create(); }").unwrap_err();
    assert!(error.to_string().contains("annotat"), "{error}");
}

fn source_module(name: &str, text: &str) -> resin_ast::SourceModule {
    common::source_module(resin_source::Source::new(name, text))
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
    let analysis = common::check(resin_ast::Program {
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
fn source_drop_hooks_reject_structural_unwrapping_before_and_after_importing() {
    for (declaration, owner) in [
        (
            "struct Managed { _0: int,  }\nfn drop(self: Ptr<Managed>)  {}\n",
            "Managed",
        ),
        (
            "struct Managed<T> { _0: T,  }\nfn drop<T>(self: Ptr<Managed<T>>)  {}\n",
            "Managed<int>",
        ),
    ] {
        rejects_local_and_imported_owner(
            declaration,
            &format!("type Raw = (int,); fn unwrap(value: {owner}) -> Raw  {{ Raw(value) }}"),
            "UnwrapManaged",
        );
    }
}
