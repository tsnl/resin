mod support;

fn result(source: &str) -> i32 {
    let output = support::project::Project::new(&support::module(source), Some("main"))
        .unwrap()
        .run();
    assert!(
        output.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
        .status
        .code()
        .expect("reference program terminated normally")
}

#[test]
fn reference_calls_preserve_aliases_while_value_bindings_copy() {
    assert_eq!(
        result(
            r#"export { main };
        fn identity<T>(value: Ref<T>) -> Ref<T>  { value }
        fn observe(left: Ref<int>, right: Ref<int>) -> int  { left = 20; right }
        fn main() -> int  {
            let mut x: int = 1;
            let mut reference: Ref<_> = identity(x);
            let mut copied = reference;
            identity(x) = 7;
            observe(reference, x) + copied + x
        }
    "#
        ),
        41
    );
}

#[test]
fn dependent_reference_results_cannot_be_turned_into_pointers() {
    let error = support::pipeline::source_module(
        r#"export { main };
        struct Cell { value: int }
        fn get(cell: Ref<Cell>) -> Ref<int> { cell.value }
        fn address<T>(cell: Ref<T>) -> Ptr<int> { &cell:get() }
        fn main() { let cell = Cell { value = 1 }; address(cell); }
        "#,
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("cannot take the address of a Ref"),
        "{error}"
    );
}

#[test]
fn reference_returning_function_values_and_branches_select_storage() {
    assert_eq!(
        result(
            r#"export { main };
        fn choose(a: Ref<int>, b: Ref<int>, flag: bool) -> Ref<int>  {
            if (flag) { a } else { b }
        }
        fn main() -> int  {
            let mut a = 1_i; let mut b = 2_i;
            let mut select = choose;
            select(a, b, 1 == 0) = 41;
            a + b
        }
    "#
        ),
        42
    );
}

#[test]
fn reference_methods_and_dependent_calls_mutate_the_original() {
    assert_eq!(
        result(
            r#"export { main };
        struct Cell<T> { value: T,
            
            
        }
fn get<T>(self: Ref<Cell<T>>) -> Ref<T>  { self.value }

fn set<T>(self: Ref<Cell<T>>, other: Ref<T>)  { self.value = other; }

        fn read_cell<C>(cell: Ref<C>) -> _  { cell:get() }
        fn set_cell<C, T>(cell: Ref<C>, value: Ref<T>)  { cell:set(value); }
        fn main() -> int  {
            let mut cell = Cell<int> { value = 1 };
            cell:get() = 7;
            let mut replacement = 35_i;
            let mut old = read_cell(cell);
            set_cell(cell, replacement);
            old + cell.value
        }
    "#
        ),
        42
    );
}

#[test]
fn assignment_evaluates_a_reference_accessor_once() {
    assert_eq!(
        result(
            r#"export { main };
        fn get(count: Ref<int>, value: Ref<int>) -> Ref<int>  {
            count = count + 1;
            value
        }
        fn main() -> int  {
            let mut count = 0_i; let mut value = 0_i;
            get(count, value) = 41;
            count + value
        }
    "#
        ),
        42
    );
}

#[test]
fn references_to_pointer_slots_are_fixed_aliases() {
    assert_eq!(
        result(
            r#"export { main };
import { "$/shared.resin" };

        fn main() -> int | Err<_> {
            let a_owner = arc_ptr_alloc(1_i)?; let a: Ref<_> = a_owner:get().*; let b_owner = arc_ptr_alloc(2_i)?; let b: Ref<_> = b_owner:get().*;
            let mut pointer = a_owner:get();
            let mut slot: Ref<Ptr<int>> = pointer;
            slot = b_owner:get();
            slot.* = 41;
            a + pointer.*
        }
    "#
        ),
        42
    );
}

#[test]
fn references_borrow_managed_storage_and_replacements_destroy_the_old_value() {
    assert_eq!(
        result(
            r#"export { main };
import { "$/shared.resin" };

        struct Resource { drops: Ptr<int>,
            
        }
fn drop(self: Ref<Resource>)  { self.drops.* = self.drops.* + 1; }

        fn identity(value: Ref<Resource>) -> Ref<Resource>  { value }
        fn main() -> int | Err<_> {
            let drops_owner = arc_ptr_alloc(0_i)?; let drops: Ref<_> = drops_owner:get().*;
            {
                let mut resource = Resource { drops = drops_owner:get() };
                {
                    let mut reference: Ref<Resource> = identity(resource);
                    { let borrowed: Ref<Resource> = reference; };
                    if (drops != 0_i) { drops = 90; };
                    reference = Resource { drops = drops_owner:get() };
                };
                if (drops != 1_i) { drops = 90; };
            };
            drops
        }
    "#
        ),
        2
    );
}

#[test]
fn dependent_reference_results_and_temporary_arguments_preserve_reference_semantics() {
    for body in ["read_cell(cell)", "cell:get()"] {
        let source = format!(
            r#"export {{ main }};
            struct Cell {{ value: int,
                
            }}
fn get(self: Ref<Cell>) -> Ref<int>  {{ self.value }}

            fn read_cell<C>(cell: Ref<C>) -> Ref<int>  {{ cell:get() }}
            fn main() -> int  {{ let mut cell = Cell {{ value = 1 }}; {body} = 42; cell.value }}
        "#
        );
        assert_eq!(result(&source), 42);
    }
    let source = r#"struct Cell { value: int,
            
        }
fn take(self: Ref<Cell>, value: Ref<int>)  {}

        fn value() -> int  { 1 }
        fn call<C>(cell: Ref<C>)  { cell:take(value()); }
        fn main()  { let mut cell = Cell { value = 1 }; call(cell); }
    "#;
    support::pipeline::source_module(source).unwrap();
}

#[test]
fn shaders_cannot_return_references_to_their_own_locals() {
    let source = "export { kernel }; fn local() -> Ref<uint> { let mut value = 1_ui; value } @compute_shader fn kernel(i: ulong, output: Ptr<uint>) { output.* = local(); }";
    let error = support::pipeline::shader_error(source);
    assert!(
        error.to_string().contains("cannot return a local address"),
        "{error}"
    );
}

#[test]
fn stored_function_signatures_preserve_reference_parameters_and_results() {
    assert_eq!(
        result(
            r#"export { main };
        type Access = (Ref<int>) -> Ref<int>;
        struct Accessor { call: Access, }
        fn identity(value: Ref<int>) -> Ref<int>  { value }
        fn invoke<F>(function: F, value: Ref<int>) -> Ref<int>  { function(value) }
        fn main() -> int  {
            let mut accessor = Accessor { call = identity };
            let mut value = 1_i;
            invoke(accessor.call, value) = 41;
            (accessor.call)(value) = value + 1;
            value
        }
    "#
        ),
        42
    );
}

#[test]
fn specialization_keeps_pointer_field_access_and_rejects_local_field_addresses() {
    assert_eq!(
        result(
            r#"export { main }; import { "$/shared.resin" };
        struct Cell { value: int }
        fn address<T>(value: T) -> Ptr<int> { &value.value }
        fn borrowed<T>(value: Ref<T>) -> Ptr<int> { &value.value }
        fn main() -> int | Err<_> {
            let owner = arc_ptr_alloc(Cell { value = 1 })?;
            address(owner:get()).* = 20;
            borrowed(owner:get()).* = 42;
            owner:get().value
        }
    "#
        ),
        42
    );
    for parameter in ["T", "Ref<T>"] {
        let error = support::pipeline::source_module(&format!(
            r#"export {{ main }};
            struct Cell {{ value: int }}
            fn address<T>(value: {parameter}) -> Ptr<int> {{ &value.value }}
            fn main() {{ let cell = Cell {{ value = 1 }}; address(cell); }}
        "#
        ))
        .unwrap_err();
        assert!(
            error.to_string().contains("cannot take the address"),
            "{error}"
        );
    }
}
