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
        fn identity<T>(value: RefMut<T>) -> RefMut<T>  { value }
        fn observe(left: RefMut<i32>, right: Ref<i32>) -> i32  { left = 20; right }
        fn main() -> i32  {
            let mut x: i32 = 1;
            let mut reference: RefMut<_> = identity(x);
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
        struct Cell { value: i32 }
        fn get(cell: Ref<Cell>) -> Ref<i32> { cell.value }
        fn address<T>(cell: Ref<T>) -> PtrMut<i32> { &cell:get() }
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
        fn choose(a: RefMut<i32>, b: RefMut<i32>, flag: bool) -> RefMut<i32>  {
            if (flag) { a } else { b }
        }
        fn main() -> i32  {
            let mut a: i32 = 1; let mut b: i32 = 2;
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
fn get<T>(self: RefMut<Cell<T>>) -> RefMut<T>  { self.value }

fn set<T>(self: RefMut<Cell<T>>, other: Ref<T>)  { self.value = other; }

        fn read_cell<C>(cell: RefMut<C>) -> _  { cell:get() }
        fn set_cell<C, T>(cell: RefMut<C>, value: Ref<T>)  { cell:set(value); }
        fn main() -> i32  {
            let mut cell = Cell<i32> { value = 1 };
            cell:get() = 7;
            let mut replacement: i32 = 35;
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
        fn get(count: RefMut<i32>, value: RefMut<i32>) -> RefMut<i32>  {
            count = count + 1;
            value
        }
        fn main() -> i32  {
            let mut count: i32 = 0; let mut value: i32 = 0;
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

        fn main() -> i32 | Err<_> {
            let a_owner = arc_ptr_alloc(i32(1))?; let a: Ref<_> = a_owner:get().*; let b_owner = arc_ptr_alloc(i32(2))?; let b: Ref<_> = b_owner:get().*;
            let mut pointer = a_owner:get();
            let mut slot: RefMut<PtrMut<i32>> = pointer;
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

        struct Resource { drops: PtrMut<i32>,
            
        }
fn drop(self: RefMut<Resource>)  { self.drops.* = self.drops.* + 1; }

        fn identity(value: RefMut<Resource>) -> RefMut<Resource>  { value }
        fn main() -> i32 | Err<_> {
            let drops_owner = arc_ptr_alloc(i32(0))?; let drops: RefMut<_> = drops_owner:get().*;
            {
                let mut resource = Resource { drops = drops_owner:get() };
                {
                    let mut reference: RefMut<Resource> = identity(resource);
                    { let borrowed: Ref<Resource> = reference; };
                    if (drops != i32(0)) { drops = 90; };
                    reference = Resource { drops = drops_owner:get() };
                };
                if (drops != i32(1)) { drops = 90; };
            };
            drops
        }
    "#
        ),
        2
    );
}

#[test]
fn dependent_reference_results_preserve_places_and_reject_temporary_arguments() {
    for body in ["read_cell(cell)", "cell:get()"] {
        let source = format!(
            r#"export {{ main }};
            struct Cell {{ value: i32,
                
            }}
fn get(self: RefMut<Cell>) -> RefMut<i32>  {{ self.value }}

            fn read_cell<C>(cell: RefMut<C>) -> RefMut<i32>  {{ cell:get() }}
            fn main() -> i32  {{ let mut cell = Cell {{ value = 1 }}; {body} = 42; cell.value }}
        "#
        );
        assert_eq!(result(&source), 42);
    }
    let source = r#"struct Cell { value: i32,
            
        }
fn take(self: Ref<Cell>, value: Ref<i32>)  {}

        fn value() -> i32  { 1 }
        fn call<C>(cell: Ref<C>)  { cell:take(value()); }
        fn main()  { let mut cell = Cell { value = 1 }; call(cell); }
    "#;
    let error = support::pipeline::source_module(source)
        .unwrap_err()
        .to_string();
    assert!(error.contains("bind the temporary to a local"), "{error}");
}

#[test]
fn shaders_cannot_return_references_to_their_own_locals() {
    let source = "export { kernel }; fn local() -> Ref<u32> { let mut value: u32 = 1; value } @compute_shader fn kernel(i: u64, output: PtrMut<u32>) { output.* = local(); }";
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
        type Access = (RefMut<i32>) -> RefMut<i32>;
        struct Accessor { call: Access, }
        fn identity(value: RefMut<i32>) -> RefMut<i32>  { value }
        fn invoke<F>(function: F, value: RefMut<i32>) -> RefMut<i32>  { function(value) }
        fn main() -> i32  {
            let mut accessor = Accessor { call = identity };
            let mut value: i32 = 1;
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
        struct Cell { value: i32 }
        fn address<T>(value: T) -> PtrMut<i32> { &value.value }
        fn borrowed<T>(value: Ref<T>) -> PtrMut<i32> { &value.value }
        fn main() -> i32 | Err<_> {
            let owner = arc_ptr_alloc(Cell { value = 1 })?;
            address(owner:get()).* = 20;
            let pointer = owner:get();
            borrowed(pointer).* = 42;
            owner:get().value
        }
    "#
        ),
        42
    );
    for parameter in ["T", "Ref<T>"] {
        let error = support::pipeline::source_module(&format!(
            r#"export {{ main }};
            struct Cell {{ value: i32 }}
            fn address<T>(value: {parameter}) -> PtrMut<i32> {{ &value.value }}
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

#[test]
fn mutable_references_weaken_without_losing_aliases() {
    assert_eq!(
        result(
            r#"export { main };
        struct Cell { value: i32 }
        fn read(value: Ref<i32>) -> i32 { value }
        fn view(value: RefMut<i32>) -> Ref<i32> { value }
        fn change(value: RefMut<i32>, observer: Ref<i32>) -> i32 {
            value = 20;
            read(observer)
        }
        fn main() -> i32 {
            let mut cell = Cell { value = 1 };
            let mutable: RefMut<i32> = cell.value;
            let readonly: Ref<i32> = mutable;
            change(mutable, readonly) + view(mutable) + 2
        }
    "#
        ),
        42
    );
}

#[test]
fn generic_mutators_reject_readonly_arguments_and_immutable_locals() {
    for source in [
        "struct Cell { value: i32 } fn change(cell: RefMut<Cell>) { cell.value = 42; } fn call<T>(cell: Ref<T>) { cell:change(); } fn main() { let mut cell = Cell { value = 1 }; call(cell); }",
        "struct Cell { value: i32 } fn change(cell: RefMut<Cell>) { cell.value = 42; } fn call<T>(cell: T) { cell:change(); } fn main() { call(Cell { value = 1 }); }",
        "fn change(value: RefMut<i32>) {} fn call<F>(function: F, value: Ref<i32>) { function(value); } fn main() { let value: i32 = 1; call(change, value); }",
        "fn main() { let mut value: i32 = 1; let reference: Ref<i32> = value; let mut alias = reference; alias = 2; let writable: RefMut<i32> = reference; }",
        "fn change(value: RefMut<i32>) {} fn main() { let function: (Ref<i32>) -> () = change; }",
        "fn change(value: RefMut<i32>) {} fn call<F>(function: F) { let value: i32 = 1; function(value); } fn main() { call(change); }",
    ] {
        let source = format!("export {{ main }}; {source}");
        assert!(
            support::pipeline::source_module(&source).is_err(),
            "accepted invalid mutable borrow: {source}"
        );
    }
}

#[test]
fn readonly_array_access_and_mutable_element_access_share_storage() {
    assert_eq!(
        result(
            r#"export { main };
        fn increment(value: RefMut<i32>) { value = value + 1; }
        fn read<A>(values: Ref<A>) -> i32 { values:at(0) + values:at(1) }
        fn main() -> i32 {
            let mut values = [i32(20), i32(20)];
            increment(values:at_mut(0));
            increment(values:at_mut(1));
            read(values)
        }
    "#
        ),
        42
    );
}

#[test]
fn dependent_function_parameters_can_borrow_mutable_locals() {
    assert_eq!(
        result(
            r#"export { main };
        fn increment(value: RefMut<i32>) { value = value + 1; }
        fn apply<F>(function: F) -> i32 {
            let mut value: i32 = 41;
            function(value);
            value
        }
        fn main() -> i32 { apply(increment) }
    "#
        ),
        42
    );
}
