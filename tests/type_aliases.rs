mod support;

#[test]
fn generic_aliases_borrow_storage_and_preserve_nominal_methods() {
    let module = support::module(
        r#"export { main };
        struct FieldsDataLength<T0, T1> { data: T0; length: T1; }
type View<T> = FieldsDataLength<Ptr<T>, ulong>;
        struct Item { value: int;
            
        }
fn read(self: Item) -> int  { self.value }

        type Renamed<T> = Item;
        fn first<T>(view: View<T>) -> T  { view.data.* }
        fn main() -> int  {
            let mut item = Renamed<bool> { value = 42 };
            first(View<Item> { data = &item, length = 1 }):read()
        }
    "#,
    );
    let output = support::project::Project::new(&module, Some("main"))
        .unwrap()
        .run();
    assert_eq!(
        output.status.code(),
        Some(42),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn aliases_reuse_the_same_function_instance() {
    let module = support::module(
        "type First<T> = Ptr<T>; type Second<U> = First<U>; fn measure<T>() -> ulong  { size_of(T) } fn main() -> ulong  { measure::<First<int>>() + measure::<Second<int>>() }",
    );
    assert_eq!(
        module
            .functions
            .iter()
            .filter(|function| function.name.as_deref() == Some("measure"))
            .count(),
        1
    );
}
