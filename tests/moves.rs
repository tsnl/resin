mod support;

#[test]
fn moves_and_partial_replacements_destroy_each_value_once() {
    let source = r#"export { main };
        struct Item { trace: Ptr<int>; digit: int;
            
        }
fn drop(value: Ptr<Item>) { value.trace.* = value.trace.* * 10 + value.digit; }

        struct Pair { a: Item; b: Item; }
        fn main() -> int {
            let mut trace: int = 0;
            {
                let mut pair = Pair { a= Item { trace= &trace, digit= 1 }, b= Item { trace= &trace, digit= 2 } };
                let saved = pair.a;
                pair.a = Item { trace= &trace, digit= 3 };
                let moved = pair;
            };
            if (trace == 231) { 0 } else { 1 }
        }
    "#;
    let module = support::module(source);
    let output = support::project::Project::new(&module, Some("main"))
        .unwrap()
        .run();
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn returned_error_payloads_keep_exactly_one_owner() {
    let source = r#"export { main };
        struct Item { trace: Ptr<int>; }
        fn drop(value: Ptr<Item>) { value.trace.* = value.trace.* + 1; }
        fn route(value: Item, failed: bool) -> Item | Err<Item> {
            if (failed) { return Err(value); };
            value
        }
        fn main() -> int {
            let mut trace: int = 0;
            {
                let result = route(Item { trace = &trace }, true);
                match (result) {
                    Err(error) => { let moved = error; },
                    Item(value) => { let moved = value; },
                };
            };
            if (trace == 1) { 0 } else { 1 }
        }
    "#;
    let module = support::module(source);
    let output = support::project::Project::new(&module, Some("main"))
        .unwrap()
        .run();
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
