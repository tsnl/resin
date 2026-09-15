#[test]
fn every_constant_specification_requires_an_explicit_initializer() {
    for declaration in [
        "const value;",
        "const value: int;",
        "const _;",
        "const value =;",
        "const ( first; second = iota; );",
        "const ( first = iota; second; );",
        "const ( first = iota; second: int; );",
        "const ( first = iota; _; );",
        "const ( a, b = iota, iota; c, d; );",
        "const ( first = iota; second =; );",
    ] {
        for source in [
            declaration.to_owned(),
            format!("def f() = {{ {declaration} }};"),
        ] {
            let document = resin_cst::build_cst(&source, None);
            assert!(document.tree().root_node().has_error(), "{source}");
            assert!(resin_cst::format_source(&source).is_none(), "{source}");
        }
    }
}

#[test]
fn explicit_initializers_allow_optional_types_and_discarded_names() {
    let declaration =
        "const ( first: uint = iota; _ = iota; next = iota; a, b: uint = iota, iota + 10; );";
    for source in [
        declaration.to_owned(),
        format!("def f() = {{ {declaration} }};"),
    ] {
        let document = resin_cst::build_cst(&source, None);
        assert!(!document.tree().root_node().has_error(), "{source}");
    }
}
