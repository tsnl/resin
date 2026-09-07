pub const MARKERS: [&str; 4] = [
    "",
    "defer ();",
    "var unused: _; unused := 1;",
    "var unused: Result<(), Never>; unused := ok(()); match (unused) { ok(v) => {}, err(e) => { absurd(e) } };",
];

pub fn variants() -> Vec<String> {
    let mut result = Vec::new();
    for source in [
        include_str!("../fixtures/compound_control.resin"),
        include_str!("../fixtures/numeric_conversions.resin"),
        include_str!("../fixtures/never_elimination.resin"),
        include_str!("../fixtures/nested_cleanup.resin"),
    ] {
        for marker in MARKERS {
            let start = source.find("def kernel(").unwrap();
            let body = start + source[start..].find('{').unwrap() + 1;
            let mut source = source.to_string();
            source.insert_str(body, marker);
            result.push(source);
        }
    }
    result
}
