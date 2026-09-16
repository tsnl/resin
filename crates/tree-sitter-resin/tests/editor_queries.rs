use std::{fs, path::Path};
use tree_sitter::Query;

#[test]
fn editor_queries_compile_against_current_grammar() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let language = tree_sitter_resin::LANGUAGE.into();
    let mut errors = Vec::new();
    for directory in [
        "editors/zed/languages/resin",
        "editors/helix/runtime/queries/resin",
    ] {
        let mut paths: Vec<_> = fs::read_dir(root.join(directory))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().is_some_and(|extension| extension == "scm"))
            .collect();
        paths.sort();
        assert!(!paths.is_empty(), "no queries in {directory}");
        for path in paths {
            let source = fs::read_to_string(&path).unwrap();
            if let Err(error) = Query::new(&language, &source) {
                errors.push(format!(
                    "{}: {error}",
                    path.strip_prefix(&root).unwrap().display()
                ));
            }
        }
    }
    assert!(errors.is_empty(), "{}", errors.join("\n\n"));
}
