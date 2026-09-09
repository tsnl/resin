use resin_source::Source;
use resin_source::{Loader, normalize_path};
use std::fs;
use tempfile::TempDir;

#[test]
fn disk_reads_reuse_versions_and_preserve_old_contents_after_edits() {
    let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
    let path = directory.path().join("main.resin");
    fs::write(&path, "first").unwrap();
    let mut loader = Loader::new(directory.path().into());
    let first = loader.load_file(&path).unwrap();
    assert_eq!(first, loader.load_file(&path).unwrap());
    fs::write(&path, "second").unwrap();
    let second = loader.load_file(&path).unwrap();
    assert_eq!(first.id(), second.id());
    assert_ne!(first, second);
    assert_eq!(first.text(), "first");
    assert_eq!(second.text(), "second");
    assert_eq!(
        loader.path(&first),
        Some(normalize_path(&path).unwrap().as_path())
    );
}

#[test]
fn unsaved_buffers_share_their_identity_with_later_disk_contents() {
    let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
    let path = directory.path().join("unsaved.resin");
    let mut loader = Loader::new(directory.path().into());
    let buffer = loader.source_from_text(&path, "buffer").unwrap();
    assert!(!path.exists());
    fs::write(&path, "buffer").unwrap();
    assert_eq!(buffer, loader.load_file(&path).unwrap());
    let edited = loader.source_from_text(&path, "edited").unwrap();
    assert_eq!(buffer.id(), edited.id());
    assert_eq!(loader.path(&edited), loader.path(&buffer));
    assert_eq!(fs::read_to_string(path).unwrap(), "buffer");
}

#[test]
fn imports_use_the_registered_origin_and_configured_library_root() {
    let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
    let mut loader = Loader::new(directory.path().join("libraries"));
    let entry = loader
        .source_from_text(&directory.path().join("project/main.resin"), "")
        .unwrap();
    for (reference, expected) in [
        ("helper.resin", "project/helper.resin"),
        ("../shared.resin", "shared.resin"),
        ("std/math.resin", "project/std/math.resin"),
        ("$/math.resin", "libraries/math.resin"),
        ("$/math/vector.resin", "libraries/math/vector.resin"),
        ("$/vendor/module.resin", "libraries/vendor/module.resin"),
        ("$/stdlib/module.resin", "libraries/stdlib/module.resin"),
    ] {
        let expected = normalize_path(&directory.path().join(expected)).unwrap();
        assert_eq!(loader.resolve_import(&entry, reference).unwrap(), expected);
    }
    for reference in [
        "$/",
        "$/../escape.resin",
        "$/math/../../escape.resin",
        "$//absolute",
    ] {
        assert!(
            loader.resolve_import(&entry, reference).is_err(),
            "{reference}"
        );
    }
    let unrelated = Source::new(entry.name(), "");
    assert!(loader.resolve_import(&unrelated, "helper.resin").is_err());
}

#[test]
fn import_loading_reads_current_disk_contents() {
    let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
    let path = directory.path().join("helper.resin");
    let mut loader = Loader::new(directory.path().into());
    let root = loader
        .source_from_text(&directory.path().join("main.resin"), "")
        .unwrap();
    fs::write(&path, "first").unwrap();
    let first = loader.load_import(&root, "helper.resin").unwrap();
    fs::write(path, "second").unwrap();
    let second = loader.load_import(&root, "helper.resin").unwrap();
    assert_eq!(first.id(), second.id());
    assert_eq!(second.text(), "second");
}

#[test]
#[cfg(unix)]
fn symlinked_paths_share_identity_even_before_a_file_is_saved() {
    use std::os::unix::fs::symlink;
    let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
    fs::create_dir(directory.path().join("real")).unwrap();
    symlink(
        directory.path().join("real"),
        directory.path().join("alias"),
    )
    .unwrap();
    let mut loader = Loader::new(directory.path().into());
    let real = loader
        .source_from_text(&directory.path().join("real/new.resin"), "")
        .unwrap();
    let alias = loader
        .source_from_text(&directory.path().join("alias/new.resin"), "")
        .unwrap();
    assert_eq!(real, alias);
    assert_eq!(
        normalize_path(std::path::Path::new(".")).unwrap(),
        std::env::current_dir().unwrap()
    );
}

#[test]
#[cfg(unix)]
fn non_utf8_paths_with_equal_display_names_have_distinct_identities() {
    use std::{ffi::OsStr, os::unix::ffi::OsStrExt};
    let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
    let first_path = directory.path().join(OsStr::from_bytes(b"file\xff.resin"));
    let second_path = directory.path().join(OsStr::from_bytes(b"file\xfe.resin"));
    let mut loader = Loader::new(directory.path().into());
    let first = loader.source_from_text(&first_path, "").unwrap();
    let second = loader.source_from_text(&second_path, "").unwrap();
    assert_eq!(first.name(), second.name());
    assert_ne!(first.id(), second.id());
    assert_eq!(
        loader.path(&first),
        Some(normalize_path(&first_path).unwrap().as_path())
    );
    assert_eq!(
        loader.path(&second),
        Some(normalize_path(&second_path).unwrap().as_path())
    );
}

#[test]
fn unknown_import_namespaces_are_rejected_without_filesystem_fallback() {
    let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
    let mut loader = Loader::new(directory.path().into());
    let source = loader
        .source_from_text(&directory.path().join("main.resin"), "")
        .unwrap();
    for reference in ["$", "$std/module.resin", "$vendor/module.resin"] {
        let error = loader.resolve_import(&source, reference).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(
            error.to_string().contains("unknown import namespace"),
            "{error}"
        );
    }
}

#[test]
fn supplied_text_survives_disk_reads_and_removal_restores_current_disk_contents() {
    let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
    let root_path = directory.path().join("root.resin");
    let path = directory.path().join("helper.resin");
    fs::write(&path, "disk before").unwrap();
    let mut loader = Loader::new(directory.path().into());
    let root = loader.source_from_text(&root_path, "").unwrap();
    let before = loader.load_file(&path).unwrap();
    let supplied = loader.source_from_text(&path, "editor").unwrap();
    assert_eq!(loader.load_import(&root, "helper.resin").unwrap(), supplied);
    fs::write(&path, "disk after").unwrap();
    let after = loader.load_file(&path).unwrap();
    assert_eq!(after.text(), "disk after");
    assert_eq!(before.id(), supplied.id());
    assert_eq!(supplied.id(), after.id());
    assert_eq!(loader.load_import(&root, "helper.resin").unwrap(), supplied);
    assert_eq!(loader.source_from_text(&path, "editor").unwrap(), supplied);
    assert_eq!(loader.load_file(&path).unwrap(), after);
    loader.remove_source(&path).unwrap();
    assert_eq!(loader.load_import(&root, "helper.resin").unwrap(), after);
    assert_eq!(supplied.text(), "editor");
    assert_eq!(before.text(), "disk before");
}

#[test]
fn closing_an_unsaved_source_preserves_its_identity_without_serving_cached_text() {
    let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
    let mut loader = Loader::new(directory.path().into());
    let root = loader
        .source_from_text(&directory.path().join("root.resin"), "")
        .unwrap();
    let path = directory.path().join("unsaved.resin");
    let unsaved = loader.source_from_text(&path, "editor").unwrap();
    assert_eq!(loader.load_import(&root, "unsaved.resin").unwrap(), unsaved);
    assert!(loader.load_file(&path).is_err());
    assert_eq!(loader.load_import(&root, "unsaved.resin").unwrap(), unsaved);
    loader.remove_source(&path).unwrap();
    assert_eq!(
        loader
            .load_import(&root, "unsaved.resin")
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::NotFound
    );
    fs::write(&path, "saved later").unwrap();
    let saved = loader.load_import(&root, "unsaved.resin").unwrap();
    assert_eq!(saved.id(), unsaved.id());
    assert_eq!(saved.text(), "saved later");
}

#[test]
fn named_source_bindings_are_explicit_and_survive_unrelated_disk_reads() {
    let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
    let mut loader = Loader::new(directory.path().into());
    let first = Source::new("same label", "first root");
    let second = Source::new("same label", "second root");
    let left = Source::new("module", "left");
    let right = Source::new("module", "right");
    loader.set_import(&first, "module", left.clone()).unwrap();
    loader.set_import(&second, "module", right.clone()).unwrap();
    let path = directory.path().join("module");
    fs::write(&path, "disk").unwrap();
    loader.load_file(&path).unwrap();
    assert_eq!(
        loader
            .load_import(&first.with_text("edited root"), "module")
            .unwrap(),
        left
    );
    assert_eq!(loader.load_import(&second, "module").unwrap(), right);
    let changed = left.with_text("changed");
    loader
        .set_import(&first, "module", changed.clone())
        .unwrap();
    assert_eq!(loader.load_import(&first, "module").unwrap(), changed);
    assert_eq!(left.text(), "left");
    loader
        .set_import(&first, "$/vendor/module", right.clone())
        .unwrap();
    assert_eq!(
        loader.load_import(&first, "$/vendor/module").unwrap(),
        right
    );
    for reference in ["$vendor/module", "$/../escape", "$/"] {
        assert!(loader.set_import(&first, reference, right.clone()).is_err());
        assert!(loader.load_import(&first, reference).is_err());
    }
}

#[test]
fn named_sources_can_import_rooted_libraries_without_a_file_origin() {
    let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
    let library = directory.path().join("math/library.resin");
    fs::create_dir(library.parent().unwrap()).unwrap();
    fs::write(&library, "math library").unwrap();
    let mut loader = Loader::new(directory.path().into());
    let source = Source::new("generated source", "");
    assert!(loader.path(&source).is_none());
    let imported = loader.load_import(&source, "$/math/library.resin").unwrap();
    assert_eq!(imported.text(), "math library");
    assert!(loader.load_import(&source, "std/library.resin").is_err());
}

#[test]
#[cfg(unix)]
fn closing_a_source_after_its_path_becomes_a_symlink_does_not_resurrect_its_buffer() {
    use std::os::unix::fs::symlink;
    let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
    let path = directory.path().join("library.resin");
    let target = directory.path().join("target.resin");
    fs::write(&path, "original disk").unwrap();
    fs::write(&target, "symlink target").unwrap();
    let mut loader = Loader::new(directory.path().into());
    let root = loader
        .source_from_text(&directory.path().join("root.resin"), "")
        .unwrap();
    let supplied = loader.source_from_text(&path, "editor buffer").unwrap();
    let registered = loader.path(&supplied).unwrap().to_path_buf();
    fs::remove_file(&path).unwrap();
    symlink(&target, &path).unwrap();
    loader.remove_source(&registered).unwrap();
    assert_eq!(
        loader.load_import(&root, "library.resin").unwrap().text(),
        "symlink target"
    );
    fs::remove_file(&path).unwrap();
    fs::write(&path, "restored disk").unwrap();
    let restored = loader.load_import(&root, "library.resin").unwrap();
    assert_eq!(restored.text(), "restored disk");
    assert_eq!(restored.id(), supplied.id());
    assert_eq!(supplied.text(), "editor buffer");
}

#[test]
#[cfg(unix)]
fn editing_an_open_path_after_symlink_replacement_preserves_its_registered_identity() {
    use std::os::unix::fs::symlink;
    let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
    let path = directory.path().join("library.resin");
    let target = directory.path().join("target.resin");
    fs::write(&path, "original disk").unwrap();
    fs::write(&target, "symlink target").unwrap();
    let mut loader = Loader::new(directory.path().into());
    let original = loader.source_from_text(&path, "first edit").unwrap();
    let registered = loader.path(&original).unwrap().to_path_buf();
    fs::remove_file(&path).unwrap();
    symlink(&target, &path).unwrap();
    let edited = loader.source_from_text(&registered, "second edit").unwrap();
    assert_eq!(edited.id(), original.id());
    assert_eq!(loader.path(&edited), Some(registered.as_path()));
    assert_eq!(original.text(), "first edit");
    loader.remove_source(&registered).unwrap();
    let root = loader
        .source_from_text(&directory.path().join("root.resin"), "")
        .unwrap();
    assert_eq!(
        loader.load_import(&root, "library.resin").unwrap().text(),
        "symlink target"
    );
    assert_eq!(
        loader.load_import(&root, "target.resin").unwrap().text(),
        "symlink target"
    );
}
