use resin_common::TempDir;

#[test]
fn temporary_directories_own_only_their_unique_contents() {
    let parent = TempDir::new(&std::env::temp_dir()).unwrap();
    let first = TempDir::new(parent.path()).unwrap();
    let second = TempDir::new(parent.path()).unwrap();
    assert_ne!(first.path(), second.path());
    let removed = first.path().to_path_buf();
    std::fs::write(first.path().join("owned"), "first").unwrap();
    std::fs::write(second.path().join("owned"), "second").unwrap();
    drop(first);
    assert!(!removed.exists());
    assert_eq!(
        std::fs::read_to_string(second.path().join("owned")).unwrap(),
        "second"
    );
}
