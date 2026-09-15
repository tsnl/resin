use resin_executor::{Cancellation, Execution};
use resin_source::Loader;
use tempfile::TempDir;

#[tokio::test]
async fn user_upload_names_do_not_inherit_the_local_library_namespace() {
    let directory = TempDir::new().unwrap();
    let mut loader = Loader::new(directory.path().into());
    let source = loader
        .source_from_text(&directory.path().join("main.resin"), "text")
        .unwrap();
    let execution = Execution::default();
    let cancellation = Cancellation::new();
    let managed = loader
        .logical_sources(
            [source.clone()],
            directory.path(),
            &execution,
            &cancellation,
        )
        .await
        .unwrap();
    let user = loader
        .logical_user_sources(
            [source.clone()],
            directory.path(),
            &execution,
            &cancellation,
        )
        .await
        .unwrap();
    assert_eq!(managed[&source.id()].name(), "$/main.resin");
    assert_eq!(user[&source.id()].name(), "main.resin");
    assert_eq!(user[&source.id()].text(), source.text());
    assert_eq!(user[&source.id()].content_hash(), source.content_hash());
    assert!(source.name().contains(directory.path().to_str().unwrap()));
}

#[cfg(unix)]
#[tokio::test]
async fn user_naming_never_inspects_an_unusable_library_root() {
    let directory = TempDir::new().unwrap();
    let broken = directory.path().join("broken-library");
    std::os::unix::fs::symlink(&broken, &broken).unwrap();
    let mut loader = Loader::new(broken);
    let source = loader
        .source_from_text(&directory.path().join("main.resin"), "text")
        .unwrap();
    let user = loader
        .logical_user_sources(
            [source.clone()],
            directory.path(),
            &Execution::default(),
            &Cancellation::new(),
        )
        .await
        .unwrap();
    assert_eq!(user[&source.id()].name(), "main.resin");
}
