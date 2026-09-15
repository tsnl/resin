use super::*;
use resin_toolchain::Environment;
use std::{fs, num::NonZeroUsize, time::Duration};
use tempfile::TempDir;

fn tools(directory: &TempDir) -> Toolchain {
    let mut environment = Environment::capture().unwrap();
    environment.directory = directory.path().into();
    environment.temporary = directory.path().into();
    environment.toolchain(None, None)
}

fn request(source: &Path, destination: &Path) -> BuildRequest {
    BuildRequest {
        uri: url::Url::from_file_path(source)
            .unwrap()
            .as_str()
            .parse()
            .unwrap(),
        entry: "main".into(),
        destination: destination.into(),
        profile: BuildProfile::Debug,
    }
}

fn executable(directory: &Path, name: &str) -> PathBuf {
    directory.join(format!("{name}{}", std::env::consts::EXE_SUFFIX))
}

async fn exit_code(path: &Path) -> i32 {
    tokio::time::timeout(
        Duration::from_secs(10),
        tokio::process::Command::new(path)
            .kill_on_drop(true)
            .status(),
    )
    .await
    .expect("tiny test executable must exit")
    .unwrap()
    .code()
    .unwrap()
}

fn evict_all(caches: &Caches) {
    let empty = Caches::default();
    caches.sources.store(empty.sources.load_full());
    caches.syntax.store(empty.syntax.load_full());
    caches.ast.store(empty.ast.load_full());
    caches.hir.store(empty.hir.load_full());
    caches.verified.store(empty.verified.load_full());
    caches.generated.store(empty.generated.load_full());
}

#[tokio::test]
async fn repeated_builds_reuse_later_passes_and_retained_generation_survives_eviction() {
    let directory = TempDir::new().unwrap();
    let source = directory.path().join("main.resin");
    fs::write(&source, "export { main }; def main() -> int = { 7 };").unwrap();
    let first_output = executable(directory.path(), "first output");
    let second_output = executable(directory.path(), "second output");
    let tools = tools(&directory);
    let caches = Caches::default();
    let execution = Execution::new(NonZeroUsize::new(2).unwrap());
    let cancellation = Cancellation::new();
    let library = resin_source::library_root();

    run(
        request(&source, &first_output),
        &library,
        &tools,
        directory.path(),
        &caches,
        &execution,
        &cancellation,
    )
    .await
    .unwrap();
    let (graph, hir) = caches
        .hir
        .load()
        .iter()
        .next()
        .map(|(key, value)| (key.clone(), value.clone()))
        .unwrap();
    let (lir_key, verified) = caches
        .verified
        .load()
        .iter()
        .next()
        .map(|(key, value)| (key.clone(), value.clone()))
        .unwrap();
    let (generation_key, generated) = caches
        .generated
        .load()
        .iter()
        .next()
        .map(|(key, value)| (key.clone(), value.clone()))
        .unwrap();
    let generated_directory = generated.directory().to_path_buf();
    let c_source = generated.c_source().unwrap().to_path_buf();
    let c_text = fs::read(&c_source).unwrap();
    let weak = Arc::downgrade(&generated);

    let response = run(
        request(&source, &second_output),
        &library,
        &tools,
        directory.path(),
        &caches,
        &execution,
        &cancellation,
    )
    .await
    .unwrap();
    let reported = url::Url::parse(response["outputUri"].as_str().unwrap())
        .unwrap()
        .to_file_path()
        .unwrap();
    assert_eq!(
        reported,
        resin_source::normalize_path(&second_output).unwrap()
    );
    assert_eq!(caches.hir.load().len(), 1);
    assert_eq!(caches.verified.load().len(), 1);
    assert_eq!(caches.generated.load().len(), 1);
    assert!(Arc::ptr_eq(&hir, caches.hir.load().get(&graph).unwrap()));
    assert!(Arc::ptr_eq(
        &verified,
        caches.verified.load().get(&lir_key).unwrap()
    ));
    assert!(Arc::ptr_eq(
        &generated,
        caches.generated.load().get(&generation_key).unwrap()
    ));
    assert_eq!(exit_code(&first_output).await, 7);
    assert_eq!(exit_code(&second_output).await, 7);

    evict_all(&caches);
    assert!(weak.upgrade().is_some());
    assert_eq!(fs::read(&c_source).unwrap(), c_text);
    drop(generated);
    assert!(weak.upgrade().is_none());
    assert!(!generated_directory.exists());
    assert_eq!(exit_code(&first_output).await, 7);
    assert_eq!(exit_code(&second_output).await, 7);
    execution.wait_idle().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_builds_keep_distinct_versions_and_independent_artifact_owners() {
    let directory = TempDir::new().unwrap();
    let left = directory.path().join("left/main.resin");
    let right = directory.path().join("right/main.resin");
    fs::create_dir(left.parent().unwrap()).unwrap();
    fs::create_dir(right.parent().unwrap()).unwrap();
    fs::write(&left, "export { main }; def main() -> int = { 7 };").unwrap();
    fs::write(&right, "export { main }; def main() -> int = { 11 };").unwrap();
    let left_output = executable(directory.path(), "left output");
    let right_output = executable(directory.path(), "right output");
    let tools = tools(&directory);
    let caches = Caches::default();
    let execution = Execution::new(NonZeroUsize::new(2).unwrap());
    let cancellation = Cancellation::new();
    let library = resin_source::library_root();
    let (left, right) = tokio::join!(
        run(
            request(&left, &left_output),
            &library,
            &tools,
            directory.path(),
            &caches,
            &execution,
            &cancellation
        ),
        run(
            request(&right, &right_output),
            &library,
            &tools,
            directory.path(),
            &caches,
            &execution,
            &cancellation
        ),
    );
    left.unwrap();
    right.unwrap();
    assert_eq!(exit_code(&left_output).await, 7);
    assert_eq!(exit_code(&right_output).await, 11);
    assert_eq!(caches.hir.load().len(), 2);
    assert_eq!(caches.verified.load().len(), 2);
    let mut generations: Vec<_> = caches
        .generated
        .load()
        .iter()
        .map(|(_, value)| value.clone())
        .collect();
    assert_eq!(generations.len(), 2);
    let first = generations.remove(0);
    let second = generations.remove(0);
    let first_directory = first.directory().to_path_buf();
    let second_directory = second.directory().to_path_buf();
    assert_ne!(first_directory, second_directory);
    let first_weak = Arc::downgrade(&first);
    let second_weak = Arc::downgrade(&second);

    evict_all(&caches);
    assert!(first_directory.exists());
    assert!(second_directory.exists());
    drop(first);
    assert!(first_weak.upgrade().is_none());
    assert!(!first_directory.exists());
    assert!(second_weak.upgrade().is_some());
    assert!(!fs::read(second.c_source().unwrap()).unwrap().is_empty());
    drop(second);
    assert!(second_weak.upgrade().is_none());
    assert!(!second_directory.exists());
    assert_eq!(exit_code(&left_output).await, 7);
    assert_eq!(exit_code(&right_output).await, 11);
    execution.wait_idle().await;
}
