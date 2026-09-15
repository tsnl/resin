#![cfg(unix)]

use resin_executor::{Cancellation, Execution};
use resin_toolchain::{BuiltProject, CProfile, Environment, Error, Toolchain};
use std::{
    fs,
    num::NonZeroUsize,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Duration,
};
use tempfile::TempDir;

struct Native {
    temporary: TempDir,
    tools: Toolchain,
    project: PathBuf,
}

impl Native {
    fn new(compiler: &str) -> Self {
        let temporary = TempDir::new().unwrap();
        let root = temporary.path();
        let mut environment = Environment::capture().unwrap();
        environment.directory = root.into();
        environment.temporary = root.into();
        environment
            .variables
            .insert("RESIN_TEST_CONTROL".into(), root.into());
        let original_ninja = environment
            .variables
            .get(std::ffi::OsStr::new("NINJA"))
            .cloned()
            .unwrap_or_else(|| "ninja".into());
        environment
            .variables
            .insert("RESIN_TEST_NINJA".into(), original_ninja);
        let runner = root.join("runner");
        script(
            &runner,
            r#"printf '%s' "$$" > "$RESIN_TEST_CONTROL/ninja.pid"
exec "$RESIN_TEST_NINJA" "$@""#,
        );
        environment.variables.insert("NINJA".into(), runner.into());
        let compiler_path = root.join("compiler");
        script(&compiler_path, compiler);
        let project = root.join("project");
        fs::create_dir(&project).unwrap();
        fs::write(project.join("input"), "11").unwrap();
        fs::write(project.join("build.ninja"), "include toolchain.ninja\nrule emit\n  command = $cc $in $out\nbuild program: emit input | toolchain.state\n").unwrap();
        let tools = environment.toolchain(Some(compiler_path.as_os_str()), None);
        Self {
            temporary,
            tools,
            project,
        }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.temporary.path().join(name)
    }

    fn build(
        &self,
        name: &str,
        execution: &Execution,
        cancellation: &Cancellation,
    ) -> tokio::task::JoinHandle<Result<BuiltProject, Error>> {
        let (tools, project, name, execution, cancellation) = (
            self.tools.clone(),
            self.project.clone(),
            name.to_owned(),
            execution.clone(),
            cancellation.clone(),
        );
        tokio::spawn(async move {
            tools
                .build(
                    &project,
                    &name,
                    "main",
                    CProfile::Debug,
                    &execution,
                    &cancellation,
                )
                .await
        })
    }
}

fn script(path: &Path, body: &str) {
    fs::write(path, format!("#!/bin/sh\nset -eu\n{body}\n")).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}

fn execution(jobs: usize) -> Execution {
    Execution::new(NonZeroUsize::new(jobs).unwrap())
}

async fn appeared(path: &Path) {
    tokio::time::timeout(Duration::from_secs(10), async {
        while !path.exists() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("native process never created {}", path.display()));
}

const GATED_COMPILER: &str = r#"
value=$(cat "$1")
printf 'call\n' >> "$RESIN_TEST_CONTROL/calls"
printf started > "$RESIN_TEST_CONTROL/started-$value"
while [ -f "$RESIN_TEST_CONTROL/block-$value" ]; do sleep 0.01; done
printf '#!/bin/sh\nexit %s\n' "$value" > "$2"
chmod +x "$2"
"#;

#[tokio::test]
async fn retained_outputs_run_while_the_same_cache_builds_a_new_generation() {
    let native = Native::new(GATED_COMPILER);
    let execution = execution(2);
    let cancellation = Cancellation::new();
    let first = native
        .build("module", &execution, &cancellation)
        .await
        .unwrap()
        .unwrap();
    let original = first.executable("program").unwrap();
    let clone = original.clone();
    let directory = first.directory().to_owned();
    let before = fs::read(original.path()).unwrap();
    fs::write(native.project.join("input"), "22").unwrap();
    fs::write(native.path("block-22"), "").unwrap();
    let second = native.build("module", &execution, &cancellation);
    appeared(&native.path("started-22")).await;
    assert_eq!(fs::read(original.path()).unwrap(), before);
    assert_eq!(
        tokio::time::timeout(
            Duration::from_secs(2),
            original.run(&execution, &cancellation)
        )
        .await
        .unwrap()
        .unwrap(),
        11
    );
    fs::remove_file(native.path("block-22")).unwrap();
    let second = second.await.unwrap().unwrap();
    assert_ne!(first.directory(), second.directory());
    assert_eq!(fs::read(original.path()).unwrap(), before);
    assert_eq!(
        second
            .executable("program")
            .unwrap()
            .run(&execution, &cancellation)
            .await
            .unwrap(),
        22
    );
    drop(first);
    drop(original);
    assert!(
        directory.exists(),
        "an executable clone retains the generation"
    );
    assert_eq!(clone.run(&execution, &cancellation).await.unwrap(), 11);
    drop(clone);
    assert!(
        !directory.exists(),
        "the final owner removes only its generation"
    );
    assert!(second.path("program").exists());
}

#[tokio::test]
async fn separate_cache_slots_build_in_parallel_and_queued_cancellation_starts_no_process() {
    let native = Native::new(GATED_COMPILER);
    fs::write(native.path("block-11"), "").unwrap();
    let execution = execution(2);
    let token = Cancellation::new();
    let first = native.build("first", &execution, &token);
    appeared(&native.path("started-11")).await;
    // A second project directory prevents a source edit racing the first staging pass.
    let other = native.path("other");
    fs::create_dir(&other).unwrap();
    fs::copy(
        native.project.join("build.ninja"),
        other.join("build.ninja"),
    )
    .unwrap();
    fs::write(other.join("input"), "22").unwrap();
    fs::write(native.path("block-22"), "").unwrap();
    let (tools, second_execution, second_token) =
        (native.tools.clone(), execution.clone(), token.clone());
    let second = tokio::spawn(async move {
        tools
            .build(
                &other,
                "second",
                "main",
                CProfile::Debug,
                &second_execution,
                &second_token,
            )
            .await
    });
    appeared(&native.path("started-22")).await;
    let queued_token = Cancellation::new();
    let queued = native.build("third", &execution, &queued_token);
    tokio::time::sleep(Duration::from_millis(30)).await;
    queued_token.cancel();
    assert!(
        tokio::time::timeout(Duration::from_secs(2), queued)
            .await
            .unwrap()
            .unwrap()
            .unwrap_err()
            .is_cancelled()
    );
    fs::remove_file(native.path("block-11")).unwrap();
    fs::remove_file(native.path("block-22")).unwrap();
    assert!(first.await.unwrap().is_ok());
    assert!(second.await.unwrap().is_ok());
    execution.wait_idle().await;
    assert_eq!(
        fs::read_to_string(native.path("calls")).unwrap(),
        "call\ncall\n"
    );
}

#[tokio::test]
async fn cancellation_while_waiting_for_the_staging_lock_leaves_the_active_build_alone() {
    let native = Native::new(GATED_COMPILER);
    fs::write(native.path("block-11"), "").unwrap();
    let execution = execution(2);
    let first = native.build("module", &execution, &Cancellation::new());
    appeared(&native.path("started-11")).await;
    let token = Cancellation::new();
    let queued = native.build("module", &execution, &token);
    tokio::time::sleep(Duration::from_millis(30)).await;
    token.cancel();
    assert!(
        tokio::time::timeout(Duration::from_secs(2), queued)
            .await
            .unwrap()
            .unwrap()
            .unwrap_err()
            .is_cancelled()
    );
    assert!(!first.is_finished());
    fs::remove_file(native.path("block-11")).unwrap();
    assert!(first.await.unwrap().is_ok());
}

const STUBBORN_COMPILER: &str = r#"
trap '' TERM INT
"$RESIN_TEST_CONTROL/child" &
printf '%s' "$!" > "$RESIN_TEST_CONTROL/child.pid"
printf '%s' "$$" > "$RESIN_TEST_CONTROL/compiler.pid"
wait
"#;

fn stubborn_native() -> Native {
    let native = Native::new(STUBBORN_COMPILER);
    script(
        &native.path("child"),
        r#"
trap '' TERM INT
while [ -f "$RESIN_TEST_CONTROL/block" ]; do sleep 0.01; done
printf escaped > "$RESIN_TEST_CONTROL/late"
"#,
    );
    fs::write(native.path("block"), "").unwrap();
    native
}

fn alive(pid: i32) -> bool {
    #[cfg(target_os = "linux")]
    {
        fs::read_to_string(format!("/proc/{pid}/stat"))
            .ok()
            .and_then(|text| {
                text.rsplit_once(')')
                    .map(|(_, rest)| rest.trim_start().starts_with('Z'))
            })
            .is_some_and(|zombie| !zombie)
    }
    #[cfg(not(target_os = "linux"))]
    unsafe {
        libc::kill(pid, 0) == 0
    }
}

async fn assert_stopped(native: &Native) {
    for name in ["ninja.pid", "compiler.pid", "child.pid"] {
        let pid: i32 = fs::read_to_string(native.path(name))
            .unwrap()
            .parse()
            .unwrap();
        assert!(
            !alive(pid),
            "{name} ({pid}) survived completed cancellation"
        );
    }
    fs::remove_file(native.path("block")).unwrap();
    tokio::time::sleep(Duration::from_millis(30)).await;
    assert!(!native.path("late").exists());
    let artifacts = native.path("build/.artifacts");
    if artifacts.exists() {
        assert_eq!(fs::read_dir(artifacts).unwrap().count(), 0);
    }
}

#[tokio::test]
async fn cancellation_terminates_ninja_compiler_groups_and_stubborn_descendants() {
    let native = stubborn_native();
    let execution = execution(1);
    let cancellation = Cancellation::new();
    let build = native.build("module", &execution, &cancellation);
    appeared(&native.path("compiler.pid")).await;
    cancellation.cancel();
    assert!(
        tokio::time::timeout(Duration::from_secs(5), build)
            .await
            .unwrap()
            .unwrap()
            .unwrap_err()
            .is_cancelled()
    );
    assert_stopped(&native).await;
    execution.wait_idle().await;
    script(&native.path("compiler"), GATED_COMPILER);
    assert!(
        native
            .build("module", &execution, &Cancellation::new())
            .await
            .unwrap()
            .is_ok(),
        "cancellation must release the staging lock and capacity"
    );
}

#[tokio::test]
async fn abandoning_a_build_future_retains_capacity_until_its_process_tree_is_stopped() {
    let native = stubborn_native();
    let execution = execution(1);
    let build = native.build("module", &execution, &Cancellation::new());
    appeared(&native.path("compiler.pid")).await;
    build.abort();
    assert!(build.await.unwrap_err().is_cancelled());
    tokio::time::timeout(Duration::from_secs(5), execution.wait_idle())
        .await
        .unwrap();
    assert_stopped(&native).await;
}

#[test]
fn shutting_down_the_runtime_terminates_owned_process_trees() {
    let native = stubborn_native();
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let _build = native.build("module", &execution(1), &Cancellation::new());
        appeared(&native.path("compiler.pid")).await;
    });
    drop(runtime);
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(assert_stopped(&native));
}

#[tokio::test]
async fn cancelling_a_fake_ninja_also_stops_its_children() {
    let native = stubborn_native();
    script(
        &native.path("runner"),
        &format!("printf '%s' \"$$\" > \"$RESIN_TEST_CONTROL/ninja.pid\"\n{STUBBORN_COMPILER}"),
    );
    let execution = execution(1);
    let token = Cancellation::new();
    let build = native.build("module", &execution, &token);
    appeared(&native.path("compiler.pid")).await;
    token.cancel();
    assert!(
        tokio::time::timeout(Duration::from_secs(5), build)
            .await
            .unwrap()
            .unwrap()
            .unwrap_err()
            .is_cancelled()
    );
    assert_stopped(&native).await;
}

#[tokio::test]
async fn removing_an_incremental_cache_slot_does_not_remove_retained_artifacts() {
    let native = Native::new(GATED_COMPILER);
    let execution = execution(2);
    let token = Cancellation::new();
    let first = native
        .build("module", &execution, &token)
        .await
        .unwrap()
        .unwrap();
    for entry in fs::read_dir(native.path("build")).unwrap() {
        let entry = entry.unwrap();
        if entry.file_name() != ".artifacts" {
            fs::remove_dir_all(entry.path()).unwrap();
        }
    }
    assert_eq!(
        first
            .executable("program")
            .unwrap()
            .run(&execution, &token)
            .await
            .unwrap(),
        11
    );
    let second = native
        .build("module", &execution, &token)
        .await
        .unwrap()
        .unwrap();
    drop(first);
    assert_eq!(
        second
            .executable("program")
            .unwrap()
            .run(&execution, &token)
            .await
            .unwrap(),
        11
    );
}
