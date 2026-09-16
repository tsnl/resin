#![cfg(unix)]

#[allow(dead_code)]
mod support;

use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
};
use tempfile::TempDir;

use support::{service::Service, shaders};

const WRAPPER: &str = "#!/bin/sh\nfor arg in \"$@\"; do if [ \"$arg\" = -E ]; then exec \"$RESIN_TEST_COMPILER\" \"$@\"; fi; done\nprintf 'compile\\n' >> \"$RESIN_TEST_COUNT\"\nprintf '%s\\n' \"$*\" >> \"$RESIN_TEST_FLAGS\"\nexec \"$RESIN_TEST_COMPILER\" \"$@\"\n";

#[test]
fn foreign_header_changes_reanalyze_including_nested_dependencies() {
    let project = Project::new();
    let header = project.input.parent().unwrap().join("foreign.h");
    let nested = project.input.parent().unwrap().join("value.h");
    fs::write(&nested, "int abs(int value);\n").unwrap();
    fs::write(&header, "#include \"value.h\"\n").unwrap();
    fs::write(
        &project.input,
        format!(
            "export {{ main }}; extern {{ \"{}\": {{ def abs(value: int) -> int; }} }}; import {{ \"$/string.resin\" }}; def main() -> () = {{ print(fmt(\"{{0}}\", (abs(-41),))); }};",
            header.display()
        ),
    )
    .unwrap();
    printed(&project.run(), b"41");
    printed(&project.run(), b"41");
    assert_eq!(project.calls(), 1);
    let analyzed = project.service.server.counters().foreign_builds;
    fs::write(&nested, "int abs(int value); // changed declaration text\n").unwrap();
    printed(&project.run(), b"41");
    assert_eq!(
        project.service.server.counters().foreign_builds,
        analyzed + 1
    );
    assert_eq!(project.calls(), 1);
    fs::write(&nested, "unsigned int abs(unsigned int value);\n").unwrap();
    let output = project.run();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert_eq!(
        project.calls(),
        1,
        "an incompatible header cannot reuse the executable"
    );
    fs::remove_file(nested).unwrap();
    let output = project.run();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    // Preprocessing detects the missing header before another link.
    assert_eq!(project.calls(), 1);
    fs::write(
        &project.input,
        "export { main }; import { \"$/string.resin\" }; def main() -> () = { print(\"no header\"); };",
    )
    .unwrap();
    printed(&project.run(), b"no header");
    printed(&project.run(), b"no header");
    assert_eq!(project.calls(), 2);
}

#[test]
fn shader_objects_are_deduplicated_cached_and_rebuilt_with_imported_helpers() {
    let Some(spirv_opt) = shaders::optimizer() else {
        return;
    };
    let mut project = Project::new();
    let shader_compiler = project.temp.path().join("shader-compiler");
    let count = project.temp.path().join("shader-calls");
    fs::write(&shader_compiler, "#!/bin/sh\nprintf 'compile\\n' >> \"$RESIN_TEST_SHADER_COUNT\"\nexec \"$RESIN_TEST_SHADER_COMPILER\" \"$@\"\n").unwrap();
    fs::set_permissions(&shader_compiler, fs::Permissions::from_mode(0o755)).unwrap();
    let helper = project.input.parent().unwrap().join("helper.resin");
    fs::write(
        &helper,
        "export { pixel }; def pixel (i: uint) -> uint = { i + uint (1) };",
    )
    .unwrap();
    fs::write(
        &project.input,
        r#"
        export { main };
        import { "helper.resin", "$/string.resin", "$/gpu.resin" };
        @compute_shader def kernel(invocation: ulong, output: Ptr<uint>) = { var i = uint(invocation); output.* := { pixel(i) }; };
        def main() -> Result<(), _> = {
            if (0 == 1) {
                var gpu = Gpu.new()?;
                gpu.create_compute_pipeline(kernel)?;
                gpu.create_compute_pipeline(kernel)?;
            };
            print("true");
            ok(())
        };
        "#,
    )
    .unwrap();
    project.configure(|_, environment| {
        environment.variables.extend([
            ("SPIRV_OPT".into(), shader_compiler.clone().into()),
            ("RESIN_TEST_SHADER_COUNT".into(), count.clone().into()),
            ("RESIN_TEST_SHADER_COMPILER".into(), spirv_opt),
        ]);
    });
    let run = || project.run();
    let calls = || fs::read_to_string(&count).unwrap().lines().count();
    printed(&run(), b"true");
    printed(&run(), b"true");
    assert_eq!(calls(), 1);
    assert_eq!(project.calls(), 1);
    // A changed optimizer rebuilds identical SPIR-V. The retained native object
    // and executable depend on those bytes, so neither needs rebuilding.
    let wrapper = fs::read_to_string(&shader_compiler).unwrap();
    fs::write(&shader_compiler, format!("{wrapper}# updated wrapper\n")).unwrap();
    printed(&run(), b"true");
    printed(&run(), b"true");
    assert_eq!(calls(), 2);
    assert_eq!(project.calls(), 1);
    fs::write(
        &helper,
        "export { pixel }; def pixel (i: uint) -> uint = { i + uint (2) };",
    )
    .unwrap();
    printed(&run(), b"true");
    assert_eq!(calls(), 3);
    assert_eq!(project.calls(), 2);
    fs::write(
        &helper,
        "export { pixel }; def pixel (i: uint) -> uint = { i / uint (2) };",
    )
    .unwrap();
    let output = run();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert_eq!(calls(), 3);
    assert_eq!(project.calls(), 2);
}

struct Project {
    service: Service,
    temp: TempDir,
    input: PathBuf,
    compiler: PathBuf,
}

impl Project {
    fn new() -> Self {
        let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
        fs::create_dir(temp.path().join("sources")).unwrap();
        let input = temp.path().join("sources/main.resin");
        let compiler = temp.path().join("compiler");
        fs::write(
            &input,
            r#"export { main }; import { "$/string.resin" }; def main() -> () = { print("first"); };"#,
        )
        .unwrap();
        fs::write(&compiler, WRAPPER).unwrap();
        fs::set_permissions(&compiler, fs::Permissions::from_mode(0o755)).unwrap();
        let service = Self::service(temp.path(), &compiler, |_, _| {});
        Self {
            service,
            temp,
            input,
            compiler,
        }
    }

    fn service(
        directory: &Path,
        compiler: &Path,
        configure: impl FnOnce(&mut resin_server::Config, &mut resin_toolchain::Environment),
    ) -> Service {
        Service::configured(|config, environment| {
            environment.variables.extend([
                ("CC".into(), compiler.as_os_str().to_owned()),
                ("RESIN_TEST_COUNT".into(), directory.join("calls").into()),
                ("RESIN_TEST_FLAGS".into(), directory.join("flags").into()),
                (
                    "RESIN_TEST_COMPILER".into(),
                    std::env::var_os("CC").unwrap_or_else(|| "cc".into()),
                ),
            ]);
            configure(config, environment);
        })
    }

    fn configure(
        &mut self,
        configure: impl FnOnce(&mut resin_server::Config, &mut resin_toolchain::Environment),
    ) {
        self.service = Self::service(self.temp.path(), &self.compiler, configure);
    }

    fn command(&self) -> Command {
        let mut command = self.service.command();
        command.current_dir(self.temp.path()).arg(&self.input);
        command
    }

    fn run(&self) -> Output {
        self.command().output().unwrap()
    }

    fn calls(&self) -> usize {
        fs::read_to_string(self.temp.path().join("calls"))
            .unwrap()
            .lines()
            .count()
    }

    fn executable(&self) -> PathBuf {
        self.profile_executable("debug")
    }

    fn profile_executable(&self, _profile: &str) -> PathBuf {
        self.service.artifacts().pop().expect("retained executable")
    }
}

fn printed(output: &Output, text: &[u8]) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, text);
}

#[test]
fn unchanged_programs_reuse_the_executable_and_still_run() {
    let project = Project::new();
    printed(&project.run(), b"first");
    let cold = project.service.server.counters();
    let executable = project.executable();
    let modified = fs::metadata(&executable).unwrap().modified().unwrap();
    printed(&project.run(), b"first");
    assert_eq!(project.calls(), 1);
    assert_eq!(project.service.server.counters(), cold);
    assert_eq!(
        fs::metadata(&executable).unwrap().modified().unwrap(),
        modified
    );
    fs::write(
        &project.input,
        "export { main };\nimport { \"$/string.resin\" };\n\n// comment only\n\ndef main() -> () = {\n    print(\"first\");\n};",
    )
    .unwrap();
    printed(&project.run(), b"first");
    assert_eq!(project.calls(), 1);
    assert_eq!(
        project.service.server.counters().syntax_builds,
        cold.syntax_builds + 1
    );
    assert_eq!(
        project.service.server.counters().hir_builds,
        cold.hir_builds + 1
    );
    assert!(!project.temp.path().join("build").exists());
}

#[test]
fn entry_points_have_separate_reusable_artifacts() {
    let mut project = Project::new();
    fs::write(
        &project.input,
        r#"
        export { main, second }; import { "$/string.resin" };
        def main() -> () = { print("first"); };
        def second() -> () = { print("second"); };
    "#,
    )
    .unwrap();
    let original = project.input.clone();
    printed(&project.run(), b"first");
    for (entry, expected) in [
        ("second", &b"second"[..]),
        ("main", &b"first"[..]),
        ("second", &b"second"[..]),
    ] {
        let mut input = original.as_os_str().to_os_string();
        input.push(format!(":{entry}"));
        project.input = input.into();
        printed(&project.run(), expected);
    }
    assert_eq!(project.calls(), 2);
    assert_eq!(project.service.server.counters().hir_builds, 1);
    assert_eq!(project.service.server.counters().verified_builds, 2);
    assert_eq!(project.service.artifacts().len(), 2);
    let mut input = original.as_os_str().to_os_string();
    input.push(":missing");
    project.input = input.into();
    let output = project.run();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert_eq!(project.calls(), 2);
}

#[test]
fn executable_output_optimizes_and_both_profiles_stay_cached() {
    let project = Project::new();
    printed(&project.run(), b"first");
    let debug = project.executable();
    let debug_modified = fs::metadata(&debug).unwrap().modified().unwrap();
    printed(
        &project.command().args(["-o", "dist/"]).output().unwrap(),
        b"",
    );
    let release = project.profile_executable("release");
    let release_modified = fs::metadata(&release).unwrap().modified().unwrap();
    assert_eq!(
        fs::read(&release).unwrap(),
        fs::read(project.temp.path().join("dist/main")).unwrap()
    );
    assert_eq!(project.calls(), 2);

    printed(&project.run(), b"first");
    printed(
        &project.command().args(["--out", "copy"]).output().unwrap(),
        b"",
    );
    let output = project.command().args(["-o", "exported"]).output().unwrap();
    printed(&output, b"");
    assert_eq!(
        fs::read(&release).unwrap(),
        fs::read(project.temp.path().join("exported")).unwrap()
    );
    assert_eq!(
        fs::metadata(debug).unwrap().modified().unwrap(),
        debug_modified
    );
    assert_eq!(
        fs::metadata(release).unwrap().modified().unwrap(),
        release_modified
    );
    assert_eq!(project.calls(), 2);

    assert_eq!(project.service.server.counters().native_object_builds, 2);
    assert_eq!(project.service.server.counters().verified_builds, 1);
}

#[test]
fn native_objects_and_artifacts_stay_on_the_server() {
    let project = Project::new();
    printed(&project.run(), b"first");
    let directory = project.executable().parent().unwrap().to_path_buf();
    assert!(directory.join("input0.o").is_file());
    assert!(!directory.join("main.c").exists());
    assert_eq!(project.calls(), 1);
    assert!(!project.temp.path().join("build").exists());
    assert!(!project.temp.path().join("main.c").exists());
}

#[test]
fn removing_shaders_keeps_the_previous_generation_usable() {
    if shaders::optimizer().is_none() {
        return;
    }
    let project = Project::new();
    let host = fs::read_to_string(&project.input).unwrap();
    fs::write(
        &project.input,
        r#"
        export { main };
        import { "$/gpu.resin" };
        @compute_shader def kernel(index: ulong, output: Ptr<uint>) = {
            output.* := uint(index);
        };
        def main() -> Result<(), _> = {
            if (0 == 1) { Gpu.new()?.create_compute_pipeline(kernel)?; };
            ok(())
        };
        "#,
    )
    .unwrap();
    printed(&project.run(), b"");
    let previous = project.executable();
    let bytes = fs::read(&previous).unwrap();
    let shaders = project.service.server.counters().shader_builds;
    assert_eq!(shaders, 1);
    fs::write(&project.input, host).unwrap();
    printed(&project.run(), b"first");
    assert_eq!(project.service.server.counters().shader_builds, shaders);
    assert_eq!(fs::read(&previous).unwrap(), bytes);
    printed(&Command::new(previous).output().unwrap(), b"");
}

#[test]
fn changed_source_preserves_previous_executable_generation() {
    let project = Project::new();
    printed(&project.run(), b"first");
    let executable = project.executable();
    fs::write(
        &project.input,
        r#"export { main }; import { "$/string.resin" }; def main() -> () = { print("second"); };"#,
    )
    .unwrap();
    printed(&project.run(), b"second");
    assert_eq!(project.calls(), 2);
    assert_ne!(project.executable(), executable);
    printed(&Command::new(executable).output().unwrap(), b"first");
    printed(&project.run(), b"second");
    assert_eq!(project.calls(), 2);
}

#[test]
fn server_compiler_changes_invalidate_while_client_environment_is_ignored() {
    let project = Project::new();
    printed(&project.run(), b"first");
    fs::write(&project.compiler, format!("{WRAPPER}# changed wrapper\n")).unwrap();
    printed(&project.run(), b"first");
    assert_eq!(project.calls(), 2);
    printed(
        &project.command().env("CFLAGS", "changed").output().unwrap(),
        b"first",
    );
    assert_eq!(project.calls(), 2);
}

#[test]
fn compiler_symlinks_preserve_the_invocation_name() {
    let mut project = Project::new();
    let alias = project.temp.path().join("driver");
    let wrapper = WRAPPER.replacen(
        "#!/bin/sh\n",
        "#!/bin/sh\ncase \"$0\" in */driver) ;; *) exit 9 ;; esac\n",
        1,
    );
    fs::write(&project.compiler, wrapper).unwrap();
    std::os::unix::fs::symlink(&project.compiler, &alias).unwrap();
    project.compiler = alias;
    project.configure(|_, _| {});
    printed(&project.run(), b"first");
    printed(&project.run(), b"first");
    assert_eq!(project.calls(), 1);
}

#[test]
fn runtime_headers_and_archive_changes_invalidate_the_cache() {
    let mut project = Project::new();
    let include = project.temp.path().join("include");
    fs::create_dir_all(include.join("resin_runtime")).unwrap();
    fs::copy(
        PathBuf::from(resin_runtime::INCLUDE_DIR).join("resin_runtime.h"),
        include.join("resin_runtime.h"),
    )
    .unwrap();
    for entry in
        fs::read_dir(PathBuf::from(resin_runtime::INCLUDE_DIR).join("resin_runtime")).unwrap()
    {
        let entry = entry.unwrap();
        fs::copy(
            entry.path(),
            include.join("resin_runtime").join(entry.file_name()),
        )
        .unwrap();
    }
    let library = project.temp.path().join("runtime.a");
    let original = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .join("libresin_runtime.a");
    fs::copy(&original, &library).unwrap();
    project.configure(|_, environment| {
        environment.variables.extend([
            ("RESIN_RUNTIME_INCLUDE".into(), include.clone().into()),
            ("RESIN_RUNTIME_LIB".into(), library.clone().into()),
        ]);
    });
    let run = || project.run();
    printed(&run(), b"first");
    printed(&run(), b"first");
    assert_eq!(project.calls(), 1);
    let before = project.service.server.counters();
    let header = include.join("resin_runtime/print.h");
    let mut text = fs::read_to_string(&header).unwrap();
    text.push_str("\n// changed header\n");
    fs::write(header, text).unwrap();
    printed(&run(), b"first");
    let after = project.service.server.counters();
    assert_eq!(after.foreign_builds, before.foreign_builds + 1);
    assert_eq!(after.native_object_builds, before.native_object_builds);
    assert_eq!(
        project.calls(),
        1,
        "a header comment leaves the validated C bindings unchanged"
    );

    fs::write(&library, "not a library").unwrap();
    let output = run();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert_eq!(project.calls(), 2);
    fs::copy(original, &library).unwrap();
    printed(&run(), b"first");
    assert_eq!(
        project.calls(),
        2,
        "restoring the original runtime reuses its retained executable"
    );
}

#[test]
fn failed_rebuilds_preserve_the_old_executable_but_never_run_it() {
    let project = Project::new();
    printed(&project.run(), b"first");
    let executable = project.executable();
    let original = fs::read(&executable).unwrap();
    fs::write(&project.compiler, "#!/bin/sh\nexit 9\n").unwrap();
    for _ in 0..2 {
        let output = project.run();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert_eq!(fs::read(&executable).unwrap(), original);
    }
    fs::write(&project.compiler, WRAPPER).unwrap();
    printed(&project.run(), b"first");
    assert_eq!(
        project.calls(),
        1,
        "restoring the original tool reuses its retained executable"
    );
}

#[test]
fn missing_artifacts_are_relinked_from_retained_objects() {
    let project = Project::new();
    printed(&project.run(), b"first");
    let native = project.service.server.counters().native_object_builds;
    fs::remove_file(project.executable()).unwrap();
    printed(&project.run(), b"first");
    assert!(project.executable().is_file());
    assert_eq!(project.calls(), 2);
    assert_eq!(
        project.service.server.counters().native_object_builds,
        native
    );
}

#[test]
fn concurrent_runs_share_one_build() {
    let project = Project::new();
    let first = project
        .command()
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let second = project
        .command()
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    printed(&first.wait_with_output().unwrap(), b"first");
    printed(&second.wait_with_output().unwrap(), b"first");
    assert_eq!(project.calls(), 1);
    assert!(project.executable().is_file());
}

#[test]
fn all_spirv_is_generated_before_native_tools_run() {
    let mut project = Project::new();
    project.configure(|_, environment| {
        environment
            .variables
            .insert("SPIRV_OPT".into(), "/missing/spirv-opt".into());
    });
    fs::write(
        &project.input,
        r#"
        export { main };
        import { "$/gpu.resin" };
        @compute_shader def good(invocation: ulong, output: Ptr<uint>) = { var i = uint(invocation); output.* := { i + 1_ui }; };
        @compute_shader def bad(invocation: ulong, output: Ptr<uint>) = { var i = uint(invocation); output.* := { i / 2_ui }; };
        def main() -> Result<(), _> = {
            if (0 == 1) {
                var gpu = Gpu.new()?;
                gpu.create_compute_pipeline(good)?;
                gpu.create_compute_pipeline(bad)?;
            };
            ok(())
        };
    "#,
    )
    .unwrap();
    let output = project.run();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("unsupported shader builtin"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!project.temp.path().join("build").exists());
    assert!(!project.temp.path().join("calls").exists());
}

#[test]
fn embedded_builds_resolve_header_dependencies_from_the_compiler_directory() {
    use resin_toolchain::{CProfile, Environment};
    let project = Project::new();
    let header = project.temp.path().join("foreign.h");
    fs::write(&header, "#define VALUE 41\n").unwrap();
    let mut environment = Environment::capture().unwrap();
    environment.directory = project.temp.path().into();
    environment.variables.extend([
        ("CPATH".into(), ".".into()),
        (
            "RESIN_TEST_COUNT".into(),
            project.temp.path().join("calls").into(),
        ),
        (
            "RESIN_TEST_FLAGS".into(),
            project.temp.path().join("flags").into(),
        ),
        (
            "RESIN_TEST_COMPILER".into(),
            std::env::var_os("CC").unwrap_or_else(|| "cc".into()),
        ),
    ]);
    let settings = environment.toolchain(Some(project.compiler.as_os_str()), None);
    let generated = project.temp.path().join("generated");
    fs::create_dir(&generated).unwrap();
    fs::write(
        generated.join("main.c"),
        "#include \"foreign.h\"\nint main(void) { return VALUE; }\n",
    )
    .unwrap();
    fs::write(generated.join("build.ninja"), "include toolchain.ninja\nbuild program: compile_preprocessed_program main.i | toolchain.state native-inputs.state $runtime_library\n").unwrap();
    fs::write(
        generated.join("native-inputs.json"),
        r#"{"translation_units":[{"source":"main.c","preprocessed":"main.i"}]}"#,
    )
    .unwrap();
    let run = || {
        let build = support::frontend::build(
            &settings,
            &generated,
            &project.input.to_string_lossy(),
            "main",
            CProfile::Debug,
        )
        .unwrap();
        Some(support::frontend::run(&build.executable("program").unwrap()).unwrap())
    };
    assert_eq!(run(), Some(41));
    assert_eq!(run(), Some(41));
    assert_eq!(project.calls(), 1);
    fs::write(header, "#define VALUE 42\n").unwrap();
    assert_eq!(run(), Some(42));
    assert_eq!(run(), Some(42));
    assert_eq!(project.calls(), 2);
}
