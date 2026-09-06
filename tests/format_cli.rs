use resin::toolchain::TempDir;
use std::{
    fs,
    path::Path,
    process::{Command, Output},
};

fn fmt(root: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_resin"))
        .current_dir(root)
        .arg("fmt")
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn recursive_check_is_read_only_and_formatting_is_idempotent() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let root = temp.path();
    fs::create_dir_all(root.join("examples/nested")).unwrap();
    let a = "def main()={var x=[1,2,];unknown(x);};";
    let b = "// 😀\ndef f()={};\r\n";
    fs::write(root.join("examples/a.resin"), a).unwrap();
    fs::write(root.join("examples/nested/b.resin"), b).unwrap();
    fs::write(root.join("examples/readme.txt"), "leave me alone").unwrap();
    let check = fmt(root, &["--check", "examples"]);
    assert_eq!(check.status.code(), Some(1));
    assert!(check.stderr.is_empty());
    assert_eq!(String::from_utf8(check.stdout).unwrap().lines().count(), 2);
    assert_eq!(
        fs::read_to_string(root.join("examples/a.resin")).unwrap(),
        a
    );
    assert_eq!(
        fs::read_to_string(root.join("examples/nested/b.resin")).unwrap(),
        b
    );
    let result = fmt(root, &["examples", "examples/a.resin"]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        String::from_utf8(result.stdout).unwrap().lines().count(),
        2,
        "duplicate paths are formatted once"
    );
    for (name, input) in [("examples/a.resin", a), ("examples/nested/b.resin", b)] {
        assert_eq!(
            fs::read_to_string(root.join(name)).unwrap(),
            resin::formatting::format_source(input).unwrap()
        );
    }
    for args in [&["--check", "examples"][..], &["examples"][..]] {
        let result = fmt(root, args);
        assert!(result.status.success());
        assert!(result.stdout.is_empty());
        assert!(result.stderr.is_empty());
    }
    assert_eq!(
        fs::read_to_string(root.join("examples/readme.txt")).unwrap(),
        "leave me alone"
    );
    assert!(
        !root.join("build").exists(),
        "formatting must not compile programs"
    );
}

#[test]
fn errors_preserve_invalid_files_and_do_not_hide_other_inputs() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let root = temp.path();
    let invalid = "def broken() = { var x = ; };";
    fs::write(root.join("invalid.resin"), invalid).unwrap();
    fs::write(root.join("valid.resin"), "def f()={};").unwrap();
    fs::write(root.join("binary.resin"), [0xff, 0xfe]).unwrap();
    let result = fmt(
        root,
        &[
            "missing.resin",
            "invalid.resin",
            "binary.resin",
            "valid.resin",
        ],
    );
    assert_eq!(result.status.code(), Some(1));
    let errors = String::from_utf8(result.stderr).unwrap();
    for path in ["missing.resin", "invalid.resin", "binary.resin"] {
        assert!(errors.contains(path), "{errors}");
    }
    assert!(errors.contains("syntax errors"));
    assert_eq!(
        fs::read_to_string(root.join("invalid.resin")).unwrap(),
        invalid
    );
    assert_eq!(fs::read(root.join("binary.resin")).unwrap(), [0xff, 0xfe]);
    assert_eq!(
        fs::read_to_string(root.join("valid.resin")).unwrap(),
        "def f() = {};\n"
    );
}

#[test]
fn help_paths_and_legacy_invocation() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let root = temp.path();
    let help = fmt(root, &["--help"]);
    assert!(help.status.success());
    assert!(String::from_utf8(help.stdout).unwrap().contains("--check"));
    assert_eq!(fmt(root, &[]).status.code(), Some(2));
    assert_eq!(fmt(root, &["--unknown"]).status.code(), Some(2));
    for name in ["with spaces.resin", "-dash.resin", "fmt"] {
        fs::write(root.join(name), "def main()={};").unwrap();
    }
    assert!(
        fmt(root, &["--", "with spaces.resin", "-dash.resin", "fmt"])
            .status
            .success()
    );
    let legacy = Command::new(env!("CARGO_BIN_EXE_resin"))
        .current_dir(root)
        .args(["--output", "check", "--", "fmt"])
        .output()
        .unwrap();
    assert!(
        legacy.status.success(),
        "{}",
        String::from_utf8_lossy(&legacy.stderr)
    );
}

#[cfg(unix)]
#[test]
fn skips_symlinks_preserves_modes_and_accepts_non_utf8_paths() {
    use std::os::unix::{
        ffi::OsStringExt,
        fs::{PermissionsExt, symlink},
    };
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let root = temp.path();
    fs::create_dir(root.join("examples")).unwrap();
    let raw = "def main()={};";
    fs::write(root.join("outside.resin"), raw).unwrap();
    symlink("../outside.resin", root.join("examples/link.resin")).unwrap();
    symlink(".", root.join("examples/cycle")).unwrap();
    let odd = root.join("examples").join(std::ffi::OsString::from_vec(
        b"non-utf8-\xff.resin".to_vec(),
    ));
    fs::write(&odd, raw).unwrap();
    fs::set_permissions(&odd, fs::Permissions::from_mode(0o640)).unwrap();
    let result = fmt(root, &["examples"]);
    assert!(result.status.success());
    assert_eq!(fs::read_to_string(root.join("outside.resin")).unwrap(), raw);
    assert!(
        fs::symlink_metadata(root.join("examples/link.resin"))
            .unwrap()
            .is_symlink()
    );
    assert_eq!(fs::read_to_string(&odd).unwrap(), "def main() = {};\n");
    assert_eq!(
        fs::metadata(&odd).unwrap().permissions().mode() & 0o777,
        0o640
    );
    assert_eq!(fmt(root, &["examples/link.resin"]).status.code(), Some(1));
    fs::write(&odd, raw).unwrap();
    fs::set_permissions(&odd, fs::Permissions::from_mode(0o440)).unwrap();
    assert_eq!(fmt(root, &["examples"]).status.code(), Some(1));
    assert_eq!(fs::read_to_string(&odd).unwrap(), raw);
}
