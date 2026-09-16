//! Validate the real preprocessor's dependency report before publishing captured C.
use crate::{Error, Settings, platform};
use resin_executor::Cancellation;
use std::path::{Component, Path, PathBuf};

pub(super) struct Roots {
    canonical: Vec<PathBuf>,
    system: Vec<PathBuf>,
}

pub(super) async fn allowed_roots(
    settings: &Settings,
    directory: &Path,
    cancellation: &Cancellation,
) -> Result<Roots, Error> {
    let mut command = settings.command(&settings.cc)?;
    // Search-variable user overlays must not expand this compiler-installation allow list.
    // MSVC's INCLUDE remains the toolchain's explicit Windows SDK installation setting.
    for variable in [
        "CPATH",
        "C_INCLUDE_PATH",
        "CPLUS_INCLUDE_PATH",
        "OBJC_INCLUDE_PATH",
    ] {
        command.env_remove(variable);
    }
    command
        .args(platform::C_FLAGS)
        .args(["-E", "-x", "c", "-v", "-"]);
    command.env("LC_ALL", "C").env("LANG", "C");
    let (_, diagnostic) =
        crate::process::capture(command, "compiler include-root discovery", cancellation).await?;
    let mut roots = vec![
        tokio::fs::canonicalize(directory).await?,
        tokio::fs::canonicalize(&settings.runtime_include).await?,
    ];
    let mut system = Vec::new();
    for path in search_roots(&diagnostic)? {
        cancellation.check()?;
        let declared = settings.directory.join(path);
        let path = tokio::fs::canonicalize(&declared).await?;
        if path.parent().is_none() {
            return Err(Error::new(
                "compiler reported a filesystem root as a system include directory".into(),
            ));
        }
        system.push(declared);
        roots.push(path);
    }
    roots.sort();
    roots.dedup();
    Ok(Roots {
        canonical: roots,
        system,
    })
}

fn search_roots(diagnostic: &[u8]) -> Result<Vec<PathBuf>, Error> {
    let diagnostic = std::str::from_utf8(diagnostic)
        .map_err(|_| Error::new("compiler include-root report is not UTF-8".into()))?;
    let mut active = false;
    let mut complete = false;
    let mut roots = Vec::new();
    for line in diagnostic.lines() {
        let line = line.trim();
        if line == "#include <...> search starts here:" {
            active = true;
        } else if active && line == "End of search list." {
            complete = true;
            break;
        } else if active {
            let line = line.strip_suffix(" (framework directory)").unwrap_or(line);
            if !line.is_empty() {
                roots.push(PathBuf::from(line));
            }
        }
    }
    if !complete || roots.is_empty() {
        return Err(Error::new(
            "compiler did not report its system include search roots".into(),
        ));
    }
    Ok(roots)
}

pub(super) async fn validate(
    depfile: &Path,
    directory: &Path,
    roots: &Roots,
    cancellation: &Cancellation,
) -> Result<(), Error> {
    let bytes = tokio::fs::read(depfile).await?;
    for path in dependencies(&bytes)? {
        cancellation.check()?;
        let original = directory.join(path);
        let resolved = tokio::fs::canonicalize(&original).await?;
        // Trusted SDKs can contain symlinked header subtrees (e.g. glibc's Linux
        // headers in Nix). Admit that declared route, without trusting symlinks
        // inside uploaded staging or widening permission to its destination tree.
        let system_alias = !original
            .components()
            .any(|part| part == Component::ParentDir)
            && roots.system.iter().any(|root| original.starts_with(root));
        if !system_alias
            && !roots
                .canonical
                .iter()
                .any(|root| resolved.starts_with(root))
        {
            return Err(Error::new(format!(
                "native header dependency is outside staged inputs and configured toolchain roots: {}",
                resolved.display()
            )));
        }
    }
    Ok(())
}

// GCC and Clang emit Make dependency escaping, independent of C #line directives.
// The fixed -MT target avoids parsing drive-letter colons or caller target names.
pub(super) fn dependencies(bytes: &[u8]) -> Result<Vec<PathBuf>, Error> {
    let body = bytes
        .strip_prefix(b"resin-input:")
        .ok_or_else(|| Error::new("compiler emitted an invalid dependency report".into()))?;
    let mut names = Vec::new();
    let mut name = Vec::new();
    let mut position = 0;
    while position < body.len() {
        let byte = body[position];
        match byte {
            b'\\' if body.get(position + 1) == Some(&b'\n') => position += 1,
            b'\\'
                if body.get(position + 1) == Some(&b'\r')
                    && body.get(position + 2) == Some(&b'\n') =>
            {
                position += 2
            }
            b'\\'
                if body.get(position + 1).is_some_and(|next| {
                    next.is_ascii_whitespace() || matches!(next, b'#' | b'\\')
                }) =>
            {
                position += 1;
                name.push(body[position]);
            }
            b'$' if body.get(position + 1) == Some(&b'$') => {
                position += 1;
                name.push(b'$');
            }
            byte if byte.is_ascii_whitespace() => finish_name(&mut names, &mut name)?,
            byte => name.push(byte),
        }
        position += 1;
    }
    finish_name(&mut names, &mut name)?;
    if names.is_empty() {
        return Err(Error::new(
            "compiler emitted an empty dependency report".into(),
        ));
    }
    Ok(names)
}

fn finish_name(names: &mut Vec<PathBuf>, name: &mut Vec<u8>) -> Result<(), Error> {
    if !name.is_empty() {
        let text = String::from_utf8(std::mem::take(name))
            .map_err(|_| Error::new("compiler dependency path is not UTF-8".into()))?;
        names.push(PathBuf::from(text));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiler_search_reports_require_complete_explicit_system_roots() {
        assert_eq!(search_roots(b"compiler banner\n#include \"...\" search starts here:\n ignored-quote-root\n#include <...> search starts here:\n /compiler/include\n /SDK/Frameworks (framework directory)\nEnd of search list.\n").unwrap(), vec![PathBuf::from("/compiler/include"), PathBuf::from("/SDK/Frameworks")]);
        assert!(search_roots(b"#include <...> search starts here:\n /truncated").is_err());
    }

    #[test]
    fn compiler_dependency_paths_decode_make_escaping_and_continuations() {
        assert_eq!(dependencies(b"resin-input: main.c path\\ with\\ space.h \\\n dollar$$name.h escaped\\#name.h C:/sdk/include.h\n").unwrap(), ["main.c", "path with space.h", "dollar$name.h", "escaped#name.h", "C:/sdk/include.h"].map(PathBuf::from));
        assert!(dependencies(b"unexpected-target: header.h").is_err());
    }
}

#[cfg(all(test, unix))]
mod aliases {
    use super::*;
    #[tokio::test]
    async fn sdk_symlinks_are_allowed_without_allowing_staged_symlink_escape() {
        let temporary = tempfile::tempdir().unwrap();
        let sdk = temporary.path().join("sdk");
        let staging = temporary.path().join("staging");
        let external = temporary.path().join("kernel");
        for path in [&sdk, &staging, &external] {
            tokio::fs::create_dir(path).await.unwrap();
        }
        tokio::fs::write(external.join("errno.h"), b"#define ERRNO 1")
            .await
            .unwrap();
        std::os::unix::fs::symlink(&external, sdk.join("linux")).unwrap();
        std::os::unix::fs::symlink(&external, staging.join("escape")).unwrap();
        let roots = Roots {
            canonical: vec![sdk.clone(), staging.clone()],
            system: vec![sdk.clone()],
        };
        let dependency = staging.join("input.d");
        for (path, allowed) in [
            (sdk.join("linux/errno.h"), true),
            (staging.join("escape/errno.h"), false),
            (external.join("errno.h"), false),
        ] {
            tokio::fs::write(&dependency, format!("resin-input: {}\n", path.display()))
                .await
                .unwrap();
            assert_eq!(
                validate(&dependency, &staging, &roots, &Cancellation::new())
                    .await
                    .is_ok(),
                allowed
            );
        }
    }
}
