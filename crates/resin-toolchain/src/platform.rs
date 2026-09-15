#[cfg(not(target_env = "msvc"))]
pub(crate) const RUNTIME_ARCHIVE: &str = "libresin_runtime.a";
#[cfg(target_env = "msvc")]
pub(crate) const RUNTIME_ARCHIVE: &str = "resin_runtime.lib";

#[cfg(not(target_env = "msvc"))]
pub(super) const C_FLAGS: &[&str] = &[];
#[cfg(target_env = "msvc")]
pub(super) const C_FLAGS: &[&str] = &["-fms-runtime-lib=dll"];

#[cfg(not(target_env = "msvc"))]
pub(super) const PREPROCESSING_FLAGS: &[&str] = &[];
#[cfg(target_env = "msvc")]
pub(super) const PREPROCESSING_FLAGS: &[&str] = &["-D_CRT_SECURE_NO_WARNINGS"];

#[cfg(not(target_env = "msvc"))]
pub(super) const LINK_FLAGS: &[&str] = &[];
#[cfg(target_env = "msvc")]
// Clang's GNU driver still injects libcmt when selecting the DLL runtime.
pub(super) const LINK_FLAGS: &[&str] = &["-Wl,/NODEFAULTLIB:libcmt"];

#[cfg(target_os = "linux")]
pub(super) const LIBRARIES: &[&str] = &["-ldl", "-lpthread", "-lm", "-lrt", "-lutil"];

#[cfg(target_os = "macos")]
pub(super) const LIBRARIES: &[&str] = &[
    "-liconv",
    "-framework",
    "Cocoa",
    "-framework",
    "IOKit",
    "-framework",
    "CoreFoundation",
    "-framework",
    "OpenGL",
    "-framework",
    "QuartzCore",
];

#[cfg(all(windows, target_env = "msvc"))]
pub(super) const LIBRARIES: &[&str] = &[
    "-llegacy_stdio_definitions",
    "-lgdi32",
    "-lopengl32",
    "-luser32",
    "-lshell32",
    "-lkernel32",
    "-lntdll",
    "-luserenv",
    "-lws2_32",
    "-ldbghelp",
];

#[cfg(all(windows, not(target_env = "msvc")))]
pub(super) const LIBRARIES: &[&str] = &[
    "-ladvapi32",
    "-lbcrypt",
    "-lkernel32",
    "-lntdll",
    "-luserenv",
    "-lws2_32",
    "-lsynchronization",
    "-lgdi32",
    "-lopengl32",
    "-luser32",
    "-lshell32",
];
