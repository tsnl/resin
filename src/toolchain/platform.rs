#[cfg(not(windows))]
pub const DEFAULT_C_COMPILER: &str = "cc";
#[cfg(all(windows, target_env = "msvc"))]
pub const DEFAULT_C_COMPILER: &str = "clang";
#[cfg(all(windows, not(target_env = "msvc")))]
pub const DEFAULT_C_COMPILER: &str = "gcc";

#[cfg(not(target_env = "msvc"))]
pub(super) const RUNTIME_ARCHIVE: &str = "libresin_runtime.a";
#[cfg(target_env = "msvc")]
pub(super) const RUNTIME_ARCHIVE: &str = "resin_runtime.lib";

#[cfg(not(target_env = "msvc"))]
pub(super) const C_FLAGS: &[&str] = &[];
#[cfg(target_env = "msvc")]
pub(super) const C_FLAGS: &[&str] = &["-fms-runtime-lib=dll", "-D_CRT_SECURE_NO_WARNINGS"];

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

#[cfg(windows)]
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
