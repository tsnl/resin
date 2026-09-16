//! Platform access to the C streams shared with ordinary libc input/output calls.
//! Rust's standard streams have separate buffers and cannot replace these handles.

#[cfg(unix)]
unsafe extern "C" {
    #[cfg_attr(target_os = "linux", link_name = "stdin")]
    #[cfg_attr(target_os = "macos", link_name = "__stdinp")]
    static mut C_STDIN: *mut libc::FILE;
    #[cfg_attr(target_os = "linux", link_name = "stdout")]
    #[cfg_attr(target_os = "macos", link_name = "__stdoutp")]
    static mut C_STDOUT: *mut libc::FILE;
}

#[cfg(windows)]
unsafe extern "C" {
    // The supported Windows MSVC target uses the Universal C Runtime.
    fn __acrt_iob_func(index: u32) -> *mut libc::FILE;
}

pub(super) fn stdin() -> *mut libc::FILE {
    #[cfg(unix)]
    unsafe {
        C_STDIN
    }
    #[cfg(windows)]
    unsafe {
        __acrt_iob_func(0)
    }
}

pub(super) fn stdout() -> *mut libc::FILE {
    #[cfg(unix)]
    unsafe {
        C_STDOUT
    }
    #[cfg(windows)]
    unsafe {
        __acrt_iob_func(1)
    }
}
