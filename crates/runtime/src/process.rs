//! Startup arguments and environment, copied into Resin-owned process-lifetime storage.
use std::ffi::{CStr, c_char};
use std::ptr;

use crate::host::{fail, resin_alloc};

fn strings(values: &[Vec<u8>]) -> *mut *mut c_char {
    let bytes = values
        .len()
        .checked_add(1)
        .and_then(|n| n.checked_mul(size_of::<*mut c_char>()))
        .unwrap_or_else(|| fail("process input too large"));
    let pointers = resin_alloc(bytes).cast::<*mut c_char>();
    for (index, value) in values.iter().enumerate() {
        let length = value
            .len()
            .checked_add(1)
            .unwrap_or_else(|| fail("process input too large"));
        let text = resin_alloc(length).cast::<c_char>();
        unsafe {
            ptr::copy_nonoverlapping(value.as_ptr(), text.cast::<u8>(), value.len());
            *pointers.add(index) = text;
        }
    }
    // resin_alloc zeroes both the strings' terminators and the array sentinel.
    pointers
}

fn os_bytes(value: std::ffi::OsString) -> Vec<u8> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        value.into_vec()
    }
    #[cfg(windows)]
    {
        value.to_string_lossy().into_owned().into_bytes()
    }
}

/// # Safety
/// `argv` must hold `argc` readable NUL-terminated strings; the two output pointers
/// must be writable. Call once at startup, before user code or foreign threads can
/// mutate the process environment. Results live until resin_cleanup and are borrowed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_process_init(
    argc: i32,
    argv: *const *const c_char,
    out_argv: *mut *mut *mut c_char,
    out_envp: *mut *mut *mut c_char,
) -> i32 {
    if argc < 0 || (argc > 0 && argv.is_null()) || out_argv.is_null() || out_envp.is_null() {
        fail("invalid process inputs");
    }
    #[cfg(unix)]
    let arguments: Vec<Vec<u8>> = (0..argc as usize)
        .map(|i| {
            let value = unsafe { *argv.add(i) };
            if value.is_null() {
                fail("null process argument");
            }
            unsafe { CStr::from_ptr(value) }.to_bytes().to_vec()
        })
        .collect();
    // Read the native wide command line, independently of the CRT's ANSI code page.
    #[cfg(windows)]
    let arguments: Vec<Vec<u8>> = std::env::args_os().map(os_bytes).collect();
    let environment: Vec<Vec<u8>> = std::env::vars_os()
        .map(|(key, value)| {
            let mut entry = os_bytes(key);
            entry.push(b'=');
            entry.extend(os_bytes(value));
            entry
        })
        .collect();
    let count =
        i32::try_from(arguments.len()).unwrap_or_else(|_| fail("too many process arguments"));
    unsafe {
        *out_argv = strings(&arguments);
        *out_envp = strings(&environment);
    }
    count
}

/// # Safety
/// `values` must be a readable, null-terminated pointer array, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_string_array_length(values: *const *const c_char) -> usize {
    if values.is_null() {
        return 0;
    }
    let mut length = 0;
    while !unsafe { *values.add(length) }.is_null() {
        length += 1;
    }
    length
}

/// # Safety
/// `envp` must be a readable null-terminated string array. `name` must be a readable
/// NUL-terminated string. Returns a borrowed value or null when absent/invalid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_environment_get(
    envp: *const *const c_char,
    name: *const c_char,
) -> *const c_char {
    if envp.is_null() || name.is_null() {
        return ptr::null();
    }
    let name = unsafe { CStr::from_ptr(name) }.to_bytes();
    if name.is_empty() || name.contains(&b'=') {
        return ptr::null();
    }
    let mut index = 0;
    loop {
        let entry = unsafe { *envp.add(index) };
        if entry.is_null() {
            return ptr::null();
        }
        let bytes = unsafe { CStr::from_ptr(entry) }.to_bytes();
        if bytes.len() > name.len() && bytes[name.len()] == b'=' && &bytes[..name.len()] == name {
            return unsafe { entry.add(name.len() + 1) };
        }
        index += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_uses_exact_names_and_preserves_empty_values_and_equals() {
        let entries = [
            c"PREFIX_LONG=wrong".as_ptr(),
            c"PREFIX=a=b".as_ptr(),
            c"EMPTY=".as_ptr(),
            ptr::null(),
        ];
        unsafe {
            assert_eq!(resin_string_array_length(entries.as_ptr()), 3);
            assert_eq!(resin_string_array_length(ptr::null()), 0);
            assert_eq!(
                CStr::from_ptr(resin_environment_get(entries.as_ptr(), c"PREFIX".as_ptr())),
                c"a=b"
            );
            assert_eq!(
                CStr::from_ptr(resin_environment_get(entries.as_ptr(), c"EMPTY".as_ptr())),
                c""
            );
            for name in [c"prefix", c"PRE", c"MISSING", c"", c"PREFIX="] {
                assert!(resin_environment_get(entries.as_ptr(), name.as_ptr()).is_null());
            }
            assert!(resin_environment_get(ptr::null(), c"PREFIX".as_ptr()).is_null());
            assert!(resin_environment_get(entries.as_ptr(), ptr::null()).is_null());
        }
    }
}
