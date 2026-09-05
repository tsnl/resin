use std::{
    ffi::{CStr, c_char, c_void},
    sync::Mutex,
};

static ALLOCATIONS: Mutex<Vec<usize>> = Mutex::new(Vec::new());

pub(crate) fn fail(message: &str) -> ! {
    eprintln!("resin: {message}");
    std::process::exit(1);
}

/// # Safety
/// `message` must point to a valid NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_fail(message: *const c_char) -> ! {
    fail(&unsafe { CStr::from_ptr(message) }.to_string_lossy());
}

#[unsafe(no_mangle)]
pub extern "C" fn resin_alloc(size: usize) -> *mut c_void {
    let allocation = unsafe { libc::calloc(1, size.max(1)) };
    if allocation.is_null() {
        fail("out of memory");
    }
    ALLOCATIONS.lock().unwrap().push(allocation as usize);
    allocation
}

/// # Safety
/// No allocation from `resin_alloc` may be accessed during or after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_cleanup() {
    let allocations = std::mem::take(&mut *ALLOCATIONS.lock().unwrap());
    for address in allocations {
        unsafe { libc::free(address as *mut c_void) };
    }
}

macro_rules! signed {
    ($($name:ident: $ty:ty),* $(,)?) => {$(
        #[unsafe(no_mangle)]
        pub extern "C" fn $name(value: u64) -> $ty { value as $ty }
    )*};
}

signed!(resin_i8: i8, resin_i16: i16, resin_i32: i32, resin_i64: i64);

#[unsafe(no_mangle)]
pub extern "C" fn resin_idiv(a: i64, b: i64) -> i64 {
    if b == 0 {
        fail("division by zero");
    }
    a.wrapping_div(b)
}

#[unsafe(no_mangle)]
pub extern "C" fn resin_imod(a: i64, b: i64) -> i64 {
    if b == 0 {
        fail("division by zero");
    }
    a.wrapping_rem(b)
}

#[unsafe(no_mangle)]
pub extern "C" fn resin_udiv(a: u64, b: u64) -> u64 {
    if b == 0 {
        fail("division by zero");
    }
    a / b
}

#[unsafe(no_mangle)]
pub extern "C" fn resin_umod(a: u64, b: u64) -> u64 {
    if b == 0 {
        fail("division by zero");
    }
    a % b
}

#[unsafe(no_mangle)]
pub extern "C" fn resin_shift(count: u64, width: u32) -> u32 {
    if count >= u64::from(width) {
        fail("shift count out of range");
    }
    count as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn resin_index(index: u64, length: usize) -> usize {
    if index >= length as u64 {
        fail("array index out of bounds");
    }
    index as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_arithmetic_wraps() {
        assert_eq!(resin_i8(128), -128);
        assert_eq!(resin_i16(32768), -32768);
        assert_eq!(resin_i32(2147483648), i32::MIN);
        assert_eq!(resin_i64(u64::MAX), -1);
        assert_eq!(resin_idiv(i64::MIN, -1), i64::MIN);
        assert_eq!(resin_imod(i64::MIN, -1), 0);
    }
}
