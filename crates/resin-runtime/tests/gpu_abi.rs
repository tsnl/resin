use resin_runtime::*;
use std::ffi::c_void;
use std::io::Write;
use std::mem::{align_of, offset_of, size_of};
use std::process::{Command, Stdio};
use std::ptr;

#[test]
fn compiler_operations_have_only_pointer_and_scalar_abi_arguments() {
    let _: unsafe extern "C" fn(*const ResinGpuPtr, usize, usize, u32) -> *mut c_void =
        resin_gpu_ptr_host_ref;
    let _: unsafe extern "C" fn(*const ResinGpuPtr, usize, usize, usize, *mut ResinGpuPtr) =
        resin_gpu_ptr_offset_into;
    let _: unsafe extern "C" fn(*const ResinGpuPtr) -> *mut ResinArc = resin_gpu_projection_new_ref;
    let _: unsafe extern "C" fn(
        *mut ResinArc,
        *const ResinGpuPtr,
        usize,
        usize,
    ) -> ResinDeviceAddress = resin_gpu_projection_pointer_ref;
    let _: unsafe extern "C" fn(
        *mut ResinCommandBuffer,
        *mut ResinImage,
        *const ResinGpuSpan,
    ) -> ResinStatus = resin_gpu_copy_image_to_span_ref;
}

#[test]
fn c_header_signatures_and_view_layouts_match_the_rust_abi() {
    let mut source = String::from(
        r#"#include <resin_runtime/gpu.h>
typedef void *(*Host)(const ResinGpuPtr *, size_t, size_t, uint32_t);
typedef void (*Offset)(const ResinGpuPtr *, size_t, size_t, size_t, ResinGpuPtr *);
typedef ResinArc *(*Projection)(const ResinGpuPtr *);
typedef ResinDeviceAddress (*Pointer)(ResinArc *, const ResinGpuPtr *, size_t, size_t);
typedef ResinStatus (*Copy)(ResinCommandBuffer *, ResinImage *, const ResinGpuSpan *);
_Static_assert(_Generic(&resin_gpu_ptr_host_ref, Host: 1, default: 0), "host signature");
_Static_assert(_Generic(&resin_gpu_ptr_offset_into, Offset: 1, default: 0), "offset signature");
_Static_assert(_Generic(&resin_gpu_projection_new_ref, Projection: 1, default: 0), "projection signature");
_Static_assert(_Generic(&resin_gpu_projection_pointer_ref, Pointer: 1, default: 0), "pointer signature");
_Static_assert(_Generic(&resin_gpu_copy_image_to_span_ref, Copy: 1, default: 0), "copy signature");
"#,
    );
    for (expression, expected) in [
        ("sizeof(ResinGpuPtr)", size_of::<ResinGpuPtr>()),
        ("_Alignof(ResinGpuPtr)", align_of::<ResinGpuPtr>()),
        (
            "offsetof(ResinGpuPtr, owner)",
            offset_of!(ResinGpuPtr, owner),
        ),
        (
            "offsetof(ResinGpuPtr, offset)",
            offset_of!(ResinGpuPtr, offset),
        ),
        (
            "offsetof(ResinGpuPtr, access)",
            offset_of!(ResinGpuPtr, access),
        ),
        ("sizeof(ResinGpuSpan)", size_of::<ResinGpuSpan>()),
        ("_Alignof(ResinGpuSpan)", align_of::<ResinGpuSpan>()),
        (
            "offsetof(ResinGpuSpan, data)",
            offset_of!(ResinGpuSpan, data),
        ),
        (
            "offsetof(ResinGpuSpan, length)",
            offset_of!(ResinGpuSpan, length),
        ),
        ("sizeof(ResinStatus)", size_of::<ResinStatus>()),
        (
            "sizeof(ResinDeviceAddress)",
            size_of::<ResinDeviceAddress>(),
        ),
    ] {
        use std::fmt::Write;
        writeln!(
            source,
            "_Static_assert({expression} == {expected}, \"{expression}\");"
        )
        .unwrap();
    }
    let compiler =
        std::env::var_os("CC").unwrap_or_else(|| if cfg!(windows) { "clang" } else { "cc" }.into());
    let mut child = Command::new(compiler)
        .args([
            "-std=c11",
            "-pedantic-errors",
            "-Wall",
            "-Wextra",
            "-Werror",
        ])
        .args(["-x", "c", "-fsyntax-only", "-I"])
        .arg(INCLUDE_DIR)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start the configured C compiler for the public ABI check");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(source.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn empty_view() -> ResinGpuPtr {
    ResinGpuPtr {
        owner: ptr::null_mut(),
        offset: 17,
        access: RESIN_GPU_ACCESS_READ | RESIN_GPU_ACCESS_WRITE,
    }
}

#[test]
fn failed_copy_borrows_the_span_and_preserves_status_and_input() {
    let span = ResinGpuSpan {
        data: empty_view(),
        length: 31,
    };
    let original = (
        span.data.owner,
        span.data.offset,
        span.data.access,
        span.length,
    );
    // Null commands and image are explicitly rejected before using their objects.
    let legacy = unsafe { resin_gpu_copy_image_to_span(ptr::null_mut(), ptr::null_mut(), span) };
    let indirect =
        unsafe { resin_gpu_copy_image_to_span_ref(ptr::null_mut(), ptr::null_mut(), &span) };
    assert_eq!(legacy, ResinStatus::InvalidArgument);
    assert_eq!(indirect, legacy);
    assert_eq!(
        (
            span.data.owner,
            span.data.offset,
            span.data.access,
            span.length
        ),
        original
    );
}

#[test]
fn empty_owners_and_overflow_keep_existing_failures_without_a_gpu() {
    for (operation, message) in [
        ("host", "access through an empty GpuPtr"),
        ("offset", "access through an empty GpuPtr"),
        ("offset_overflow", "GPU pointer offset overflow"),
        ("projection", "access through an empty GpuPtr"),
        ("pointer", "access through empty GPU arguments"),
    ] {
        let mut diagnostics = Vec::new();
        for variant in ["value", "pointer"] {
            let output = Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "gpu_failure_child", "--nocapture"])
                .env("RESIN_GPU_ABI_FAILURE", format!("{operation}:{variant}"))
                .output()
                .unwrap();
            assert_eq!(output.status.code(), Some(1), "{operation}:{variant}");
            let stderr = String::from_utf8(output.stderr).unwrap();
            assert!(stderr.contains(message), "{operation}:{variant}: {stderr}");
            diagnostics.push(stderr);
        }
        assert_eq!(diagnostics[0], diagnostics[1], "{operation}");
    }
}

#[test]
fn gpu_failure_child() {
    let Ok(case) = std::env::var("RESIN_GPU_ABI_FAILURE") else {
        return;
    };
    let (operation, variant) = case.split_once(':').unwrap();
    let indirect = variant == "pointer";
    let mut value = empty_view();
    if operation == "offset_overflow" {
        value.offset = usize::MAX;
    }
    // Empty owners/projections are checked errors in the existing private routines.
    // Struct pointers themselves remain initialized, aligned and non-null.
    unsafe {
        match operation {
            "host" if indirect => {
                resin_gpu_ptr_host_ref(&value, 1, 1, RESIN_GPU_ACCESS_READ);
            }
            "host" => {
                resin_gpu_ptr_host(value, 1, 1, RESIN_GPU_ACCESS_READ);
            }
            "offset" | "offset_overflow" if indirect => {
                let pointer = &mut value as *mut ResinGpuPtr;
                resin_gpu_ptr_offset_into(pointer, 1, 1, 1, pointer);
            }
            "offset" | "offset_overflow" => {
                resin_gpu_ptr_offset(value, 1, 1, 1);
            }
            "projection" if indirect => {
                resin_gpu_projection_new_ref(&value);
            }
            "projection" => {
                resin_gpu_projection_new(value);
            }
            "pointer" if indirect => {
                resin_gpu_projection_pointer_ref(ptr::null_mut(), &value, 1, 1);
            }
            "pointer" => {
                resin_gpu_projection_pointer(ptr::null_mut(), value, 1, 1);
            }
            _ => panic!("unknown failure case {case}"),
        }
    }
    panic!("{case} unexpectedly returned");
}
