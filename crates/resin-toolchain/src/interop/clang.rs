//! Load an explicit CIndex library without reading or mutating process-global settings.
use crate::{Error, ForeignDeclaration, ForeignFunction, ForeignScalar};
use clang_sys::*;
use resin_executor::Cancellation;
use std::{
    collections::BTreeMap,
    ffi::{CStr, CString, c_char, c_int, c_uint},
    path::Path,
    ptr,
};

struct Clang {
    _library: libloading::Library,
    create_index: unsafe extern "C" fn(c_int, c_int) -> CXIndex,
    dispose_index: unsafe extern "C" fn(CXIndex),
    parse: unsafe extern "C" fn(
        CXIndex,
        *const c_char,
        *const *const c_char,
        c_int,
        *mut CXUnsavedFile,
        c_uint,
        c_uint,
    ) -> CXTranslationUnit,
    dispose_unit: unsafe extern "C" fn(CXTranslationUnit),
    cursor: unsafe extern "C" fn(CXTranslationUnit) -> CXCursor,
    visit: unsafe extern "C" fn(CXCursor, CXCursorVisitor, CXClientData) -> c_uint,
    spelling: unsafe extern "C" fn(CXCursor) -> CXString,
    mangling: unsafe extern "C" fn(CXCursor) -> CXString,
    linkage: unsafe extern "C" fn(CXCursor) -> CXLinkageKind,
    definition: unsafe extern "C" fn(CXCursor) -> c_uint,
    cursor_type: unsafe extern "C" fn(CXCursor) -> CXType,
    canonical_type: unsafe extern "C" fn(CXType) -> CXType,
    type_declaration: unsafe extern "C" fn(CXType) -> CXCursor,
    enum_integer_type: unsafe extern "C" fn(CXCursor) -> CXType,
    type_spelling: unsafe extern "C" fn(CXType) -> CXString,
    result_type: unsafe extern "C" fn(CXType) -> CXType,
    argument_type: unsafe extern "C" fn(CXType, c_uint) -> CXType,
    argument_count: unsafe extern "C" fn(CXType) -> c_int,
    calling_convention: unsafe extern "C" fn(CXType) -> CXCallingConv,
    variadic: unsafe extern "C" fn(CXType) -> c_uint,
    size: unsafe extern "C" fn(CXType) -> i64,
    inlined: unsafe extern "C" fn(CXCursor) -> c_uint,
    string: unsafe extern "C" fn(CXString) -> *const c_char,
    dispose_string: unsafe extern "C" fn(CXString),
    diagnostic_count: unsafe extern "C" fn(CXTranslationUnit) -> c_uint,
    diagnostic: unsafe extern "C" fn(CXTranslationUnit, c_uint) -> CXDiagnostic,
    severity: unsafe extern "C" fn(CXDiagnostic) -> CXDiagnosticSeverity,
    format_diagnostic: unsafe extern "C" fn(CXDiagnostic, c_uint) -> CXString,
    dispose_diagnostic: unsafe extern "C" fn(CXDiagnostic),
}

impl Clang {
    unsafe fn load(path: &Path) -> Result<Self, Error> {
        unsafe {
            let library = libloading::Library::new(path).map_err(error)?;
            Ok(Self {
                create_index: *library.get(b"clang_createIndex\0").map_err(error)?,
                dispose_index: *library.get(b"clang_disposeIndex\0").map_err(error)?,
                parse: *library
                    .get(b"clang_parseTranslationUnit\0")
                    .map_err(error)?,
                dispose_unit: *library
                    .get(b"clang_disposeTranslationUnit\0")
                    .map_err(error)?,
                cursor: *library
                    .get(b"clang_getTranslationUnitCursor\0")
                    .map_err(error)?,
                visit: *library.get(b"clang_visitChildren\0").map_err(error)?,
                spelling: *library.get(b"clang_getCursorSpelling\0").map_err(error)?,
                mangling: *library.get(b"clang_Cursor_getMangling\0").map_err(error)?,
                linkage: *library.get(b"clang_getCursorLinkage\0").map_err(error)?,
                definition: *library.get(b"clang_isCursorDefinition\0").map_err(error)?,
                cursor_type: *library.get(b"clang_getCursorType\0").map_err(error)?,
                canonical_type: *library.get(b"clang_getCanonicalType\0").map_err(error)?,
                type_declaration: *library.get(b"clang_getTypeDeclaration\0").map_err(error)?,
                enum_integer_type: *library
                    .get(b"clang_getEnumDeclIntegerType\0")
                    .map_err(error)?,
                type_spelling: *library.get(b"clang_getTypeSpelling\0").map_err(error)?,
                result_type: *library.get(b"clang_getResultType\0").map_err(error)?,
                argument_type: *library.get(b"clang_getArgType\0").map_err(error)?,
                argument_count: *library.get(b"clang_getNumArgTypes\0").map_err(error)?,
                calling_convention: *library
                    .get(b"clang_getFunctionTypeCallingConv\0")
                    .map_err(error)?,
                variadic: *library
                    .get(b"clang_isFunctionTypeVariadic\0")
                    .map_err(error)?,
                size: *library.get(b"clang_Type_getSizeOf\0").map_err(error)?,
                inlined: *library
                    .get(b"clang_Cursor_isFunctionInlined\0")
                    .map_err(error)?,
                string: *library.get(b"clang_getCString\0").map_err(error)?,
                dispose_string: *library.get(b"clang_disposeString\0").map_err(error)?,
                diagnostic_count: *library.get(b"clang_getNumDiagnostics\0").map_err(error)?,
                diagnostic: *library.get(b"clang_getDiagnostic\0").map_err(error)?,
                severity: *library
                    .get(b"clang_getDiagnosticSeverity\0")
                    .map_err(error)?,
                format_diagnostic: *library.get(b"clang_formatDiagnostic\0").map_err(error)?,
                dispose_diagnostic: *library.get(b"clang_disposeDiagnostic\0").map_err(error)?,
                _library: library,
            })
        }
    }

    unsafe fn text(&self, text: CXString) -> String {
        unsafe {
            let pointer = (self.string)(text);
            let output = if pointer.is_null() {
                String::new()
            } else {
                CStr::from_ptr(pointer).to_string_lossy().into_owned()
            };
            (self.dispose_string)(text);
            output
        }
    }
}

struct Translation<'a> {
    clang: &'a Clang,
    index: CXIndex,
    unit: CXTranslationUnit,
}
impl Drop for Translation<'_> {
    fn drop(&mut self) {
        unsafe {
            if !self.unit.is_null() {
                (self.clang.dispose_unit)(self.unit);
            }
            (self.clang.dispose_index)(self.index);
        }
    }
}

pub(super) fn inspect(
    library: &Path,
    source: &[u8],
    functions: &[ForeignFunction],
    directory: &Path,
    cancellation: &Cancellation,
) -> Result<(Vec<ForeignDeclaration>, Vec<String>), Error> {
    cancellation.check()?;
    // Every borrowed CIndex value remains inside this job and is disposed before
    // unloading the library; outputs below contain only owned Rust strings.
    unsafe {
        let clang = Clang::load(library)?;
        let filename = CString::new(directory.join("foreign.i").to_string_lossy().as_bytes())
            .map_err(error)?;
        let arguments = [
            c"-x".as_ptr(),
            c"c".as_ptr(),
            c"-std=c11".as_ptr(),
            c"-Wno-gnu-line-marker".as_ptr(),
            c"-Werror=implicit-function-declaration".as_ptr(),
        ];
        let mut unsaved = CXUnsavedFile {
            Filename: filename.as_ptr(),
            Contents: source.as_ptr().cast(),
            Length: source.len() as _,
        };
        let index = (clang.create_index)(0, 0);
        if index.is_null() {
            return Err(error("cannot create CIndex"));
        }
        let mut translation = Translation {
            clang: &clang,
            index,
            unit: ptr::null_mut(),
        };
        translation.unit = (clang.parse)(
            index,
            filename.as_ptr(),
            arguments.as_ptr(),
            arguments.len() as _,
            &mut unsaved,
            1,
            0,
        );
        if translation.unit.is_null() {
            return Err(error("cannot parse captured C interoperability input"));
        }
        let diagnostics = diagnostics(&clang, translation.unit, directory)?;
        cancellation.check()?;
        let mut context = Context {
            clang: &clang,
            functions: BTreeMap::new(),
        };
        (clang.visit)(
            (clang.cursor)(translation.unit),
            collect_functions,
            (&mut context as *mut Context<'_>).cast(),
        );
        let result = functions
            .iter()
            .map(|function| declaration(&clang, &context.functions, function))
            .collect::<Result<Vec<_>, _>>()?;
        Ok((result, diagnostics))
    }
}

struct Context<'a> {
    clang: &'a Clang,
    functions: BTreeMap<String, Vec<CXCursor>>,
}
extern "C" fn collect_functions(
    cursor: CXCursor,
    _: CXCursor,
    data: CXClientData,
) -> CXChildVisitResult {
    unsafe {
        let context = &mut *data.cast::<Context<'_>>();
        if cursor.kind == CXCursor_FunctionDecl {
            let name = context.clang.text((context.clang.spelling)(cursor));
            context.functions.entry(name).or_default().push(cursor);
        }
    }
    CXChildVisit_Continue
}

unsafe fn diagnostics(
    clang: &Clang,
    unit: CXTranslationUnit,
    directory: &Path,
) -> Result<Vec<String>, Error> {
    unsafe {
        let mut messages = Vec::new();
        let mut failed = false;
        for index in 0..(clang.diagnostic_count)(unit) {
            let diagnostic = (clang.diagnostic)(unit, index);
            failed |= (clang.severity)(diagnostic) >= CXDiagnostic_Error;
            let text = clang.text((clang.format_diagnostic)(diagnostic, 1 | 2 | 4));
            messages.push(text.replace(directory.to_string_lossy().as_ref(), "<foreign>"));
            (clang.dispose_diagnostic)(diagnostic);
        }
        if failed {
            return Err(Error::new(format!(
                "C interoperability header analysis failed:\n{}",
                messages.join("\n")
            )));
        }
        Ok(messages)
    }
}

unsafe fn declaration(
    clang: &Clang,
    declarations: &BTreeMap<String, Vec<CXCursor>>,
    function: &ForeignFunction,
) -> Result<ForeignDeclaration, Error> {
    unsafe {
        let declarations = declarations.get(&function.name).ok_or_else(|| {
            error(format!(
                "no C function declaration for {}; macro-only names cannot be linked directly",
                function.name
            ))
        })?;
        let cursor = *declarations.last().expect("collected declaration");
        let ty = (clang.cursor_type)(cursor);
        // A header body is not a link contract: its object is never compiled here.
        // Preserve explicit external prototypes even when libc later supplies an
        // inline definition of the same function in the captured translation unit.
        if (clang.linkage)(cursor) != CXLinkage_External
            || !declarations.iter().any(|cursor| {
                (clang.linkage)(*cursor) == CXLinkage_External
                    && (clang.definition)(*cursor) == 0
                    && (clang.inlined)(*cursor) == 0
            })
        {
            return Err(error(format!(
                "{} needs an external non-inline C function prototype; static and inline-only definitions cannot be linked directly",
                function.name
            )));
        }
        // A later redeclaration may add an asm label. Resolve the final declaration
        // so the native import uses Clang's actual symbol, never just its source name.
        let params = (clang.argument_count)(ty);
        if !platform_calling_convention((clang.calling_convention)(ty))
            || (clang.variadic)(ty) != 0
            || params != function.params.len() as i32
            || !matches_scalar(clang, (clang.result_type)(ty), &function.result)
            || function.params.iter().enumerate().any(|(index, expected)| {
                !matches_scalar(clang, (clang.argument_type)(ty, index as _), expected)
            })
        {
            return Err(error(format!(
                "C declaration {} does not match the requested scalar ABI: {}",
                function.name,
                clang.text((clang.type_spelling)(ty))
            )));
        }
        let symbol = import_symbol(&clang.text((clang.mangling)(cursor)))?;
        Ok(ForeignDeclaration {
            name: function.name.clone(),
            symbol,
            c_signature: clang.text((clang.type_spelling)((clang.canonical_type)(ty))),
        })
    }
}

fn import_symbol(mangled: &str) -> Result<String, Error> {
    // CIndex returns the object-file spelling, while native emitters apply the
    // platform global prefix themselves. Explicit asm labels omit that prefix;
    // an unprefixed Mach-O label cannot be represented by this import contract.
    let symbol = if cfg!(target_os = "macos") {
        mangled.strip_prefix('_').ok_or_else(|| {
            error("explicit C symbol without the Mach-O underscore prefix is unsupported")
        })?
    } else {
        mangled
    };
    if symbol.is_empty() || !symbol.is_ascii() || symbol.bytes().any(|byte| byte.is_ascii_control())
    {
        return Err(error("C declaration has an unsupported linker symbol name"));
    }
    Ok(symbol.into())
}

fn platform_calling_convention(convention: CXCallingConv) -> bool {
    convention == CXCallingConv_C
        || convention == CXCallingConv_Default
        || cfg!(all(target_arch = "x86_64", target_os = "windows"))
            && convention == CXCallingConv_Win64
        || cfg!(all(target_arch = "x86_64", not(target_os = "windows")))
            && convention == CXCallingConv_X86_64SysV
}

unsafe fn matches_scalar(clang: &Clang, ty: CXType, expected: &ForeignScalar) -> bool {
    unsafe {
        let mut ty = (clang.canonical_type)(ty);
        if ty.kind == CXType_Enum {
            ty = (clang.enum_integer_type)((clang.type_declaration)(ty));
        }
        match expected {
            ForeignScalar::Void => ty.kind == CXType_Void,
            ForeignScalar::Bool => ty.kind == CXType_Bool && (clang.size)(ty) == 1,
            ForeignScalar::Pointer => ty.kind == CXType_Pointer && (clang.size)(ty) == 8,
            ForeignScalar::Integer { bits, signed } => {
                (clang.size)(ty) * 8 == i64::from(*bits)
                    && if *signed {
                        [
                            CXType_Char_S,
                            CXType_SChar,
                            CXType_Short,
                            CXType_Int,
                            CXType_Long,
                            CXType_LongLong,
                        ]
                        .contains(&ty.kind)
                    } else {
                        [
                            CXType_Char_U,
                            CXType_UChar,
                            CXType_UShort,
                            CXType_UInt,
                            CXType_ULong,
                            CXType_ULongLong,
                        ]
                        .contains(&ty.kind)
                    }
            }
            ForeignScalar::Float { bits } => {
                [CXType_Float, CXType_Double].contains(&ty.kind)
                    && (clang.size)(ty) * 8 == i64::from(*bits)
            }
        }
    }
}

fn error(error: impl std::fmt::Display) -> Error {
    Error::new(format!("libclang C interoperability: {error}"))
}
