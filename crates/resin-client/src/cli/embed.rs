//! Binary-file conversion used by generated Ninja rules.
use super::Result;
use std::{fmt::Write as _, fs, io::Write as _, path::Path};

pub(super) fn run(input: &Path, output: &Path, symbol: &str) -> Result<i32> {
    validate_symbol(symbol)?;
    if resin_source::normalize_path(input)? == resin_source::normalize_path(output)? {
        return Err("embedding output would overwrite its input".into());
    }
    let bytes = fs::read(input)?;
    publish(output, &header(&bytes, symbol))?;
    Ok(0)
}

fn header(bytes: &[u8], symbol: &str) -> String {
    let mut text = format!(
        "#include <stdint.h>\n\n_Alignas(4) static const uint8_t {symbol}[{}] = {{\n",
        bytes.len().max(1)
    );
    for chunk in bytes.chunks(12) {
        text.push_str("    ");
        for byte in chunk {
            write!(text, "0x{byte:02x}, ").unwrap();
        }
        text.push('\n');
    }
    if bytes.is_empty() {
        text.push_str("    0\n");
    }
    writeln!(
        text,
        "}};\n#define {symbol}_length UINT64_C({})",
        bytes.len()
    )
    .unwrap();
    text
}

fn publish(output: &Path, text: &str) -> Result<()> {
    if fs::read(output).ok().as_deref() == Some(text.as_bytes()) {
        return Ok(());
    }
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let mut file = tempfile::NamedTempFile::new_in(parent)?;
    file.write_all(text.as_bytes())?;
    file.persist(output)?;
    Ok(())
}

fn validate_symbol(symbol: &str) -> Result<()> {
    let first = symbol.as_bytes().first().copied().unwrap_or_default();
    if !(first.is_ascii_alphabetic() || first == b'_')
        || !symbol
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'_')
        || C_KEYWORDS.contains(&symbol)
    {
        return Err(format!("invalid C array identifier: {symbol:?}").into());
    }
    Ok(())
}

const C_KEYWORDS: &[&str] = &[
    "auto",
    "break",
    "case",
    "char",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extern",
    "float",
    "for",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "register",
    "restrict",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "struct",
    "switch",
    "typedef",
    "union",
    "unsigned",
    "void",
    "volatile",
    "while",
    "_Alignas",
    "_Alignof",
    "_Atomic",
    "_Bool",
    "_Complex",
    "_Generic",
    "_Imaginary",
    "_Noreturn",
    "_Static_assert",
    "_Thread_local",
];
