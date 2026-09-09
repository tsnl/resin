//! C source tree → text. This module does not inspect LIR or Resin types.
use crate::c::{CBody, CFunction, CModule, CStatement};
use std::fmt::Write;

pub fn module(module: &CModule) -> String {
    let mut out = String::new();
    for header in &module.includes {
        writeln!(out, "#include <{header}>").unwrap();
    }
    for header in &module.local_includes {
        writeln!(out, "#include \"{header}\"").unwrap();
    }
    for (condition, message) in &module.assertions {
        writeln!(out, "_Static_assert({condition}, {message:?});").unwrap();
    }
    out.push_str(&module.declarations);
    for signature in &module.prototypes {
        writeln!(out, "{signature};").unwrap();
    }
    out.push_str(&module.lifecycle);
    for function in &module.functions {
        print_function(&mut out, function);
    }
    print_function(&mut out, &module.entry);
    out
}

fn print_function(out: &mut String, function: &CFunction) {
    writeln!(out, "{} {{", function.signature).unwrap();
    match &function.body {
        CBody::Inline(body) => out.push_str(body),
        CBody::Structured { locals, statements } => {
            out.push_str(locals);
            print_statements(out, statements, 1);
        }
    }
    out.push_str("}\n");
}

fn print_statements(out: &mut String, statements: &[CStatement], depth: usize) {
    let indent = "  ".repeat(depth);
    for statement in statements {
        match statement {
            CStatement::Text { source } => {
                for line in source.lines() {
                    writeln!(out, "{indent}{}", line.strip_prefix("  ").unwrap_or(line)).unwrap();
                }
            }
            CStatement::Return { value } => writeln!(out, "{indent}return {value};").unwrap(),
            CStatement::Break => writeln!(out, "{indent}break;").unwrap(),
            CStatement::If {
                condition,
                then,
                els,
            } => {
                writeln!(out, "{indent}if ({condition}) {{").unwrap();
                print_statements(out, then, depth + 1);
                if !els.is_empty() {
                    writeln!(out, "{indent}}} else {{").unwrap();
                    print_statements(out, els, depth + 1);
                }
                writeln!(out, "{indent}}}").unwrap();
            }
            CStatement::Loop { body } => {
                writeln!(out, "{indent}while (true) {{").unwrap();
                print_statements(out, body, depth + 1);
                writeln!(out, "{indent}}}").unwrap();
            }
        }
    }
}
