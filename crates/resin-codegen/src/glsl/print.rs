//! Shader source tree → GLSL text; no upstream compiler state is needed.
use crate::glsl::{GlslFunction, GlslModule, GlslStatement};
use std::fmt::Write;

pub fn module(module: &GlslModule) -> String {
    let mut out = "#version 460\n#extension GL_EXT_buffer_reference : require\n#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n".to_string();
    out.push_str(&module.extensions);
    out.push_str(&module.declarations);
    out.push_str(&module.globals);
    for function in &module.functions {
        print_function(&mut out, function);
    }
    out.push_str(&module.entry);
    out
}

fn print_function(out: &mut String, function: &GlslFunction) {
    writeln!(out, "{} {{", function.signature).unwrap();
    out.push_str(&function.locals);
    print_statements(out, &function.statements, 1);
    out.push_str("}\n");
}

fn print_statements(out: &mut String, statements: &[GlslStatement], depth: usize) {
    let indent = "  ".repeat(depth);
    for statement in statements {
        match statement {
            GlslStatement::Text { source } => {
                for line in source.lines() {
                    writeln!(
                        out,
                        "{indent}{}",
                        line.strip_prefix("      ").unwrap_or(line)
                    )
                    .unwrap();
                }
            }
            GlslStatement::Return { value } => writeln!(out, "{indent}return {value};").unwrap(),
            GlslStatement::Break => writeln!(out, "{indent}break;").unwrap(),
            GlslStatement::If {
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
            GlslStatement::Loop { body } => {
                writeln!(out, "{indent}while (true) {{").unwrap();
                print_statements(out, body, depth + 1);
                writeln!(out, "{indent}}}").unwrap();
            }
        }
    }
}
