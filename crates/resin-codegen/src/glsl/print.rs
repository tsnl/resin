//! Shader source tree → GLSL text; no upstream compiler state is needed.
use super::*;
use std::fmt::Write;

pub fn module(module: &Module) -> String {
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

fn print_function(out: &mut String, function: &Function) {
    writeln!(out, "{} {{", function.signature).unwrap();
    out.push_str(&function.locals);
    writeln!(
        out,
        "  int pc = {};\n  while (true) {{\n    switch (pc) {{",
        function.entry
    )
    .unwrap();
    for block in &function.blocks {
        print_block(out, block);
    }
    writeln!(
        out,
        "    default: return {};\n    }}\n  }}\n}}",
        function.default_result
    )
    .unwrap();
}

fn print_block(out: &mut String, block: &Block) {
    writeln!(out, "    case {}: {{", block.label).unwrap();
    out.push_str(&block.statements);
    print_exit(out, &block.exit);
    out.push_str("    }\n");
}

fn print_exit(out: &mut String, exit: &Exit) {
    match exit {
        Exit::Return(value) => writeln!(out, "      return {value};").unwrap(),
        Exit::Jump(edge) => print_edge(out, edge),
        Exit::Branch {
            condition,
            then,
            els,
        } => {
            writeln!(out, "      if ({condition}) {{").unwrap();
            print_edge(out, then);
            out.push_str("      } else {\n");
            print_edge(out, els);
            out.push_str("      }\n");
        }
        Exit::Unreachable => {}
    }
}

fn print_edge(out: &mut String, edge: &Edge) {
    out.push_str("      {\n");
    for value in &edge.values {
        writeln!(
            out,
            "        {} edge{} = {};",
            value.ty, value.slot, value.value
        )
        .unwrap();
    }
    for value in &edge.values {
        writeln!(
            out,
            "        r_b{}_{} = edge{};",
            edge.target, value.slot, value.slot
        )
        .unwrap();
    }
    writeln!(out, "        pc = {}; continue;\n      }}", edge.target).unwrap();
}
