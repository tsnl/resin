//! C source tree → text. This module does not inspect LIR or Resin types.
use crate::c::{CBlock, CBody, CEdge, CEdgeValue, CExit, CFunction, CModule};
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
        CBody::Blocks {
            locals,
            entry,
            blocks,
        } => {
            out.push_str(locals);
            writeln!(out, "  goto r_b{entry};").unwrap();
            for block in blocks {
                print_block(out, block);
            }
        }
    }
    out.push_str("}\n");
}

fn print_block(out: &mut String, block: &CBlock) {
    writeln!(out, "r_b{}: {{", block.label).unwrap();
    out.push_str(&block.statements);
    print_exit(out, &block.exit);
    out.push_str("}\n");
}

fn print_exit(out: &mut String, exit: &CExit) {
    match exit {
        CExit::Return(value) => writeln!(out, "  return {value};").unwrap(),
        CExit::Jump(edge) => print_edge(out, edge),
        CExit::Branch {
            condition,
            then,
            els,
        } => {
            writeln!(out, "  if ({condition}) {{").unwrap();
            print_edge(out, then);
            out.push_str("  } else {\n");
            print_edge(out, els);
            out.push_str("  }\n");
        }
        CExit::Unreachable => {}
    }
}

fn print_edge(out: &mut String, edge: &CEdge) {
    out.push_str("  {\n");
    for (i, value) in edge.values.iter().enumerate() {
        edge_temporary(out, i, value);
    }
    for (i, value) in edge.values.iter().enumerate() {
        edge_assignment(out, edge.target, i, value);
    }
    writeln!(out, "    goto r_b{};\n  }}", edge.target).unwrap();
}

fn edge_temporary(out: &mut String, i: usize, value: &CEdgeValue) {
    writeln!(out, "    {} r_edge{i} = {};", value.ty, value.value).unwrap();
    if let Some(live) = &value.live {
        writeln!(out, "    bool *r_edge{i}_live = {live};").unwrap();
    }
}

fn edge_assignment(out: &mut String, target: usize, i: usize, value: &CEdgeValue) {
    writeln!(out, "    r_b{target}_{i} = r_edge{i};").unwrap();
    if value.live.is_some() {
        writeln!(out, "    r_b{target}_{i}_live = r_edge{i}_live;").unwrap();
    }
}
