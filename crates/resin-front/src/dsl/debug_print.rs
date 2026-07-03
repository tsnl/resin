//! Graph dump for debugging (Python `debug_print` / toposort tree).

use std::collections::HashMap;
use std::fmt::Write;

use super::node::{NodeKind, RemapInfo};
use super::node_ref::NodeRef;
use super::view::View;

/// Reference counts for nodes reachable from `roots` (by identity).
pub fn refcount(roots: &[View]) -> HashMap<NodeRef, usize> {
    let mut ref_counts: HashMap<NodeRef, usize> = HashMap::new();

    fn visit(node: &NodeRef, ref_counts: &mut HashMap<NodeRef, usize>) {
        *ref_counts.entry(node.clone()).or_insert(0) += 1;
        if ref_counts[node] == 1 {
            for operand in &node.args {
                visit(&operand.node_ref, ref_counts);
            }
        }
    }

    for root in roots {
        visit(&root.node_ref, &mut ref_counts);
    }
    ref_counts
}

/// Print a full tree dump of `root`, matching the Python DSL debugger:
/// non-identity views, shared nodes as `%tid`, and definitions for shared nodes.
pub fn debug_print(root: &View, out: &mut dyn Write) -> std::fmt::Result {
    let reference_count_map = refcount(std::slice::from_ref(root));
    let root_ref_count = *reference_count_map
        .get(&root.node_ref)
        .expect("root node in refcount");
    if root_ref_count != 1 {
        return writeln!(
            out,
            "error: root tensor must have reference count 1, got {root_ref_count}"
        );
    }

    let mut tid_map: HashMap<NodeRef, usize> = HashMap::new();
    for (node, &ref_count) in &reference_count_map {
        if ref_count < 1 {
            return writeln!(out, "error: node has invalid reference count {ref_count}");
        }
        if ref_count == 1 {
            continue;
        }
        let tid = tid_map.len();
        tid_map.insert(node.clone(), tid);
    }

    visit_view(root, "", "", "", true, &tid_map, out)?;

    // Emit definitions for shared nodes (same order as Python: HashMap iteration
    // is unstable; sort by tid for determinism).
    let mut shared: Vec<(&NodeRef, usize)> = tid_map.iter().map(|(n, &t)| (n, t)).collect();
    shared.sort_by_key(|(_, tid)| *tid);
    for (node, _) in shared {
        visit_node(node, "", "", "", true, &tid_map, out)?;
    }

    Ok(())
}

fn headline(node: &NodeRef) -> String {
    let kind = match &node.kind {
        NodeKind::Const(_) => "const()".to_string(),
        NodeKind::Param => "param".into(),
        NodeKind::Elementwise(k) => format!("elementwise(op={:?})", k.op),
        NodeKind::Matmul(_) => "matmul()".to_string(),
        NodeKind::Reduction(k) => format!("reduction(op={:?}, axes={:?})", k.op, k.axes),
        NodeKind::Remap(k) => match &k.info {
            RemapInfo::Scatter(s) => format!(
                "remap(scatter, operator={:?}, accessor={:?})",
                s.operator,
                s.accessor.as_ref().map(|a| format!(
                    "offset={} shape={:?} pitch={:?}",
                    a.offset, a.shape, a.pitch
                ))
            ),
            RemapInfo::Gather(g) => format!(
                "remap(gather, source_shape={:?}, accessor={:?})",
                g.source_shape,
                g.accessor.as_ref().map(|a| format!(
                    "offset={} shape={:?} pitch={:?}",
                    a.offset, a.shape, a.pitch
                ))
            ),
        },
    };
    format!("{kind} :: {:?}{:?}", node.element_type, node.shape)
}

fn visit_view(
    view: &View,
    prefix: &str,
    connector: &str,
    prefix_ext: &str,
    is_root: bool,
    tid_map: &HashMap<NodeRef, usize>,
    out: &mut dyn Write,
) -> std::fmt::Result {
    if view.is_identity() {
        return visit_node(
            &view.node_ref,
            prefix,
            connector,
            prefix_ext,
            is_root,
            tid_map,
            out,
        );
    }

    let a = &view.accessor;
    writeln!(
        out,
        "{prefix}{connector}view(offset={}, shape={:?}, pitch={:?})",
        a.offset, a.shape, a.pitch
    )?;
    let child_prefix = format!("{prefix}{prefix_ext}");
    visit_node(
        &view.node_ref,
        &child_prefix,
        "└ ",
        "  ",
        is_root,
        tid_map,
        out,
    )
}

fn visit_node(
    node: &NodeRef,
    prefix: &str,
    connector: &str,
    prefix_ext: &str,
    is_root: bool,
    tid_map: &HashMap<NodeRef, usize>,
    out: &mut dyn Write,
) -> std::fmt::Result {
    let tid = tid_map.get(node).copied();

    if let Some(tid) = tid {
        if !is_root {
            writeln!(out, "{prefix}{connector}%{tid}")?;
            return Ok(());
        }
        writeln!(out, "{prefix}{connector}%{tid} := {}", headline(node))?;
    } else {
        writeln!(out, "{prefix}{connector}{}", headline(node))?;
    }

    let child_prefix = format!("{prefix}{prefix_ext}");
    let n = node.args.len();
    for (operand_index, operand) in node.args.iter().enumerate() {
        let is_last = operand_index + 1 == n;
        let c_connector = if is_last { "└ " } else { "├ " };
        let c_ext = if is_last { "  " } else { "│ " };
        visit_view(
            operand,
            &child_prefix,
            c_connector,
            c_ext,
            false,
            tid_map,
            out,
        )?;
    }
    Ok(())
}

impl View {
    /// Full graph dump (tree + shared `%tid` nodes), Python `debug_print` style.
    pub fn debug_print(&self, out: &mut dyn Write) -> std::fmt::Result {
        debug_print(self, out)
    }
}

#[cfg(test)]
mod tests {
    use crate::dsl::prelude::param;
    use resin_core::F4;

    #[test]
    fn dump_shared_param_uses_tid() {
        let x = param([2], F4);
        let y = &x + &x;
        let mut s = String::new();
        y.debug_print(&mut s).unwrap();
        assert!(s.contains("%0"), "{s}");
        assert!(s.contains("elementwise"), "{s}");
        assert!(s.contains("param"), "{s}");
        // Shared node definition and use as %0
        assert!(s.matches("%0").count() >= 2, "{s}");
    }

    #[test]
    fn dump_non_identity_view_line() {
        let x = param([2, 3], F4);
        let v = x.broadcast(&[4]);
        let mut s = String::new();
        // Root must be identity for Python's refcount==1 rule on the node; wrap by
        // using a unary on the broadcast view so the printed root is identity.
        let root = v.exp();
        root.debug_print(&mut s).unwrap();
        assert!(s.contains("view(offset="), "{s}");
        assert!(s.contains("elementwise"), "{s}");
    }
}
