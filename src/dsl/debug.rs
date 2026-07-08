//! Multi-line graph dumps for debugging and expect-tests.

use std::collections::HashMap;
use std::fmt::Write;

use super::{Tensor, TensorKind};
use crate::ops::{AssocOp, BinaryOp, Op, UnaryOp};

/// Render `tensor` as a multi-line graph dump. Shared nodes print as `%id`
/// references, with their definitions listed after the main tree.
pub fn debug_str(tensor: &Tensor) -> String {
    let mut out = String::new();
    debug_print(tensor, &mut out).expect("write to string");
    out.trim_end().to_string()
}

pub fn debug_print(root: &Tensor, out: &mut dyn Write) -> std::fmt::Result {
    // Nodes referenced more than once get a `%id` and print once.
    let mut ref_counts: HashMap<Tensor, usize> = HashMap::new();
    fn count(node: &Tensor, counts: &mut HashMap<Tensor, usize>) {
        let n = counts.entry(node.clone()).or_insert(0);
        *n += 1;
        if *n == 1 {
            for arg in node.args() {
                count(&arg, counts);
            }
        }
    }
    count(root, &mut ref_counts);

    let mut ids = HashMap::new();
    for node in root.toposort().iter().rev() {
        if ref_counts[node] > 1 {
            let id = ids.len();
            ids.insert(node.clone(), id);
        }
    }

    visit(root, "", "", "", true, &ids, out)?;

    let mut shared: Vec<(&Tensor, usize)> = ids.iter().map(|(t, &id)| (t, id)).collect();
    shared.sort_by_key(|&(_, id)| id);
    for (node, _) in shared {
        visit(node, "", "", "", true, &ids, out)?;
    }
    Ok(())
}

fn visit(
    node: &Tensor,
    prefix: &str,
    connector: &str,
    prefix_ext: &str,
    is_root: bool,
    ids: &HashMap<Tensor, usize>,
    out: &mut dyn Write,
) -> std::fmt::Result {
    match ids.get(node) {
        Some(id) if !is_root => return writeln!(out, "{prefix}{connector}%{id}"),
        Some(id) => writeln!(out, "{prefix}{connector}%{id} := {}", headline(node))?,
        None => writeln!(out, "{prefix}{connector}{}", headline(node))?,
    }

    let child_prefix = format!("{prefix}{prefix_ext}");
    let args = node.args();
    for (i, arg) in args.iter().enumerate() {
        let last = i + 1 == args.len();
        let connector = if last { "└ " } else { "├ " };
        let ext = if last { "  " } else { "│ " };
        visit(arg, &child_prefix, connector, ext, false, ids, out)?;
    }
    Ok(())
}

fn headline(tensor: &Tensor) -> String {
    let kind = match tensor.kind() {
        TensorKind::Constant { values } => {
            format!("constant(value={})", format_values(values, tensor.shape()))
        }
        TensorKind::Parameter => "parameter()".to_string(),
        TensorKind::Elementwise { op, .. } => format!("elementwise(op='{}')", op_name(*op)),
        TensorKind::Matmul { .. } => "matmul()".to_string(),
        TensorKind::Reduction { op, axes, .. } => {
            format!("reduction(op='{}', axes={})", assoc_op_name(*op), tuple(axes))
        }
        TensorKind::Broadcast { axes, .. } => format!(
            "broadcast(axes={}, target_shape={})",
            tuple(axes),
            tuple(tensor.shape())
        ),
        TensorKind::Transpose { .. } => "transpose()".to_string(),
        TensorKind::Squeeze { axes, .. } => format!("squeeze(axes={})", tuple(axes)),
    };
    format!("{kind} :: f32{}", shape(tensor.shape()))
}

fn op_name(op: Op) -> &'static str {
    match op {
        Op::Unary(op) => match op {
            UnaryOp::Neg => "neg",
            UnaryOp::Exp => "exp",
            UnaryOp::Log => "log",
            UnaryOp::Relu => "relu",
            UnaryOp::Abs => "abs",
            UnaryOp::Sqrt => "sqrt",
            UnaryOp::Sin => "sin",
            UnaryOp::Cos => "cos",
        },
        Op::Binary(op) => match op {
            BinaryOp::Assoc(op) => assoc_op_name(op),
            BinaryOp::Sub => "sub",
            BinaryOp::Div => "div",
            BinaryOp::Pow => "pow",
        },
    }
}

fn assoc_op_name(op: AssocOp) -> &'static str {
    match op {
        AssocOp::Add => "add",
        AssocOp::Mul => "mul",
        AssocOp::Max => "max",
        AssocOp::Min => "min",
    }
}

fn shape(dims: &[usize]) -> String {
    match dims {
        [] => "()".to_string(),
        [dim] => format!("({dim},)"),
        dims => format!(
            "({})",
            dims.iter().map(usize::to_string).collect::<Vec<_>>().join(", ")
        ),
    }
}

fn tuple(values: &[usize]) -> String {
    format!(
        "({})",
        values.iter().map(usize::to_string).collect::<Vec<_>>().join(", ")
    )
}

fn format_values(values: &[f32], shape: &[usize]) -> String {
    if shape.is_empty() {
        return format!("{}", values.first().copied().unwrap_or(0.0));
    }
    if shape.len() == 1 {
        return format!("{values:?}");
    }
    let cols = shape[shape.len() - 1];
    let rows: Vec<String> = values
        .chunks(cols)
        .map(|row| {
            let cells = row.iter().map(f32::to_string).collect::<Vec<_>>().join(", ");
            format!("[{cells}]")
        })
        .collect();
    format!("[{}]", rows.join(", "))
}

/// Python `textwrap.dedent(...).strip()` for multiline expect literals.
pub fn dedent(s: &str) -> String {
    let s = s.strip_prefix('\n').unwrap_or(s);
    let margin = s
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.len() - line.trim_start().len())
        .min()
        .unwrap_or(0);
    s.lines()
        .map(|line| if line.trim().is_empty() { "" } else { &line[margin..] })
        .collect::<Vec<_>>()
        .join("\n")
        .trim_end()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedent_strips_uniform_indent() {
        let expected = dedent(
            r#"
            line1
            ├ child
            "#,
        );
        assert_eq!(expected, "line1\n├ child");
    }

    #[test]
    fn dump_scalar_constant() {
        assert_eq!(debug_str(&Tensor::scalar(42.0)), "constant(value=42) :: f32()");
    }

    #[test]
    fn dump_1d_constant() {
        assert_eq!(
            debug_str(&Tensor::constant(&[3], &[1.0, 2.0, 3.0])),
            "constant(value=[1.0, 2.0, 3.0]) :: f32(3,)"
        );
    }

    #[test]
    fn dump_chained_elementwise() {
        let t1 = Tensor::constant(&[2], &[1.0, 2.0]);
        let t2 = Tensor::constant(&[2], &[3.0, 4.0]);
        let t3 = Tensor::constant(&[2], &[5.0, 6.0]);
        let expected = dedent(
            r#"
            elementwise(op='mul') :: f32(2,)
            ├ elementwise(op='add') :: f32(2,)
            │ ├ constant(value=[1.0, 2.0]) :: f32(2,)
            │ └ constant(value=[3.0, 4.0]) :: f32(2,)
            └ constant(value=[5.0, 6.0]) :: f32(2,)
            "#,
        );
        assert_eq!(debug_str(&((t1 + t2) * t3)), expected);
    }

    #[test]
    fn dump_shared_subgraph() {
        let sum = Tensor::constant(&[2], &[1.0, 2.0]) + Tensor::constant(&[2], &[3.0, 4.0]);
        let expected = dedent(
            r#"
            elementwise(op='mul') :: f32(2,)
            ├ %0
            └ %0
            %0 := elementwise(op='add') :: f32(2,)
            ├ constant(value=[1.0, 2.0]) :: f32(2,)
            └ constant(value=[3.0, 4.0]) :: f32(2,)
            "#,
        );
        assert_eq!(debug_str(&(sum.clone() * sum)), expected);
    }

    #[test]
    fn dump_parameter() {
        assert_eq!(debug_str(&Tensor::parameter(&[4])), "parameter() :: f32(4,)");
    }
}
