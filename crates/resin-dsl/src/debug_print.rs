//! Graph dump for debugging (Python `resin.dsl.view.debug_print`).

use std::collections::HashMap;
use std::fmt::Write;

use crate::tensor::{
    ElementOperator, ElementType, IndexKeyElement, ScatterOp, Tensor, TensorKind,
};

/// Reference counts for tensors reachable from `roots` (by pointer identity).
pub fn refcount(roots: &[&Tensor]) -> HashMap<Tensor, usize> {
    let mut ref_counts = HashMap::new();

    fn visit(tensor: &Tensor, ref_counts: &mut HashMap<Tensor, usize>) {
        *ref_counts.entry(tensor.clone()).or_insert(0) += 1;
        if ref_counts[tensor] == 1 {
            for dep in tensor.data_dependencies() {
                visit(&dep, ref_counts);
            }
        }
    }

    for root in roots {
        visit(root, &mut ref_counts);
    }
    ref_counts
}

/// Python `textwrap.dedent(...).strip()` for multiline expect literals.
///
/// Write expected output in a raw string block indented with the surrounding code:
///
/// ```ignore
/// let expected = dedent(r#"
///     elementwise(operator='add') :: f32(2,)
///     ├ constant(value=[1.0, 2.0]) :: f32(2,)
/// "#);
/// ```
pub fn dedent(s: &str) -> String {
    let s = s.strip_prefix('\n').unwrap_or(s);
    let lines: Vec<&str> = s.lines().collect();
    if lines.is_empty() {
        return String::new();
    }

    let margin = lines
        .iter()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.len() - line.trim_start().len())
        .min()
        .unwrap_or(0);

    lines
        .iter()
        .map(|line| {
            if line.trim().is_empty() {
                ""
            } else if line.len() >= margin {
                &line[margin..]
            } else {
                *line
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim_end()
        .to_string()
}

/// Render `tensor` as a multi-line graph dump.
pub fn debug_str(tensor: &Tensor) -> String {
    let mut out = String::new();
    debug_print(tensor, &mut out).expect("write to string");
    out.trim_end().to_string()
}

/// Print a full tree dump of `root`, matching the Python DSL debugger:
/// shared nodes as `%tid`, with definitions for shared nodes after the main tree.
pub fn debug_print(root: &Tensor, out: &mut dyn Write) -> std::fmt::Result {
    let reference_count_map = refcount(&[root]);
    let root_ref_count = *reference_count_map
        .get(root)
        .expect("root tensor in refcount");
    if root_ref_count != 1 {
        return writeln!(
            out,
            "error: root tensor must have reference count 1, got {root_ref_count}"
        );
    }

    let mut tid_map = HashMap::new();
    for (tensor, &ref_count) in &reference_count_map {
        if ref_count < 1 {
            return writeln!(out, "error: tensor has invalid reference count {ref_count}");
        }
        if ref_count == 1 {
            continue;
        }
        let tid = tid_map.len();
        tid_map.insert(tensor.clone(), tid);
    }

    visit_tensor(root, "", "", "", true, &tid_map, out)?;

    let mut shared: Vec<(&Tensor, usize)> = tid_map.iter().map(|(t, &id)| (t, id)).collect();
    shared.sort_by_key(|(_, tid)| *tid);
    for (tensor, _) in shared {
        visit_tensor(tensor, "", "", "", true, &tid_map, out)?;
    }

    Ok(())
}

fn visit_tensor(
    tensor: &Tensor,
    prefix: &str,
    connector: &str,
    prefix_ext: &str,
    is_root: bool,
    tid_map: &HashMap<Tensor, usize>,
    out: &mut dyn Write,
) -> std::fmt::Result {
    let tid = tid_map.get(tensor).copied();

    if let Some(tid) = tid {
        if !is_root {
            writeln!(out, "{prefix}{connector}%{tid}")?;
            return Ok(());
        }
        writeln!(out, "{prefix}{connector}%{tid} := {}", headline(tensor))?;
    } else {
        writeln!(out, "{prefix}{connector}{}", headline(tensor))?;
    }

    let child_prefix = format!("{prefix}{prefix_ext}");
    let deps = tensor.data_dependencies();
    let n = deps.len();
    for (operand_index, dep) in deps.iter().enumerate() {
        let is_last = operand_index + 1 == n;
        let c_connector = if is_last { "└ " } else { "├ " };
        let c_ext = if is_last { "  " } else { "│ " };
        visit_tensor(dep, &child_prefix, c_connector, c_ext, false, tid_map, out)?;
    }
    Ok(())
}

fn headline(tensor: &Tensor) -> String {
    let kind = match tensor.kind() {
        TensorKind::Constant { bytes } => format!("constant({})", format_constant(bytes, tensor)),
        TensorKind::Parameter => "parameter()".to_string(),
        TensorKind::Elementwise { operator, .. } => {
            format!("elementwise(operator='{}')", operator_name(*operator))
        }
        TensorKind::Reduction { operator, axes, .. } => format!(
            "reduction(operator='{}', axes={})",
            operator_name(*operator),
            format_tuple(axes)
        ),
        TensorKind::Index { key, .. } => format!("index(key={})", format_index_key(key)),
        TensorKind::Gather { .. } => "gather_rows()".to_string(),
        TensorKind::Scatter { op, .. } => {
            format!("scatter_rows(op='{}')", scatter_op_name(*op))
        }
        TensorKind::Broadcast { axes, target_shape, .. } => format!(
            "broadcast(axes={}, target_shape={})",
            format_tuple(axes),
            format_tuple(target_shape)
        ),
        TensorKind::ScatterIndex { key, target_shape, .. } => format!(
            "scatter_index(key={}, target_shape={})",
            format_index_key(key),
            format_tuple(target_shape)
        ),
        TensorKind::Transpose { .. } => "transpose()".to_string(),
        TensorKind::Squeeze { axes, .. } => {
            format!("squeeze(axes={})", format_tuple(axes))
        }
    };
    format!(
        "{kind} :: {}{}",
        element_type_name(tensor.element_type()),
        format_shape(tensor.shape())
    )
}

fn element_type_name(element_type: ElementType) -> &'static str {
    match element_type {
        ElementType::F32 => "f32",
        ElementType::U32 => "u32",
    }
}

fn operator_name(operator: ElementOperator) -> &'static str {
    match operator {
        ElementOperator::Neg => "neg",
        ElementOperator::Log => "log",
        ElementOperator::Exp => "exp",
        ElementOperator::Relu => "relu",
        ElementOperator::Abs => "abs",
        ElementOperator::Sqrt => "sqrt",
        ElementOperator::Floor => "floor",
        ElementOperator::Ceil => "ceil",
        ElementOperator::Cast => "cast",
        ElementOperator::Bitcast => "bitcast",
        ElementOperator::Pow => "pow",
        ElementOperator::Mul => "mul",
        ElementOperator::Div => "div",
        ElementOperator::Rem => "rem",
        ElementOperator::Add => "add",
        ElementOperator::Sub => "sub",
        ElementOperator::Min => "min",
        ElementOperator::Max => "max",
        ElementOperator::Matmul => "matmul",
        ElementOperator::CmpEq => "cmp_eq",
        ElementOperator::CmpNe => "cmp_ne",
        ElementOperator::CmpLt => "cmp_lt",
        ElementOperator::CmpLe => "cmp_le",
        ElementOperator::CmpGt => "cmp_gt",
        ElementOperator::CmpGe => "cmp_ge",
        ElementOperator::BitAnd => "bit_and",
        ElementOperator::BitOr => "bit_or",
        ElementOperator::BitXor => "bit_xor",
        ElementOperator::Shl => "shl",
        ElementOperator::Shr => "shr",
    }
}

fn scatter_op_name(op: ScatterOp) -> &'static str {
    match op {
        ScatterOp::Write => "write",
        ScatterOp::Add => "add",
    }
}

fn format_shape(shape: &[usize]) -> String {
    match shape {
        [] => "()".to_string(),
        [dim] => format!("({dim},)"),
        dims => format!(
            "({})",
            dims.iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn format_tuple(values: &[usize]) -> String {
    format!(
        "({})",
        values
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn format_index_key(key: &[IndexKeyElement]) -> String {
    let parts = key
        .iter()
        .map(|element| match element {
            IndexKeyElement::Single(index) => index.to_string(),
            IndexKeyElement::Slice(range) => format!("{}:{}", range.start, range.end),
        })
        .collect::<Vec<_>>();
    format!("({})", parts.join(", "))
}

fn format_constant(bytes: &[u8], tensor: &Tensor) -> String {
    match tensor.element_type() {
        ElementType::F32 => format_f32_constant(bytes, tensor.shape()),
        ElementType::U32 => format_u32_constant(bytes, tensor.shape()),
    }
}

fn format_u32_constant(bytes: &[u8], shape: &[usize]) -> String {
    let values: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("u32 chunk")))
        .collect();
    if shape.is_empty() {
        return format!("value={}", values.first().copied().unwrap_or(0));
    }
    format!("value={values:?}")
}

fn format_f32_constant(bytes: &[u8], shape: &[usize]) -> String {
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 chunk")))
        .collect();

    if shape.is_empty() {
        return format!("value={}", values.first().copied().unwrap_or(0.0));
    }

    if shape.len() == 1 {
        return format!("value={values:?}");
    }

    let cols = shape[shape.len() - 1];
    let rows: Vec<String> = values
        .chunks(cols)
        .map(|row| {
            let cells = row
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{cells}]")
        })
        .collect();
    format!("value=[{}]", rows.join(", "))
}

impl Tensor {
    /// Full graph dump (tree + shared `%tid` nodes), Python `debug_print` style.
    pub fn debug_print(&self, out: &mut dyn Write) -> std::fmt::Result {
        debug_print(self, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

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
        let tensor = Tensor::full(&[], 42.0, ElementType::F32);
        assert_eq!(debug_str(&tensor), "constant(value=42) :: f32()");
    }

    #[test]
    fn dump_1d_constant() {
        let tensor = Tensor::constant_f32(&[3], &[1.0, 2.0, 3.0]);
        assert_eq!(
            debug_str(&tensor),
            "constant(value=[1.0, 2.0, 3.0]) :: f32(3,)"
        );
    }

    #[test]
    fn dump_elementwise_add() {
        let left = Tensor::constant_f32(&[2], &[1.0, 2.0]);
        let right = Tensor::constant_f32(&[2], &[3.0, 4.0]);
        let tensor = left + right;
        let expected = dedent(
            r#"
            elementwise(operator='add') :: f32(2,)
            ├ constant(value=[1.0, 2.0]) :: f32(2,)
            └ constant(value=[3.0, 4.0]) :: f32(2,)
            "#,
        );
        assert_eq!(debug_str(&tensor), expected);
    }

    #[test]
    fn dump_chained_elementwise() {
        let t1 = Tensor::constant_f32(&[2], &[1.0, 2.0]);
        let t2 = Tensor::constant_f32(&[2], &[3.0, 4.0]);
        let t3 = Tensor::constant_f32(&[2], &[5.0, 6.0]);
        let tensor = (t1 + t2) * t3;
        let expected = dedent(
            r#"
            elementwise(operator='mul') :: f32(2,)
            ├ elementwise(operator='add') :: f32(2,)
            │ ├ constant(value=[1.0, 2.0]) :: f32(2,)
            │ └ constant(value=[3.0, 4.0]) :: f32(2,)
            └ constant(value=[5.0, 6.0]) :: f32(2,)
            "#,
        );
        assert_eq!(debug_str(&tensor), expected);
    }

    #[test]
    fn dump_shared_subgraph() {
        let left = Tensor::constant_f32(&[2], &[1.0, 2.0]);
        let right = Tensor::constant_f32(&[2], &[3.0, 4.0]);
        let sum = left + right;
        let tensor = sum.clone() * sum;
        let expected = dedent(
            r#"
            elementwise(operator='mul') :: f32(2,)
            ├ %0
            └ %0
            %0 := elementwise(operator='add') :: f32(2,)
            ├ constant(value=[1.0, 2.0]) :: f32(2,)
            └ constant(value=[3.0, 4.0]) :: f32(2,)
            "#,
        );
        assert_eq!(debug_str(&tensor), expected);
    }

    #[test]
    fn dump_parameter() {
        let tensor = Tensor::parameter(&[4], ElementType::F32);
        assert_eq!(debug_str(&tensor), "parameter() :: f32(4,)");
    }
}