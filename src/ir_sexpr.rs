use std::fmt::{self, Write};

use crate::ir;

pub fn print(program: &ir::Program) -> String {
    let mut buf = String::new();
    write_program(&mut buf, program).unwrap();
    buf
}

fn write_program(f: &mut String, program: &ir::Program) -> fmt::Result {
    f.write_str("(program")?;
    for func in &program.functions {
        f.write_char(' ')?;
        write_function(f, func)?;
    }
    f.write_char(')')
}

fn write_function(f: &mut String, func: &ir::Function) -> fmt::Result {
    write!(f, "(func {} {}", func.id.0, func.name)?;
    // Params
    f.write_str(" (params")?;
    for p in &func.params {
        write!(f, " {p}")?;
    }
    f.write_char(')')?;
    // Nodes
    f.write_str(" (nodes")?;
    for (i, node) in func.nodes.iter().enumerate() {
        f.write_char(' ')?;
        write_node(f, i as u32, node)?;
    }
    f.write_char(')')?;
    // Outputs
    f.write_str(" (outputs")?;
    for r in &func.outputs {
        f.write_char(' ')?;
        write_ref(f, r)?;
    }
    f.write_str("))")
}

fn write_node(f: &mut String, _id: u32, node: &ir::Node) -> fmt::Result {
    match node {
        ir::Node::Param { idx } => write!(f, "(param {idx})"),
        ir::Node::Const { val } => {
            f.write_str("(const ")?;
            match val {
                ir::ConstVal::Int(v) => write!(f, "{v}")?,
                ir::ConstVal::Float(v) => write!(f, "{v}")?,
                ir::ConstVal::Bool(v) => write!(f, "{v}")?,
            }
            f.write_char(')')
        }
        ir::Node::Elem { op, args } => {
            write!(f, "(elem {} ", elem_op_name(*op))?;
            for (i, a) in args.iter().enumerate() {
                if i > 0 {
                    f.write_char(' ')?;
                }
                write_ref(f, a)?;
            }
            f.write_char(')')
        }
        ir::Node::Reduce { op, input, dim } => {
            write!(f, "(reduce {} ", reduce_op_name(*op))?;
            write_ref(f, input)?;
            write!(f, " {dim})")
        }
        ir::Node::Gather { data, indices, dim } => {
            f.write_str("(gather ")?;
            write_ref(f, data)?;
            f.write_char(' ')?;
            write_ref(f, indices)?;
            write!(f, " {dim})")
        }
        ir::Node::Scatter {
            op,
            data,
            indices,
            dim,
            dim_size,
        } => {
            write!(f, "(scatter {} ", reduce_op_name(*op))?;
            write_ref(f, data)?;
            f.write_char(' ')?;
            write_ref(f, indices)?;
            write!(f, " {dim} {dim_size})")
        }
        ir::Node::Cond {
            pred,
            then_refs,
            else_refs,
        } => {
            f.write_str("(cond ")?;
            write_ref(f, pred)?;
            f.write_str(" (then")?;
            for r in then_refs {
                f.write_char(' ')?;
                write_ref(f, r)?;
            }
            f.write_str(") (else")?;
            for r in else_refs {
                f.write_char(' ')?;
                write_ref(f, r)?;
            }
            f.write_str("))")
        }
        ir::Node::Call { func, args } => {
            write!(f, "(call {}", func.0)?;
            for a in args {
                f.write_char(' ')?;
                write_ref(f, a)?;
            }
            f.write_char(')')
        }
    }
}

fn write_ref(f: &mut String, r: &ir::Ref) -> fmt::Result {
    if r.view.is_identity() {
        write!(f, "(ref {} {})", r.node.0, r.output.0)
    } else {
        write!(
            f,
            "(ref {} {} (view {:?} {:?} {}))",
            r.node.0, r.output.0, r.view.shape, r.view.strides, r.view.offset
        )
    }
}

fn elem_op_name(op: ir::ElemOp) -> &'static str {
    match op {
        ir::ElemOp::Neg => "neg",
        ir::ElemOp::Recip => "recip",
        ir::ElemOp::Exp => "exp",
        ir::ElemOp::Log => "log",
        ir::ElemOp::Sqrt => "sqrt",
        ir::ElemOp::Abs => "abs",
        ir::ElemOp::Not => "not",
        ir::ElemOp::Add => "add",
        ir::ElemOp::Sub => "sub",
        ir::ElemOp::Mul => "mul",
        ir::ElemOp::Div => "div",
        ir::ElemOp::Rem => "rem",
        ir::ElemOp::Pow => "pow",
        ir::ElemOp::Max => "max",
        ir::ElemOp::Min => "min",
        ir::ElemOp::Eq => "eq",
        ir::ElemOp::Ne => "ne",
        ir::ElemOp::Lt => "lt",
        ir::ElemOp::Gt => "gt",
        ir::ElemOp::Le => "le",
        ir::ElemOp::Ge => "ge",
        ir::ElemOp::And => "and",
        ir::ElemOp::Or => "or",
        ir::ElemOp::Where => "where",
    }
}

fn reduce_op_name(op: ir::ReduceOp) -> &'static str {
    match op {
        ir::ReduceOp::Sum => "sum",
        ir::ReduceOp::Prod => "prod",
        ir::ReduceOp::Max => "max",
        ir::ReduceOp::Min => "min",
    }
}
