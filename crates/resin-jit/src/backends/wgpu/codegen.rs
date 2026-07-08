//! Naive WGSL emission for IR kernels.
//!
//! No fusion, tiling heuristics, or shared-memory opts — one IR kernel → one
//! compute shader that indexes storage via the kernel's arg accessors.

use resin_core::{
    c_contiguous_pitch_for_shape, Accessor, BinaryAssocElementOperator, BinaryBitwiseOperator,
    BinaryCompareOperator, BinaryElementOperator, ElementOperator, ElementType,
    UnaryElementOperator,
};
use resin_ir::{
    IrElementwiseRpnKernel, IrKernel, IrMatmulKernel, IrReductionKernel, IrRemapKernel,
    RemapInfo, RpnAtom,
};

use super::error::WgpuLowerError;

#[derive(Debug, Clone, Copy)]
pub struct WgslKernelConfig {
    /// log2 of elements processed per thread (naive strip-mining).
    pub lg2_items_per_thread: u32,
    pub workgroup_size: u32,
}

impl Default for WgslKernelConfig {
    fn default() -> Self {
        Self {
            lg2_items_per_thread: 3,
            workgroup_size: 64,
        }
    }
}

pub fn dispatch_size_for_kernel(kernel: &IrKernel, config: &WgslKernelConfig) -> [u32; 3] {
    // Scatter remaps iterate the source, not the output.
    let n = shape_len(kernel.thread_shape());
    if n == 0 {
        return [0, 1, 1];
    }
    let items_per_thread = 1u64 << config.lg2_items_per_thread;
    let threads = n.div_ceil(items_per_thread);
    let workgroups_x = threads.div_ceil(u64::from(config.workgroup_size));
    [workgroups_x.max(1) as u32, 1, 1]
}

pub fn emit_wgsl_for_kernel(
    kernel: &IrKernel,
    config: &WgslKernelConfig,
) -> Result<String, WgpuLowerError> {
    let mut w = WgslWriter::new();
    match kernel {
        IrKernel::ElementwiseRpn(k) => emit_elementwise(&mut w, k, config)?,
        IrKernel::Matmul(k) => emit_matmul(&mut w, k, config)?,
        IrKernel::Reduction(k) => emit_reduction(&mut w, k, config)?,
        IrKernel::Remap(k) => emit_remap(&mut w, k, config)?,
    }
    Ok(w.finish())
}

fn shape_len(shape: &[u32]) -> u64 {
    if shape.is_empty() {
        1
    } else {
        shape.iter().map(|&d| u64::from(d)).product()
    }
}

fn spell_etype(etype: ElementType) -> Result<&'static str, WgpuLowerError> {
    match etype {
        ElementType::F4 => Ok("f32"),
        ElementType::F2 => Err(WgpuLowerError::Message(
            "f16 WGSL not enabled in naive backend".into(),
        )),
        ElementType::U4 => Ok("u32"),
    }
}

struct WgslWriter {
    lines: Vec<String>,
}

impl WgslWriter {
    fn new() -> Self {
        Self { lines: Vec::new() }
    }

    fn print(&mut self, text: &str) {
        let dedented: String = text
            .lines()
            .map(|l| l.trim_start())
            .collect::<Vec<_>>()
            .join("\n");
        self.lines.push(dedented);
    }

    fn block<F: FnOnce(&mut Self)>(&mut self, prefix: &str, body: F) {
        if !prefix.is_empty() {
            self.print(prefix);
        }
        self.print("{");
        body(self);
        self.print("}");
    }

    fn finish(self) -> String {
        self.lines.join("\n")
    }
}

fn emit_bindings(
    w: &mut WgslWriter,
    output_etype: ElementType,
    arg_etypes: &[ElementType],
) -> Result<(), WgpuLowerError> {
    let t = spell_etype(output_etype)?;
    w.print(&format!(
        "@group(0) @binding(0)\nvar<storage, read_write> output: array<{t}>;"
    ));
    for (i, arg_etype) in arg_etypes.iter().enumerate() {
        let arg_t = spell_etype(*arg_etype)?;
        w.print(&format!(
            "@group(0) @binding({})\nvar<storage, read> arg{i}: array<{arg_t}>;",
            i + 1
        ));
    }
    Ok(())
}

fn define_address_function(w: &mut WgslWriter, name: &str, accessor: &Accessor) {
    let n = accessor.rank();
    if n == 0 {
        w.block(&format!("fn {name}() -> u32"), |w| {
            w.print(&format!("return {}u;", accessor.offset));
        });
        return;
    }
    w.block(&format!("fn {name}(index: array<u32, {n}>) -> u32"), |w| {
        w.print(&format!("var acc: u32 = {}u;", accessor.offset));
        for i in 0..n {
            w.print(&format!(
                "acc += index[{i}] * {}u;",
                accessor.pitch[i]
            ));
        }
        w.print("return acc;");
    });
}

fn define_cc_index_function(w: &mut WgslWriter, name: &str, shape: &[u32]) {
    let pitch = c_contiguous_pitch_for_shape(shape);
    let n = shape.len();
    if n == 0 {
        return;
    }
    w.block(&format!("fn {name}(address: u32) -> array<u32, {n}>"), |w| {
        w.print("var addr: u32 = address;");
        w.print(&format!("var index: array<u32, {n}>;"));
        for i in 0..n {
            w.print(&format!("index[{i}] = addr / {}u;", pitch[i]));
            w.print(&format!("addr = addr % {}u;", pitch[i]));
        }
        w.print("return index;");
    });
}

fn emit_arg_address_functions(w: &mut WgslWriter, accessors: &[Accessor]) {
    for (i, acc) in accessors.iter().enumerate() {
        define_address_function(w, &format!("address_arg{i}"), acc);
    }
}

fn arg_address_expr(arg_index: usize, accessor: &Accessor, index_expr: &str) -> String {
    if accessor.rank() == 0 {
        format!("address_arg{arg_index}()")
    } else {
        format!("address_arg{arg_index}({index_expr})")
    }
}

fn per_output_element<F: FnOnce(&mut WgslWriter, &str)>(
    w: &mut WgslWriter,
    shape: &[u32],
    config: &WgslKernelConfig,
    body: F,
) {
    if shape.is_empty() {
        w.block(
            "@compute @workgroup_size(1)\nfn main(\n@builtin(global_invocation_id) global_id: vec3<u32>\n)",
            |w| {
                w.block("if (global_id.x > 0u)", |w| {
                    w.print("return;");
                });
                body(w, "0u");
            },
        );
        return;
    }

    define_cc_index_function(w, "index", shape);
    let items_per_thread = 1u32 << config.lg2_items_per_thread;
    w.block(
        &format!(
            "@compute @workgroup_size({})\nfn main(\n@builtin(global_invocation_id) global_id: vec3<u32>\n)",
            config.workgroup_size
        ),
        |w| {
            w.print(&format!(
                "let out_address_beg = global_id.x << {}u;\nlet out_address_end = out_address_beg + {items_per_thread}u;",
                config.lg2_items_per_thread
            ));
            w.block(
                "for (\nvar out_address = out_address_beg;\nout_address < out_address_end;\nout_address += 1u\n)",
                |w| {
                    w.block("if (out_address >= arrayLength(&output))", |w| {
                        w.print("return;");
                    });
                    w.print("let out_index = index(out_address);");
                    body(w, "out_address");
                },
            );
        },
    );
}

fn emit_elementwise(
    w: &mut WgslWriter,
    kernel: &IrElementwiseRpnKernel,
    config: &WgslKernelConfig,
) -> Result<(), WgpuLowerError> {
    emit_bindings(w, kernel.element_type, &kernel.arg_element_types)?;
    emit_arg_address_functions(w, &kernel.arg_accessors);
    emit_eval_rpn(
        w,
        &kernel.rpn_expr.atoms,
        kernel.element_type,
        &kernel.arg_element_types,
    )?;
    let n = kernel.arg_accessors.len();
    per_output_element(w, &kernel.shape, config, |w, out_addr| {
        let args: Vec<String> = (0..n)
            .map(|i| {
                format!(
                    "arg{i}[{}]",
                    arg_address_expr(i, &kernel.arg_accessors[i], "out_index")
                )
            })
            .collect();
        w.print(&format!(
            "output[{out_addr}] = eval_rpn_expr({});",
            args.join(", ")
        ));
    });
    Ok(())
}

fn emit_eval_rpn(
    w: &mut WgslWriter,
    atoms: &[RpnAtom],
    etype: ElementType,
    arg_etypes: &[ElementType],
) -> Result<(), WgpuLowerError> {
    let t = spell_etype(etype)?;
    let mut params = Vec::new();
    for (i, e) in arg_etypes.iter().enumerate() {
        params.push(format!("a{i}: {}", spell_etype(*e)?));
    }
    w.block(
        &format!("fn eval_rpn_expr({}) -> {t}", params.join(", ")),
        |w| {
            let mut stack: Vec<String> = Vec::new();
            for atom in atoms {
                match atom {
                    RpnAtom::Arg(i) => stack.push(format!("a{i}")),
                    RpnAtom::Op(op) => apply_op(&mut stack, *op, t),
                }
            }
            assert_eq!(stack.len(), 1, "RPN must leave one value");
            w.print(&format!("return {};", stack[0]));
        },
    );
    Ok(())
}

fn apply_op(stack: &mut Vec<String>, op: ElementOperator, t: &str) {
    match op {
        ElementOperator::Unary(u) => {
            let x = stack.pop().expect("rpn unary");
            let e = match u {
                UnaryElementOperator::Neg => format!("(-({x}))"),
                UnaryElementOperator::Exp => format!("exp({x})"),
                UnaryElementOperator::Log => format!("log({x})"),
                UnaryElementOperator::Relu => format!("max({x}, {t}(0))"),
                UnaryElementOperator::Abs => format!("abs({x})"),
                UnaryElementOperator::Sqrt => format!("sqrt({x})"),
                UnaryElementOperator::Sin => format!("sin({x})"),
                UnaryElementOperator::Cos => format!("cos({x})"),
                UnaryElementOperator::Not => format!("abs({t}(1) - ({x}))"),
                UnaryElementOperator::Floor => format!("floor({x})"),
                UnaryElementOperator::Ceil => format!("ceil({x})"),
                UnaryElementOperator::Bitcast => format!("bitcast<{t}>({x})"),
                UnaryElementOperator::Convert => format!("{t}({x})"),
            };
            stack.push(e);
        }
        ElementOperator::Binary(b) => {
            let rhs = stack.pop().expect("rpn bin rhs");
            let lhs = stack.pop().expect("rpn bin lhs");
            let e = match b {
                BinaryElementOperator::Pow => format!("pow({lhs}, {rhs})"),
                BinaryElementOperator::Div => format!("(({lhs}) / ({rhs}))"),
                BinaryElementOperator::Sub => format!("(({lhs}) - ({rhs}))"),
                BinaryElementOperator::Assoc(BinaryAssocElementOperator::Mul) => {
                    format!("(({lhs}) * ({rhs}))")
                }
                BinaryElementOperator::Assoc(BinaryAssocElementOperator::Add) => {
                    format!("(({lhs}) + ({rhs}))")
                }
                BinaryElementOperator::Assoc(BinaryAssocElementOperator::Max) => {
                    format!("max({lhs}, {rhs})")
                }
                BinaryElementOperator::Assoc(BinaryAssocElementOperator::Min) => {
                    format!("min({lhs}, {rhs})")
                }
            };
            stack.push(e);
        }
        ElementOperator::Compare(c) => {
            let rhs = stack.pop().expect("rpn cmp rhs");
            let lhs = stack.pop().expect("rpn cmp lhs");
            let pred = match c {
                BinaryCompareOperator::Eq => format!("({lhs}) == ({rhs})"),
                BinaryCompareOperator::Ne => format!("({lhs}) != ({rhs})"),
                BinaryCompareOperator::Gt => format!("({lhs}) > ({rhs})"),
                BinaryCompareOperator::Lt => format!("({lhs}) < ({rhs})"),
                BinaryCompareOperator::Ge => format!("({lhs}) >= ({rhs})"),
                BinaryCompareOperator::Le => format!("({lhs}) <= ({rhs})"),
            };
            stack.push(format!("select({t}(0), {t}(1), {pred})"));
        }
        ElementOperator::Bitwise(b) => {
            let rhs = stack.pop().expect("rpn bit rhs");
            let lhs = stack.pop().expect("rpn bit lhs");
            let e = match b {
                BinaryBitwiseOperator::Band => format!("(({lhs}) & ({rhs}))"),
                BinaryBitwiseOperator::Bor => format!("(({lhs}) | ({rhs}))"),
                BinaryBitwiseOperator::Bxor => format!("(({lhs}) ^ ({rhs}))"),
                BinaryBitwiseOperator::Shl => format!("(({lhs}) << ({rhs}))"),
                BinaryBitwiseOperator::Shr => format!("(({lhs}) >> ({rhs}))"),
            };
            stack.push(e);
        }
    }
}

/// Loop over `thread_shape` elements with a literal element-count guard
/// (used by scatter kernels, whose thread count differs from the output size).
fn per_thread_element<F: Fn(&mut WgslWriter, &str)>(
    w: &mut WgslWriter,
    thread_shape: &[u32],
    config: &WgslKernelConfig,
    body: F,
) {
    let total = shape_len(thread_shape);
    let items_per_thread = 1u32 << config.lg2_items_per_thread;
    w.block(
        &format!(
            "@compute @workgroup_size({})\nfn main(\n@builtin(global_invocation_id) global_id: vec3<u32>\n)",
            config.workgroup_size
        ),
        |w| {
            w.print(&format!(
                "let t_beg = global_id.x << {}u;\nlet t_end = t_beg + {items_per_thread}u;",
                config.lg2_items_per_thread
            ));
            w.block(
                "for (\nvar t = t_beg;\nt < t_end;\nt += 1u\n)",
                |w| {
                    w.block(&format!("if (t >= {total}u)"), |w| {
                        w.print("return;");
                    });
                    body(w, "t");
                },
            );
        },
    );
}

/// `fn {name}(index) -> u32`: dense C-contiguous address into `shape`.
fn define_dense_address_function(w: &mut WgslWriter, name: &str, shape: &[u32]) {
    let accessor = Accessor::dense(shape, 0);
    define_address_function(w, name, &accessor);
}

fn emit_remap(
    w: &mut WgslWriter,
    kernel: &IrRemapKernel,
    config: &WgslKernelConfig,
) -> Result<(), WgpuLowerError> {
    let t = spell_etype(kernel.element_type)?;
    match &kernel.info {
        RemapInfo::GatherRows => {
            let rank = kernel.shape.len();
            let src_rows = kernel.arg_accessors[0].shape[0];
            emit_bindings(w, kernel.element_type, &kernel.arg_element_types)?;
            emit_arg_address_functions(w, &kernel.arg_accessors);
            per_output_element(w, &kernel.shape, config, |w, out_addr| {
                w.print(&format!(
                    "let row = min(arg1[{}], {}u);",
                    arg_address_expr(1, &kernel.arg_accessors[1], "array<u32, 1>(out_index[0])"),
                    src_rows.saturating_sub(1),
                ));
                w.print(&format!("var src_index: array<u32, {rank}> = out_index;"));
                w.print("src_index[0] = row;");
                w.print(&format!(
                    "output[{out_addr}] = arg0[{}];",
                    arg_address_expr(0, &kernel.arg_accessors[0], "src_index")
                ));
            });
        }
        RemapInfo::ScatterRows { operator } => {
            let src_shape = kernel.arg_accessors[0].shape.clone();
            let rank = src_shape.len();
            let out_rows = kernel.shape[0];
            let atomic_output = operator.is_some();
            if atomic_output {
                // Atomic accumulation binds the output as atomic<u32> words
                // (f32 adds go through a compare-exchange loop on the bits).
                w.print(
                    "@group(0) @binding(0)\nvar<storage, read_write> output: array<atomic<u32>>;",
                );
                for (i, arg_etype) in kernel.arg_element_types.iter().enumerate() {
                    let arg_t = spell_etype(*arg_etype)?;
                    w.print(&format!(
                        "@group(0) @binding({})\nvar<storage, read> arg{i}: array<{arg_t}>;",
                        i + 1
                    ));
                }
            } else {
                emit_bindings(w, kernel.element_type, &kernel.arg_element_types)?;
            }
            emit_arg_address_functions(w, &kernel.arg_accessors);
            define_cc_index_function(w, "src_index_of", &src_shape);
            define_dense_address_function(w, "out_address_of", &kernel.shape);
            if atomic_output && kernel.element_type == ElementType::F4 {
                w.block("fn atomic_add_f32(addr: u32, value: f32)", |w| {
                    w.print("var old = atomicLoad(&output[addr]);");
                    w.block("loop", |w| {
                        w.print(
                            "let new_bits = bitcast<u32>(bitcast<f32>(old) + value);\nlet result = atomicCompareExchangeWeak(&output[addr], old, new_bits);",
                        );
                        w.block("if (result.exchanged)", |w| {
                            w.print("break;");
                        });
                        w.print("old = result.old_value;");
                    });
                });
            }
            per_thread_element(w, &src_shape, config, |w, thread| {
                w.print(&format!("var s_index = src_index_of({thread});"));
                w.print(&format!(
                    "let row = arg1[{}];",
                    arg_address_expr(1, &kernel.arg_accessors[1], "array<u32, 1>(s_index[0])")
                ));
                w.block(&format!("if (row < {out_rows}u)"), |w| {
                    w.print(&format!("var o_index: array<u32, {rank}> = s_index;"));
                    w.print("o_index[0] = row;");
                    w.print("let out_addr = out_address_of(o_index);");
                    let src_value = format!(
                        "arg0[{}]",
                        arg_address_expr(0, &kernel.arg_accessors[0], "s_index")
                    );
                    match (operator, kernel.element_type) {
                        (None, _) => w.print(&format!("output[out_addr] = {src_value};")),
                        (Some(_), ElementType::U4) => {
                            w.print(&format!("atomicAdd(&output[out_addr], {src_value});"))
                        }
                        (Some(_), _) => {
                            w.print(&format!("atomic_add_f32(out_addr, {src_value});"))
                        }
                    }
                });
            });
        }
        RemapInfo::ScatterView { accessor } => {
            let src_shape = kernel.arg_accessors[0].shape.clone();
            emit_bindings(w, kernel.element_type, &kernel.arg_element_types)?;
            emit_arg_address_functions(w, &kernel.arg_accessors);
            define_cc_index_function(w, "src_index_of", &src_shape);
            define_address_function(w, "out_address_of", accessor);
            per_thread_element(w, &src_shape, config, |w, thread| {
                if src_shape.is_empty() {
                    w.print(&format!(
                        "let _ = {thread};\noutput[out_address_of()] = arg0[{}];",
                        arg_address_expr(0, &kernel.arg_accessors[0], "")
                    ));
                } else {
                    w.print(&format!("var s_index = src_index_of({thread});"));
                    w.print(&format!(
                        "output[out_address_of(s_index)] = arg0[{}];",
                        arg_address_expr(0, &kernel.arg_accessors[0], "s_index")
                    ));
                }
            });
        }
    }
    let _ = t;
    Ok(())
}

fn emit_matmul(
    w: &mut WgslWriter,
    kernel: &IrMatmulKernel,
    config: &WgslKernelConfig,
) -> Result<(), WgpuLowerError> {
    let t = spell_etype(kernel.element_type)?;
    let rank = kernel.shape.len();
    if rank < 2 {
        return Err(WgpuLowerError::Message("matmul rank < 2".into()));
    }
    let k_dim = kernel.arg_accessors[0].shape[rank - 1];
    let arg_etypes = vec![kernel.element_type; 2];
    emit_bindings(w, kernel.element_type, &arg_etypes)?;
    emit_arg_address_functions(w, &kernel.arg_accessors);
    define_matmul_arg_index(w, "arg0_index", rank, rank - 1, Some(rank - 2), None);
    define_matmul_arg_index(w, "arg1_index", rank, rank - 2, None, Some(rank - 1));
    per_output_element(w, &kernel.shape, config, |w, out_addr| {
        w.print(&format!("var sum: {t} = {t}(0);"));
        w.block(
            &format!("for (var ki: u32 = 0u; ki < {k_dim}u; ki += 1u)"),
            |w| {
                w.print(&format!(
                    "let a_idx = arg0_index(out_index, ki);\nlet b_idx = arg1_index(out_index, ki);\nsum += arg0[{}] * arg1[{}];",
                    arg_address_expr(0, &kernel.arg_accessors[0], "a_idx"),
                    arg_address_expr(1, &kernel.arg_accessors[1], "b_idx"),
                ));
            },
        );
        w.print(&format!("output[{out_addr}] = sum;"));
    });
    Ok(())
}

fn define_matmul_arg_index(
    w: &mut WgslWriter,
    name: &str,
    rank: usize,
    k_axis: usize,
    m_axis: Option<usize>,
    n_axis: Option<usize>,
) {
    w.block(
        &format!("fn {name}(out_index: array<u32, {rank}>, k: u32) -> array<u32, {rank}>"),
        |w| {
            w.print(&format!("var idx: array<u32, {rank}>;"));
            for i in 0..rank.saturating_sub(2) {
                w.print(&format!("idx[{i}] = out_index[{i}];"));
            }
            if let Some(m) = m_axis {
                w.print(&format!("idx[{m}] = out_index[{m}];"));
            }
            if let Some(n) = n_axis {
                w.print(&format!("idx[{n}] = out_index[{n}];"));
            }
            w.print(&format!("idx[{k_axis}] = k;"));
            w.print("return idx;");
        },
    );
}

fn emit_reduction(
    w: &mut WgslWriter,
    kernel: &IrReductionKernel,
    config: &WgslKernelConfig,
) -> Result<(), WgpuLowerError> {
    let t = spell_etype(kernel.element_type)?;
    let rank = kernel.shape.len();
    let input_shape = &kernel.arg_accessors[0].shape;
    let count: u32 = kernel
        .axes
        .iter()
        .map(|&ax| input_shape[ax as usize])
        .product();
    let arg_etypes = vec![kernel.element_type];
    emit_bindings(w, kernel.element_type, &arg_etypes)?;
    emit_arg_address_functions(w, &kernel.arg_accessors);
    define_reduction_input_index(w, "input_index", rank, input_shape, &kernel.axes);
    per_output_element(w, &kernel.shape, config, |w, out_addr| {
        if count == 0 {
            w.print(&format!("output[{out_addr}] = {t}(0);"));
            return;
        }
        w.print(&format!(
            "var acc: {t} = arg0[{}];",
            arg_address_expr(0, &kernel.arg_accessors[0], "input_index(out_index, 0u)")
        ));
        if count > 1 {
            w.block(
                &format!("for (var ri: u32 = 1u; ri < {count}u; ri += 1u)"),
                |w| {
                    w.print(&format!(
                        "let v = arg0[{}];",
                        arg_address_expr(0, &kernel.arg_accessors[0], "input_index(out_index, ri)")
                    ));
                    w.print(&reduction_acc(kernel.operator, "acc", "v"));
                },
            );
        }
        w.print(&format!("output[{out_addr}] = acc;"));
    });
    Ok(())
}

fn define_reduction_input_index(
    w: &mut WgslWriter,
    name: &str,
    rank: usize,
    input_shape: &[u32],
    axes: &[u32],
) {
    let axes_set: std::collections::HashSet<u32> = axes.iter().copied().collect();
    let mut sorted_axes: Vec<u32> = axes.to_vec();
    sorted_axes.sort_unstable();
    w.block(
        &format!("fn {name}(out_index: array<u32, {rank}>, ri: u32) -> array<u32, {rank}>"),
        |w| {
            w.print(&format!("var idx: array<u32, {rank}>;"));
            for i in 0..rank {
                if !axes_set.contains(&(i as u32)) {
                    w.print(&format!("idx[{i}] = out_index[{i}];"));
                }
            }
            if !sorted_axes.is_empty() {
                w.print("var remaining: u32 = ri;");
                for &axis in sorted_axes.iter().rev() {
                    let dim = input_shape[axis as usize];
                    w.print(&format!("idx[{axis}] = remaining % {dim}u;"));
                    w.print(&format!("remaining = remaining / {dim}u;"));
                }
            }
            w.print("return idx;");
        },
    );
}

fn reduction_acc(op: BinaryAssocElementOperator, acc: &str, value: &str) -> String {
    match op {
        BinaryAssocElementOperator::Add => format!("{acc} += {value};"),
        BinaryAssocElementOperator::Mul => format!("{acc} *= {value};"),
        BinaryAssocElementOperator::Max => format!("{acc} = max({acc}, {value});"),
        BinaryAssocElementOperator::Min => format!("{acc} = min({acc}, {value});"),
    }
}

#[cfg(test)]
mod tests {
    use resin_core::{Accessor, BinaryElementOperator, ElementOperator, F4};
    use resin_ir::{ElementRpnExpr, IrElementwiseRpnKernel, RpnAtom};

    use super::*;

    #[test]
    fn elementwise_wgsl_contains_eval() {
        let kernel = IrKernel::ElementwiseRpn(IrElementwiseRpnKernel {
            arg_accessors: vec![Accessor::dense([4], 0), Accessor::dense([4], 0)],
            arg_element_types: vec![F4, F4],
            element_type: F4,
            shape: Box::from([4]),
            rpn_expr: ElementRpnExpr {
                atoms: vec![
                    RpnAtom::Arg(0),
                    RpnAtom::Arg(1),
                    RpnAtom::Op(ElementOperator::Binary(BinaryElementOperator::Assoc(
                        BinaryAssocElementOperator::Add,
                    ))),
                ],
            },
            clear_output_before_dispatch: false,
        });
        let wgsl = emit_wgsl_for_kernel(&kernel, &WgslKernelConfig::default()).unwrap();
        assert!(wgsl.contains("eval_rpn_expr"), "{wgsl}");
        assert!(wgsl.contains("f32"), "{wgsl}");
        assert!(wgsl.contains("@compute"), "{wgsl}");
    }

    #[test]
    fn convert_emits_value_cast() {
        let kernel = IrKernel::ElementwiseRpn(IrElementwiseRpnKernel {
            arg_accessors: vec![Accessor::dense([4], 0)],
            arg_element_types: vec![resin_core::U4],
            element_type: F4,
            shape: Box::from([4]),
            rpn_expr: ElementRpnExpr {
                atoms: vec![
                    RpnAtom::Arg(0),
                    RpnAtom::Op(ElementOperator::Unary(UnaryElementOperator::Convert)),
                ],
            },
            clear_output_before_dispatch: false,
        });
        let wgsl = emit_wgsl_for_kernel(&kernel, &WgslKernelConfig::default()).unwrap();
        assert!(wgsl.contains("f32(a0)"), "{wgsl}");
    }

    fn gather_kernel() -> IrKernel {
        IrKernel::Remap(IrRemapKernel {
            arg_accessors: vec![Accessor::dense([8, 2], 0), Accessor::dense([4], 0)],
            arg_element_types: vec![F4, resin_core::U4],
            element_type: F4,
            shape: Box::from([4, 2]),
            info: RemapInfo::GatherRows,
            clear_output_before_dispatch: false,
        })
    }

    fn scatter_kernel(operator: Option<BinaryAssocElementOperator>, etype: ElementType) -> IrKernel {
        IrKernel::Remap(IrRemapKernel {
            arg_accessors: vec![Accessor::dense([4, 2], 0), Accessor::dense([4], 0)],
            arg_element_types: vec![etype, resin_core::U4],
            element_type: etype,
            shape: Box::from([8, 2]),
            info: RemapInfo::ScatterRows { operator },
            clear_output_before_dispatch: true,
        })
    }

    #[test]
    fn gather_rows_clamps_indices() {
        let wgsl = emit_wgsl_for_kernel(&gather_kernel(), &WgslKernelConfig::default()).unwrap();
        assert!(wgsl.contains("min(arg1["), "{wgsl}");
        assert!(wgsl.contains("@compute"), "{wgsl}");
    }

    #[test]
    fn scatter_add_f32_uses_cas_loop() {
        let kernel = scatter_kernel(Some(BinaryAssocElementOperator::Add), F4);
        let wgsl = emit_wgsl_for_kernel(&kernel, &WgslKernelConfig::default()).unwrap();
        assert!(wgsl.contains("array<atomic<u32>>"), "{wgsl}");
        assert!(wgsl.contains("atomicCompareExchangeWeak"), "{wgsl}");
    }

    #[test]
    fn scatter_add_u32_uses_atomic_add() {
        let kernel = scatter_kernel(Some(BinaryAssocElementOperator::Add), resin_core::U4);
        let wgsl = emit_wgsl_for_kernel(&kernel, &WgslKernelConfig::default()).unwrap();
        assert!(wgsl.contains("atomicAdd"), "{wgsl}");
    }

    #[test]
    fn scatter_write_is_plain_store() {
        let kernel = scatter_kernel(None, F4);
        let wgsl = emit_wgsl_for_kernel(&kernel, &WgslKernelConfig::default()).unwrap();
        assert!(!wgsl.contains("atomic"), "{wgsl}");
        assert!(wgsl.contains("output[out_addr] ="), "{wgsl}");
    }

    #[test]
    fn scatter_dispatch_iterates_source() {
        // Source has 4×2 = 8 elements; output has 8×2 = 16. Threads follow the source.
        let config = WgslKernelConfig {
            lg2_items_per_thread: 0,
            workgroup_size: 1,
        };
        let kernel = scatter_kernel(None, F4);
        assert_eq!(dispatch_size_for_kernel(&kernel, &config), [8, 1, 1]);
        assert_eq!(dispatch_size_for_kernel(&gather_kernel(), &config), [8, 1, 1]);
    }

    #[test]
    fn relu_emits_max() {
        let kernel = IrKernel::ElementwiseRpn(IrElementwiseRpnKernel {
            arg_accessors: vec![Accessor::dense([8], 0)],
            arg_element_types: vec![F4],
            element_type: F4,
            shape: Box::from([8]),
            rpn_expr: ElementRpnExpr {
                atoms: vec![
                    RpnAtom::Arg(0),
                    RpnAtom::Op(ElementOperator::Unary(UnaryElementOperator::Relu)),
                ],
            },
            clear_output_before_dispatch: false,
        });
        let wgsl = emit_wgsl_for_kernel(&kernel, &WgslKernelConfig::default()).unwrap();
        assert!(wgsl.contains("max("), "{wgsl}");
    }
}
