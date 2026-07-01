//! WGSL emission for IR kernels.

use resin_core::{
    etype_needs_enable_f16, spell_etype_in_wgsl, Accessor, BinaryAssocElementOperator,
    BinaryBitwiseOperator, BinaryCompareOperator, BinaryElementOperator, ElementOperator,
    ElementType, UnaryElementOperator, c_contiguous_pitch_for_shape,
};
use resin_dsl::{RemapGatherInfo, RemapInfo, RemapScatterInfo};
use resin_ir::{
    ElementRpnExpr, IrElementwiseRpnKernel, IrKernel, IrMatmulKernel, IrReductionKernel,
    IrRemapKernel, RpnAtom,
};

#[derive(Debug, Clone, Copy)]
pub struct WgslKernelConfig {
    pub lg2_items_per_thread: u32,
    pub workgroup_size: u32,
}

impl Default for WgslKernelConfig {
    fn default() -> Self {
        Self {
            lg2_items_per_thread: 3,
            workgroup_size: 8,
        }
    }
}

pub fn dispatch_size_for_kernel(kernel: &IrKernel, config: &WgslKernelConfig) -> [u32; 3] {
    let n: u64 = match kernel {
        IrKernel::Remap(IrRemapKernel {
            info: RemapInfo::Scatter(_),
            arg_accessors,
            ..
        }) => arg_accessors[0]
            .shape
            .iter()
            .map(|&d| u64::from(d))
            .product(),
        _ => kernel.shape().iter().map(|&d| u64::from(d)).product(),
    };
    if n == 0 {
        return [0, 1, 1];
    }
    let items_per_thread = 1u64 << config.lg2_items_per_thread;
    let threads = n.div_ceil(items_per_thread);
    let workgroups_x = threads.div_ceil(u64::from(config.workgroup_size));
    [workgroups_x as u32, 1, 1]
}

pub fn emit_wgsl_for_kernel(kernel: &IrKernel, config: &WgslKernelConfig) -> String {
    let mut w = WgslWriter::new(etype_needs_enable_f16(kernel.etype()));
    match kernel {
        IrKernel::ElementwiseRpn(k) => emit_elementwise(&mut w, k, config),
        IrKernel::Matmul(k) => emit_matmul(&mut w, k, config),
        IrKernel::Reduction(k) => emit_reduction(&mut w, k, config),
        IrKernel::Remap(k) => emit_remap(&mut w, k, config),
    }
    w.finish()
}

struct WgslWriter {
    lines: Vec<String>,
}

impl WgslWriter {
    fn new(enable_f16: bool) -> Self {
        let mut lines = Vec::new();
        if enable_f16 {
            lines.push("enable f16;".into());
        }
        Self { lines }
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

fn emit_bindings(w: &mut WgslWriter, kernel: &IrKernel) {
    let t = spell_etype_in_wgsl(kernel.etype());
    w.print(&format!(
        "@group(0) @binding(0)\nvar<storage, read_write> output: array<{t}>;"
    ));
    for (i, arg_etype) in kernel.operand_etypes().iter().enumerate() {
        let arg_t = spell_etype_in_wgsl(*arg_etype);
        w.print(&format!(
            "@group(0) @binding({})\nvar<storage, read> arg{i}: array<{arg_t}>;",
            i + 1
        ));
    }
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
                "acc += index[{i}] * {}u;  // dim {i}",
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
            w.print(&format!("index[{i}] = addr / {}u;  // dim {i}", pitch[i]));
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

fn emit_elementwise(w: &mut WgslWriter, kernel: &IrElementwiseRpnKernel, config: &WgslKernelConfig) {
    let k = IrKernel::ElementwiseRpn(kernel.clone());
    emit_bindings(w, &k);
    emit_arg_address_functions(w, &kernel.arg_accessors);
    emit_eval_rpn(
        w,
        &kernel.rpn_expr,
        kernel.etype,
        &kernel.arg_etypes,
    );
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
}

fn emit_eval_rpn(
    w: &mut WgslWriter,
    rpn: &ElementRpnExpr,
    etype: ElementType,
    arg_etypes: &[ElementType],
) {
    let t = spell_etype_in_wgsl(etype);
    let params: Vec<String> = arg_etypes
        .iter()
        .enumerate()
        .map(|(i, e)| format!("a{i}: {}", spell_etype_in_wgsl(*e)))
        .collect();
    w.block(
        &format!("fn eval_rpn_expr({}) -> {t}", params.join(", ")),
        |w| {
            let mut stack: Vec<String> = Vec::new();
            for atom in &rpn.atoms {
                match atom {
                    RpnAtom::Arg(i) => stack.push(format!("a{i}")),
                    RpnAtom::Op(op) => apply_op(&mut stack, *op, t),
                }
            }
            assert_eq!(stack.len(), 1);
            w.print(&format!("return {};", stack[0]));
        },
    );
}

fn apply_op(stack: &mut Vec<String>, op: ElementOperator, t: &str) {
    match op {
        ElementOperator::Unary(u) => {
            let x = stack.pop().expect("rpn unary");
            let e = match u {
                UnaryElementOperator::Neg => format!("(-{x})"),
                UnaryElementOperator::Exp => format!("(exp({x}))"),
                UnaryElementOperator::Log => format!("(log({x}))"),
                UnaryElementOperator::Sqrt => format!("(sqrt({x}))"),
                UnaryElementOperator::Sin => format!("(sin({x}))"),
                UnaryElementOperator::Cos => format!("(cos({x}))"),
                UnaryElementOperator::Not => format!("(abs(1.0 - {x}))"),
                UnaryElementOperator::Floor => format!("floor({x})"),
                UnaryElementOperator::Ceil => format!("ceil({x})"),
                UnaryElementOperator::Bitcast => format!("bitcast<{t}>({x})"),
            };
            stack.push(e);
        }
        ElementOperator::Binary(b) => {
            let rhs = stack.pop().expect("rpn bin rhs");
            let lhs = stack.pop().expect("rpn bin lhs");
            let e = match b {
                BinaryElementOperator::Pow => format!("pow({lhs}, {rhs})"),
                BinaryElementOperator::Div => format!("({lhs} / {rhs})"),
                BinaryElementOperator::Sub => format!("({lhs} - {rhs})"),
                BinaryElementOperator::Assoc(BinaryAssocElementOperator::Mul) => {
                    format!("({lhs} * {rhs})")
                }
                BinaryElementOperator::Assoc(BinaryAssocElementOperator::Add) => {
                    format!("({lhs} + {rhs})")
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
                BinaryCompareOperator::Eq => format!("{lhs} == {rhs}"),
                BinaryCompareOperator::Ne => format!("{lhs} != {rhs}"),
                BinaryCompareOperator::Gt => format!("{lhs} > {rhs}"),
                BinaryCompareOperator::Lt => format!("{lhs} < {rhs}"),
                BinaryCompareOperator::Ge => format!("{lhs} >= {rhs}"),
                BinaryCompareOperator::Le => format!("{lhs} <= {rhs}"),
            };
            stack.push(format!("select({t}(0), {t}(1), {pred})"));
        }
        ElementOperator::Bitwise(b) => {
            let rhs = stack.pop().expect("rpn bit rhs");
            let lhs = stack.pop().expect("rpn bit lhs");
            let e = match b {
                BinaryBitwiseOperator::Band => format!("({lhs} & {rhs})"),
                BinaryBitwiseOperator::Bor => format!("({lhs} | {rhs})"),
                BinaryBitwiseOperator::Bxor => format!("({lhs} ^ {rhs})"),
                BinaryBitwiseOperator::Shl => format!("({lhs} << {rhs})"),
                BinaryBitwiseOperator::Shr => format!("({lhs} >> {rhs})"),
            };
            stack.push(e);
        }
    }
}

fn emit_matmul(w: &mut WgslWriter, kernel: &IrMatmulKernel, config: &WgslKernelConfig) {
    let t = spell_etype_in_wgsl(kernel.etype);
    let rank = kernel.shape.len();
    let k = kernel.arg_accessors[0].shape[rank - 1];
    let as_kernel = IrKernel::Matmul(kernel.clone());
    emit_bindings(w, &as_kernel);
    emit_arg_address_functions(w, &kernel.arg_accessors);
    define_matmul_arg_index(w, "arg0_index", rank, rank - 1, Some(rank - 2), None);
    define_matmul_arg_index(w, "arg1_index", rank, rank - 2, None, Some(rank - 1));
    per_output_element(w, &kernel.shape, config, |w, out_addr| {
        w.print(&format!("var sum: {t} = {t}(0);"));
        w.block(&format!("for (var ki: u32 = 0u; ki < {k}u; ki += 1u)"), |w| {
            w.print(&format!(
                "let a_idx = arg0_index(out_index, ki);\nlet b_idx = arg1_index(out_index, ki);\nsum += arg0[{}] * arg1[{}];",
                arg_address_expr(0, &kernel.arg_accessors[0], "a_idx"),
                arg_address_expr(1, &kernel.arg_accessors[1], "b_idx"),
            ));
        });
        w.print(&format!("output[{out_addr}] = sum;"));
    });
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

fn emit_reduction(w: &mut WgslWriter, kernel: &IrReductionKernel, config: &WgslKernelConfig) {
    let t = spell_etype_in_wgsl(kernel.etype);
    let rank = kernel.shape.len();
    let input_shape = &kernel.arg_accessors[0].shape;
    let count: u32 = kernel
        .axes
        .iter()
        .map(|&ax| input_shape[ax as usize])
        .product();
    let as_kernel = IrKernel::Reduction(kernel.clone());
    emit_bindings(w, &as_kernel);
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

fn emit_remap_bindings(
    w: &mut WgslWriter,
    output_etype: ElementType,
    arg_etypes: &[ElementType],
    operator: Option<BinaryAssocElementOperator>,
) {
    let t = spell_etype_in_wgsl(output_etype);
    if operator.is_none() {
        w.print(&format!(
            "@group(0) @binding(0)\nvar<storage, read_write> output: array<{t}>;"
        ));
    } else {
        w.print(
            "@group(0) @binding(0)\nvar<storage, read_write> output: array<atomic<u32>>;",
        );
    }
    for (i, arg_etype) in arg_etypes.iter().enumerate() {
        let arg_t = spell_etype_in_wgsl(*arg_etype);
        w.print(&format!(
            "@group(0) @binding({})\nvar<storage, read> arg{i}: array<{arg_t}>;",
            i + 1
        ));
    }
}

fn scatter_atomic(op: BinaryAssocElementOperator, out_addr: &str, value: &str) -> String {
    let combine = match op {
        BinaryAssocElementOperator::Add => format!("old_val + ({value})"),
        BinaryAssocElementOperator::Mul => format!("old_val * ({value})"),
        BinaryAssocElementOperator::Max => format!("max(old_val, {value})"),
        BinaryAssocElementOperator::Min => format!("min(old_val, {value})"),
    };
    format!(
        "{{
let out_slot = &output[{out_addr}];
loop {{
let old_bits = atomicLoad(out_slot);
let old_val = bitcast<f32>(old_bits);
let new_val = {combine};
let new_bits = bitcast<u32>(new_val);
let exchanged = atomicCompareExchangeWeak(out_slot, old_bits, new_bits).exchanged;
if (exchanged) {{ break; }}
}}
}}"
    )
}

fn emit_remap_store(
    w: &mut WgslWriter,
    operator: Option<BinaryAssocElementOperator>,
    out_addr: &str,
    value: &str,
) {
    match operator {
        None => w.print(&format!("output[{out_addr}] = {value};")),
        Some(op) => w.print(&scatter_atomic(op, out_addr, value)),
    }
}

fn define_out_address(w: &mut WgslWriter, name: &str, offset: u32, pitch: &[u32]) {
    let rank = pitch.len();
    w.block(
        &format!("fn {name}(index: array<u32, {rank}>) -> u32"),
        |w| {
            w.print(&format!("var acc: u32 = {offset}u;"));
            for (i, p) in pitch.iter().enumerate() {
                w.print(&format!("acc += index[{i}] * {p}u;"));
            }
            w.print("return acc;");
        },
    );
}

fn emit_source_driven_remap(
    w: &mut WgslWriter,
    kernel: &IrRemapKernel,
    config: &WgslKernelConfig,
    operator: Option<BinaryAssocElementOperator>,
    out_address_for_src_index: &str,
) {
    let source_accessor = &kernel.arg_accessors[0];
    let source_shape = &source_accessor.shape;
    let rank = source_shape.len();
    emit_remap_bindings(w, kernel.etype, &kernel.arg_etypes, operator);
    emit_arg_address_functions(w, &kernel.arg_accessors);

    if rank == 0 {
        w.block(
            "@compute @workgroup_size(1)\nfn main(\n@builtin(global_invocation_id) global_id: vec3<u32>\n)",
            |w| {
                w.block("if (global_id.x > 0u)", |w| w.print("return;"));
                let value = format!(
                    "arg0[{}]",
                    arg_address_expr(0, source_accessor, "")
                );
                // rank 0 address helpers
                let value = if source_accessor.rank() == 0 {
                    "arg0[address_arg0()]".to_string()
                } else {
                    value
                };
                emit_remap_store(w, operator, out_address_for_src_index, &value);
            },
        );
        return;
    }

    define_cc_index_function(w, "source_index", source_shape);
    let source_count: u32 = source_shape.iter().product();
    let items_per_thread = 1u32 << config.lg2_items_per_thread;
    w.block(
        &format!(
            "@compute @workgroup_size({})\nfn main(\n@builtin(global_invocation_id) global_id: vec3<u32>\n)",
            config.workgroup_size
        ),
        |w| {
            w.print(&format!(
                "let src_address_beg = global_id.x << {}u;\nlet src_address_end = src_address_beg + {items_per_thread}u;",
                config.lg2_items_per_thread
            ));
            w.block(
                "for (\nvar src_address = src_address_beg;\nsrc_address < src_address_end;\nsrc_address += 1u\n)",
                |w| {
                    w.block(&format!("if (src_address >= {source_count}u)"), |w| {
                        w.print("return;");
                    });
                    w.print("let src_index = source_index(src_address);");
                    let value = format!(
                        "arg0[{}]",
                        arg_address_expr(0, source_accessor, "src_index")
                    );
                    let out = out_address_for_src_index.replace("SRC_INDEX", "src_index");
                    emit_remap_store(w, operator, &out, &value);
                },
            );
        },
    );
}

fn emit_remap(w: &mut WgslWriter, kernel: &IrRemapKernel, config: &WgslKernelConfig) {
    match &kernel.info {
        RemapInfo::Scatter(RemapScatterInfo {
            accessor: Some(accessor),
            operator,
        }) => {
            if !accessor.pitch.is_empty() {
                define_out_address(w, "scatter_out_address", accessor.offset, &accessor.pitch);
            }
            let out_expr = if kernel.arg_accessors[0].shape.is_empty() {
                format!("{}u", accessor.offset)
            } else {
                "scatter_out_address(SRC_INDEX)".to_string()
            };
            emit_source_driven_remap(w, kernel, config, *operator, &out_expr);
        }
        RemapInfo::Scatter(RemapScatterInfo {
            accessor: None,
            operator,
        }) => {
            let out_pitch = c_contiguous_pitch_for_shape(&kernel.shape);
            define_out_address(w, "out_address_from_indices", 0, &out_pitch);
            // Indices-driven scatter: use contiguous out address from indices in arg1 — simplified:
            // treat as dense copy path for indices accessor scatter via source-driven with identity out
            // Full indices_at support: emit densify-style using out_address from multi-index in arg1.
            // For port v1 of indexed scatter, require indices and use out_address_from_indices(indices).
            let source_rank = kernel.arg_accessors[0].rank();
            let out_rank = kernel.shape.len();
            let indices_accessor = &kernel.arg_accessors[1];
            define_indices_at(w, source_rank, out_rank, indices_accessor);
            let out_expr = if source_rank == 0 {
                "out_address_from_indices(indices_at(array<u32, 0>()))".to_string()
            } else {
                "out_address_from_indices(indices_at(SRC_INDEX))".to_string()
            };
            emit_source_driven_remap(w, kernel, config, *operator, &out_expr);
        }
        RemapInfo::Gather(RemapGatherInfo {
            accessor: None,
            source_shape: None,
        })
        | RemapInfo::Gather(RemapGatherInfo {
            accessor: None,
            source_shape: Some(_),
        }) => {
            emit_remap_bindings(w, kernel.etype, &kernel.arg_etypes, None);
            emit_arg_address_functions(w, &kernel.arg_accessors);
            per_output_element(w, &kernel.shape, config, |w, out_addr| {
                w.print(&format!(
                    "output[{out_addr}] = arg0[{}];",
                    arg_address_expr(0, &kernel.arg_accessors[0], "out_index")
                ));
            });
        }
        RemapInfo::Gather(RemapGatherInfo {
            accessor: Some(accessor),
            ..
        }) => {
            let source_rank = kernel.shape.len();
            let out_rank = accessor.rank();
            let indices_accessor = &kernel.arg_accessors[1];
            emit_remap_bindings(w, kernel.etype, &kernel.arg_etypes, None);
            emit_arg_address_functions(w, &kernel.arg_accessors);
            define_indices_at(w, source_rank, out_rank, indices_accessor);
            define_out_address(
                w,
                "out_address_from_indices",
                accessor.offset,
                &accessor.pitch,
            );
            per_output_element(w, &kernel.shape, config, |w, out_addr| {
                let indices_expr = if source_rank == 0 {
                    "indices_at(array<u32, 0>())".to_string()
                } else {
                    "indices_at(out_index)".to_string()
                };
                w.print(&format!(
                    "output[{out_addr}] = arg0[out_address_from_indices({indices_expr})];"
                ));
            });
        }
    }
}

fn define_indices_at(
    w: &mut WgslWriter,
    source_rank: usize,
    out_rank: usize,
    indices_accessor: &Accessor,
) {
    let indices_rank = indices_accessor.rank();
    w.block(
        &format!(
            "fn indices_at(source_index: array<u32, {source_rank}>) -> array<u32, {out_rank}>"
        ),
        |w| {
            w.print(&format!("var result: array<u32, {out_rank}>;"));
            w.block(&format!("for (var k: u32 = 0u; k < {out_rank}u; k += 1u)"), |w| {
                if indices_rank == 0 {
                    w.print(&format!(
                        "result[k] = arg1[{}];",
                        arg_address_expr(1, indices_accessor, "")
                    ));
                } else {
                    w.print(&format!("var idx: array<u32, {indices_rank}>;"));
                    for d in 0..source_rank {
                        w.print(&format!("idx[{d}] = source_index[{d}];"));
                    }
                    w.print(&format!("idx[{source_rank}] = k;"));
                    w.print(&format!(
                        "result[k] = arg1[{}];",
                        arg_address_expr(1, indices_accessor, "idx")
                    ));
                }
            });
            w.print("return result;");
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use resin_core::{BinaryElementOperator, ElementOperator, F4};
    use resin_ir::{ElementRpnExpr, IrElementwiseRpnKernel, RpnAtom};

    #[test]
    fn elementwise_wgsl_contains_eval() {
        let kernel = IrKernel::ElementwiseRpn(IrElementwiseRpnKernel {
            arg_accessors: vec![
                Accessor::dense([4], 0),
                Accessor::dense([4], 0),
            ],
            arg_etypes: vec![F4, F4],
            etype: F4,
            shape: Box::new([4]),
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
        let wgsl = emit_wgsl_for_kernel(&kernel, &WgslKernelConfig::default());
        assert!(wgsl.contains("eval_rpn_expr"), "{wgsl}");
        assert!(wgsl.contains("f32"), "{wgsl}");
    }
}
