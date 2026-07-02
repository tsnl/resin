//! WGSL emission for IR kernels.
//!
//! Prefer multi-line string literals in `WgslWriter::print` / `block`; the writer
//! trims shared leading indentation (simple dedent) so shaders stay readable.
//! A structured naga AST would be clearer still — follow-up, not this port.

use resin_core::{c_contiguous_pitch_for_shape, Accessor, ElementOperator, ElementType};
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
        IrKernel::ElementwiseRpn(k) => k.shape.iter().map(|&d| u64::from(d)).product(),
        IrKernel::Matmul(k) => k.shape.iter().map(|&d| u64::from(d)).product(),
        IrKernel::Reduction(k) => k.shape.iter().map(|&d| u64::from(d)).product(),
        IrKernel::Remap(k) => k.shape.iter().map(|&d| u64::from(d)).product(),
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
    let etype = match kernel {
        IrKernel::ElementwiseRpn(k) => k.etype,
        IrKernel::Matmul(k) => k.etype,
        IrKernel::Reduction(k) => k.etype,
        IrKernel::Remap(k) => k.etype,
    };
    let mut w = WgslWriter::new(etype.needs_enable_f16());
    match kernel {
        IrKernel::ElementwiseRpn(k) => emit_elementwise(&mut w, k, config),
        IrKernel::Matmul(k) => emit_matmul(&mut w, k, config),
        IrKernel::Reduction(k) => emit_reduction(&mut w, k, config),
        IrKernel::Remap(k) => emit_remap(&mut w, k, config),
    }
    w.finish()
}

/// Strip the minimum common leading whitespace from non-empty lines (and leading/trailing blank lines).
fn dedent(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let min_indent = lines
        .iter()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.len() - l.trim_start().len())
        .min()
        .unwrap_or(0);
    lines
        .iter()
        .map(|l| {
            if l.trim().is_empty() {
                ""
            } else {
                l.get(min_indent..).unwrap_or(l.trim_start())
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

/// Line-oriented WGSL source builder (single growable buffer, not `Vec<String>` lines).
struct StringBuilder {
    buf: String,
}

impl StringBuilder {
    fn new(enable_f16: bool) -> Self {
        let mut buf = String::new();
        if enable_f16 {
            buf.push_str("enable f16;\n");
        }
        Self { buf }
    }

    fn print(&mut self, text: &str) {
        let chunk = dedent(text);
        if !self.buf.is_empty() && !self.buf.ends_with('\n') {
            self.buf.push('\n');
        }
        self.buf.push_str(&chunk);
        if !self.buf.ends_with('\n') {
            self.buf.push('\n');
        }
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
        self.buf.trim_end().to_string()
    }
}

type WgslWriter = StringBuilder;

fn emit_bindings(w: &mut WgslWriter, etype: ElementType, arg_etypes: &[ElementType]) {
    let t = etype.wgsl_name();
    w.print(&format!(
        "@group(0) @binding(0)\nvar<storage, read_write> output: array<{t}>;"
    ));
    for (i, arg_etype) in arg_etypes.iter().enumerate() {
        let arg_t = arg_etype.wgsl_name();
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
    w.block(
        &format!("fn {name}(address: u32) -> array<u32, {n}>"),
        |w| {
            w.print("var addr: u32 = address;");
            w.print(&format!("var index: array<u32, {n}>;"));
            for i in 0..n {
                w.print(&format!("index[{i}] = addr / {}u;  // dim {i}", pitch[i]));
                w.print(&format!("addr = addr % {}u;", pitch[i]));
            }
            w.print("return index;");
        },
    );
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
) {
    emit_bindings(w, kernel.etype, &kernel.arg_etypes);
    emit_arg_address_functions(w, &kernel.arg_accessors);
    emit_eval_rpn(w, &kernel.rpn_expr, kernel.etype, &kernel.arg_etypes);
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
    let t = etype.wgsl_name();
    let params: Vec<String> = arg_etypes
        .iter()
        .enumerate()
        .map(|(i, e)| format!("a{i}: {}", e.wgsl_name()))
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
        // unary
        ElementOperator::Relu
        | ElementOperator::Neg
        | ElementOperator::Exp
        | ElementOperator::Log
        | ElementOperator::Sqrt
        | ElementOperator::Sin
        | ElementOperator::Cos
        | ElementOperator::Not
        | ElementOperator::Floor
        | ElementOperator::Ceil
        | ElementOperator::Bitcast => {
            let x = stack.pop().expect("rpn unary");
            let e = match op {
                ElementOperator::Relu => format!("max({x}, {t}(0))"),
                ElementOperator::Neg => format!("(-{x})"),
                ElementOperator::Exp => format!("(exp({x}))"),
                ElementOperator::Log => format!("(log({x}))"),
                ElementOperator::Sqrt => format!("(sqrt({x}))"),
                ElementOperator::Sin => format!("(sin({x}))"),
                ElementOperator::Cos => format!("(cos({x}))"),
                ElementOperator::Not => format!("(abs(1.0 - {x}))"),
                ElementOperator::Floor => format!("floor({x})"),
                ElementOperator::Ceil => format!("ceil({x})"),
                ElementOperator::Bitcast => format!("bitcast<{t}>({x})"),
                _ => unreachable!(),
            };
            stack.push(e);
        }
        // binary arithmetic
        ElementOperator::Add
        | ElementOperator::Sub
        | ElementOperator::Mul
        | ElementOperator::Div
        | ElementOperator::Pow
        | ElementOperator::Max
        | ElementOperator::Min => {
            let rhs = stack.pop().expect("rpn bin rhs");
            let lhs = stack.pop().expect("rpn bin lhs");
            let e = match op {
                ElementOperator::Pow => format!("pow({lhs}, {rhs})"),
                ElementOperator::Div => format!("({lhs} / {rhs})"),
                ElementOperator::Sub => format!("({lhs} - {rhs})"),
                ElementOperator::Mul => format!("({lhs} * {rhs})"),
                ElementOperator::Add => format!("({lhs} + {rhs})"),
                ElementOperator::Max => format!("max({lhs}, {rhs})"),
                ElementOperator::Min => format!("min({lhs}, {rhs})"),
                _ => unreachable!(),
            };
            stack.push(e);
        }
        // binary comparison
        ElementOperator::Eq
        | ElementOperator::Ne
        | ElementOperator::Gt
        | ElementOperator::Lt
        | ElementOperator::Ge
        | ElementOperator::Le => {
            let rhs = stack.pop().expect("rpn cmp rhs");
            let lhs = stack.pop().expect("rpn cmp lhs");
            let pred = match op {
                ElementOperator::Eq => format!("{lhs} == {rhs}"),
                ElementOperator::Ne => format!("{lhs} != {rhs}"),
                ElementOperator::Gt => format!("{lhs} > {rhs}"),
                ElementOperator::Lt => format!("{lhs} < {rhs}"),
                ElementOperator::Ge => format!("{lhs} >= {rhs}"),
                ElementOperator::Le => format!("{lhs} <= {rhs}"),
                _ => unreachable!(),
            };
            stack.push(format!("select({t}(0), {t}(1), {pred})"));
        }
        // binary bitwise
        ElementOperator::Band
        | ElementOperator::Bor
        | ElementOperator::Bxor
        | ElementOperator::Shl
        | ElementOperator::Shr => {
            let rhs = stack.pop().expect("rpn bit rhs");
            let lhs = stack.pop().expect("rpn bit lhs");
            let e = match op {
                ElementOperator::Band => format!("({lhs} & {rhs})"),
                ElementOperator::Bor => format!("({lhs} | {rhs})"),
                ElementOperator::Bxor => format!("({lhs} ^ {rhs})"),
                ElementOperator::Shl => format!("({lhs} << {rhs})"),
                ElementOperator::Shr => format!("({lhs} >> {rhs})"),
                _ => unreachable!(),
            };
            stack.push(e);
        }
    }
}

fn emit_matmul(w: &mut WgslWriter, kernel: &IrMatmulKernel, config: &WgslKernelConfig) {
    let t = kernel.etype.wgsl_name();
    let rank = kernel.shape.len();
    let k = kernel.arg_accessors[0].shape[rank - 1];
    emit_bindings(w, kernel.etype, &[kernel.etype, kernel.etype]);
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
    let t = kernel.etype.wgsl_name();
    let rank = kernel.shape.len();
    let input_shape = &kernel.arg_accessors[0].shape;
    let count: u32 = kernel
        .axes
        .iter()
        .map(|&ax| input_shape[ax as usize])
        .product();
    emit_bindings(w, kernel.etype, &[kernel.etype]);
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

fn reduction_acc(op: ElementOperator, acc: &str, value: &str) -> String {
    match op {
        ElementOperator::Add => format!("{acc} += {value};"),
        ElementOperator::Mul => format!("{acc} *= {value};"),
        ElementOperator::Max => format!("{acc} = max({acc}, {value});"),
        ElementOperator::Min => format!("{acc} = min({acc}, {value});"),
        other => panic!("reduction expects associative op, got {other:?}"),
    }
}

fn emit_remap_bindings(
    w: &mut WgslWriter,
    output_etype: ElementType,
    arg_etypes: &[ElementType],
    operator: Option<ElementOperator>,
) {
    let t = output_etype.wgsl_name();
    if operator.is_none() {
        w.print(&format!(
            "@group(0) @binding(0)\nvar<storage, read_write> output: array<{t}>;"
        ));
    } else {
        // CAS loop bitcasts atomic<u32> <-> f32; only F4 is implemented.
        assert!(
            output_etype == ElementType::F4,
            "scatter-accumulate Remap only supports F4 output (got {output_etype:?}); \
             atomic bitcast path is f32-specific"
        );
        w.print("@group(0) @binding(0)\nvar<storage, read_write> output: array<atomic<u32>>;");
    }
    for (i, arg_etype) in arg_etypes.iter().enumerate() {
        let arg_t = arg_etype.wgsl_name();
        w.print(&format!(
            "@group(0) @binding({})\nvar<storage, read> arg{i}: array<{arg_t}>;",
            i + 1
        ));
    }
}

/// f32 CAS accumulate into `atomic<u32>` (caller must ensure output etype is F4).
fn scatter_atomic(op: ElementOperator, out_addr: &str, value: &str) -> String {
    let combine = match op {
        ElementOperator::Add => format!("old_val + ({value})"),
        ElementOperator::Mul => format!("old_val * ({value})"),
        ElementOperator::Max => format!("max(old_val, {value})"),
        ElementOperator::Min => format!("min(old_val, {value})"),
        other => panic!("scatter accumulate expects associative op, got {other:?}"),
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
    operator: Option<ElementOperator>,
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
    operator: Option<ElementOperator>,
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
            w.block(
                &format!("for (var k: u32 = 0u; k < {out_rank}u; k += 1u)"),
                |w| {
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
                },
            );
            w.print("return result;");
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use resin_core::{ElementOperator, F4};
    use resin_ir::{ElementRpnExpr, IrElementwiseRpnKernel, RpnAtom};

    #[test]
    fn elementwise_wgsl_contains_eval() {
        let kernel = IrKernel::ElementwiseRpn(IrElementwiseRpnKernel {
            arg_accessors: vec![
                Accessor::new_c_contiguous([4], 0),
                Accessor::new_c_contiguous([4], 0),
            ],
            arg_etypes: vec![F4, F4],
            etype: F4,
            shape: Box::new([4]),
            rpn_expr: ElementRpnExpr {
                atoms: vec![
                    RpnAtom::Arg(0),
                    RpnAtom::Arg(1),
                    RpnAtom::Op(ElementOperator::Add),
                ],
            },
            clear_output_before_dispatch: false,
        });
        let wgsl = emit_wgsl_for_kernel(&kernel, &WgslKernelConfig::default());
        assert!(wgsl.contains("eval_rpn_expr"), "{wgsl}");
        assert!(wgsl.contains("f32"), "{wgsl}");
    }
}
