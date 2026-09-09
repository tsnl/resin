use crate::{GlslBlock, GlslEdge, GlslEdgeValue, GlslExit, GlslFunction};
use resin_lir_verifier::FunctionTypes;
use std::fmt::Write;

use crate::Error;
use resin_lir::{Case, Function, Instr, Terminator, Ty};

use super::{
    Slot,
    ops::{instruction, is_variant},
    symbols,
    types::Types,
};

pub(super) fn lower(
    types: &mut Types<'_>,
    function: &Function,
    flow: &FunctionTypes,
    name: &str,
    index: usize,
) -> Result<GlslFunction, Error> {
    let inputs = symbols::inputs(types, function, flow)?;
    let signature = format!(
        "{} {name}({} arg)",
        types.name(&function.result),
        types.name(&function.locals[0].ty)
    );
    let locals = locals(types, function, flow);
    let blocks = function
        .blocks
        .iter()
        .enumerate()
        .map(|(b, _)| lower_block(types, function, flow, index, b, &inputs[b]))
        .collect::<Result<_, _>>()?;
    Ok(GlslFunction {
        signature,
        locals,
        entry: function.entry.index(),
        blocks,
        default_result: types.zero(&function.result),
    })
}

fn locals(types: &Types<'_>, function: &Function, flow: &FunctionTypes) -> String {
    let mut out = String::new();
    for (i, local) in function.locals.iter().enumerate() {
        writeln!(
            out,
            "  {} r_l{i} = {};",
            types.name(&local.ty),
            types.zero(&local.ty)
        )
        .unwrap();
    }
    writeln!(out, "  r_l0 = arg;").unwrap();
    for (b, inputs) in flow.inputs.iter().enumerate() {
        for (i, ty) in inputs.iter().enumerate() {
            if matches!(ty, Ty::Function { .. }) {
                continue;
            }
            writeln!(out, "  {} r_b{b}_{i};", types.name(ty)).unwrap();
        }
        for (i, ty) in flow.results[b].iter().enumerate() {
            if let Some(ty) = ty
                && !matches!(ty, Ty::Function { .. })
            {
                writeln!(out, "  {} r_v{b}_{i};", types.name(ty)).unwrap();
            }
        }
    }
    out
}

fn lower_block(
    types: &mut Types<'_>,
    function: &Function,
    flow: &FunctionTypes,
    index: usize,
    b: usize,
    inputs: &[Slot],
) -> Result<GlslBlock, Error> {
    let block = &function.blocks[b];
    let mut out = String::new();
    let mut stack = inputs.to_vec();
    let mut diverged = false;
    for (i, instr) in block.instrs.iter().enumerate() {
        if matches!(instr, Instr::Eliminate { .. }) {
            writeln!(
                out,
                "      r_failed = true; return {};",
                types.zero(&function.result)
            )
            .unwrap();
            diverged = true;
            break;
        }
        let args = stack.split_off(stack.len() - instr.stack_effect().pops);
        let result = flow.results[b][i].as_ref();
        check_instruction(types, instr, &args, result)
            .map_err(|error| Error::at(types.module, index, Some((b, i)), error))?;
        let local = Slot::local_result(instr, &args);
        runtime_checks(types, instr, &args, &function.result, &mut out);
        let expr = instruction(types, instr, &args, result, &mut out)
            .map_err(|error| Error::at(types.module, index, Some((b, i)), error))?;
        if let Some(ty) = result {
            let mut expr = expr.unwrap();
            if !local && !matches!(ty, Ty::Function { .. }) {
                let name = format!("r_v{b}_{i}");
                writeln!(out, "      {name} = {expr};").unwrap();
                expr = name;
            }
            if matches!(instr, Instr::Call) {
                writeln!(
                    out,
                    "      if (r_failed) return {};",
                    types.zero(&function.result)
                )
                .unwrap();
            }
            stack.push(Slot {
                ty: ty.clone(),
                expr,
                local,
            });
        }
    }

    let exit = if diverged {
        GlslExit::Unreachable
    } else {
        lower_exit(types, &block.terminator, &mut stack)
            .map_err(|e| Error::at(types.module, index, Some((b, block.instrs.len())), e))?
    };
    Ok(GlslBlock {
        label: b,
        statements: out,
        exit,
    })
}

fn lower_exit(
    types: &Types<'_>,
    term: &Terminator,
    stack: &mut Vec<Slot>,
) -> Result<GlslExit, Error> {
    Ok(match term {
        Terminator::Return => {
            if stack[0].local {
                return Err(Error("shader cannot return a local address".into()));
            }
            GlslExit::Return(stack[0].expr.clone())
        }
        Terminator::Break { target } => GlslExit::Jump(edge(types, target.index(), stack)),
        Terminator::Branch { then, els } => {
            let condition = stack.pop().unwrap();
            GlslExit::Branch {
                condition: types.unwrap(&condition.ty, condition.expr),
                then: edge(types, then.index(), stack),
                els: edge(types, els.index(), stack),
            }
        }
    })
}

fn edge(types: &Types<'_>, target: usize, stack: &[Slot]) -> GlslEdge {
    let values = stack
        .iter()
        .enumerate()
        .filter(|(_, slot)| !slot.symbolic())
        .map(|(slot, value)| GlslEdgeValue {
            slot,
            ty: types.name(&value.ty),
            value: value.expr.clone(),
        })
        .collect();
    GlslEdge { target, values }
}

fn check_instruction(
    types: &Types<'_>,
    instr: &Instr,
    args: &[Slot],
    result: Option<&Ty>,
) -> Result<(), Error> {
    if args
        .iter()
        .enumerate()
        .any(|(i, arg)| arg.local && !permits_local_address(instr, i))
    {
        return Err(Error(
            "shader-local addresses cannot escape through values, casts, or calls".into(),
        ));
    }
    let managed = args
        .iter()
        .map(|arg| &arg.ty)
        .chain(result)
        .any(|ty| ty.needs_drop(&types.module.types));
    if managed || manages_ownership(instr) {
        return Err(Error(
            "shader cannot consume a managed value or invoke automatic destruction".into(),
        ));
    }
    Ok(())
}

fn permits_local_address(instr: &Instr, operand: usize) -> bool {
    matches!(instr, Instr::Discard)
        || operand == 0
            && matches!(
                instr,
                Instr::Load
                    | Instr::TransferLoad
                    | Instr::IsVariant { .. }
                    | Instr::Store
                    | Instr::Replace
                    | Instr::AccessStatic { .. }
                    | Instr::AccessDynamic
            )
}

fn manages_ownership(instr: &Instr) -> bool {
    matches!(
        instr,
        Instr::ArcNew
            | Instr::ArcData
            | Instr::Downgrade
            | Instr::Upgrade
            | Instr::WeakEmpty { .. }
            | Instr::DropLocal { .. }
    )
}

fn runtime_checks(types: &Types<'_>, instr: &Instr, args: &[Slot], result: &Ty, out: &mut String) {
    let invalid = match instr {
        Instr::ExcludeNone => Some(is_variant(
            types,
            &args[0].ty,
            &Case::Type(Ty::None),
            &args[0].expr,
        )),
        Instr::NumericCast { ty } => crate::numeric::invalid(
            &args[0].ty,
            ty,
            &args[0].expr,
            |from, value| format!("{}({value})", types.name(from)),
            "trunc",
        ),
        _ => None,
    };
    if let Some(invalid) = invalid {
        writeln!(
            out,
            "      if ({invalid}) {{ r_failed = true; return {}; }}",
            types.zero(result)
        )
        .unwrap();
    }
}
