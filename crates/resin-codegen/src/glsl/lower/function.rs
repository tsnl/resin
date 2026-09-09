use crate::glsl::{GlslFunction, GlslStatement};
use resin_lir::FunctionTypes;
use resin_types::prelude::*;
use std::fmt::Write;

use crate::Error;
use resin_lir::{Function, Instr, Terminator};

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
    let mut lowering = FunctionLowering {
        types,
        function,
        flow,
        inputs,
        index,
    };
    let statements = lowering.region(function.entry.index(), None)?;
    Ok(GlslFunction {
        signature,
        locals,
        statements,
    })
}

#[derive(Clone, Copy)]
enum YieldTarget {
    Block { target: usize },
    Condition { body: usize, test: usize },
}

struct FunctionLowering<'a, 'm> {
    types: &'a mut Types<'m>,
    function: &'a Function,
    flow: &'a FunctionTypes,
    inputs: Vec<Vec<Slot>>,
    index: usize,
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

impl FunctionLowering<'_, '_> {
    // Continuations are a sequence, so only child regions recurse.
    fn region(
        &mut self,
        mut b: usize,
        yield_target: Option<YieldTarget>,
    ) -> Result<Vec<GlslStatement>, Error> {
        let mut statements = Vec::new();
        loop {
            let types = &mut *self.types;
            let function = self.function;
            let flow = self.flow;
            let index = self.index;
            let block = &function.blocks[b];
            let mut out = String::new();
            let mut stack = self.inputs[b].clone();
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
                let args = stack.split_off(
                    stack.len() - flow.operand_count(resin_lir::BlockId::from_index(b), i),
                );
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

            statements.push(GlslStatement::Text { source: out });
            if diverged {
                return Ok(statements);
            }
            let next = self.exit(b, stack, yield_target, &mut statements)?;
            let Some(next) = next else {
                return Ok(statements);
            };
            b = next;
        }
    }

    fn exit(
        &mut self,
        b: usize,
        mut stack: Vec<Slot>,
        yield_target: Option<YieldTarget>,
        statements: &mut Vec<GlslStatement>,
    ) -> Result<Option<usize>, Error> {
        let next = match self.function.blocks[b].terminator {
            Terminator::Return => {
                if stack[0].local {
                    return Err(Error::at(
                        self.types.module,
                        self.index,
                        Some((b, self.function.blocks[b].instrs.len())),
                        Error("shader cannot return a local address".into()),
                    ));
                }
                statements.push(GlslStatement::Return {
                    value: stack[0].expr.clone(),
                });
                None
            }
            Terminator::Yield => {
                statements.push(GlslStatement::Text {
                    source: self.yield_values(&stack, yield_target.unwrap()),
                });
                None
            }
            Terminator::If { then, els, next } => {
                let condition = stack.pop().unwrap();
                let target = next
                    .map(|id| YieldTarget::Block { target: id.index() })
                    .or(yield_target);
                statements.push(GlslStatement::If {
                    condition: self.types.unwrap(&condition.ty, condition.expr),
                    then: self.enter(then.index(), &stack, target)?,
                    els: self.enter(els.index(), &stack, target)?,
                });
                next
            }
            Terminator::Loop {
                condition,
                body,
                next,
            } => {
                let condition = condition.index();
                let body = body.index();
                statements.push(GlslStatement::Text {
                    source: self.transfer(condition, &stack),
                });
                statements.push(GlslStatement::Text {
                    source: format!("      bool r_test{b} = false;\n"),
                });
                let mut repeated =
                    self.region(condition, Some(YieldTarget::Condition { body, test: b }))?;
                repeated.push(GlslStatement::If {
                    condition: format!("!r_test{b}"),
                    then: vec![GlslStatement::Break],
                    els: vec![],
                });
                repeated.extend(self.region(body, Some(YieldTarget::Block { target: condition }))?);
                statements.push(GlslStatement::Loop { body: repeated });
                // These slots hold the final condition's operands on the false exit.
                let output = self.inputs[body].clone();
                if let Some(next) = next {
                    statements.push(GlslStatement::Text {
                        source: self.transfer(next.index(), &output),
                    });
                } else {
                    statements.push(GlslStatement::Text {
                        source: self.yield_values(&output, yield_target.unwrap()),
                    });
                }
                next
            }
        };
        Ok(next.map(|id| id.index()))
    }

    fn enter(
        &mut self,
        block: usize,
        stack: &[Slot],
        target: Option<YieldTarget>,
    ) -> Result<Vec<GlslStatement>, Error> {
        let mut statements = vec![GlslStatement::Text {
            source: self.transfer(block, stack),
        }];
        statements.extend(self.region(block, target)?);
        Ok(statements)
    }

    fn yield_values(&self, stack: &[Slot], target: YieldTarget) -> String {
        match target {
            YieldTarget::Block { target } => self.transfer(target, stack),
            YieldTarget::Condition { body, test } => {
                let (condition, operands) = stack.split_last().unwrap();
                let mut source = self.transfer(body, operands);
                writeln!(
                    source,
                    "      r_test{test} = {};",
                    self.types.unwrap(&condition.ty, condition.expr.clone())
                )
                .unwrap();
                source
            }
        }
    }

    fn transfer(&self, target: usize, stack: &[Slot]) -> String {
        if stack.iter().all(Slot::symbolic) {
            return String::new();
        }
        let mut source = String::from("      {\n");
        for (i, value) in stack
            .iter()
            .enumerate()
            .filter(|(_, slot)| !slot.symbolic())
        {
            writeln!(
                source,
                "        {} edge{i} = {};",
                self.types.name(&value.ty),
                value.expr
            )
            .unwrap();
        }
        for (i, _) in stack
            .iter()
            .enumerate()
            .filter(|(_, slot)| !slot.symbolic())
        {
            writeln!(source, "        r_b{target}_{i} = edge{i};").unwrap();
        }
        source.push_str("      }\n");
        source
    }
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
