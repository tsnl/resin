//! Cleanup partitions for locals with statically addressed field moves.
//! Every partition has one live flag. A whole-value transfer changes every flag;
//! a field transfer changes only the partitions it contains.
use super::types::Types;
use resin_lir::{Function, Instr};
use resin_types::prelude::*;
use std::fmt::Write;

struct Part {
    path: Vec<usize>,
    ty: Ty,
}

fn parts(types: &Types<'_>, function: &Function, local: LocalId) -> Vec<Part> {
    let paths = function
        .blocks
        .iter()
        .flat_map(|block| &block.instrs)
        .filter_map(|instr| match instr {
            Instr::TakeField {
                local: target,
                path,
            } if *target == local => Some(path.as_slice()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut result = Vec::new();
    partition(
        types,
        &function.locals[local.index()].ty,
        &[],
        &paths,
        &mut result,
    );
    result
}

fn partition(
    types: &Types<'_>,
    ty: &Ty,
    path: &[usize],
    paths: &[&[usize]],
    result: &mut Vec<Part>,
) {
    if !ty.needs_drop(&types.module.types) {
        return;
    }
    let splits = paths
        .iter()
        .any(|other| other.starts_with(path) && other.len() > path.len());
    if splits && let Ty::Record { fields } = types.shape(ty) {
        for (index, field) in fields.iter().enumerate() {
            let mut child = path.to_vec();
            child.push(index);
            partition(types, &field.ty, &child, paths, result);
        }
    } else {
        result.push(Part {
            path: path.to_vec(),
            ty: ty.clone(),
        });
    }
}

fn flag(local: LocalId, path: &[usize]) -> String {
    let mut name = format!("r_live{}", local.index());
    for index in path {
        write!(name, "_{index}").unwrap();
    }
    name
}

pub(super) fn declarations(types: &Types<'_>, function: &Function, out: &mut String) {
    for index in 0..function.locals.len() {
        let local = LocalId::from_index(index);
        for part in parts(types, function, local)
            .iter()
            .filter(|part| !part.path.is_empty())
        {
            writeln!(
                out,
                "  bool {} = {};",
                flag(local, &part.path),
                index < function.parameter_count
            )
            .unwrap();
        }
    }
}

pub(super) fn project(
    types: &Types<'_>,
    function: &Function,
    local: LocalId,
    path: &[usize],
) -> String {
    let mut ty = &function.locals[local.index()].ty;
    let mut expression = format!("r_l{}", local.index());
    for index in path {
        expression = types.unwrap(ty, expression);
        let Ty::Record { fields } = types.shape(ty) else {
            unreachable!("verified owned field path");
        };
        expression = format!("({expression}).f{index}");
        ty = &fields[*index].ty;
    }
    expression
}

pub(super) fn mark(
    types: &Types<'_>,
    function: &Function,
    local: LocalId,
    path: &[usize],
    live: bool,
    out: &mut String,
) {
    for part in parts(types, function, local)
        .iter()
        .filter(|part| part.path.starts_with(path))
    {
        writeln!(out, "  {} = {live};", flag(local, &part.path)).unwrap();
    }
    if path.is_empty()
        && function.locals[local.index()]
            .ty
            .needs_drop(&types.module.types)
    {
        writeln!(out, "  r_live{} = {live};", local.index()).unwrap();
    }
}

pub(super) fn drop(
    types: &Types<'_>,
    function: &Function,
    local: LocalId,
    path: &[usize],
    out: &mut String,
) {
    let parts = parts(types, function, local);
    if parts
        .iter()
        .any(|part| path.starts_with(&part.path) && path.len() > part.path.len())
    {
        let mut ty = &function.locals[local.index()].ty;
        for index in path {
            let Ty::Record { fields } = types.shape(ty) else {
                unreachable!("verified field path");
            };
            ty = &fields[*index].ty;
        }
        types.drop_value(ty, &project(types, function, local, path), out);
        return;
    }
    for part in parts
        .iter()
        .rev()
        .filter(|part| part.path.starts_with(path))
    {
        let flag = flag(local, &part.path);
        writeln!(out, "  if ({flag}) {{").unwrap();
        types.drop_value(&part.ty, &project(types, function, local, &part.path), out);
        writeln!(out, "    {flag} = false; }}").unwrap();
    }
}
