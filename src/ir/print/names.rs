use std::{collections::HashSet, sync::Arc};

use crate::ir::{Function, Module};

pub(super) struct Names {
    pub(super) types: Vec<Arc<str>>,
    pub(super) globals: Vec<Arc<str>>,
    pub(super) functions: Vec<Arc<str>>,
}

pub(super) struct FunctionNames {
    pub(super) locals: Vec<Arc<str>>,
    pub(super) blocks: Vec<Arc<str>>,
}

impl Names {
    pub(super) fn new(module: &Module) -> Self {
        Self {
            types: uniquify(
                module
                    .types
                    .iter()
                    .map(|def| Some(def.name.clone()))
                    .collect(),
                "type",
            ),
            globals: uniquify(
                module
                    .globals
                    .iter()
                    .map(|global| Some(global.name.clone()))
                    .collect(),
                "g",
            ),
            functions: uniquify(
                module
                    .functions
                    .iter()
                    .map(|function| function.name.clone())
                    .collect(),
                "fn",
            ),
        }
    }
}

impl FunctionNames {
    pub(super) fn new(function: &Function) -> Self {
        Self {
            locals: uniquify(
                function
                    .locals
                    .iter()
                    .map(|local| local.name.clone())
                    .collect(),
                "l",
            ),
            blocks: uniquify(
                function
                    .blocks
                    .iter()
                    .map(|block| block.name.clone())
                    .collect(),
                "b",
            ),
        }
    }
}

fn uniquify(preferred: Vec<Option<Arc<str>>>, fallback_prefix: &str) -> Vec<Arc<str>> {
    let mut used = HashSet::new();
    let mut names = Vec::with_capacity(preferred.len());
    for (index, name) in preferred.into_iter().enumerate() {
        let mut candidate: Arc<str> =
            name.unwrap_or_else(|| format!("{fallback_prefix}.{index}").into());
        if used.contains(&candidate) {
            let base = candidate.clone();
            let mut suffix = 1;
            loop {
                let next: Arc<str> = format!("{base}.{suffix}").into();
                if !used.contains(&next) {
                    candidate = next;
                    break;
                }
                suffix += 1;
            }
        }
        used.insert(candidate.clone());
        names.push(candidate);
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_names_and_fallbacks_share_one_namespace() {
        let names = uniquify(
            vec![
                Some("l.1".into()),
                None,
                Some("l.1".into()),
                Some("l.1.1".into()),
            ],
            "l",
        );
        let names: Vec<_> = names.iter().map(AsRef::as_ref).collect();
        assert_eq!(names, ["l.1", "l.1.1", "l.1.2", "l.1.1.1"]);
    }
}
