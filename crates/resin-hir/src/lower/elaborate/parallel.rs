//! Establish capture and iteration ownership before parallel regions leave HIR.
use super::*;

pub(super) struct Captures {
    pub enclosing: BTreeSet<DeclarationId>,
    pub used: BTreeSet<DeclarationId>,
}

impl Completion<'_> {
    pub(super) fn captured_place(&self, term: &Term) -> bool {
        let Some(captures) = self.parallel_captures.last() else {
            return false;
        };
        capture_root(term).is_some_and(|binding| captures.enclosing.contains(&binding))
    }

    pub(super) fn parallel_input(&mut self, source: &typed::Term) -> Result<Box<Term>> {
        let input = self.boxed(source)?;
        let crate::Type::Array { element, .. } = &input.ty else {
            return Err(GenerateError::inference(
                source.span,
                "parallel blocks require an inline array",
            ));
        };
        self.require_copy(
            element,
            element,
            source.span,
            "parallel block input elements must be copyable",
        )?;
        Ok(input)
    }

    pub(super) fn parallel_body(
        &mut self,
        parameters: &[&typed::ParallelParameter],
        source: &typed::Term,
    ) -> Result<(
        Vec<crate::ParallelParameter>,
        Box<Term>,
        Vec<crate::BindingId>,
    )> {
        let entry = self.state();
        let reachable = self.reachable;
        self.parallel_captures.push(Captures {
            enclosing: self.initialization.keys().copied().collect(),
            used: BTreeSet::new(),
        });
        let result = (|| {
            let parameters = parameters
                .iter()
                .map(|parameter| {
                    self.initialization
                        .insert(parameter.binding, Initialization::Initialized);
                    self.written.insert(parameter.binding);
                    self.mutable
                        .insert(parameter.binding, parameter.pattern.mutable);
                    Ok(crate::ParallelParameter {
                        binding: parameter.binding,
                        name: parameter.pattern.name.clone(),
                        ty: self
                            .solver
                            .require_complete(&parameter.ty, parameter.pattern.name.span)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let body = self.boxed(source)?;
            if matches!(body.ty, crate::Type::Reference { .. }) {
                return Err(GenerateError::inference(
                    source.span,
                    "parallel block results must be values, not references",
                ));
            }
            Ok((parameters, body))
        })();
        let captures = self
            .parallel_captures
            .pop()
            .expect("current parallel region");
        self.restore(entry);
        self.reachable = reachable;
        result.map(|(parameters, body)| (parameters, body, captures.used.into_iter().collect()))
    }
}

// A raw pointer introduces its own access contract. Capturing its value does
// not freeze its target; this check is deliberately not a general effect check.
fn capture_root(term: &Term) -> Option<DeclarationId> {
    match &term.kind {
        TermKind::Local { binding, .. } => Some(*binding),
        TermKind::Use { arg } => capture_root(arg),
        TermKind::Field { base, .. } if !pointer_access(&base.ty) => capture_root(base),
        _ => None,
    }
}

fn pointer_access(ty: &crate::Type) -> bool {
    match ty {
        crate::Type::Pointer { .. } => true,
        crate::Type::Reference { referent, .. } => pointer_access(referent),
        _ => false,
    }
}
