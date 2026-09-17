//! Parallel blocks introduce ordinary local bindings, never callable closure values.
use super::*;

impl Expression<'_, '_> {
    pub(super) fn parallel_map(
        &mut self,
        input: &resin_ast::Term,
        pattern: &resin_ast::BindingPattern,
        body: &resin_ast::Term,
        span: Span,
        out: &Type,
    ) -> Result<TermKind> {
        let input = self.child(input, None);
        let element = self.checker.typing.solver.fresh();
        let (mut parameters, body) = self.parallel_body(&[pattern], &element, body, None)?;
        self.constrain((
            span,
            Constraint::ParallelArray {
                input: input.ty.clone(),
                element,
                mapped: Some((body.ty.clone(), out.clone())),
            },
        ));
        Ok(TermKind::ParallelMap {
            input: Box::new(input),
            element: parameters.remove(0),
            body: Box::new(body),
        })
    }

    pub(super) fn parallel_reduce(
        &mut self,
        input: &resin_ast::Term,
        identity: &resin_ast::Term,
        patterns: [&resin_ast::BindingPattern; 2],
        body: &resin_ast::Term,
        span: Span,
    ) -> Result<(TermKind, Type)> {
        let input = self.child(input, None);
        let element = self.checker.typing.solver.fresh();
        let identity = self.child(identity, Some(element.clone()));
        let (parameters, body) =
            self.parallel_body(&patterns, &element, body, Some(element.clone()))?;
        self.constrain((
            span,
            Constraint::ParallelArray {
                input: input.ty.clone(),
                element: element.clone(),
                mapped: None,
            },
        ));
        let [left, right]: [_; 2] = parameters.try_into().unwrap();
        Ok((
            TermKind::ParallelReduce {
                input: Box::new(input),
                identity: Box::new(identity),
                left,
                right,
                body: Box::new(body),
            },
            element,
        ))
    }

    fn parallel_body(
        &mut self,
        patterns: &[&resin_ast::BindingPattern],
        ty: &Type,
        body: &resin_ast::Term,
        expected: Option<Type>,
    ) -> Result<(Vec<typed::ParallelParameter>, Term)> {
        self.checker.scopes.push_at(body.span);
        let depth = std::mem::replace(&mut self.checker.loop_depth, 0);
        self.checker.parallel_depth += 1;
        let result = (|| {
            let parameters = patterns
                .iter()
                .map(|pattern| {
                    Ok(typed::ParallelParameter {
                        binding: self.checker.bind(
                            &pattern.name,
                            ty.clone(),
                            DefinitionKind::Variable,
                        )?,
                        pattern: (*pattern).clone(),
                        ty: ty.clone(),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok((parameters, self.child(body, expected)))
        })();
        self.checker.parallel_depth -= 1;
        self.checker.loop_depth = depth;
        self.checker.scopes.pop();
        result
    }
}
